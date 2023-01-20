"""
This files contains a series of Wrapper classes for wrapping a transformer language model and
provides convenient methods for evaluation.
"""

import os
from typing import List, Dict, Tuple
from tqdm import tqdm, trange
import json
import jsonpickle

from preprocessor import MLMPreprocessor, SequenceClassifierPreprocessor
from utils import InputExample, InputFeatures, DictDataset
import log

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import BertConfig, AutoTokenizer, BertTokenizer, BertForSequenceClassification, \
    BertForMaskedLM, AdamW, get_linear_schedule_with_warmup, AutoConfig, AutoModelForSequenceClassification, \
    AutoModelForMaskedLM
from transformers.data.metrics import simple_accuracy

CONFIG_NAME = 'wrapper_config.json'

SEQUENCE_CLASSIFIER_WRAPPER = 'sequence_classifier'
MLM_WRAPPER = 'mlm'

WRAPPER_TYPES = [SEQUENCE_CLASSIFIER_WRAPPER, MLM_WRAPPER]

PREPROCESSORS = {
    SEQUENCE_CLASSIFIER_WRAPPER: SequenceClassifierPreprocessor,
    MLM_WRAPPER: MLMPreprocessor
}

MODEL_CLASSES = {
    'bert':{
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: BertForSequenceClassification,
        MLM_WRAPPER: BertForMaskedLM
    },
    'xlm-r':{
        'config': AutoConfig,
        'tokenizer': AutoTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: AutoModelForSequenceClassification,
        MLM_WRAPPER: AutoModelForMaskedLM
    }
}

logger = log.get_logger('root')

class WrapperConfig:
    """A configuration for a :class:'TransformerWrapper'."""

    def __init__(self, model_type: str, model_name_or_path: str, wrapper_type: str, task_name: str, max_seq_length: int,
                 label_list: List[str], pattern_id: int = 0, verbalizer_file=None, cache_dir: str = None, seed: str = 42):
        """
        Create a new config.

        :param model_type: the model type (e.g., 'bert' etc.)
        :param model_name_or_path: the model name or path (e.g., 'bert-base-multilingual-cased')
        :param wrapper_type: the wrapper type (e.g., 'mlm')
        :param task_name: the name of the task to solve
        :param max_seq_length: the maximum number of tokens in a sequence
        :param label_list: the list of labels for the task
        :param pattern_id: the id of pattern model to use
        :param verbalizer_file: optional path to a verbalizer file if different from the task default
        :param cache_dir: optional path to a cache dir
        :param seed: random seed for initialization
        """
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.wrapper_type = wrapper_type
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.label_list = label_list
        self.pattern_id = pattern_id
        self.verbalizer_file = verbalizer_file
        self.cache_dir = cache_dir
        self.seed = seed

class TransformerModelWrapper:
    """
    A wrapper for a Transformer-based language model.
    """

    def __init__(self, config: WrapperConfig):
        """Create a new wrapper from the given wrapper config."""
        self.config = config
        config_class = MODEL_CLASSES[self.config.model_type]['config']
        tokenizer_class = MODEL_CLASSES[self.config.model_type]['tokenizer']
        model_class = MODEL_CLASSES[self.config.model_type][self.config.wrapper_type]

        model_config = config_class.from_pretrained(
            config.model_name_or_path, num_labels=len(config.label_list), finetuning_task=config.task_name,
            cache_dir=config.cache_dir if config.cache_dir else None, use_cache=False
        )

        self.tokenizer = tokenizer_class.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.cache_dir if config.cache_dir else None
        )                    # Type: PreTrainedTokenizer

        self.model = model_class.from_pretrained(
            config.model_name_or_path, config=model_config,
            cache_dir=config.cache_dir if config.cache_dir else None
        )

        self.preprocessor = PREPROCESSORS[self.config.wrapper_type](self, self.config.task_name, self.config.pattern_id,
                                                                    self.config.verbalizer_file)

    @classmethod
    def from_pretrained(cls, path: str) -> 'TransformerModelWrapper':
        """Load a pretraining wrapper from a given path."""

        wrapper = TransformerModelWrapper.__new__(TransformerModelWrapper)
        wrapper.config = wrapper._load_config(path)
        tokenizer_class = MODEL_CLASSES[wrapper.config.model_type]['tokenizer']
        model_class = MODEL_CLASSES[wrapper.config.model_type][wrapper.config.wrapper_type]
        wrapper.model = model_class.from_pretrained(path)
        wrapper.tokenizer = tokenizer_class.from_pretrained(path)
        wrapper.preprocessor = PREPROCESSORS[wrapper.config.wrapper_type](
            wrapper, wrapper.config.task_name, wrapper.config.pattern_id, wrapper.config.verbalizer_file)

        return wrapper


    def self_predict(self, ori_data: List[InputExample], device, batch_size: int = 8) -> List[InputExample]:
        labeled_data = []

        pred_data = self._generate_dataset(ori_data)
        pred_sampler = SequentialSampler(pred_data)
        pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=batch_size)

        preds = None

        for batch in pred_dataloader:
            self.model.to(device)
            self.model.eval()
            batch = {k: t.to(device) for k, t in batch.items()}

            with torch.no_grad():
                logits = self.mlm_eval_step(batch)

            if preds is None:
                preds = logits.detach().cpu().numpy()

            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

        # turn preds into labels
        if self.config.task_name == 'product-review-polarity':
            labels = [str(np.argmax(pred)+1) for pred in preds]
        else:
            labels = [str(np.argmax(pred)) for pred in preds]

        # method 1
        for idx, (sent, label) in enumerate(zip(ori_data, labels)):
            labeled_data.append(InputExample(guid=idx, text_a=sent.text_a, label=label))

        return labeled_data

    def train(self, task_train_data: List[InputExample], device, train_batch_size: int = 8, num_train_epochs: int = 1,
              gradient_accumulation_steps: int = 1, weight_decay: float = 0.0, learning_rate: float = 1e-5,
              adam_epsilon: float = 1e-08, warmup_steps=0, max_grad_norm: float = 1.0, logging_steps: int = 50,
              max_steps = -1) -> Tuple[int, float]:
        """
        Train the language model

        :param task_train_data: the training examples to use
        :param device: the training device (cpu/gpu)
        :param train_batch_size: the number of training examples per batch
        :param num_train_epochs: the number of epochs to train
        :param gradient_accumulation_steps: the number of gradient accumulation steps before performing an update
        :param weight_decay: the weight decay to use
        :param learning_rate: the learning rate to use
        :param adam_epsilon: epsilon pamameter for Adam optimizer
        :param warmup_steps: the number of warmup steps
        :param max_grad_norm: the maximum norm for the gradient
        :param logging_steps: the number of steps after which logging information is printed
        :param max_steps: the maximum number of training steps, overrides num_train_epochs
        :return: a tuple consisting of the total number of the steps and the average training loss
        """

        train_dataset = self._generate_dataset(task_train_data)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        if max_steps > 0:
            t_total = max_steps
            num_train_epochs = max_steps // (max(1, len(train_dataloader) // gradient_accumulation_steps)) + 1

        else:
            t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

        # Prepare optimizer and schedule
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n,p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

        step, global_step = 0, 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()

        train_iterator = trange(num_train_epochs, desc='Epoch')

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc='Iteration')
            for _, batch in enumerate(epoch_iterator):
                self.model.train()

                batch = {k: t.to(device) for k, t in batch.items()}

                loss = self.mlm_train_step(batch=batch)

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    if logging_steps > 0 and global_step % logging_steps == 0:
                        logs = dict()
                        loss_scalar = (tr_loss - logging_loss) / logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        logs['learning_rate'] = learning_rate_scalar
                        logs['loss'] = loss_scalar
                        logging_loss = tr_loss

                        print(json.dumps({**logs, **{'step': global_step}}))

                if 0 < max_steps < global_step:
                    epoch_iterator.close()
                    break
                step += 1
            if 0 < max_steps < global_step:
                train_iterator.close()
                break

        return global_step, (tr_loss / global_step if global_step > 0 else -1)

    def eval(self, eval_data: List[InputExample], device, priming_idx: int = -1, batch_size: int = 8,
             priming: bool = False, num_priming: int = 1, conc:bool=False):
        """
        Evaluate the language model

        :param eval_data: the evaluation examples to use
        :param device:  the evaluation device (cpu/gpu)
        :param batch_size: the number of evaluation examples per batch
        :param primimng: wheather to use priming
        :return: a dictionary of numpy array containing the indices, logits, labels for each evaluation example
        """

        eval_data = self._generate_dataset(eval_data, priming_idx=priming_idx, priming=priming, num_priming=num_priming,
                                           conc=conc)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        preds = None

        for batch in tqdm(eval_dataloader, desc='Evaluating'):
            self.model.eval()

            batch = {k: t.to(device) for k, t in batch.items()}
            labels = batch['labels']
            indices = batch['idx']
            with torch.no_grad():
                logits = self.mlm_eval_step(batch)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
                all_indices = indices.detach().cpu().numpy()

            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
                all_indices = np.append(all_indices, indices.detach().cpu().numpy(), axis=0)

        predictions = np.argmax(preds, axis=1)

        temp_results = {}
        temp_results['indices'] = all_indices
        temp_results['acc'] = simple_accuracy(predictions, out_label_ids)
        temp_results['predictions'] = np.expand_dims(predictions, 0)
        temp_results['logits'] = preds
        temp_results['labels'] = out_label_ids

        return temp_results

    def save(self, path: str) -> None:
        """Save a pretrained model."""

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self._save_config(path)

    def _save_config(self, path: str) -> None:
        with open(os.path.join(path, CONFIG_NAME), 'w') as f:
            f.write(jsonpickle.encode(self.config))

    @staticmethod
    def _load_config(path: str) -> WrapperConfig:
        with open(os.path.join(path, CONFIG_NAME), 'r') as f:
            return jsonpickle.decode(f.read())

    def _generate_dataset(self, data: List[InputExample], priming_idx: int = -1, labelled: bool = True,
                          priming: bool = False, num_priming: int=0, conc:bool=False):
        features = self._convert_example_to_features(data, priming_idx=priming_idx, labelled=labelled, priming=priming,
                                                     num_priming=num_priming, conc=conc)
        feature_dict = {
            'input_ids': torch.tensor([f.input_ids for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            'token_type_ids': torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            'labels': torch.tensor([f.label for f in features], dtype=torch.long),
            'mlm_labels': torch.tensor([f.mlm_labels for f in features], dtype=torch.long),
            'idx': torch.tensor([int(f.idx) for f in features], dtype=torch.long)
        }

        return DictDataset(**feature_dict)

    def _convert_example_to_features(self, examples: List[InputExample], priming_idx: int = -1, labelled: bool = True,
                                     priming: bool = False, num_priming: int=0, conc:bool=False) -> List[InputFeatures]:
        features = []
        for (ex_index, example) in enumerate(examples):
            # if ex_index % 10000 == 0:
            #     logger.info(f"Writing example {ex_index}")
            input_features = self.preprocessor.get_input_features(example, labelled=labelled, priming_idx=priming_idx,
                                                                  priming=priming, num_priming=num_priming, conc=conc)
            features.append(input_features)

        return features

    def generate_default_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate the default inputs required by almost every language model."""
        inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
        if self.config.model_type in ['bert', 'xlnet']:
            inputs['token_type_ids'] = batch['token_type_ids']
        return inputs

    def mlm_train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform an MLM training step.
        :return: loss
        """

        inputs = self.generate_default_inputs(batch)
        mlm_labels, labels = batch['mlm_labels'], batch['labels']

        outputs = self.model(**inputs)
        prediction_scores = self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(mlm_labels, outputs[0])
        loss = nn.CrossEntropyLoss()(prediction_scores.view(-1, len(self.config.label_list)), labels.view(-1))
        return loss

    def mlm_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a MLM evaluation step."""
        inputs = self.generate_default_inputs(batch)
        outputs = self.model(**inputs)

        return self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(batch['mlm_labels'], outputs[0])


