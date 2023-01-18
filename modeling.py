# -*- coding: utf-8 -*-
# Author: Niel
# Date: 2022/6/29  11:57
import os.path
from abc import ABC
import json
from typing import List, Dict
from collections import defaultdict
import statistics

from wrapper import WrapperConfig, TransformerModelWrapper
from utils import InputExample, save_logits, save_evaluations, set_seed
from retrieve import add_priming_data
import log
from tqdm import tqdm

import torch
import numpy as np
from transformers.data.metrics import simple_accuracy

logger = log.get_logger('root')


class ModelingConfig(ABC):
    """Abstract class for a modeling configuration that can be saved and loaded from a json file."""

    def __repr__(self):
        return repr(self.__dict__)

    def save(self, path: str):
        """Save this config to a file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str):
        """Load a config from a file."""
        cfg = cls.__new__(cls)
        with open(path, 'r', encoding='utf-8') as f:
            cfg.__dict__ = json.load(f)
        return cfg


class TrainConfig(ModelingConfig):
    """Configuration for training a model."""

    def __init__(self, device, train_batch_size: int = 8, num_train_epochs: int = 3, max_steps: int = -1,
                 gradient_accumulation_steps: int = 1, weight_decay: float = 0.0, learning_rate: float = 1e-5,
                 adam_epsilon: float = 1e-8, warmup_steps: int = 0, max_grad_norm: float = 1.0,
                 logging_steps: int = 50):
        # TODO: No bool arguments lm_training and use_logits and temperature
        """
        Create a new training config.

        :param device: the device to use ('cpu' or 'cuda')
        :param train_batch_size: the number of labeled training examples per train batch
        :param num_train_epochs: the number of epochs to train for
        :param max_steps: the maximal number of steps to train for (overrides num_train_epochs)
        :param gradient_accumulation_steps: the number of steps to accumulate gradients for before performing an update
        :param weight_decay: the weight decay to use
        :param learning_rate: the maximum learning rate to use
        :param adam_epsilon: the epsilon value for optimizer Adam
        :param warmup_steps: the number of warmup steps to perform before reaching the maximum learning rate
        :param max_grad_norm: the maximum norm for the gradient
        :param logging_steps: Log every X updates steps
        :param model_from_ft: wheather to load a finetuned model from the directory
        """
        self.device = device
        self.train_batch_size = train_batch_size
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps


class EvalConfig(ModelingConfig):
    """Configuration for evaluation."""

    def __init__(self, device, output_dir: str, task_name: str, pattern_ids: List[int], batch_size: int = 8,
                 priming: bool = False, seed: int = 42, eval_lang: str = 'te', retrieved_lang: str = 'en',
                 model_from_ft: bool = False, self_prediction: bool = False, num_priming: int = 1,
                 random_retrieval: bool = False, baseline_ft: bool = False,
                 sentence_transformer_name: str = 'sentence-transformers/distiluse-base-multilingual-cased-v2',
                 conc: bool = False):
        """Create a new evaluation config."""

        self.device = device
        self.output_dir = output_dir
        self.task_name = task_name
        self.pattern_ids = pattern_ids
        self.batch_size = batch_size
        self.priming = priming
        self.seed = seed
        self.eval_lang = eval_lang
        self.retrieved_lang = retrieved_lang
        self.model_from_ft = model_from_ft
        self.self_prediction = self_prediction
        self.num_priming = num_priming
        self.random_retrieval = random_retrieval
        self.sentence_transformer_name = sentence_transformer_name
        self.baseline_ft = baseline_ft
        self.conc = conc


def train(model_config: WrapperConfig, train_data: List[InputExample], eval_data: List[InputExample],
          config: TrainConfig, eval_config: EvalConfig, output_dir: str, pattern_ids: List[int], repetitions: int = 1,
          do_train: bool = True, do_eval: bool = True, seed: int = 42):
    """
    Used to train a series of pvp models given different pattern ids.

    :param model_config: the model to train (in wrapper form)
    :param repetitions: the number of training repetitions
    :param train_data: the training examples to use
    :param eval_data: the examples used for evaluation
    :param config: the training config
    :param eval_config: the evaluation config
    :param output_dir: the output directory
    :param pattern_ids: the list of pattern ids
    :param do_train: wheather to train the model
    :param do_eval: wheather to evaluate the model
    :param seed: random seed
    """

    results = defaultdict(list)
    set_seed(seed)

    for pattern_id in pattern_ids:
        for iteration in range(repetitions):

            model_config.pattern_id = pattern_id  # reset the pattern_id in the model config
            results_dict = {}

            pattern_iter_output_dir = '{}/p{}-i{}'.format(output_dir, pattern_id, iteration)

            if os.path.exists(pattern_iter_output_dir):
                logger.warning(f'Path {pattern_iter_output_dir} already exists...')

            if not os.path.exists(pattern_iter_output_dir):
                os.mkdir(pattern_iter_output_dir)

            wrapper = init_model(model_config)

            # Training
            if do_train:
                results_dict.update(train_single_model(wrapper, train_data, config, eval_config))

            # save the results
            with open(os.path.join(pattern_iter_output_dir, 'results.txt'), 'w') as f:
                f.write(str(results_dict))

            # save all things (model including model config and tokenizer and train config and eval config
            logger.info('Saving trained model at {}...'.format(pattern_iter_output_dir))
            wrapper.save(pattern_iter_output_dir)
            config.save(os.path.join(pattern_iter_output_dir, 'train_config.json'))
            eval_config.save(os.path.join(pattern_iter_output_dir, 'eval_config.json'))
            logger.info('Saving complete.')

            if not do_eval:
                wrapper.model = None
                wrapper = None
                torch.cuda.empty_cache()

        # Evaluation
        if do_eval:
            logger.info('Starting evaluation...')
            if not wrapper:
                wrapper = TransformerModelWrapper.from_pretrained(pattern_iter_output_dir)

            eval_result = eval_single_model(wrapper, eval_data, eval_config, pattern_iter_output_dir)

            save_evaluations(os.path.join(pattern_iter_output_dir, 'predictions.json1'), wrapper, eval_result)
            save_logits(os.path.join(pattern_iter_output_dir, 'eval_logits.txt'), eval_result['logits'])

            acc = eval_result['acc']
            logger.info("--- RESULT (pattern_id={}) --- ".format(pattern_id))
            logger.info('acc={}'.format(acc))

            results_dict['test_set_after_training'] = acc
            with open(os.path.join(pattern_iter_output_dir, 'results.json'), 'w') as f:
                json.dump(results_dict, f)

            results[pattern_id].append(acc)

            wrapper.model = None
            wrapper = None
            torch.cuda.empty_cache()

    if do_eval:
        logger.info('++++OVERALL RESULTS++++')
        _write_results(os.path.join(output_dir, 'result_test.txt'), results)
    else:
        logger.info('====TRAINING COMPLETE====')


def train_single_model(model: TransformerModelWrapper, train_data: List[InputExample], config: TrainConfig,
                       eval_config: EvalConfig):
    # TODO: no unlabeled_data and bool argument return_train_set_results
    """
    Used to train a single model.

    :param model: the model to train (in wrapper form）
    :param train_data: the training examples to use
    :param config: the training config
    :param eval_config: the evaluation config
    """

    device = torch.device(config.device if config.device else 'cuda' if torch.cuda.is_available() else 'cpu')

    results_dict = dict()

    model.model.to(device)

    if train_data:
        # evaluation during finetuning does not need priming data
        results_dict['train_set_before_training'] = model.eval(eval_data=train_data, device=eval_config.device,
                                                               batch_size=eval_config.batch_size, priming=False)['acc']

    global_step, tr_loss = model.train(task_train_data=train_data, device=device,
                                       train_batch_size=config.train_batch_size,
                                       num_train_epochs=config.num_train_epochs,
                                       gradient_accumulation_steps=config.gradient_accumulation_steps,
                                       weight_decay=config.weight_decay, learning_rate=config.learning_rate,
                                       adam_epsilon=config.adam_epsilon, warmup_steps=config.warmup_steps,
                                       max_grad_norm=config.max_grad_norm, max_steps=config.max_steps,
                                       logging_steps=config.logging_steps)
    results_dict['global'] = global_step
    results_dict['average_loss'] = tr_loss

    if train_data:
        results_dict['train_set_after_training'] = model.eval(eval_data=train_data, device=eval_config.device,
                                                              batch_size=eval_config.batch_size, priming=False)['acc']

    return results_dict


def evaluate(model_config: WrapperConfig, eval_data: List[InputExample], eval_config: EvalConfig,
             train_config: TrainConfig):
    """
    Evaluate a task and record the results in the output file

    :param model_config: the configuration for the model used for evaluation
    :param eval_data: the examples for evaluation
    :param eval_config: the evaluation configuration
    :return: a dictionary containing the model's logits, predictions and evaluation metrics
    """

    for pattern_id in eval_config.pattern_ids:
        accs = []
        # initialize a model for the current prompt pattern mode.
        model_config.pattern_id = pattern_id
        if eval_config.model_from_ft:
            model_config.model_name_or_path = f'output_ft/p{pattern_id}-i0'
            logger.info(f'Loading finetuned model {model_config.model_name_or_path}...')
        model = init_model(model_config)
        logger.info(f"model type: {type(model.model)}")

        # set up the output directory
        output_dir = "{}/{}/{}/p{}".format(eval_config.output_dir, eval_config.task_name,
                                           eval_config.eval_lang, pattern_id)

        if os.path.exists((output_dir)):
            logger.warning(f"Path {output_dir} already exists, skipping it")
            # continue

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # adding priming data (i.e. cross-lingual retrieval result) into input example
        if eval_config.priming:
            eval_data_priming = add_priming_data(model, eval_config.device, eval_data, lang=eval_config.retrieved_lang,
                                                 seed=eval_config.seed, self_prediction=eval_config.self_prediction,
                                                 num_priming=eval_config.num_priming, task_name=eval_config.task_name,
                                                 random_retrieval=eval_config.random_retrieval, transformer_name=eval_config.sentence_transformer_name)
        else:
            eval_data_priming = eval_data

        if eval_config.baseline_ft:
            eval_results = eval_baseline_ft(model_config, eval_data_priming, eval_config, train_config)
        else:
            eval_results = eval_single_model(model, eval_data_priming, eval_config, output_dir)

        # natural accuracy
        # labels = [example.label for example in eval_data_priming]
        # prime_labels = [example.meta['priming_data'][0].label for example in eval_data_priming]
        # nat_acc = simple_accuracy(np.array(prime_labels), np.array(labels))


        acc = eval_results['acc']
        logger.info("--- RESULT (pattern_id={}, lang={}) --- ".format(pattern_id, eval_config.eval_lang))
        logger.info('acc={}'.format(acc))
        # logger.info('nat_acc={}'.format(nat_acc))
        accs.append(round(acc*100, 2))

        with open(f'result_xlmr.txt', 'a', encoding='utf-8') as f:
            result_record = f'Task {eval_config.task_name} | Num of Priming {eval_config.num_priming} | ' \
                            f'Lang {eval_config.eval_lang} | Labeled {not eval_config.self_prediction}' \
                            f'\nAccuracy {acc} \n'
            f.writelines(result_record)

    return accs

def eval_single_model(model: TransformerModelWrapper, eval_data: List[InputExample], eval_config: EvalConfig,
                      output_dir: str):
    """Evaluate one single model."""
    # get the evaluation results
    model.model.to(eval_config.device)
    ###---###
    results = dict()
    for idx in range(eval_config.num_priming):
    ###---###
        temp_results = model.eval(eval_data=eval_data, device=eval_config.device, priming_idx=idx,
                         batch_size=eval_config.batch_size, priming=eval_config.priming,
                                  num_priming=eval_config.num_priming, conc=eval_config.conc)
        if results:
            results['logits'] = np.concatenate((results['logits'], temp_results['logits']), axis=1)
            results['predictions'] = np.concatenate((results['predictions'], temp_results['predictions']), axis=0)
        else:
            results.update(temp_results)
        if eval_config.conc:
            break

    results['final_predictions'] = np.array([np.bincount(l).argmax() for l in results['predictions'].T])
    results['acc'] = simple_accuracy(results['final_predictions'], results['labels'])
    # # save results
    # num_priming = eval_config.num_priming if eval_config.priming else 0
    # save_evaluations(os.path.join(output_dir, f'record_{num_priming}.txt'), model, results, eval_data, eval_config.self_prediction,
    #                  eval_config.eval_lang)
    # save_logits(os.path.join(output_dir, 'eval_logits.txt'), results['logits'])

    return results

def eval_baseline_ft(model_config: WrapperConfig, eval_data_priming: List[InputExample], eval_config: EvalConfig,
                     train_config: TrainConfig):

    results = []
    for eval_ex in tqdm(eval_data_priming, desc='Evaluation with finetuning'):
        model = init_model(model_config)
        device = torch.device(train_config.device if train_config.device else 'cuda' if torch.cuda.is_available() else 'cpu')
        model.model.to(device)

        priming_data = eval_ex.meta['priming_data'][:eval_config.num_priming]  # List of InputExample

        if priming_data:
            _ = model.train(task_train_data=priming_data, device=device,
                                               train_batch_size=train_config.train_batch_size,
                                               num_train_epochs=train_config.num_train_epochs,
                                               gradient_accumulation_steps=train_config.gradient_accumulation_steps,
                                               weight_decay=train_config.weight_decay,
                                               learning_rate=train_config.learning_rate,
                                               adam_epsilon=train_config.adam_epsilon,
                                               warmup_steps=train_config.warmup_steps,
                                               max_grad_norm=train_config.max_grad_norm,
                                               max_steps=train_config.max_steps,
                                               logging_steps=train_config.logging_steps)

            temp_result = model.eval(eval_data=[eval_ex], device=eval_config.device, batch_size=eval_config.batch_size,
                                     priming=False)['acc']
            results.append(temp_result.item())

        model.model = None
        model = None
        torch.cuda.empty_cache()

    return {'acc': np.array(results).mean()}



def _write_results(path: str, results: Dict):
    with open(path, 'w') as f:
        for pattern_id, accs in results.items():
            mean = statistics.mean(accs)
            stdev = statistics.stdev(accs) if len(accs) > 1 else 0
            result_str = 'Acc-p{}: {} +- {}'.format(pattern_id, mean, stdev)
            logger.info(result_str)
            f.write(result_str + '\n')


def init_model(config: WrapperConfig) -> TransformerModelWrapper:
    """Initialize a new model wrapper from the given config."""
    assert config.pattern_id is not None, 'A patter_id must be set for initializing a new PET model'
    model = TransformerModelWrapper(config)
    return model
