from typing import Dict, Optional, Union, List
import json
import csv
import random

import numpy as np
import torch
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset


class InputExample:
    """
    A raw input example consisting of one or two segments of text and a label.
    """

    def __init__(self, guid, text_a, text_b=None, label=None, meta: Optional[Dict] = None):
        """
        Create a new InputExample.

        :param guid: textual identifier (e.g., train, test...)
        :param text_a: the sequence of text
        :param text_b: optional, the second sequence of text if applicable
        :param label: an optional label
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.meta = meta

class InputFeatures:
    """
    A set of numeric features obtained from an :class:'InputExample'
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label, mlm_labels=None, meta: Optional[Dict] = None,
                 idx: int = -1):
        """
        Create new InputFeatures.

        :param input_ids: the input ids corresponding to the original text or text sequence
        :param attention_mask: an attention mask, with 0 = no attention, 1 = attention
        :param token_type_ids: segment ids as used by BERT
        :param label: the label
        :param mlm_labels: an optional sequence of labels used for auxiliary language modeling
        """
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.mlm_labels = mlm_labels
        self.meta = meta if meta else {}
        self.idx = idx

class DictDataset(Dataset):
    """A dataset of tensors that uses a dictionary for key-value mappings"""

    def __init__(self, **tensors):
        tensors.values()

        assert all(next(iter(tensors.values())).size(0) == tensor.size(0) for tensor in tensors.values())
        self.tensors = tensors

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).size(0)


def get_verbalization_ids(verbalizer: str, tokenizer: PreTrainedTokenizer, force_single_token: bool = True) \
        -> Union[int, List[int]]:
    """
    Get the token ids corresponding to a verbalization.

    :param verbalizer: the verbalized word
    :param tokenizer: the tokenizer to use
    :param force_single_token: wheather it should be enforced that the verbalization corresponds to a single token.
           If set to true, this method returns a single int instead of a list and throws an error if the word
           corresponds to multiple tokens.
    :return: either the list of token ids or the single token id corresponding to this word
    """

    ids = tokenizer.encode(verbalizer, add_special_tokens=False)
    if not force_single_token:
        return ids
    assert len(ids) == 1, \
        f"Verbalizer '{verbalizer}' does not correspond to a single token, got {tokenizer.convert_ids_to_tokens(ids)}."
    verbalizer_id = ids[0]
    assert verbalizer_id not in tokenizer.all_special_ids, \
        f"Verbalizer {verbalizer} is mapped to a special token {tokenizer.convert_ids_to_tokens(verbalizer_id)}."
    return verbalizer_id

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    emb2 = emb2.T
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def save_evaluations(path: str,  wrapper, results: Dict, eval_data: List[InputExample], self_prediction: bool,
                     eval_lang: str):
    """Save a sequence of predictions into a file"""
    predictions_with_idx = []
    inv_label_map = {idx: label for label, idx in wrapper.preprocessor.label_map.items()}
    for idx, prediction_idx, label_idx, example in zip(results['indices'], results['final_predictions'], results['labels'], eval_data):
        prediction = inv_label_map[prediction_idx]
        label = inv_label_map[label_idx]
        idx = idx.tolist() if isinstance(idx, np.ndarray) else int(idx)
        line = dict()
        # for p_idx, priming_data in enumerate(example.meta['priming_data']):
        #     priming_label = priming_data.label
        #     priming_sample = priming_data.text_a
        #     line[f"priming_label_{p_idx}"] = priming_label
        #     line[f"priming_sample_{p_idx}"] = priming_sample
        input_sample = example.text_a
        line.update({'idx': idx, 'pred': prediction, 'label': label, 'input_sample': input_sample})
        predictions_with_idx.append(line)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"Acc: {results['acc']}"+'\n')
        # f.write('\t'.join(['Idx', 'Priming Label 1', 'Priming Label 2', 'Priming Label 3', 'Pred', 'Label',
        #                 'Priming Sample 1', 'Priming Sample 2','Priming Sample 3', 'Input Sample'])+'\n')
        # for line in predictions_with_idx:
        #     f.write('\t'.join([str(line['idx']), str(line['priming_label_1']), str(line['priming_label_2']), str(line['priming_label_3']),
        #                        str(line['pred']), str(line['label']), line['priming_sample_1'], line['priming_sample_2'],
        #                        line['priming_sample_3'], line['input_sample']])+'\n')

        # f.write('\t'.join(['Idx', 'Priming Label 1', 'Pred', 'Label',
        #                 'Priming Sample 1', 'Input Sample'])+'\n')
        # for line in predictions_with_idx:
        #     f.write('\t'.join([str(line['idx']), str(line['priming_label_1']), str(line['pred']), str(line['label']),
        #                        line['priming_sample_1'],line['input_sample']])+'\n')

        f.write('\t'.join(['Idx', 'Pred', 'Label',
                        'Input Sample'])+'\n')
        for line in predictions_with_idx:
            f.write('\t'.join([str(line['idx']), str(line['pred']), str(line['label']),
                               line['input_sample']])+'\n')


def save_logits(path: str, logits: np.ndarray):
    """Save an array of logits to a file"""
    with open(path, 'w') as f:
        for example_logits in logits:
            f.write(' '.join(str(logit) for logit in example_logits) + '\n')
    pass

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)






