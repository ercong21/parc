# -*- coding: utf-8 -*-
# Author: Niel
# Date: 2022/6/15  15:14
"""
This file contains the pattern (prompt template) verbalizer pairs for different tasks.
"""
import random
import string
from abc import ABC, abstractmethod
from collections import defaultdict

from typing import Tuple, Union, List
import torch

from utils import InputExample, get_verbalization_ids
import log

logger = log.get_logger('root')

# used for designing the prompt template for data example
FilledPattern = Tuple[List[Union[str, Tuple[str, bool]]], List[Union[str, Tuple[str, bool]]]]

class PVP(ABC):
    """
    This class contains functions to apply patterns and verbalizers as required by prompt learning.
    Each task requires its own custom implementation (processor) of pvp.
    """

    def __init__(self, wrapper, pattern_id: int = 0, verbalizer_file: str = None, seed: int = 42):
        """
        Create a new PVP.

        :param wrapper: the wrapper for the underlying language model
        :param pattern_id: the pattern id to use
        :param verbalizer_file: an optional file that contains the verbalizer to be used
        :param seed: a seed to be used for generating random numbers if necessary
        """
        self.wrapper = wrapper
        self.pattern_id = pattern_id
        self.rng = random.Random(seed)  # random number generator

        if verbalizer_file:
            self.verbalize = PVP._load_verbalizer_from_file(verbalizer_file, self.pattern_id)

        self.mlm_logits_to_cls_logits_tensor = self._build_mlm_logits_to_cls_logits_tensor()

    def _build_mlm_logits_to_cls_logits_tensor(self):
        label_list = self.wrapper.config.label_list
        m2c_tensor = torch.ones([len(label_list), self.max_num_verbalizers], dtype=torch.long) * -1

        for label_idx, label in enumerate(label_list):
            verbalizers = self.verbalize(label)
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = get_verbalization_ids(verbalizer, self.wrapper.tokenizer, force_single_token=True)
                assert verbalizer_id != self.wrapper.tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id

        return m2c_tensor

    @property
    def mask(self) -> str:
        """Return the underlying LM's special mask token."""
        return self.wrapper.tokenizer.mask_token

    @property
    def mask_id(self) -> int:
        """Return the underlying LM's mask token id."""
        return self.wrapper.tokenizer.mask_token_id

    @property
    def max_num_verbalizers(self) -> int:
        """Return the maximum number of the verbalizers across all labels."""
        return max(len(self.verbalize(label)) for label in self.wrapper.config.label_list)

    def encode(self, example: InputExample, priming: bool = False, labeled: bool = False,
               max_length = None) -> Tuple[List[int], List[int]]:
        """
        Encode an input example using this pattern verbalizer pair

        :param example: an input example to encode
        :param priming: wheather to use this example for priming
        :param labeled: if "priming=True", wheather the label should be appended to this example
        :return: A tuple, consisting of a list of input ids and a list of token type ids
        """

        if not priming:
            assert not labeled, "'labeled' can only be set to true if 'priming' is also set to true."

        tokenizer = self.wrapper.tokenizer  # type: PreTrainedTokenizer
        parts_a, parts_b = self.get_parts(example)

        parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
        parts_a = [(tokenizer.encode(x, add_special_tokens=False), s) for x, s in parts_a if x]

        if parts_b:
            parts_b = [x if isinstance(x, tuple) else (x, False) for x in parts_b]
            parts_b = [(tokenizer.encode(x, add_special_tokens=False), s) for x, s in parts_b if x]

        if max_length:
            self.truncate(parts_a, parts_b, max_length=max_length)
        else:
            self.truncate(parts_a, parts_b, max_length=self.wrapper.config.max_seq_length)

        tokens_a = [token_id for part, _ in parts_a for token_id in part]
        tokens_b = [token_id for part, _ in parts_b for token_id in part]

        if priming:
            input_ids = tokens_a
            if tokens_b:
                input_ids += tokens_b
            if labeled:
                assert self.mask_id in input_ids, 'sequence of input_ids must contain a mask token'
                mask_idx = input_ids.index(self.mask_id)
                # assert len(self.verbalize(example.label)) == 1, 'priming only supports one verbalization per label'
                verbalizer = self.verbalize(example.label)[0]

                verbalizer_id = get_verbalization_ids(verbalizer, self.wrapper.tokenizer, force_single_token=True)
                input_ids[mask_idx] = verbalizer_id
            return input_ids, []

        input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
        # input_ids.append(102)
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)


        return input_ids, token_type_ids


    @staticmethod
    def shortenable(s: str) -> Tuple[str, bool]:
        """
        Return an instance of this string that is marked as shortenable
        :param s: the given string to be marked
        :return: a tuple
        """
        return s, True

    @staticmethod
    def remove_final_punc(s: Union[str, Tuple[str, bool]]):
        """Remove the final punctuation mark."""
        if isinstance(s, tuple):
            return PVP.remove_final_punc(s[0]), s[1]
        return s.rsplit(string.punctuation)

    # TODO: data type of the first element in the tuple: List[int] or str???
    def truncate(self, parts_a: List[Tuple[List[int], bool]], parts_b: List[Tuple[List[int], bool]], max_length: int):
        """
        Truncate two sequences of text to a predefined total maximum of length.
        :param parts_a: the first text
        :param parts_b: the second text
        :param max_length: predefined total maximum length
        :return: truncated parts_a and parts_b
        """

        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)
        # total_len += self.wrapper.tokenizer.num_special_tokens_to_add(bool(parts_b))
        total_len += self.wrapper.tokenizer.num_special_tokens_to_add(bool(parts_a))
        num_tokens_to_remove = total_len - max_length

        if num_tokens_to_remove <= 0:
            return parts_a, parts_b

        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(parts_b, only_shortenable=True):
                self._remove_last(parts_a)
            else:
                self._remove_last(parts_b)

    @staticmethod
    def _seq_length(parts: List[Tuple[str, bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    @staticmethod
    def _remove_last(parts: List[Tuple[str, bool]]):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    def convert_mlm_logits_to_cls_logits(self, mlm_labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        masked_logits = logits[mlm_labels >= 0]
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml) for ml in masked_logits])
        return cls_logits

    def _convert_single_mlm_logits_to_cls_logits(self, logits: torch.Tensor) -> torch.Tensor:
        m2c = self.mlm_logits_to_cls_logits_tensor
        m2c = m2c.to(logits.device)

        # filler_len.shape() == max_fillers
        filler_len = torch.tensor([len(self.verbalize(label)) for label in self.wrapper.config.label_list],
                                  dtype=torch.float)
        filler_len = filler_len.to(logits.device)

        # cls_logits.shape() == num_labels * max_fillers
        cls_logits = logits[torch.max(torch.zeros_like(m2c), m2c)]
        cls_logits = cls_logits * (m2c > 0).float()


        # cls_logits.shape() == num_labels
        cls_logits = cls_logits.sum(axis=1) / filler_len
        return cls_logits

    def get_mask_positions(self, input_ids: List[int]) -> List[int]:
        label_idx = input_ids.index(self.mask_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels

    @abstractmethod
    def get_parts(self, example: InputExample) -> FilledPattern:
        """
        Given an input example, apply a pattern to obtain two text sequences text_a and text_b, containing exactly one
        mask token for a single task. If a task requires only a single sequence of text, then the second sequence
        should be an empty list.

        :param example: the input example to be processed
        :return: Two sequences of texts. All text segments can optionally be marked as being shortenable.
        """
        pass

    @abstractmethod
    def verbalize(self, label) -> List[str]:
        """
        Return all verbalizations for a given label

        :param label: the label
        :return: the list of all verbalizations to the label
        """
        pass

    @staticmethod
    def _load_verbalizer_from_file(path: str, pattern_id: int):

        verbalizers = defaultdict(dict)  # type: Dict[int, Dict[str, List[str]]]
        current_pattern_ids = None

        with open(path, 'r') as fh:
            for line in fh.read().splitlines():
                if line.isdigit():
                    current_pattern_ids = int(line)
                elif line:
                    label, *realizations = line.split()
                    verbalizers[current_pattern_ids][label] = realizations

        logger.info('Automatically loaded from the following verbalizer: \n {}'.format(verbalizers[pattern_id]))

        def verbalize(label) -> List[str]:
            return verbalizers[pattern_id][label]

        return verbalize

class ProductPVP(PVP):
	# 1: terrible 2: great
    VERBALIZER = {
        '1': ['bad'],
        '2': ['great']
    }
# verbalizer pool: good - 'positive', 'great', 'super'
                #  bad - 'terrible', 'negative'

    def get_parts(self, example: InputExample) -> FilledPattern:
        text = self.shortenable(example.text_a)

        # e.g. text = 'The best laptop I have ever used!'
        # pattern 0: It was [MASK]. The best laptop I have ever used!
        if self.pattern_id == 0:
            return [text, self.mask], []
        elif self.pattern_id == 1:
            return ['It was', self.mask, '.', text], []
        # pattern 1: The best laptop I have ever used! All in all, it was [MASK].
        elif self.pattern_id == 2:
            return [text, 'All in all, it was', self.mask, '.'], []
        # pattern 2: Just [MASK]! The best laptop I have ever used!
        elif self.pattern_id == 3:
            return ['Just', self.mask, '!'], [text]
        # pattern 3: The best laptop I have ever used! In summary, the product is [MASK].
        elif self.pattern_id == 4:
            return [text], ['In summary, the product is', self.mask, '.']
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return ProductPVP.VERBALIZER[label]


class XnliPVP(PVP):
    VERBALIZER_A = {
        '0': ['Yes'],
        '1': ['Maybe'],
        '2': ['No']
    }

    VERBALIZER_B = {
        '0': ['Right'],
        '1': ['Maybe'],
        '2': ['Wrong']
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(self.remove_final_punc(example.text_a))
        text_a = (' '.join(text_a[0]), text_a[1])
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0:
            return [text_a, '.', self.mask, text_b], []
        elif self.pattern_id == 1 or self.pattern_id == 2:
            return [text_a, '?'], [self.mask, ',', text_b]
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 1 or self.pattern_id == 0:
            return XnliPVP.VERBALIZER_A[label]
        return XnliPVP.VERBALIZER_B[label]


class AgNewsPVP(PVP):
    VERBALIZER = {
        '0': ['World'],
        '1': ['Sports'],
        '2': ['Business'],
        '3': ['Tech']
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(example.text_a)

        # example text: Germany won the 2014 Wolrd Cup.
        # pattern 0: Germany won the 2014 Wolrd Cup. [MASK]
        if self.pattern_id == 0:
            return [text_a, self.mask], []
        # pattern 1: [MASK]: Germany won the 2014 Wolrd Cup.
        elif self.pattern_id == 1:
            return [self.mask, ':', text_a], []
        # pattern 2: [MASK] News: Germany won the 2014 Wolrd Cup.
        elif self.pattern_id == 2:
            return [self.mask, 'News', ':', text_a], []
        # pattern 3: Germany won the 2014 Wolrd Cup. Category: [MASK]
        elif self.pattern_id == 3:
            return [text_a, 'Category', ':', self.mask], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))


    def verbalize(self, label) -> List[str]:
        return AgNewsPVP.VERBALIZER[label]


class XtcPVP(PVP):
    VERBALIZER = {
         '1': ['Politics'],
         '2': ['Military'],
         '3': ['Law'],
         '4': ['Economics'],
         '5': ['Education'],
         '6': ['Medicine'],
         '7': ['Religion'],
         '8': ['Literature'],
         '9': ['Culture'],
         '10': ['Transportation'],
         '11': ['Sport'],
         '12': ['History'],
         '13': ['Landscape'],
         '14': ['Science'],
         '15': ['Daily'],
         '16': ['Media'],
         '17': ['Entertainment'],
         '18': ['Food'],
         '19': ['Philosophy'],
         '20': ['News'],
         '21': ['Person'],
         '22': ['Popular'],
         '23': ['Organization']
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(example.text_a)

        # example text: Germany won the 2014 Wolrd Cup.
        # pattern 0: Germany won the 2014 Wolrd Cup. [MASK]
        if self.pattern_id == 0:
            return [text_a, self.mask], []
        # pattern 1: [MASK]: Germany won the 2014 Wolrd Cup.
        elif self.pattern_id == 1:
            return [self.mask, ':', text_a], []
        # pattern 2: [MASK] News: Germany won the 2014 Wolrd Cup.
        elif self.pattern_id == 2:
            return [self.mask, 'News', ':', text_a], []
        # pattern 3: Germany won the 2014 Wolrd Cup. Category: [MASK]
        elif self.pattern_id == 3:
            return [text_a, 'Category', ':', self.mask], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return XtcPVP.VERBALIZER[label]


PVPS = {
    'product-review-polarity': ProductPVP,
    'xnli': XnliPVP,
    'ag_news': AgNewsPVP,
    'xtc':XtcPVP
}
