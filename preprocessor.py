# -*- coding: utf-8 -*-
# Author: Niel
# Date: 2022/6/14  14:29

from abc import ABC, abstractmethod

from utils import InputExample, InputFeatures
from pvp import PVPS  # TODO: prepare pvp.py

class Preprocessor(ABC):
    """
    A preprocessor that transforms an :class:"InputExample" into a :class:"InputFeatures" object so that it can be
    processed by the model being used.
    """
    # TODO: prepare model wrappers in wrapper.py file
    def __init__(self, wrapper, task_name, pattern_id: int = 0, verbalizer_file: str = None):
        """
        Create a new preprocessor.

        :param wrapper: the wrapper for the language model to use
        :param task_name:  the name of the task
        :param pattern_id: the id of prompt patterns to be used
        :param verbalizer_file: path to a file containing a verbalizer that overrides the default verbalizer (optional)
        """

        self.wrapper = wrapper
        self.pvp = PVPS[task_name](self.wrapper, pattern_id, verbalizer_file) # pvp stands for patter verbalizer pair
        self.label_map = {label: i for i, label in enumerate(self.wrapper.config.label_list)}
        # convert real label to label index

    @abstractmethod
    def get_input_features(self, example: InputExample, labelled: bool, priming_idx: int = -1, priming: bool = False,
                           **kwargs) -> InputFeatures:
        """Convert the given example into a set of input features"""
        pass

class MLMPreprocessor(Preprocessor):
    """
    Preprocess for models pretrained using a masked language modeling objective, e.g., BERT.
    """
    def get_input_features(self, example: InputExample, labelled: bool, priming_idx: int = -1, priming: bool = False,
                           num_priming: int=0, conc: bool = False, **kwargs) -> InputFeatures:
        """Convert the given example into a set of input features"""

        #TODO: ???what if the total sequence length of priming sequenece + original sequence larger than max_length???
        if priming:
            priming_data = example.meta['priming_data'][:num_priming]  # type of priming_data: List[InputExample]
            if conc:
                priming_input_ids = []
                max_length = int(self.wrapper.config.max_seq_length / (num_priming + 1))
                for priming_example in priming_data:
                    priming_input_ids += \
                        self.pvp.encode(priming_example, priming=True, labeled=True, max_length=max_length)[0]

            else:
                max_length = int(self.wrapper.config.max_seq_length / 2)
                priming_example = priming_data[priming_idx]
                priming_input_ids, _ = self.pvp.encode(priming_example, priming=True, labeled=True,
                                                       max_length=max_length)

            input_ids, token_type_ids = self.pvp.encode(example, max_length=max_length)
            input_ids = priming_input_ids + input_ids

            token_type_ids = self.wrapper.tokenizer.create_token_type_ids_from_sequences(input_ids)
            input_ids = self.wrapper.tokenizer.build_inputs_with_special_tokens(input_ids)

        else:
            input_ids, token_type_ids = self.pvp.encode(example)

        assert len(input_ids) == len(token_type_ids), f"length of input ids: {len(input_ids)}, " \
                                                      f"length of tokens: {len(token_type_ids)}."

        attention_mask = [1] * len(input_ids)
        padding_length = self.wrapper.config.max_seq_length - len(input_ids)

        if padding_length < 0:
            raise ValueError(f"Maximum sequence length is too small, got {len(input_ids)} input ids.")

        input_ids = input_ids + ([self.wrapper.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        assert len(input_ids) == self.wrapper.config.max_seq_length
        assert len(attention_mask) == self.wrapper.config.max_seq_length
        assert len(token_type_ids) == self.wrapper.config.max_seq_length

        label = self.label_map[example.label] if example.label is not None else -100  # convert label to label index

        if labelled:
            mlm_labels = self.pvp.get_mask_positions(input_ids)
        else:
            mlm_labels = [-1] * self.wrapper.config.max_seq_length

        return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             label=label, mlm_labels=mlm_labels, idx=example.guid)

class SequenceClassifierPreprocessor(Preprocessor):
    """Preprocessor for a regular sequence classification model."""

    def get_input_features(self, example: InputExample, **kwargs) -> InputFeatures:
        # TODO: prepare task_helper
        inputs = self.wrapper.task_helper.get_sequence_classifier_inputs(example) if self.wrapper.task_helper else None
        if inputs is None:
            inputs = self.wrapper.tokenizer.encode_plus(
                example.text_a if example.text_a else None,
                example.text_b if example.text_b else None,
                add_special_tokens=True,
                max_length=self.wrapper.max_seq_length
            )
        input_ids, token_type_ids = inputs['input_ids'], inputs.get('token_type_ids')

        attention_mask = [1] * len(input_ids)
        padding_length = self.wrapper.config.max_seq_length - len(input_ids)

        input_ids = input_ids + ([self.wrapper.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        if not token_type_ids:
            token_type_ids = [0] * self.wrapper.max_seq_length
        else:
            token_type_ids = token_type_ids + ([0] * padding_length)
        mlm_labels = [-1] * len(input_ids)  # mlm_labels padded with -1 if no MASK is needed.

        assert len(input_ids) == self.wrapper.config.max_seq_length
        assert len(attention_mask) == self.wrapper.config.max_seq_length
        assert len(token_type_ids) == self.wrapper.config.max_seq_length

        label = self.label_map[example.label] if example.label is not None else -100  # -100 represents no label

        return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             label=label, mlm_labels=mlm_labels)

