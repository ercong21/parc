# -*- coding: utf-8 -*-
# Author: Niel
# Date: 2022/6/17  17:00

import os

from abc import ABC, abstractmethod
from typing import List

import log
from utils import InputExample

from datasets import load_dataset

logger = log.get_logger('root')

class DataProcessor(ABC):
    """
    Abstract class that provides methodss for loading data
    """
    @abstractmethod
    def get_examples(self, data_dir, set_type, lang, **kwargs) -> List[InputExample]:
        """
        Get a collection of data 'InputExample'
        """
        pass

    @abstractmethod
    def get_labels(self) -> List[str]:
        """
        Get the list of labels for this data set.
        """
        pass

class ProductProcessor(DataProcessor):
    """
    Processor for the product review binary classification task data set.
    """
    def get_examples(self, data_dir, set_type: str, from_datasets: bool = False, lang: str = 'en',
                     seed: int = 42, num_data: int = 500) -> List[InputExample]:
        if from_datasets:
            return self.get_examples_from_datasets(data_dir, set_type, lang, seed=seed, num_data=num_data)
        else:
            return self.get_examples_from_dir(data_dir, lang)

    def get_examples_from_dir(self, data_dir, lang):
        NEG_POS = {
            'neg': {'label': '1', 'file_path': 'neg.txt'},
            'pos': {'label': '2', 'file_path': 'pos.txt'}
        }

        examples = list()
        idx = 0
        for label in NEG_POS:
            path = os.path.join(data_dir, lang, NEG_POS[label]['file_path'])

            with open(path, 'r', encoding='utf-8') as f:
                text = ''
                for line in f:
                    if line.startswith('______'):
                        if text:
                            guid = str(idx)
                            text_a = text
                            example = InputExample(guid=guid, text_a=text_a, label=NEG_POS[label]['label'])
                            examples.append(example)

                            idx += 1
                            text = ''
                    else:
                        text = ' '.join([text.strip(), line.strip()])
        return examples


    def get_examples_from_datasets(self, data_dir, set_type, lang, seed: int = 42, num_data: int = 200):
        dataset = load_dataset(data_dir, lang, split=set_type)

        # shuffle the data and select num_data examples as dataset
        dataset = dataset.shuffle(seed)

        examples = list()
        idx = 0
        c_neg = 0
        c_pos = 0
        for text, star in zip(dataset['review_body'], dataset['stars']):
            if star == 1 and c_neg < num_data:
                label = '1'
                c_neg += 1
            elif star == 5 and c_pos < num_data:
                label = '2'
                c_pos += 1
            else:
                continue
            examples.append(InputExample(guid=idx, text_a=text, label=label))
            idx += 1

        return examples


    def get_labels(self) -> List[str]:
        return ['1', '2']


class AgNewsProcessor(DataProcessor):
    """
    Processor for text topic classification task on ag_news dataset
    """

    def get_examples(self, data_dir, set_type, lang, **kwargs) -> List[InputExample]:
        path = os.path.join(data_dir, lang+'.txt')
        examples = []
        idx = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                try:
                  text, label = line.split('\t')
                except:
                  print(idx, text)
                example = InputExample(guid=str(idx), text_a=text.strip(), label=label.strip())
                idx += 1
                examples.append(example)

        return examples

    def get_labels(self) -> List[str]:
        """
        '0':  'world',
        '1': 'sports',
        '2': 'business',
        '3': 'sci/tech'
        """
        return ['0', '1', '2', '3']


# 注意读label的时候要strip
class XnliProcessor(DataProcessor):
    """Processor for xnli task."""

    def get_examples(self, data_dir, set_type, lang, **kwargs) -> List[InputExample]:
        path = os.path.join(data_dir, lang + '.txt')
        examples = []
        idx = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if len(line.split('\t'))==3:
                    text_a, text_b, label = line.split('\t')
                    example = InputExample(guid=str(idx), text_a=text_a.strip(), text_b=text_b.strip(), label=label.strip())
                    idx += 1
                    examples.append(example)

        return examples

    def get_labels(self) -> List[str]:
        """
        '0': 'entailment',
        '1': 'neutral',
        '2': 'contradiction'
        """
        return ['0', '1', '2']


class XtcProcessor(DataProcessor):
    """Processor for XTC task."""

    def get_examples(self, data_dir, set_type, lang, **kwargs) -> List[InputExample]:
        path = os.path.join(data_dir, 'xtc_'+lang, 'xtc_'+lang+'.test_a.txt')
        examples = []
        idx = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if line.startswith('id'):
                    continue
                else:
                    idx, _, label_id, text = line.split('\t')
                    example = InputExample(guid=idx.strip(), text_a=text.strip(), label=label_id.strip())
                    examples.append(example)

        return examples

    def get_labels(self) -> List[str]:
        """
        '1': 'politics'
        '2': 'military'
        '3': 'law'
        '': 'economics'
        '4': 'education'
        '5': 'medicine'
        '6': 'religion'
        '7': 'literature'
        '8': 'culture'
        '10': 'transportation'
        '11': 'sport'
        '12': 'history'
        '13': 'landscape'
        '14': 'science'
        '15': 'daily'
        '16': 'media'
        '17': 'entertainment'
        '18': 'food'
        '19': 'philosophy
        '20': 'news'
        '21': 'person'
        '22': 'popular'
        '23'： 'organization'

        """

        return ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']



PROCESSORS = {
    'product-review-polarity': ProductProcessor,
    'ag_news': AgNewsProcessor,
    'xnli': XnliProcessor,
    'xtc':XtcProcessor
}

# def load_examples(task, data_dir: str) -> List[InputExample]:
#     """Load examples from the given data directory for a given task."""
#     processor = PROCESSORS[task]
