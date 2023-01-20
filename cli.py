"""
Command line interface
This script can be used to evaluate a pretrained multilingual model on one supported task or dataset.
"""

import argparse
import csv
from typing import Tuple

import torch

import log
from wrapper import MODEL_CLASSES, WRAPPER_TYPES, WrapperConfig
from modeling import TrainConfig, EvalConfig, evaluate, train
from tasks import PROCESSORS
from utils import set_seed

logger = log.get_logger('root')

def load_configs(args) -> Tuple[WrapperConfig, EvalConfig]:
    """Load the model and evaluation configs given the command line arguments."""
    model_cfg = WrapperConfig(model_type=args.model_type, model_name_or_path=args.model_name_or_path,
                              wrapper_type=args.wrapper_type, task_name=args.task_name, label_list=args.label_list,
                              max_seq_length=args.max_seq_length, verbalizer_file=args.verbalizer_file,
                              cache_dir=args.cache_dir)

    train_cfg = TrainConfig(device=args.device, train_batch_size=args.train_batch_size,
                            num_train_epochs=args.num_train_epochs, max_steps=args.max_steps,
                            max_grad_norm=args.max_grad_norm, gradient_accumulation_steps=args.gradient_accumulation_steps,
                            weight_decay=args.weight_decay, learning_rate=args.learning_rate, adam_epsilon=args.adam_epsilon,
                            warmup_steps=args.warmup_steps, logging_steps=args.logging_steps)

    eval_cfg = EvalConfig(device=args.device, output_dir=args.output_dir, task_name=args.task_name,
                          pattern_ids=args.pattern_ids, batch_size=args.eval_batch_size, priming=args.priming,
                          eval_lang=args.eval_lang, retrieved_lang=args.retrieved_lang, model_from_ft=args.model_from_ft,
                          self_prediction=args.self_prediction, num_priming=args.num_priming,
                          random_retrieval=args.random_retrieval, baseline_ft=args.baseline_ft,
                          sentence_transformer_name=args.sentence_transformer_name, conc=args.conc)

    return model_cfg, train_cfg, eval_cfg

def main():
    parser = argparse.ArgumentParser(description='command line interface for zero-shot evaluation')

    # General parameters
    parser.add_argument("--data_dir", default='data/amazon_reviews', type=str,
                        help="The input data directory. Should contain the data files for the task for evaluation.")
    parser.add_argument("--model_type", default='bert', type=str, choices=MODEL_CLASSES.keys(),
                        help='The type of the pretrained model to use.')
    parser.add_argument('--model_name_or_path', default='bert-base-multilingual-cased', type=str,
                        help='Path to the pretrained model or shortcut name.')
    parser.add_argument('--model_from_ft', action='store_true',
                        help='Wheather to load a finetuned model.')
    parser.add_argument('--task_name', default='product-review-polarity', type=str,
                        choices=PROCESSORS.keys(), help='The name of the task to evaluate/train on.')
    parser.add_argument('--output_dir', default='results', type=str,
                        help='The output directory where the model predictions and checkpoints are saved.')
    parser.add_argument('--seed', default=42, type=int,
                        help='set the random seed')
    parser.add_argument('--eval_from_datasets', action='store_true',
                        help='Wheather to load the evaluation data from local directory or from huggingface datasets.')
    parser.add_argument('--train_from_datasets', action='store_true',
                        help='Wheather to load the train data from a directory or from huggingface datasets')
    parser.add_argument('--retrieved_lang', default='en', type=str,
                        help='The high resource language for retrieval')
    parser.add_argument('--eval_langs', default=['en', 'af', 'ur', 'sw', 'te', 'ta', 'mn', 'uz', 'my', 'jw', 'tl'],
                        nargs='+', help='The low resource language for zero-shot evaluation')
    parser.add_argument('--finetuning', action='store_true',
                        help='Wheather to finetune the model.')
    parser.add_argument('--ft_dataset_name', default='amazon_reviews_multi', type=str,
                        help='Datset name used for finetuning.')
    parser.add_argument('--ft_lang', default='en', type=str,
                        help='Dataset language used for finetuning.')
    parser.add_argument('--num_train_data', default=-1, type=int,
                        help='The number of train_data, if < 0, all data')
    parser.add_argument('--num_eval_data', default=200, type=int,
                        help='The number of eval data.')

    # parameters specific for the prompt baseline
    parser.add_argument('--wrapper_type', default='mlm', type=str, choices=WRAPPER_TYPES,
                        help="The wrapper type. Set this to 'mlm' for a masked language model like BERT.")
    parser.add_argument('--pattern_ids', default=[0, 1, 2, 3, 4], type=int, nargs='+',
                        help='The ids of the PVPs to be used')
    parser.add_argument('--max_seq_length', default=512, type=int,
                        help='The maximum total input sequence length after tokenization for PET')
    parser.add_argument('--train_batch_size', default=4, type=int,
                        help='Batch size for training.')
    parser.add_argument('--eval_batch_size', default=8, type=int,
                        help='Batch size for evaluation.')
    parser.add_argument('--num_train_epochs', default=1, type=int,
                        help='Total number pf training epochs.')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help='Number of update steps to accumulate before performing a backpropogation.')
    parser.add_argument('--max_steps', default=-1, type=int,
                        help='If > 0: set total number of training steps to perform. Overriding max_num_epochs.')
    parser.add_argument('--metrics', default='acc', type=str,
                    help='Metrics used for evaluation')
    parser.add_argument('--verbalizer_file', default=None, type=str,
                        help='Use other verbalizers than the default.')
    parser.add_argument('--cache_dir', default="", type=str,
                        help='Where to store the pre-trained')

    # optional parameters for cross-lingual retrieval
    parser.add_argument('--priming', action='store_true',
                        help='Wheather to use priming for evaluation')
    parser.add_argument('--retrieval_dataset_name', default="amazon_reviews_multi", type=str,
                        help='The dataset name used for cross lingual retrieval (loaded from huggingface dataset)')
    parser.add_argument('--save_dir', type=str, help='The directory used to save the retrieved sentences if needed')
    parser.add_argument('--sentence_transformer_name', default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                        type=str, help='The pretrained multilingual sentence transformer used for retrieval')
    parser.add_argument('--retrieval_language', default='en', type=str,
                        help='The high resource language used for retrieval')
    parser.add_argument('--size_pool', default=10000, type=int,
                        help='Define the size of sentence pool for retrieval')
    parser.add_argument('--num_sim_sent', default=100, type=int,
                        help='Number of the similar sentences retrieved from pool for each input sequence')
    parser.add_argument('--num_priming', default=1, type=int,
                        help='Number of retrieved sentences used in the prompt')
    parser.add_argument('--retrieval_method', default='sentence_transformer', type=str,
                        help='Cross lingual retrieval method to use')
    parser.add_argument('--random_retrieval', action='store_true',
                        help='whether retrieve cross-lingual sentences randomly')

    # Other optional parameters
    parser.add_argument('--conc', action='store_true', help='whether to use concactenate strategy or BOW sentences')
    parser.add_argument('--do_train', action='store_true',
                        help='Wheather to perform training')
    parser.add_argument('--do_eval', action='store_true',
                        help='Wheather to perform evaluation')
    parser.add_argument('--weight_decay', default=0.0, type=float,
                        help='Weight decay if we apply some.')
    parser.add_argument('--learning_rate', default=1e-5, type=float,
                        help='The initial learning rate for Adam.')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float,
                        help='Epsilon for Adam optimizer.')
    parser.add_argument('--max_grad_norm', default=1.0, type=float,
                        help='Maximum gradient norm.')
    parser.add_argument('--warmup_steps', default=0, type=int,
                        help='Linear warmup over warmup_steps.')
    parser.add_argument('--logging_steps', default=50, type=int,
                        help='Log every X update steps.')
    parser.add_argument('--train_repetitions', default=1, type=int,
                        help='the number of training repetitions')
    parser.add_argument('--self_prediction', action='store_true',
                        help='Wheather to use self prediction or directly use the train data at cros-lingual retrieval.')
    parser.add_argument('--baseline_ft', action='store_true', help='whether to use baseline finetuning paradigm.')

    args = parser.parse_args()
    logger.info('Parameters: {}'.format(args))

    # Setup device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Prepare task
    if args.task_name not in PROCESSORS:
        raise ValueError("Task '{}' not found".format(args.task_name))
    processor = PROCESSORS[args.task_name]()
    args.label_list = processor.get_labels()

    set_seed(args.seed)

    accs = []
    for eval_lang in args.eval_langs:
        args.eval_lang = eval_lang

        model_cfg, train_cfg, eval_cfg = load_configs(args)
        eval_data = processor.get_examples(data_dir=args.data_dir, set_type='test', from_datasets=args.eval_from_datasets,
                                           lang=args.eval_lang, seed=args.seed, num_data=args.num_eval_data)
        if args.finetuning:
            train_data = processor.get_examples(data_dir=args.ft_dataset_name, set_type='train',
                                                from_datasets=args.train_from_datasets, lang=args.retrieved_lang,
                                                seed=args.seed, num_data=args.num_train_data)
            logger.info(f"Training set contains {len(train_data)} data examples.")
            train(model_config=model_cfg, train_data=train_data, eval_data=eval_data, config=train_cfg, eval_config=eval_cfg,
                  output_dir=args.output_dir, pattern_ids=args.pattern_ids, repetitions=args.train_repetitions,
                  do_train=args.do_train, do_eval=args.do_eval, seed=args.seed)
        else:
            accs += evaluate(model_cfg, eval_data, eval_cfg, train_cfg)

        # result_file_path = f'results_{eval_lang}.csv'
        # with open(result_file_path, 'a', encoding='utf-8') as f:
        #     writer = csv.writer(f)
        #     # accs.extend([args.])
        #     writer.writerow(accs)
    with open(f'results_{args.task_name}.csv', 'a', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(accs)

if __name__ == '__main__':
    main()
