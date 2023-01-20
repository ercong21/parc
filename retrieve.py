import os
import pickle
import random
from tqdm import tqdm
from typing import Dict, List

from sentence_transformers import SentenceTransformer, models
from datasets import load_dataset

import log
from utils import InputExample, cosine_similarity
from wrapper import TransformerModelWrapper

logger = log.get_logger('root')

# METHOD = {'sentence_transformer': 0,
#           'bm25': 1}

# LANGUAGES = {'en': 0, 'zh': 1, 'de': 2, 'es': 3, 'fr': 4, 'ja': 5}

TASK_TO_POOL_DIR = {
    'product-review-polarity': 'data/amazon_reviews',
    'xnli': 'data/xnli',
    'ag_news': 'data/ag_news',
    'xtc': '../projects/cmxt/data/xtc/xtc_en',
}

DATA_DIR = 'data/amazon_reviews'
# TRANSFORMER_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
TRANSFORMER_NAME = 'sentence-transformers/distiluse-base-multilingual-cased-v2'
DATASET_NAME = 'amazon_reviews_multi'
SIZE_POOL = 10000
NUM_SIM_SENT = 100

method = 0
LANG = 'en'
SAVE = False

SAVE_DIR = 'retrieved/product_review/'

def get_sim_sents(original_sents_ex: List[InputExample], sent_pool, model: SentenceTransformer,
                  num_sim_sents: int, self_prediction: bool = False, task_name: str = 'product-review-polarity') \
        -> Dict[InputExample, List[InputExample]]:
    """
    Retrieve cross-lingual semantically sentences from a pool of sentences in high-resource language (e.g., English)
    for a given list of sentences in low-resource language.

    :param original_sents_ex: the list of sentences for which similar sentences should be retrieved
    :param sent_pool: the list of sentences from which similar sentences are retrieved
    :param model: the sentence transformer model used for sentence encoding
    :param num_sim_sents: the number of retrieved similar sentences for each original sentence
    :param self_prediction: wheather to use self-prediction method at cross-lingual retrieval
    :return: a dictionary mapping original sentence to its 100 most similar sentences in a foreign language
    """

    ori_sents = [e.text_a for e in original_sents_ex]
    embedded_ori_sents = model.encode(ori_sents)
    if self_prediction:
        embedded_sent_pool = model.encode(sent_pool)  # output type: np.ndarray
    else:
        embedded_sent_pool = model.encode(sent_pool[0])

    sim_mat = cosine_similarity(embedded_ori_sents, embedded_sent_pool)

    # store the indices of k most similar sentences, k = num_sim_sents
    k_sim_sent_indices = list()
    for row in sim_mat:
        sim_sent_indices = row.argsort()[::-1][:num_sim_sents]
        k_sim_sent_indices.append(sim_sent_indices)

    ori_sent_to_sim_sents = dict()

    for (idx, sent) in enumerate(original_sents_ex):
        sent_indices = k_sim_sent_indices[idx]

        candidates = list()
        for c_idx in sent_indices:
            if self_prediction:
                if task_name == 'xnli':
                    text_a, text_b = sent_pool[c_idx].split('\t')
                    candidate = InputExample(guid=c_idx, text_a=text_a, text_b=text_b)
                else:
                    candidate = InputExample(guid=c_idx, text_a=sent_pool[c_idx])
            else:
                label = sent_pool[1][c_idx]
                # label = '1' if star < 3 else '2'
                if task_name == 'xnli':
                    text_a, text_b = sent_pool[0][c_idx].split('\t')
                    candidate = InputExample(guid=c_idx, text_a=text_a, text_b=text_b, label=label)
                else:
                    candidate = InputExample(guid=c_idx, text_a=sent_pool[0][c_idx], label=label)
            candidates.append(candidate)

        ori_sent_to_sim_sents[sent] = candidates

    return ori_sent_to_sim_sents

# get_sim_sents(eval_data, sent_pool, sent_encoder, num_sim_sent, self_prediction)
def get_random_sents(original_sents_ex: List[InputExample], sent_pool, num_sim_sents: int,
                     self_prediction: bool = False, seed=1213):
    sent_pool = list(zip(sent_pool[0], sent_pool[1]))
    random.seed(seed)
    random.shuffle(sent_pool)
    pool_size = len(sent_pool)

    ori_sent_to_sim_sents = dict()
    for example in original_sents_ex:
        rand_idx = random.randint(0, pool_size - num_sim_sents)
        priming_sents = sent_pool[rand_idx : rand_idx+num_sim_sents]
        candidates = []
        for idx, sent in enumerate(priming_sents):
            if self_prediction:
                candidate = InputExample(guid=idx, text_a=sent[0])
            else:
                candidate = InputExample(guid=idx, text_a=sent[0], label=sent[1])
            candidates.append(candidate)

        ori_sent_to_sim_sents[example] = candidates

    return ori_sent_to_sim_sents



def save_sim_sents(sents: Dict[str, List[str]], save_path: str):
    with open(save_path, 'wb') as f:
        pickle.dump(sents, f)

def retrieve_sim_labeled_sents(model_wrapper: TransformerModelWrapper, device, eval_data: List[InputExample],
                               dataset_name: str = 'amazon_reviews_multi', save_dir: str = 'retrieved/product_review/',
                               transformer_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                               lang: str = 'en', size_pool: int = 10000, num_sim_sent: int = 100, save: bool = False,
                               method: str = 'sentence_transformer', seed: int = 42, self_prediction: bool = False,
                               num_priming: int = 1, random_retrieval: bool = False,
                               task_name: str = 'product-review-polarity') -> Dict[InputExample, List[InputExample]]:
    """
    Retrieve the candidates from sentence pool most similar to the input sequence together with the label predicted by
    by the model.

    :param model_wrapper: the transformer model wrapper
    :param device: the device used
    :param data_dir: the directory of the input sequence data
    :param dataset_name: the name of the dataset from which the sentence pool is extracted
    :param save_dir: the directory to save the retrieved sentences if required
    :param transformer_name: the name of the sentence transformer used for sentence retrieval
    :param lang: the high-resource language of the sentence pool
    :param size_pool: the size of the sentence pool
    :param num_sim_sent: the number of the retrieved sentence
    :param save: if save the retrieved sentences or not
    :param method: which information retrieval method to use
    :param seed: random seed for initialization
    :param self_prediction: wheather to use self-prediction at cross-lingual retrieval
    :param num_priming: the number of retrieved cross-lingual priming sentences
    :return: a dictionary mapping a input sequence to its high-resource similar sentences with the label
    """

    # sent_pool = load_dataset(dataset_name, lang, split='train')

    # # randomly select SIZE_POOL sentences from the training set to comprise the sentence pool.
    # sents, stars = sent_pool['review_body'], sent_pool['stars']
    # sents_stars = list(zip(sents, stars))
    # random.seed(seed)
    # random.shuffle(sents_stars)
    #
    # sent_pool = [sent for sent, _ in sents_stars[:size_pool]]
    # if not self_prediction:
    #     sent_pool = (sent_pool, [star for _, star in sents_stars[:size_pool]])

    sent_pool_file_path = f'{TASK_TO_POOL_DIR[task_name]}/sent_pool_{lang}.txt'
    sent_pool = []
    labels = []
    with open(sent_pool_file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if task_name == 'xnli':
                text_a, text_b, label = line.split('\t')
                sent = text_a+'\t'+text_b
            elif task_name == 'xtc':
                if line.startswith('id'):
                    continue
                else:
                    idx, _, label, sent = line.split('\t')
            else:
                try:
                    sent, label = line.split('\t')
                except:
                    continue
            sent_pool.append(sent.strip())
            labels.append(label.strip())
    if not self_prediction:
        sent_pool = (sent_pool, labels)

    logger.info('Load original data from file')

    # retrieve similar sentences by sentence transformer
    if random_retrieval:
        ori_sent_to_sim_sents = get_random_sents(eval_data, sent_pool, num_priming, self_prediction, seed)
        logger.info('create dictionary mapping original sentence to random sentences.')
    else:
        # load sentence transformer
        if transformer_name == 'average_pooling':
            # load sentence transformer by combining PLM with average pooling method
            mbert = models.Transformer('bert-base-multilingual-cased')
            emb_dim = mbert.get_word_embedding_dimension()
            pooling = models.Pooling(emb_dim)
            sent_encoder = SentenceTransformer(modules=[mbert, pooling])
        else:
            # load from pretrained sentence transformer
            sent_encoder = SentenceTransformer(transformer_name)

        ori_sent_to_sim_sents = get_sim_sents(eval_data, sent_pool, sent_encoder, num_sim_sent, self_prediction,
                                              task_name)
        logger.info(f'create dictionary mapping original sentence to similar sentences.')

        if save:
            save_file_name = f'sim_sents_{str(lang)}_method{str(method)}.pk'
            save_path = os.path.join(save_dir, save_file_name)
            save_sim_sents(ori_sent_to_sim_sents, save_path)

        # self prediction and then save similar sentences with labeled by self prediction
    labeled_ori_sent_to_sim_sents = dict()
        # wrapper_config = WrapperConfig(model_type='bert', model_name_or_path='bert-base-multilingual-cased',
        #                                wrapper_type='mlm', task_name='product-review-polarity', max_seq_length=512,
        #                                label_list=['1', '2'])
        #
        # model_wrapper = TransformerModelWrapper(wrapper_config)
        # logger.info('Load a model wrapper from a preset configuration.')

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if self_prediction:
        for ori_sent, sim_sents in ori_sent_to_sim_sents.items():
        # sim_sents = list(ori_sent_to_sim_sents.values())[0]
            labeled_data = model_wrapper.self_predict(sim_sents[:num_priming], device)
            labeled_ori_sent_to_sim_sents[ori_sent] = labeled_data
        return labeled_ori_sent_to_sim_sents

    else:
        return ori_sent_to_sim_sents


        # logger.info('Saving logits as pickle file.')
        # with open('tmp.pk', 'wb') as f:
        #     pickle.dump(labeled_ori_sent_to_sim_sents, f)

def add_priming_data(model_wrapper: TransformerModelWrapper, device, eval_data: List[InputExample],
                     num_priming: int = 1, dataset_name: str = 'amazon_reviews_multi',
                     save_dir: str = 'retrieved/product_review/', random_retrieval: bool = False,
                     transformer_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                     lang: str = 'en', size_pool: int = 10000, num_sim_sent: int = 100, save: bool = False,
                     method: str = 'sentence_transformer', seed: int = 42, self_prediction: bool = False,
                     task_name: str = 'product-review-polarity') -> List[InputExample]:

    labeled_ori_sent_to_sim_sents = retrieve_sim_labeled_sents(model_wrapper=model_wrapper, device=device,
                                    eval_data=eval_data, dataset_name=dataset_name, num_sim_sent=num_sim_sent,
                                    save_dir=save_dir, transformer_name=transformer_name, lang=lang,
                                    size_pool=size_pool, save=save, method=method, seed=seed,
                                    self_prediction=self_prediction, num_priming=num_priming,
                                    random_retrieval=random_retrieval, task_name=task_name)

    for example in eval_data:
        example.meta = {'priming_data': labeled_ori_sent_to_sim_sents[example]}

    return eval_data




