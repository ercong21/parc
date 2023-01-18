from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.chrf_score import sentence_chrf
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def create_datasets(hypo_path, ref_path):
    ref_set = []
    hypo_set = []
    with open(hypo_path, 'r', encoding='utf-8') as f_hypo:
        with open(ref_path, 'r', encoding='utf-8') as f_ref:
            hypos = [tuple(line_hypo.split('\t')) for line_hypo in f_hypo.readlines()]
            refs = [(idx, line_ref.split('\t')[0], line_ref.split('\t')[1]) for idx, line_ref in enumerate(f_ref)]
            for idx_hypo, text1_hypo, text2_hypo in hypos:
                for idx_ref, text1_ref, text2_ref in refs:
                    if str(idx_hypo) == str(idx_ref):
                        hypo_set.extend([text1_hypo, text2_hypo])
                        ref_set.extend([text1_ref, text2_ref])

    return ref_set, hypo_set


def compute_bleu_chrf(ref_set, hypo_set, smooth):
    bleu_total = 0
    chrf_total = 0
    for ref, hypo in tqdm(zip(ref_set, hypo_set)):
        bleu_total += sentence_bleu([ref], hypo, smoothing_function=smooth.method1)
        chrf_total += sentence_chrf([ref], hypo)

    return round(bleu_total / len(ref_set) * 100, 4), round(chrf_total / len(ref_set) * 100, 4)


def compute_similarity(ref_set, hypo_set, model):
    ref_embeddings = model.encode(ref_set)
    hypo_embeddings = model.encode(hypo_set)
    sim = (ref_embeddings * hypo_embeddings).sum(axis=1) / (np.linalg.norm(ref_embeddings, axis=1) *
                                                            np.linalg.norm(hypo_embeddings, axis=1))
    return round(sim.mean() * 100, 4)

if __name__ == '__main__':
    hypo_path_ur = 'data/xnli/trans_ur'
    ref_path_ur = 'data/xnli/ur_ov.txt'
    hypo_path_sw = 'data/xnli/trans_sw'
    ref_path_sw = 'data/xnli/sw_ov.txt'
    model_path = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

    #  test
    # hypo_path_ur = 'test_data/test_hypo'
    # ref_path_ur = 'test_data/test_ref'
    # hypo_path_sw = 'test_data/test_hypo'
    # ref_path_sw = 'test_data/test_ref'

    ref_set_ur, hypo_set_ur = create_datasets(hypo_path_ur, ref_path_ur)
    ref_set_sw, hypo_set_sw = create_datasets(hypo_path_sw, ref_path_sw)

    model = SentenceTransformer(model_path)
    smooth = SmoothingFunction()

    bleu_ur, chrf_ur = compute_bleu_chrf(ref_set_ur, hypo_set_ur, smooth)
    bleu_sw, chrf_sw = compute_bleu_chrf(ref_set_sw, hypo_set_sw, smooth)
    sim_ur = compute_similarity(ref_set_ur, hypo_set_ur, model)
    sim_sw = compute_similarity(ref_set_sw, hypo_set_sw, model)

    results = {'ur': {'bleu': bleu_ur, 'chrf': chrf_ur, 'sim': sim_ur},
               'sw': {'bleu': bleu_sw, 'chrf': chrf_sw, 'sim': sim_sw},}

    print(results)


