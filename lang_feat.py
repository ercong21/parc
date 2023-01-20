
import numpy as np
import pandas as pd
import argparse
import lang2vec.lang2vec as l2v
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

FEATURES = ['syntax_average', 'phonology_average', 'inventory_average', 'fam', 'geo']


def cal_sim(lang1, lang2, features, weights):
    feat_vecs1 = list()
    feat_vecs2 = list()
    for feature in features:
        vec1 = l2v.get_features(lang1, feature)[lang1]
        vec2 = l2v.get_features(lang2, feature)[lang2]
        vec1_new = [0.1]
        vec2_new = [0.1]
        for v1, v2 in zip(vec1, vec2):
            if isinstance(v1, str):
                v1 = 0.0
            if isinstance(v2, str):
                v2 = 0.0
            vec1_new.append(v1)
            vec2_new.append(v2)

        feat_vecs1.append(vec1_new)
        feat_vecs2.append(vec2_new)

    sims = dict()
    for feature, vec1, vec2 in zip(features, feat_vecs1, feat_vecs2):
        sims[feature] = cosine_similarity(np.array(vec1), np.array(vec2))

    sim_score = cal_sim_score(sims, weights, features)
    sims['sim_score'] = sim_score

    return {f"{lang1}-{lang2}": sims}


def cal_sim_score(sims, weights, features):
    score = 0
    for feature in features:
        score += sims[feature] * weights[feature]

    return score


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    emb2 = emb2.T
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * (np.linalg.norm(emb2)))


def save_all_sims(hr_langs, lr_langs, features, weights, save_path):
    all_sims = dict()
    for lang1 in tqdm(hr_langs, unit='lang1'):
        for lang2 in tqdm(lr_langs, unit='lang1-lang2'):
            all_sims.update(cal_sim(lang1, lang2, features, weights))

            data = [[sim for sim in features.values()] for features in all_sims.values()]
            index = list(all_sims.keys())
            columns = features + ['sim_score']
            df = pd.DataFrame(data, columns=columns, index=index)
            df.to_csv(save_path)


def corr_analysis(file_path, show):
    data = pd.read_csv(file_path, index_col=0)

    if show:
        corr = data.corr()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0, len(data.columns), 1)
        ax.set_xticks(ticks)
        plt.xticks(rotation=90)
        ax.set_yticks(ticks)
        ax.set_xticklabels(data.columns)
        ax.set_yticklabels(data.columns)
        plt.show()

    results = dict()
    # calculate spearman and pearson correlation
    for feat in data.columns:
        if feat != 'Performance':
            s_corr = spearmanr(data['Performance'], data[feat])
            p_corr = pearsonr(data['Performance'], data[feat])
            results[feat] = {'spearman': (s_corr.correlation, s_corr.pvalue),'pearson': p_corr}

    print(results)


def main():
    parser = argparse.ArgumentParser(description='Arguments used to calculate language similarities.')

    parser.add_argument('--hr_langs', default=['eng', 'deu', 'zho', 'hin', 'ceb'], type=str, nargs='+',
                        help='list of high resource languages.')
    parser.add_argument('--lr_langs', default=['eng', 'afr', 'urd', 'swa', 'tel', 'tam', 'mon', 'uzb', 'mya', 'jav', 'tag'],
                        type=str, nargs='+', help='list of low resource langauges')
    parser.add_argument('--save_path', default='lang_sim.csv', type=str, help='file path to save the results.')
    parser.add_argument('--file_path', default='lang_corr.csv', type=str,
                        help='file path to the language correlation analysis data')
    parser.add_argument('--analyze', action='store_true', help='whether to conduct correlationa anlysis')
    parser.add_argument('--show_plt', action='store_true', help='whether to show the correlation plot')

    args = parser.parse_args()

    features = ['syntax_average', 'phonology_average', 'inventory_average', 'fam', 'geo']
    weights = {'syntax_average': 0.3, 'phonology_average': 0.1, 'inventory_average': 0.1, 'fam': 0.3, 'geo': 0.2}
    if args.analyze:
        corr_analysis(args.file_path, args.show_plt)
    else:
        save_all_sims(args.hr_langs, args.lr_langs, features=features, weights=weights, save_path=args.save_path)


if __name__ == '__main__':
    main()
