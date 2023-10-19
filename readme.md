# PARC
## Prompts augmented by retrieval crosslingually


## Description
Codes and data used for the paper *Cross-Lingual Retrieval Augmented Prompt for Low-Resource Languages*.   

## Usage
```commandline
python cli.py --task_name [TASK NAME] --pattern_ids [PATTERN IDS] --data_dir [DATA DIR} \
--eval_langs [EVAL LANGS] --priming --self_prediction
```

## Citation
```
@inproceedings{nie-etal-2023-cross,
    title = "Cross-Lingual Retrieval Augmented Prompt for Low-Resource Languages",
    author = {Nie, Ercong  and
      Liang, Sheng  and
      Schmid, Helmut  and
      Sch{\"u}tze, Hinrich},
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.528",
    doi = "10.18653/v1/2023.findings-acl.528",
    pages = "8320--8340",
    abstract = "Multilingual Pretrained Language Models (MPLMs) perform strongly in cross-lingual transfer. We propose Prompts Augmented by Retrieval Crosslingually (PARC) to improve zero-shot performance on low-resource languages (LRLs) by augmenting the context with prompts consisting of semantically similar sentences retrieved from a high-resource language (HRL). PARC improves zero-shot performance on three downstream tasks (sentiment classification, topic categorization, natural language inference) with multilingual parallel test sets across 10 LRLs covering 6 language families in unlabeled (+5.1{\%}) and labeled settings (+16.3{\%}). PARC also outperforms finetuning by 3.7{\%}. We find a significant positive correlation between cross-lingual transfer performance on one side, and the similarity between high- and low-resource languages as well as the amount of low-resource pretraining data on the other side. A robustness analysis suggests that PARC has the potential to achieve even stronger performance with more powerful MPLMs.",
}
```
