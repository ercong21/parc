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

If you make use of the code in this repository, please cite the following papers:

    @article{nie2022cross,
    title={Cross-Lingual Retrieval Augmented Prompt for Low-Resource Languages},
    author={Nie, Ercong and Liang, Sheng and Schmid, Helmut and Sch{\"u}tze, Hinrich},
    journal={arXiv preprint arXiv:2212.09651},
    year={2022}
    }