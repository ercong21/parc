#CUDA_VISIBLE_DEVICES=1 python cli.py --task_name 'ag_news' --data_dir 'data/ag_news' --pattern_ids 0 --priming --baseline_ft --eval_langs mn
#CUDA_VISIBLE_DEVICES=1 python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 2 --priming --baseline_ft --eval_langs mn
#CUDA_VISIBLE_DEVICES=1 python cli.py --pattern_ids 2 --priming --baseline_ft --eval_langs mn --num_priming 3
#CUDA_VISIBLE_DEVICES=1 python cli.py --pattern_ids 2 --priming --baseline_ft --eval_langs mn --num_priming 10

#CUDA_VISIBLE_DEVICES=2 python -u cli.py --eval_langs mn --pattern_ids 2 --priming --sentence_transformer_name 'sentence-transformers/LaBSE' --self_prediction --task_name 'ag_news' --data_dir 'data/ag_news'
#CUDA_VISIBLE_DEVICES=2 python -u cli.py --eval_langs mn --pattern_ids 2 --priming --sentence_transformer_name 'sentence-transformers/LaBSE' --num_priming 3 --self_prediction --task_name 'ag_news' --data_dir 'data/ag_news'
#CUDA_VISIBLE_DEVICES=2 python -u cli.py --eval_langs mn --pattern_ids 2 --priming --sentence_transformer_name 'sentence-transformers/LaBSE' --num_priming 5 --self_prediction --task_name 'ag_news' --data_dir 'data/ag_news'
#CUDA_VISIBLE_DEVICES=2 python -u cli.py --eval_langs mn --pattern_ids 1 --priming --sentence_transformer_name 'sentence-transformers/LaBSE' --self_prediction --task_name 'xnli' --data_dir 'data/xnli'
#CUDA_VISIBLE_DEVICES=2 python -u cli.py --eval_langs mn --pattern_ids 1 --priming --sentence_transformer_name 'sentence-transformers/LaBSE' --num_priming 3 --self_prediction --task_name 'xnli' --data_dir 'data/xnli'
#CUDA_VISIBLE_DEVICES=2 python -u cli.py --eval_langs mn --pattern_ids 1 --priming --sentence_transformer_name 'sentence-transformers/LaBSE' --num_priming 5 --self_prediction --task_name 'xnli' --data_dir 'data/xnli'

CUDA_VISIBLE_DEVICES=2 python -u cli.py --eval_langs mn --pattern_ids 0 --sentence_transformer_name 'sentence-transformers/LaBSE'
CUDA_VISIBLE_DEVICES=2 python -u cli.py --eval_langs mn --pattern_ids 0 --priming --sentence_transformer_name 'sentence-transformers/LaBSE'
CUDA_VISIBLE_DEVICES=2 python -u cli.py --eval_langs mn --pattern_ids 0 --priming --sentence_transformer_name 'sentence-transformers/LaBSE' --num_priming 3
CUDA_VISIBLE_DEVICES=2 python -u cli.py --eval_langs mn --pattern_ids 0 --priming --sentence_transformer_name 'sentence-transformers/LaBSE' --num_priming 5
CUDA_VISIBLE_DEVICES=2 python -u cli.py --eval_langs mn --pattern_ids 0 --priming --sentence_transformer_name 'sentence-transformers/LaBSE' --self_prediction
CUDA_VISIBLE_DEVICES=2 python -u cli.py --eval_langs mn --pattern_ids 0 --priming --sentence_transformer_name 'sentence-transformers/LaBSE' --num_priming 3 --self_prediction
CUDA_VISIBLE_DEVICES=2 python -u cli.py --eval_langs mn --pattern_ids 0 --priming --sentence_transformer_name 'sentence-transformers/LaBSE' --num_priming 5 --self_prediction
CUDA_VISIBLE_DEVICES=2 python -u cli.py --eval_langs mn --pattern_ids 2 --sentence_transformer_name 'sentence-transformers/LaBSE'
CUDA_VISIBLE_DEVICES=2 python -u cli.py --eval_langs mn --pattern_ids 2 --priming --sentence_transformer_name 'sentence-transformers/LaBSE'
CUDA_VISIBLE_DEVICES=2 python -u cli.py --eval_langs mn --pattern_ids 2 --priming --sentence_transformer_name 'sentence-transformers/LaBSE' --num_priming 3
CUDA_VISIBLE_DEVICES=2 python -u cli.py --eval_langs mn --pattern_ids 2 --priming --sentence_transformer_name 'sentence-transformers/LaBSE' --num_priming 5
CUDA_VISIBLE_DEVICES=2 python -u cli.py --eval_langs mn --pattern_ids 2 --priming --sentence_transformer_name 'sentence-transformers/LaBSE' --self_prediction
CUDA_VISIBLE_DEVICES=2 python -u cli.py --eval_langs mn --pattern_ids 2 --priming --sentence_transformer_name 'sentence-transformers/LaBSE' --num_priming 3 --self_prediction
CUDA_VISIBLE_DEVICES=2 python -u cli.py --eval_langs mn --pattern_ids 2 --priming --sentence_transformer_name 'sentence-transformers/LaBSE' --num_priming 5 --self_prediction
