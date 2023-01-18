#CUDA_VISIBLE_DEVICES=6 python -u cli.py --pattern_ids 0 --sentence_transformer_name 'sentence-transformers/LaBSE'
#CUDA_VISIBLE_DEVICES=6 python -u cli.py --pattern_ids 0 --priming --sentence_transformer_name 'sentence-transformers/LaBSE'
#CUDA_VISIBLE_DEVICES=6 python -u cli.py --pattern_ids 0 --priming --sentence_transformer_name 'sentence-transformers/LaBSE' --num_priming 3
#CUDA_VISIBLE_DEVICES=6 python -u cli.py --pattern_ids 0 --priming --sentence_transformer_name 'sentence-transformers/LaBSE' --num_priming 5
#CUDA_VISIBLE_DEVICES=6 python -u cli.py --pattern_ids 0 --priming --sentence_transformer_name 'sentence-transformers/LaBSE' --self_prediction
#CUDA_VISIBLE_DEVICES=6 python -u cli.py --pattern_ids 0 --priming --sentence_transformer_name 'sentence-transformers/LaBSE' --num_priming 3 --self_prediction
#CUDA_VISIBLE_DEVICES=6 python -u cli.py --pattern_ids 0 --priming --sentence_transformer_name 'sentence-transformers/LaBSE' --num_priming 5 --self_prediction


CUDA_VISIBLE_DEVICES=0 python -u cli.py --pattern_ids 2 --eval_langs ig sn mt co sm st haw zu ny
CUDA_VISIBLE_DEVICES=0 python -u cli.py --pattern_ids 2 --priming --eval_langs ig sn mt co sm st haw zu ny
CUDA_VISIBLE_DEVICES=0 python -u cli.py --pattern_ids 2 --priming --num_priming 3 --eval_langs ig sn mt co sm st haw zu ny
CUDA_VISIBLE_DEVICES=0 python -u cli.py --pattern_ids 2 --priming --num_priming 5 --eval_langs ig sn mt co sm st haw zu ny
CUDA_VISIBLE_DEVICES=0 python -u cli.py --pattern_ids 2 --priming --self_prediction --eval_langs ig sn mt co sm st haw zu ny
CUDA_VISIBLE_DEVICES=0 python -u cli.py --pattern_ids 2 --priming --num_priming 3 --self_prediction --eval_langs ig sn mt co sm st haw zu ny
CUDA_VISIBLE_DEVICES=0 python -u cli.py --pattern_ids 2 --priming --num_priming 5 --self_prediction --eval_langs ig sn mt co sm st haw zu ny
