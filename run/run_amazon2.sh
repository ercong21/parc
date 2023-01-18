CUDA_VISIBLE_DEVICES=5 python -u cli.py --pattern_ids 2 --sentence_transformer_name 'sentence-transformers/LaBSE'
CUDA_VISIBLE_DEVICES=5 python -u cli.py --pattern_ids 2 --priming --sentence_transformer_name 'sentence-transformers/LaBSE'
CUDA_VISIBLE_DEVICES=5 python -u cli.py --pattern_ids 2 --priming --sentence_transformer_name 'sentence-transformers/LaBSE' --num_priming 3
CUDA_VISIBLE_DEVICES=5 python -u cli.py --pattern_ids 2 --priming --sentence_transformer_name 'sentence-transformers/LaBSE' --num_priming 5
CUDA_VISIBLE_DEVICES=5 python -u cli.py --pattern_ids 2 --priming --sentence_transformer_name 'sentence-transformers/LaBSE' --self_prediction
CUDA_VISIBLE_DEVICES=5 python -u cli.py --pattern_ids 2 --priming --sentence_transformer_name 'sentence-transformers/LaBSE' --num_priming 3 --self_prediction
CUDA_VISIBLE_DEVICES=5 python -u cli.py --pattern_ids 2 --priming --sentence_transformer_name 'sentence-transformers/LaBSE' --num_priming 5 --self_prediction
