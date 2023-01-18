CUDA_VISIBLE_DEVICES=0 python -u cli.py --pattern_ids 0 1 2 --task_name 'xnli' --data_dir 'data/xnli' --eval_langs ur ur_ov sw sw_ov
CUDA_VISIBLE_DEVICES=0 python -u cli.py --pattern_ids 0 1 2 --priming --task_name 'xnli' --data_dir 'data/xnli' --eval_langs ur ur_ov sw sw_ov
CUDA_VISIBLE_DEVICES=0 python -u cli.py --pattern_ids 0 1 2 --priming --num_priming 3 --task_name 'xnli' --data_dir 'data/xnli' --eval_langs ur ur_ov sw sw_ov
CUDA_VISIBLE_DEVICES=0 python -u cli.py --pattern_ids 0 1 2 --priming --num_priming 5 --task_name 'xnli' --data_dir 'data/xnli' --eval_langs ur ur_ov sw sw_ov
CUDA_VISIBLE_DEVICES=0 python -u cli.py --pattern_ids 0 1 2 --priming --self_prediction --task_name 'xnli' --data_dir 'data/xnli' --eval_langs ur ur_ov sw sw_ov
CUDA_VISIBLE_DEVICES=0 python -u cli.py --pattern_ids 0 1 2 --priming --num_priming 3 --self_prediction --task_name 'xnli' --data_dir 'data/xnli' --eval_langs ur ur_ov sw sw_ov
CUDA_VISIBLE_DEVICES=0 python -u cli.py --pattern_ids 0 1 2 33.82--priming --num_priming 5 --self_prediction --task_name 'xnli' --data_dir 'data/xnli' --eval_langs ur ur_ov sw sw_ov
