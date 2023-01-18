#!/bin/bash
# python cli.py --priming --self_prediction --pattern_ids 2 --eval_langs zh
# python cli.py --priming --self_prediction --eval_from_datasets --data_dir amazon_reviews_multi --pattern_ids 2 --eval_langs de es fr ja zh
# python cli.py --priming --data_dir 'data/amazon_reviews' --pattern_ids 2 --retrieved_lang de
# python cli.py --data_dir 'data/amazon_reviews' --pattern_ids 2 --retrieved_lang de
# python cli.py --priming --data_dir 'data/amazon_reviews' --pattern_ids 2 --retrieved_lang zh
# python cli.py --data_dir 'data/amazon_reviews' --pattern_ids 2 --retrieved_lang zh
# python cli.py --priming --data_dir 'data/amazon_reviews' --pattern_ids 2 --retrieved_lang hi
# python cli.py --data_dir 'data/amazon_reviews' --pattern_ids 2 --retrieved_lang hi
# python cli.py --priming --data_dir 'data/amazon_reviews' --pattern_ids 2 --retrieved_lang ceb

# verbalizer: bad - great
# test with XLM-R model
# python cli.py --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r' --pattern_ids 0 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2'
# python cli.py --priming --self_prediction --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r' --pattern_ids 0 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2'
# python cli.py --priming --self_prediction --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r' --pattern_ids 0 --num_priming 3 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2'
# python cli.py --priming --self_prediction --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r' --pattern_ids 0 --num_priming 5 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2'
# python cli.py --priming --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r' --pattern_ids 0 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2'
# python cli.py --priming --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r' --pattern_ids 0 --num_priming 3 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2'
# python cli.py --priming --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r' --pattern_ids 0 --num_priming 5 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2'

# test for correlation alaysis
# python cli.py --pattern_ids 2 --priming --self_prediction --retrieved_lang ceb
# python cli.py --pattern_ids 2 --priming --self_prediction --retrieved_lang de
# python cli.py --pattern_ids 2 --priming --self_prediction --retrieved_lang en
# python cli.py --pattern_ids 2 --priming --self_prediction --retrieved_lang zh
# python cli.py --pattern_ids 2 --priming --self_prediction --retrieved_lang hi
#
# python cli.py --pattern_ids 2 --priming --retrieved_lang ceb --num_priming 10
# python cli.py --pattern_ids 2 --priming --retrieved_lang de --num_priming 10
# python cli.py --pattern_ids 2 --priming --retrieved_lang en --num_priming 10
# python cli.py --pattern_ids 2 --priming --retrieved_lang zh --num_priming 10
# python cli.py --pattern_ids 2 --priming --retrieved_lang hi --num_priming 10

# test with mbert
# python cli.py --pattern_ids 2
# python cli.py --priming --self_prediction --pattern_ids 2
# python cli.py --priming --self_prediction --pattern_ids 2 --num_priming 3
# python cli.py --priming --self_prediction --pattern_ids 2 --num_priming 5
# python cli.py --priming --pattern_ids 2
# python cli.py --priming --pattern_ids 2 --num_priming 3
# python cli.py --priming --pattern_ids 2 --num_priming 5

# test with mBERT and averaging pooling retriever
# python cli.py --pattern_ids 2 --sentence_transformer_name 'average_pooling'
# python cli.py --priming --self_prediction --pattern_ids 2 --sentence_transformer_name 'average_pooling'
# python cli.py --priming --self_prediction --pattern_ids 2 --num_priming 3 --sentence_transformer_name 'average_pooling'
# python cli.py --priming --self_prediction --pattern_ids 2 --num_priming 5 --sentence_transformer_name 'average_pooling'
# python cli.py --priming --pattern_ids 2 --sentence_transformer_name 'average_pooling'
# python cli.py --priming --pattern_ids 2 --num_priming 3 --sentence_transformer_name 'average_pooling'
# python cli.py --priming --pattern_ids 2 --num_priming 5 --sentence_transformer_name 'average_pooling'

# test with mBERT and distiluse retriever for AGNews task
# python cli.py --task_name 'ag_news' --data_dir 'data/ag_news' --pattern_ids 2 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2'
# python cli.py --task_name 'ag_news' --data_dir 'data/ag_news' --priming --self_prediction --pattern_ids 2 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2'
# python cli.py --task_name 'ag_news' --data_dir 'data/ag_news' --priming --self_prediction --pattern_ids 2 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2' --num_priming 3
# python cli.py --task_name 'ag_news' --data_dir 'data/ag_news' --priming --self_prediction --pattern_ids 2 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2' --num_priming 5
# python cli.py --task_name 'ag_news' --data_dir 'data/ag_news' --priming --pattern_ids 2 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2'
# python cli.py --task_name 'ag_news' --data_dir 'data/ag_news' --priming --pattern_ids 2 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2' --num_priming 3
# python cli.py --task_name 'ag_news' --data_dir 'data/ag_news' --priming --pattern_ids 2 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2' --num_priming 5

# test with XLM-R and paraphrase retriever for AGNews task
# python cli.py --task_name 'ag_news' --data_dir 'data/ag_news' --pattern_ids 2 --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r'
# python cli.py --task_name 'ag_news' --data_dir 'data/ag_news' --pattern_ids 2 --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r' --priming
# python cli.py --task_name 'ag_news' --data_dir 'data/ag_news' --pattern_ids 2 --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r' --priming --num_priming 3
# python cli.py --task_name 'ag_news' --data_dir 'data/ag_news' --pattern_ids 2 --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r' --priming --num_priming 5
# python cli.py --task_name 'ag_news' --data_dir 'data/ag_news' --pattern_ids 2 --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r' --priming --self_prediction
# python cli.py --task_name 'ag_news' --data_dir 'data/ag_news' --pattern_ids 2 --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r' --priming --self_prediction --num_priming 3
# python cli.py --task_name 'ag_news' --data_dir 'data/ag_news' --pattern_ids 2 --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r' --priming --self_prediction --num_priming 5

# test with mBERT and pooling retriever for AGNews task
# python cli.py --task_name 'ag_news' --data_dir 'data/ag_news' --pattern_ids 2 --sentence_transformer_name 'average_pooling'
# python cli.py --task_name 'ag_news' --data_dir 'data/ag_news' --priming --pattern_ids 2 --sentence_transformer_name 'average_pooling'
# python cli.py --task_name 'ag_news' --data_dir 'data/ag_news' --priming --pattern_ids 2 --sentence_transformer_name 'average_pooling' --num_priming 3
# python cli.py --task_name 'ag_news' --data_dir 'data/ag_news' --priming --pattern_ids 2 --sentence_transformer_name 'average_pooling' --num_priming 5
# python cli.py --task_name 'ag_news' --data_dir 'data/ag_news' --priming --self_prediction --pattern_ids 2 --sentence_transformer_name 'average_pooling'
# python cli.py --task_name 'ag_news' --data_dir 'data/ag_news' --priming --self_prediction --pattern_ids 2 --sentence_transformer_name 'average_pooling' --num_priming 3
# python cli.py --task_name 'ag_news' --data_dir 'data/ag_news' --priming --self_prediction --pattern_ids 2 --sentence_transformer_name 'average_pooling' --num_priming 5

# test with mBERT and distil use retriever for XNLI
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 1 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2'
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --self_prediction --pattern_ids 1 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2'
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --self_prediction --pattern_ids 1 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2' --num_priming 3
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --self_prediction --pattern_ids 1 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2' --num_priming 5
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --pattern_ids 1 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2'
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --pattern_ids 1 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2' --num_priming 3
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --pattern_ids 1 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2' --num_priming 5
#
# test with mBERT and pooling retriever for XNLI
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 1 --sentence_transformer_name 'average_pooling'
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --self_prediction --pattern_ids 1 --sentence_transformer_name 'average_pooling'
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --self_prediction --pattern_ids 1 --sentence_transformer_name 'average_pooling' --num_priming 3
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --self_prediction --pattern_ids 1 --sentence_transformer_name 'average_pooling' --num_priming 5
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --pattern_ids 1 --sentence_transformer_name 'average_pooling'
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --pattern_ids 1 --sentence_transformer_name 'average_pooling' --num_priming 3
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --pattern_ids 1 --sentence_transformer_name 'average_pooling' --num_priming 5


# test with xlm-r and default retriever for XNLI
#python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 1 --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r'
#python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --self_prediction --pattern_ids 1 --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r'
#python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --self_prediction --pattern_ids 1 --num_priming 3 --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r'
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --self_prediction --pattern_ids 1 --num_priming 5 --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r'
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --pattern_ids 1 --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r'
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --pattern_ids 1 --num_priming 3 --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r'
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --pattern_ids 1 --num_priming 5 --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r'

# test with mBERT and distil use retriever for XNLI
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --pattern_ids 2 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2' --num_priming 5

# test with mBERT and pooling retriever for XNLI
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 2 --sentence_transformer_name 'average_pooling'
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --self_prediction --pattern_ids 2 --sentence_transformer_name 'average_pooling'
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --self_prediction --pattern_ids 2 --sentence_transformer_name 'average_pooling' --num_priming 3
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --selssjf_prediction --pattern_ids 2 --sentence_transformer_name 'average_pooling' --num_priming 5
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --pattern_ids 2 --sentence_transformer_name 'average_pooling'
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --pattern_ids 2 --sentence_transformer_name 'average_pooling' --num_priming 3
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --pattern_ids 2 --sentence_transformer_name 'average_pooling' --num_priming 5

# correlation analysis with AG News task
# python cli.py --task_name 'ag_news' --data_dir 'data/ag_news' --pattern_ids 2 --priming --retrieved_lang zh-cn
# python cli.py --task_name 'ag_news' --data_dir 'data/ag_news' --pattern_ids 2 --priming --retrieved_lang zh-cn --self_prediction
# python cli.py --task_name 'ag_news' --data_dir 'data/ag_news' --pattern_ids 2 --priming --retrieved_lang hi --self_prediction

# test with xlm-r and default retriever for XNLI
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 2 --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r'
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --self_prediction --pattern_ids 2 --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r'
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --self_prediction --pattern_ids 2 --num_priming 3 --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r'
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --self_prediction --pattern_ids 2 --num_priming 5 --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r'
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --pattern_ids 2 --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r'
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --pattern_ids 2 --num_priming 3 --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r'
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --pattern_ids 2 --num_priming 5 --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r'


# python cli.py --pattern_ids 0 2 --eval_langs en ur
# python cli.py --priming --self_prediction --pattern_ids 0 2 --eval_langs en ur
# python cli.py --priming --self_prediction --pattern_ids 0 2 --eval_langs en ur --num_priming 3
# python cli.py --priming  --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r' --pattern_ids 2
# python cli.py --priming  --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r' --pattern_ids 2 --num_priming 3
# python cli.py --priming  --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r' --pattern_ids 2 --num_priming 5
# python cli.py --priming  --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r' --pattern_ids 2 --num_priming 10
# python cli.py --priming  --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r' --pattern_ids 2 --num_priming 20

# python cli.py --priming  --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r' --pattern_ids 2 --num_priming 30 --eval_langs my jw tl
# python cli.py --priming --self_prediction --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r' --pattern_ids 2
# python cli.py --priming --self_prediction --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r' --pattern_ids 2 --num_priming 3
# python cli.py --priming --self_prediction --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r' --pattern_ids 2 --num_priming 5
# python cli.py --priming --self_prediction --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r' --pattern_ids 2 --num_priming 10
# python cli.py --priming --self_prediction --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r' --pattern_ids 2 --num_priming 20
# python cli.py --priming --self_prediction --model_name_or_path 'xlm-roberta-base' --model_type 'xlm-r' --pattern_ids 2 --num_priming 30
#
# python cli.py --priming --self_prediction --pattern_ids 2 --num_priming 30


# correlation analysis with more pairs
#python cli.py --pattern_ids 2 --priming --retrieved_lang zh --num_priming 10  --eval_langs de hi ceb en
#python cli.py --pattern_ids 2 --priming --retrieved_lang de --num_priming 10  --eval_langs zh hi ceb en
#python cli.py --pattern_ids 2 --priming --retrieved_lang hi --num_priming 10  --eval_langs de zh ceb en
#python cli.py --pattern_ids 2 --priming --retrieved_lang ceb --num_priming 10  --eval_langs de hi zh en
#python cli.py --pattern_ids 2 --priming --retrieved_lang en --num_priming 10  --eval_langs de hi ceb zh

#python cli.py --pattern_ids 2 --priming --retrieved_lang my --num_priming 10  --eval_langs af ur sw te ta mn uz my jw tl de hi ceb en zh
#python cli.py --pattern_ids 2 --priming --retrieved_lang uz --num_priming 10  --eval_langs af ur sw te ta mn uz my jw tl de hi ceb en zh
#python cli.py --pattern_ids 2 --eval_langs de hi ceb zh

# test with mBERT and distil use retriever for XNLI
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 1 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2'
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --self_prediction --pattern_ids 1 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2'
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --self_prediction --pattern_ids 1 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2' --num_priming 3
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --self_prediction --pattern_ids 1 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2' --num_priming 5
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --pattern_ids 1 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2'
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --pattern_ids 1 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2' --num_priming 3
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --pattern_ids 1 --sentence_transformer_name 'sentence-transformers/distiluse-base-multilingual-cased-v2' --num_priming 5

# test with mBERT and pooling retriever for XNLI
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 1 --sentence_transformer_name 'average_pooling'
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --self_prediction --pattern_ids 1 --sentence_transformer_name 'average_pooling'
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --self_prediction --pattern_ids 1 --sentence_transformer_name 'average_pooling' --num_priming 3
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --self_prediction --pattern_ids 1 --sentence_transformer_name 'average_pooling' --num_priming 5
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --pattern_ids 1 --sentence_transformer_name 'average_pooling'
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --pattern_ids 1 --sentence_transformer_name 'average_pooling' --num_priming 3
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --priming --pattern_ids 1 --sentence_transformer_name 'average_pooling' --num_priming 5

#TODO: restest xnli task (encoding issue figured out)
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 0
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 0 --priming
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 0 --priming --self_prediction
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 0 --priming --num_priming 3
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 0 --priming --self_prediction --num_priming 3
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 0 --priming --num_priming 5
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 0 --priming --self_prediction --num_priming 5
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 0 --priming --num_priming 10
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 0 --priming --self_prediction --num_priming 10
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 0 --priming --num_priming 20
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 0 --priming --self_prediction --num_priming 20
#
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 1
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 1 --priming
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 1 --priming --self_prediction
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 1 --priming --num_priming 3
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 1 --priming --self_prediction --num_priming 3
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 1 --priming --num_priming 5
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 1 --priming --self_prediction --num_priming 5
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 1 --priming --num_priming 10
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 1 --priming --self_prediction --num_priming 10
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 1 --priming --num_priming 20
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 1 --priming --self_prediction --num_priming 20
#
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 2
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 2 --priming
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 2 --priming --self_prediction
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 2 --priming --num_priming 3
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 2 --priming --self_prediction --num_priming 3
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 2 --priming --num_priming 5
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 2 --priming --self_prediction --num_priming 5
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 2 --priming --num_priming 10
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 2 --priming --self_prediction --num_priming 10
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 2 --priming --num_priming 20
# python cli.py --task_name 'xnli' --data_dir 'data/xnli' --pattern_ids 2 --priming --self_prediction --num_priming 20

# ag news task
#python cli.py --task_name 'ag_news' --data_dir 'data/ag_news' --pattern_ids 2 --priming --num_priming 30

# random retrieval
#python cli.py --priming --random_retrieval --pattern_ids 2
#python cli.py --priming --random_retrieval --pattern_ids 2 --task_name 'ag_news' --data_dir 'data/ag_news'
#python cli.py --priming --random_retrieval --pattern_ids 2 --task_name 'xnli' --data_dir 'data/xnli'
#python cli.py --model_from_ft --pattern_ids 2
#python cli.py --model_from_ft --pattern_ids 2 --priming
python cli.py --pattern_ids 2 --priming --eval_langs en my