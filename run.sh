#!/bin/bash

# cd submodules/bert_ner
# pip3 install -r requirements.txt
# python3 run_ner.py --data_dir=data/ --bert_model=bert-base-cased --task_name=ner --output_dir=out_base --max_seq_length=128 --do_train --num_train_epochs 5 --do_eval --warmup_proportion=0.1
# cd ../..
pip3 install --no-cache-dir -q -r requirements.txt
python3 src/main.py
