#!/bin/bash

now="COQGMODEL"$(date +"%Y.%m.%d_%H.%M.%S")
mkdir -v $now
cd nmt
python -m nmt.nmt \
    --attention=scaled_luong \
    --src=para --tgt=ques \
    --vocab_prefix=../data/vocab \
    --train_prefix=../data/train/coqg_train_h2 \
    --dev_prefix=../data/dev/coqg_dev_h2 \
    --test_prefix=../data/test/coqg_test_h2 \
    --src_max_len=500 \
    --tgt_max_len=50 \
    --out_dir=../${now} \
    --num_train_steps=45000 \
    --steps_per_stats=1000 \
    --num_layers=2 \
    --num_units=256 \
    --dropout=0.2 \
    --metrics=bleu