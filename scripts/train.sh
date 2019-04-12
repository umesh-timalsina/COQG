#!/bin/bash
num_history=2
out_dir=COQGMODEL`${num_history}`
mkdir -v $out_dir
cd nmt
python -m nmt.nmt \
    --num_units=256 \
    --num_layers=2 \
    --num_encoder_layers=2 \
    --num_decoder_layers=1 \
    --encoder_type=gnmt \
    --attention=scaled_luong \
    --attention_architecture=standard \
    --num_train_steps=90000 \
    --init_op=glorot_normal \
    --src=para \
    --tgt=ques \
    --vocab_prefix=../data/vocab \
    --train_prefix=../data/train/coqg_h${num_history}_train \
    --dev_prefix=../data/dev/coqg_h${num_history}_dev \
    --test_prefix=../data/test/coqg_h${num_history}_test \
    --src_max_len=1000 \
    --tgt_max_len=100 \
    --out_dir=../${out_dir} \
    --dropout=0.2 \
    --metrics=bleu \
    --batch_size=64
