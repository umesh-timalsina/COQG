#!/bin/bash
cd ../OpenNMT-py
python3 preprocess.py \
-train_src ../data/train/coqa-src-train-v1.1.txt \
-train_tgt ../data/train/coqa-tgt-train-v1.1.txt \
-valid_src ../data/dev/coqa-src-valid-v1.1.txt \
-valid_tgt ../data/dev/coqa-tgt-valid-v1.1.txt \
-save_data ../data/data.feat.coqg \
-src_seq_length 10000 -tgt_seq_length 10000 \
-dynamic_dict

 PYTHONPATH=../OpenNMT-py python ../OpenNMT-py/tools/embeddings_to_torch.py \
 -emb_file_enc ../data/glove.42B.300d.txt \
 -emb_file_dec ../data/glove.42B.300d.txt \
 -dict_file ../data/data.feat.coqg.vocab.pt \
 -output_file ../data/data.feat.coqg.embed \
 -verbose

 python OpenNMT-py/train.py \
    -data data/data.feat.coqg \
    -save_model COQGMODEL`date "+%Y%m%d"`/coqg_copy_history_all \
    -save_checkpoint_steps 5000 \
    -copy_attn \
    -reuse_copy_attn \
    -word_vec_size 300 \
    -pre_word_vecs_enc data/data.feat.coqg.embed.enc.pt \
    -pre_word_vecs_dec data/data.feat.coqg.embed.dec.pt \
    -train_steps 150000 \
    -gpu_ranks 0 \
    -seed 123 2>&1 train_op_`date "+%Y%m%d"`.log &

 python OpenNMT-py/translate.py -model seq2seq_models/seq2seq_copy_step_35000.pt -src data/dev/coqa-src-valid-v1.0.txt -output seq2seq_models/pred.txt -replace_unk -verbose -gpu 0
