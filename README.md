# COQG Baselines
We provide baseline implementations for contextual(conversational) question generator using COQA dataset, a question generator based on the COQA dataset. We provide instructions to prepare the dataset for question generation. We use the [tensorflow-nmt](https://github.com/tensorflow/nmt "Tensorflow NMT") for our experiments. This document structure is inspired form the following paper's baseline repository. We provide guidelines on how to run our pre--trained models as well as how to use our scripts to replicate our baselines. 

This repository can be cloned using the following command:
```bash
git clone --recurse-submodules git@github.com:umesh-timalsina/COQG.git
```
We recommend creating a new virtual environment the project. 

# Requirements
```
tensorflow==1.13.1-nightly
python>=3.5
spacy==2.0.12
```

# Run our pretrained model
You can directly run our pretrained model for generating inferences. Currently, we have two models available. First is the history of 

# Training, Inference and Evaluation for Encoder-Decoder NMT With attention model
This part presents an overview of how to use the code presented to train COQG with tensorflow seq2seq. Examples presented here are for dataset which uses num_history=2. 

## Data preparation and Preprocessing
1. Download the COQA dataset
```bash
chmod +x scripts/download_coqa.sh
./scripts/download_coqa.sh
```
2. Download the vocabulary ( We use the 50K most common words in english language)
```
wget -O data/vocab.para https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/vocab.50K.en
cp data/vocab.para data/vocab.ques
```
3. Prepare source and targets for training. You can also change the n_history by changing the cmd-line argument(Takes aroud 30 mins in a standard workstation).
```bash
chomod +x scripts/preprocess.sh 
./scripts/preprocess.sh 2
```

## Training 
Following options are available with Tensorflow nmt:

|Option Name   | Meaning  | Available Values  | Default  | COQG  |
|---|---|---|---|---|
| --num_units  | Number of units in one RNN-Cell. This relates to embedding size  | integer |  32 | 256  |
| --num_layers  | Depth of Encoder-Decoder Network  | integer  | 2  | 2  |
| --num_encoder_layers | Encoder depth | integer | num_layers |  2 |
| --num_decoder_layers | Decoder depth | integer | num_layers |  1 |
| --encoder_type | use uni, bi or gnmt style encoders | string (unidirectional, bidirectional, gnmt style) | uni | gnmt |
| --residual | Whether to add residual connections | boolean | false | false |
| --time_major | Whether to use time-major mode for dynamic RNN | boolean | true | true |
| --num_embeddings_partitions | Number of partitions for embedding vars | int | 0 | 0 |
| --attention | which attention function to use | str (luong, scaled_luong, bahdanau, normed_bahdanau ) | None | scaled_luong, normed_bahdanau |
| --attention_architecture | Which attention architecture to use | str (standard, gnmt, gnmt_v2) | standard | standard |
| --output_attention | Only used in standard attention_architecture. Whether use attention as the cell output at each timestep. | boolean | True | True |
| --pass_hidden_state | Whether to pass encoder's hidden state to decoder when using an attention based model | boolean | True | True |
| --optimizer | which optimizer to use | string (sgd, adam) | sgd | sgd | 
| --learning_rate | The learning rate | float | 1.0 | 1.0 |
| --warmup_steps | Learning rate warmup scheme | str | t2t | t2t |
| --decay_scheme | How to decay learning rate | str | "" | luong234 |
| --num_train_steps | Number of steps (usually num_epochs * (num_examples // batch_size) ) | int | 12000 | 45000 |
| --colocate_gradients_with_ops | "Whether try colocating gradients with corresponding output" | Boolean | True | True |
| --init_op | How to do initialization | str(uniform, glorot_normal, glorot_uniform) | uniform | glorot_normal |
| --init_weight | Initialize values | float | 0.1 | 0.1 |
| --src | Source suffix | str | None | para |
| --tgt | Target suffix | str | None | ques |
| --train_prefix | Train prefix, expect files with src/tgt suffixes | str | None | depends on the model |
| --dev_prefix | Dev prefix, expect files with src/tgt suffixes. | str | None | depends on the model |
| --test_prefix | Test prefix, expect files with src/tgt suffixes. | str | None | depends on the model |
| --out_dir | Store log/model files. | str | None | depends on the model |
| --vocab_prefix | Vocab prefix, expect files with src/tgt suffixes | str | None | vocab |
| --src_max_len | Max length of src sequences during training. | int | 50 | 1000 |
| --tgt_max_len | Max length of tgt sequences during training. | int | 50 | 100  |
| --unit_type | LSTM, GRU or nas | str | lstm | lstm
| --forget_bias | Forget bias for BasicLSTMCell | float | 1.0 | 1.0 |
| --batch_size | batch_size | int | 128 | 128 |
| --steps_per_stats | How many training steps to do per stats logging. Save checkpoint every 10x steps_per_stats | int | 100 | 200 |