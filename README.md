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
3. Prepare source and targets for training. You can also change the n_history by changing the cmd-line argument.
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