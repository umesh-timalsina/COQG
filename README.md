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
3. Prepare source and targets for training. You can also change the, num_history by changing the cmd-line argument from 2 to 3.
```bash
chomod +x scripts/preprocess.sh 
./scripts/preprocess.sh 2
```

