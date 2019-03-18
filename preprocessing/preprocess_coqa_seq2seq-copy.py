#!/home/tumesh/anaconda3/envs/tensorflow/bin/python
# Copyright 2019-Present, College of Applied Science and Arts, SIUC

"""
A Script written to convert the COQG dataset into a text format written for COQG.
Steps:
    1. read coqa json file, extract all context, write to file, one rationale per line
"""
import argparse
import json
import os
import logging
from copy import deepcopy
from pycorenlp import StanfordCoreNLP
from tqdm import tqdm


logger = logging.getLogger("dataset.features.log")
logging.basicConfig(level=logging.INFO)
nlp = StanfordCoreNLP('http://localhost:9000')  # Starts a corenlp server 


def load_dataset(path):
    """Load the json and yeild fields separately"""
    with open(path) as f:
        data = json.load(f)['data']

    for values in data:
        yield {
            'story': values['story'],
            'questions': values['questions'],
            'answers': values['answers']
        }


def _str(s):
    """ Convert PTB tokens to normal tokens.
        Reference Implementation from
        https://github.com/stanfordnlp/coqa-baselines"""
    if (s.lower() == '-lrb-'):
        s = '('
    elif (s.lower() == '-rrb-'):
        s = ')'
    elif (s.lower() == '-lsb-'):
        s = '['
    elif (s.lower() == '-rsb-'):
        s = ']'
    elif (s.lower() == '-lcb-'):
        s = '{'
    elif (s.lower() == '-rcb-'):
        s = '}'
    elif (s.lower() == '``'):
        s = "`"
    return s


def _case(word):
    """Returns U for an uppercase word, L otherwise"""
    assert(type(word) == str)
    if(word.isalpha()):
        if word.islower():
            return 'L'
        else:
            return 'U'
    else:
        return 'L'


def tokenize_text(text, boundary_token="Ü", is_target=False):
    """Annotate the given text using corenlp server"""
    paragraph = nlp.annotate(text, properties={
        'annotators': 'tokenize, ssplit',
        'outputFormat': 'json'
    })
    tokens = []
    i = 0
    answer_begin = False
    if is_target:
        for sent in paragraph['sentences']:
            for token in sent['tokens']:
                tokens.append(_str(token['word']))

    if not is_target:
        for sent in paragraph['sentences']:
            for token in sent['tokens']:
                if token['word'] == boundary_token:
                    answer_begin = not answer_begin
                    continue
                if answer_begin:
                    tokens.append(_str(token['word']))
                else:
                    tokens.append(_str(token['word']))
    return ' '.join(tokens)


def parse_arguments():
    parser = argparse.ArgumentParser()
    # cmd options for corenlp preprocessing step
    parser.add_argument('-data_dir', type=str, help='Path to COQA dataset directory')
    parser.add_argument('-out_dir', type=str, help='Path to output file dir')
    parser.add_argument('-split', type=str, help='Filename for test/dev split',
                        default='coqa-dev-v1.0')
    parser.add_argument('-history', type=int, default=1, help="Number of questions to put in history...")

    # cmd option for processing from corenlp preprocessed data
    parser.add_argument('-num_sents', default='all', type=str,
                        help='number of sentences to select for the document')


if __name__ == '__main__':
    coqa_gen = load_dataset('../data/coqa-train-v1.0.json')
    src_train = open('../data/train/coqa-src-train-v1.1.txt', 'w')
    tgt_train = open('../data/train/coqa-tgt-train-v1.1.txt', 'w')
    print('Started Generating training set')
    for one_story in tqdm(list(coqa_gen)):
        story = one_story['story']
        prev_question = None
        prev_question_tokenized = ""
        for question, answer in zip(
                        one_story['questions'], one_story['answers']):
            # start_idx = len(' '.split(story[:answer['span_start']]))
            start_idx = len(story[:answer['span_start']-1].split(' '))
            end_idx = len(story[:answer['span_end']-1].split(' '))
            story_rationale = story[:answer['span_start']-1] + " Ü " + story[answer['span_start']:answer['span_end']] + " Ü " + story[answer['span_end']:]
            if prev_question is not None:
                prev_question_tokenized = tokenize_text(prev_question)
            target = question['input_text']
            prev_question = target
            src_train.write(tokenize_text(story_rationale) + ' || <r> ' + tokenize_text(answer['span_text']) + ' <r> ' + ' || <q> ' + prev_question_tokenized + ' <q>' + '\n')
            tgt_train.write(tokenize_text(target, is_target=True) + '\n')
    src_train.close()
    tgt_train.close()
