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

logger = logging.getLogger("dataset.features.log")
logging.basicConfig(level=logging.INFO)
nlp = StanfordCoreNLP('http://localhost:5002')  # Starts a corenlp server 


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


def tokenize_text(text):
    """Annotate the given text using corenlp server"""
    paragraph = nlp.annotate(text, properties={
        'annotators': 'tokenize, truecase, pos, ssplit, ner',
        'outputFormat': 'json'
    })
    tokens = []
    for sent in paragraph['sentences']:
        for token in sent['tokens']:
            tokens.append('|'.join([_str(token['word']),
                                    _case(token['word']),
                                    token['pos'],
                                    token['ner']]) + '|-')
    return ' '.join(tokens)


if __name__ == '__main__':
    coqa_gen = load_dataset('../data/coqa-train-v1.0.json')
    for one_story in coqa_gen:
        print(tokenize_text(one_story['story']))
        break
