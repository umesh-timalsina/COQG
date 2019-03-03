#!/home/tumesh/anaconda3/envs/tensorflow/bin/python
# Copyright 2019-Present, College of Applied Science and Arts, SIUC

"""
A Script written to convert the COQG dataset into a text format
written for COQG Paper
Steps:
    1. read coqa json file, extract all context, write to file, one rationale per line
"""

import argparse
import json
import os
import logging
from copy import deepcopy
logger = logging.getLogger("dataset.features.log")
logging.basicConfig(level=logging.INFO)
# --------------------------------------------------------------------------------------------------------
# read coqa json file, extract all context, write to file, one rationale per line
# Load the dataset 
# --------------------------------------------------------------------------------------------------------


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


def coqa_train_file(data_generator, path):
    """Take the data generator and proecess the rationale logic"""
    with open(path, "w") as outfile:
        for data in data_generator:
            story = data['story']
            prev_question = ''
            for question, answer in zip(data['questions'], data['answers']):
                if 'bad_turn' not in answer.keys() or not answer['bad_turn']:
                    _story = deepcopy(story)
                    print(_story.replace('\n\n', ''), end='|')
                    print(prev_question, end='|')
                    print(question['input_text'])
                    prev_question = question['input_text']
            break

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(
    #             "Read the COQG JSON files, output text files")
    # parser.add_argument('input', type=str,
    #                     help='Path of COQA json')
    # parser.add_argument('output', type=str,
    #                     help='Output file name')
    # parser.add_argument('--history',
    #                     type=int, choices=[1, 2], default=1,
    #                     help='How many questions to include in history')
    # args = parser.parse_args()
    coqa_generator = load_dataset('../data/coqa-train-v1.0.json')
    # print(next(final_generator))
    os.makedirs('../data/train', exist_ok=True)
    coqa_train_file(coqa_generator, '../data/train/coqa-train-v1.0.txt')
