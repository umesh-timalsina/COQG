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
import random
from copy import deepcopy
from pycorenlp import StanfordCoreNLP
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool, Lock
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("preprocess.logger")

src_train = open('../data/train/coqa-src-train-threaded-v1.0.txt', 'w')
tgt_train = open('../data/train/coqa-tgt-train-threaded-v1.0.txt', 'w')

logger = logging.getLogger("dataset.features.log")
logging.basicConfig(level=logging.INFO)
nlp_servers = [StanfordCoreNLP('http://localhost:9000'),
               StanfordCoreNLP('http://localhost:9001'),
               StanfordCoreNLP('http://localhost:9002')]


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


def tokenize_text(text, boundary_token="Ü", is_target=False):
    """Annotate the given text using corenlp server"""
    nlp = random.choice(nlp_servers)
    paragraph = None
    if not is_target:
        paragraph = nlp.annotate(text, properties={
            'annotators': 'tokenize, truecase, pos, ssplit, ner',
            'outputFormat': 'json'
        })
    else:
        paragraph = nlp.annotate(text, properties={
            'annotators': 'tokenize, ssplit',
            'outputFormat': 'json'
        })
    if isinstance(paragraph, str):
        print(paragraph)
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
                    tokens.append('|'.join([_str(token['word']),
                                            _case(token['word']),
                                            token['pos'],
                                            token['ner']]) + '|A|-')
                else:
                    tokens.append('|'.join([_str(token['word']),
                                            _case(token['word']),
                                            token['pos'],
                                            token['ner']]) + '|-')
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


def preprocess(one_story):
    story = one_story['story']
    # print(story)
    prev_question = ""
    ret_array_src = []
    ret_array_tgt = []
    for question, answer in zip(
                    one_story['questions'], one_story['answers']):
            start_idx = len(story[:answer['span_start']-1].split(' '))
            end_idx = len(story[:answer['span_end']-1].split(' '))
            story_rationale = story[:answer['span_start']-1] + " Ü " + story[answer['span_start']:answer['span_end']] + " Ü " + story[answer['span_end']:]
            if prev_question is not None:
                prev_question_tokenized = tokenize_text(prev_question)
            target = question['input_text']
            prev_question = target
            ret_array_src.append(tokenize_text(story_rationale) + ' || <q> ' + prev_question_tokenized + ' <q>' + '\n')
            ret_array_tgt.append(tokenize_text(target, is_target=True) + '\n')
    assert(len(ret_array_src) == len(ret_array_tgt))
    # logger.info("Finished Processing Story from process {}".format(os.getpid()))
    return (ret_array_src, ret_array_tgt)

if __name__ == '__main__':
    coqa_gen = load_dataset('../data/coqa-train-v1.0.json')
    print('Started Generating training set...')

    pool = Pool(32)
    time_start = datetime.now()
    final_array_src = list(tqdm(pool.imap(preprocess, list(coqa_gen)), total=8000))
    for src in final_array_src:
        src_train.write("".join(src[0]))
        tgt_train.write("".join(src[1]))
        # print(src[0][1])
    # pool.close()
    # pool.join()
    time_end = datetime.now()
    logger.info("Time it took to process: {} Seconds".format((time_end-time_start).total_seconds()))
    src_train.close()
    tgt_train.close()
