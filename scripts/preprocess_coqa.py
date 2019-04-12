"""Preprocess the COQA dataset for COQG
Steps:
    1. Read the COQA json file, yeild a generator for story, questions and answers
    2. For each story, annotate the story, questions and answers for tokenization
    3. Join every token by space and do some decontraction operations
    4. Depending upon the number of history arguments used form the src string,
       target string in the following format
            src: <story>,{<prev_rationale><prev_question>}*n_history,<current_rationale>
            tgt: <current_question>
    5. The file names are set according to command line arguments for args.type
    6. ToDo: Find a way to add extra linguistic features
"""
from __future__ import unicode_literals, print_function
import argparse
import json
import os
import logging
import spacy
import re
import time
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG)

nlp = spacy.load("en_core_web_sm")


def _decontracted(phrase):
    """ Courtesy of 
        https://stackoverflow.com/questions/43018030/replace-apostrophe-short-words-in-python
    """
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"\sn\'t", " not", phrase)
    phrase = re.sub(r"\s\'re", " are", phrase)
    phrase = re.sub(r"\s\'d", " would", phrase)
    phrase = re.sub(r"\s\'ll", " will", phrase)
    phrase = re.sub(r"\s\'t", " not", phrase)
    phrase = re.sub(r"\s\'ve", " have", phrase)
    phrase = re.sub(r"\s\'m", " am", phrase)
    phrase = re.sub(r"\s\'s", " &apos;s", phrase)
    phrase = re.sub(r"\s\"", " &quot;", phrase)
    # print(phrase)
    return phrase


def _tokenize(text, lower=True):
    """Tokenize the sentence"""
    text = text.replace('\n', '')
    doc = nlp(text)
    text_new = " ".join([token.text.lower() for token in doc])
    text_new = _decontracted(text_new)
    return text_new


def load_data(path, prefix):
    """load the coqa src files"""
    src_file = os.path.join(path, prefix)
    with open(src_file, 'r') as fp:
        coqa = json.load(fp)
        for stories in coqa['data']:
            yield {
                'story': stories['story'],
                'questions': stories['questions'],
                'answers': stories['answers']
            }


def generate_training_files(path,
                            coqa_gen,
                            prefix="coqg_h2",
                            num_history=2,
                            type_="dev"):
    """Generate the training files for COQG"""
    coqa_list = list(coqa_gen)
    if not os.path.exists(path):
        os.makedirs(path)
    # print(path+prefix)
    src_file = open(os.path.join(path, prefix) + "_" + type_ + ".para", 'w')
    tgt_file = open(os.path.join(path, prefix) + "_" + type_ + ".ques", 'w')
    start_time = time.time()
    for i, one_story in enumerate(coqa_list):
        assert(len(one_story['questions']) == len(one_story['answers']))
        if i % 10 == 0:
            logging.debug('processing %d / %d (used_time = %.2fs)...' % (
                            i, len(coqa_list), time.time() - start_time))
        story = _tokenize(one_story['story'])
        # print(story)
        history = []
        for question, answer in zip(
                one_story['questions'], one_story['answers']):
            assert(question['turn_id'] == answer['turn_id'])
            src_train = story + " "
            if num_history > 0:
                context_length = min(len(history), num_history)
            else:
                context_length = len(history)
            for j, (q, r) in enumerate(history[-context_length:]):
                d = context_length - j
                src_train += "<r{}> ".format(d) + r + " "
                src_train += "<q{}> ".format(d) + q + " "
            target_question = _tokenize(question['input_text'])
            rationale = _tokenize(answer['span_text'])
            src_train += "<r> " + rationale
            # logging.debug(src_train + " | " + target_question)
            history.append((target_question, rationale))
            src_file.write(re.sub(r'\s\s+', ' ', src_train) + '\n')
            tgt_file.write(re.sub(r'\s\s+', ' ', target_question) + '\n')
    src_file.close()
    tgt_file.close()


def parse_args():
    """Parse the command-line arguments"""
    parser = argparse.ArgumentParser("Parser for arguments")
    parser.add_argument("--src_path", "-src_path",
                        type=str, required=False,
                        default="./data",
                        help="path of coqa files")
    parser.add_argument("--type", "-type",
                        type=str, required=False,
                        default="train",
                        help="which file to work on train or dev")
    parser.add_argument("--n_history", "-n_history",
                        type=int, required=False,
                        default=2,
                        help="Number of previous questions to include\
                            in conversation history, use -1 for all history")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args.src_path)
    if args.type == "train":
        coqa_gen = load_data(args.src_path, "coqa-train-v1.0.json")
    elif args.type == "dev":
        coqa_gen = load_data(args.src_path, "coqa-dev-v1.0.json")
    file_path = args.src_path + "/" + args.type
    prefix = "coqgh{}".format(args.n_history)
    generate_training_files(file_path, coqa_gen,
                            type_=args.type,
                            num_history=args.n_history)
