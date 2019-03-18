from termcolor import colored
import argparse 
import json

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


def print_colored(dev_set_loc="../data/coqa-dev-v1.0.json",
                  num=1,
                  tgt_set_loc="../training_output2/pred.txt"):
    op_paragraph = load_dataset(dev_set_loc)
    op_question = open(tgt_set_loc, "r")
    for i, one_story in enumerate(op_paragraph):
        current_index = 0
        for j, answer in enumerate(one_story['answers']):
            print(one_story['story'][current_index: current_index+answer['span_start']], end=" ")
            print(colored(one_story['story'][answer['span_start']:answer['span_end']], 'red'), end=" ")
            print(one_story['story'][answer['span_end']:])
            print(colored("Question: " + op_question.readline(), 'green')) 
        if i == num-1:
            break

def parse_args():
    parser = argparse.ArgumentParser("Generate Colored Text")
    parser.add_argument("--dev_set_loc", 
                        type=str, 
                        help="Location of the devset",
                        default="../data/coqa-dev-v1.0.json")
    parser.add_argument("--num", 
                        type=int, 
                        help="Number of Paragraph To Generate",
                        default=1)
    parser.add_argument("--tgt_set_loc",
                        type=str,
                        help="Location of the generated Questions",
                        default='../training_output2/pred.txt')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print_colored(num=args.num)
