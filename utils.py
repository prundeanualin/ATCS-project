import torch
import numpy as np
import random
import os
import pickle
import re

SAVE_DIR = "results"
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

def seed_experiments(seed):
    """
    Seed the experiment, for reproducibility
    :param seed: the random seed to be used
    :return: nothing
    """
    # Set `pytorch` pseudo-random generator at a fixed value
    torch.manual_seed(seed)

    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)


def save_results(results, evaluation_metrics, prompts, filename):
    detailed_save_file = os.path.join(SAVE_DIR, f'{filename}.pl')
    readable_save_file = os.path.join(SAVE_DIR, f'{filename}.txt')
    with open(detailed_save_file, 'wb') as f:
        pickle.dump(results, f)
    with open(readable_save_file, 'w') as f:
        f.write("-- Some of the prompts that have been fed to the model\n")
        for p in prompts:
            f.write(p + '\n===========\n')
        f.write("============================================\n\n")
        for sample, output in results:
            f.write("- Inference:             " + sample['inference'] + '\n')
            f.write(f"- Label & alternatives:  {sample['label']} - {sample['alternatives']}" + '\n')
            f.write(f"- Generated:\n")
            f.write(output + '\n')
            f.write('\n----------------\n\n')
        f.write("### Evaluation metrics:\n")
        for category in evaluation_metrics:
            f.write(f"- {category}: {evaluation_metrics[category]}\n")
    print(f"Saved results in file: {filename}")


class DummyPipeline:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, prompt, **kwargs):
        return [
            {
                "generated_text": f"{prompt} <fake-assistant> This is a dummy generated text!"
            }
        ]

# The substrings to look for
direct_answer_pattern = ["final answer be", "correct answer be", "answer be"]

# The end characters
stop = ['.', '?', '!']
r_stop = r'?)([' + ''.join(stop) + r'])'

# Regex pattern to match any of the answer substrings and capture
# the following words until one of the end character
r_direct_answer = r'(?:\b.*\b)?\b(?:' + '|'.join(direct_answer_pattern) + r') (.*)'

# Pattern to look for the analogy inference:
# "if (optional one word) A be like (optional one word) B then (optional one word) C be like (the string we are looking for)"
r_analogy = r'if (?:\w+\s)?({}) be like (?:\w+\s)?({}) then (?:\w+\s)?({}) be like (.*)'
r_partial_analogy = r'(?:\w+\s)?({}) be like (.*)'
r_analogy_elements = r'if (.*?) be like (.*?) then (.*?) be like'

r_first_sent = r'(.*)'


def limit_to_max_nr_words(text, max_length):
    if text is None:
        return text
    text_words = text.split()
    length_text = len(text_words)
    if length_text > max_length:
        # Keep at most max_length words from the sentence
        text_words = text_words[:max_length]
    return " ".join(text_words)


def extract_regex_pattern(regex, text, match_id, max_nr_words_without_stop):
    pattern_with_stop = re.compile(regex[:-1] + r_stop)
    pattern = re.compile(regex)

    match = pattern_with_stop.search(text)
    if match:
        # print("First match entered!")
        # Extract the relevant portion of the match
        answer_fragment = match.group(match_id).strip()
        return answer_fragment
    else:
        # Look for the match in the whole text, without matching/stopping at any stop symbols
        match = pattern.search(text)
        # print("MAtch is: ", match)
        if match:
            # print("match whole text: ", match.group(0))
            # Extract the relevant portion of the match
            answer_fragment = match.group(match_id).strip()
            # print("The answer fragment: ", answer_fragment)
            return limit_to_max_nr_words(answer_fragment, max_length=max_nr_words_without_stop)
    return None


def extract_direct_answer(text, max_length):
    answer = extract_regex_pattern(r_direct_answer, text, match_id=1, max_nr_words_without_stop=max_length)
    return answer


def get_analogy_elements(inference_analogy):
    # Pattern to extract the analogy elements A, B, C from the lemmatized inference - "if A be like B then C be like"
    pattern = re.compile(r_analogy_elements)
    match = pattern.search(inference_analogy)
    response = []
    for i in [1, 2, 3]:
        analogy_elem = match.group(i).strip()
        # Escape the variable to ensure any special regex characters are treated literally
        response.append(re.escape(analogy_elem))
    return response


def extract_analogy_pattern_match_answer(text, inference_analogy, max_length):
    # The analogy pattern to look for is made up of those elements
    a, b, c = get_analogy_elements(inference_analogy)
    r_analogy_filled = r_analogy.format(a, b, c)

    answer = extract_regex_pattern(r_analogy_filled, text, match_id=4, max_nr_words_without_stop=max_length)
    # No pattern match for the whole analogy structure, trying out the last part of the analogy: "C is like"
    if answer is None:
        r_partial_analogy_filled = r_partial_analogy.format(c)
        answer = extract_regex_pattern(r_partial_analogy_filled, text, match_id=2, max_nr_words_without_stop=max_length)
    return answer


def extract_first_sentence_answer(text, max_length):
    # There is no stop symbol to mark the end of the sentence. The whole text is considered in that case
    first_sentence_candidate = extract_regex_pattern(r_first_sent, text, match_id=1, max_nr_words_without_stop=max_length)
    # The first sentence is always limited to the first max_length words, regardless of stop symbols
    return limit_to_max_nr_words(first_sentence_candidate, max_length=max_length)
