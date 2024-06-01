import pandas as pd
import nltk
import regex as re

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from utils import *

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


class EvaluationStrategy:
    def __init__(self):
        self.tokenizer = None

    def evaluate(self, generated_text: str, sample: dict):
        raise NotImplementedError


class RegexEvaluationStrategy(EvaluationStrategy):
    def __init__(self):
        super().__init__()

    def evaluate(self, generated_text: str, sample: dict):
        label = sample['label']
        alternative_labels = sample['alternatives']
        labels = [i.lower() for i in alternative_labels + [label]]
    
        for label_i in labels:
            x = re.findall(rf'\b{label_i}+?(s\b|\b)', generated_text.lower())
            if x:
                return 1
        return 0


class SimpleEvaluationStrategy(EvaluationStrategy):
    def __init__(self):
        super().__init__()

    def evaluate(self, generated_text: str, sample: dict):
        label = sample['label']
        alternative_labels = sample['alternatives']
        labels = alternative_labels + [label]
        # TODO: maybe also test other tokenization strategies
        generated_words = nltk.word_tokenize(generated_text)
        for w in generated_words:
            if w in labels:
                return 1
        return 0


def tokenize(sentence):
    return [w.lower() for w in nltk.word_tokenize(sentence)]


# POS_TAGGER_FUNCTION : TYPE 1
# Converts the Penn Treebank tag set to the WordNet tags
def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def pos_tag(words_tokenized):
    words_pos_tagged = nltk.pos_tag(words_tokenized)
    words_tagged_wordnet = list(map(lambda x: (x[0], pos_tagger(x[1])), words_pos_tagged))
    return words_tagged_wordnet


def test_label_in_answer(labels, answer):
    if answer is None:
        return None
    else:
        for label in labels:
            if label in answer:
                return 1
        return 0


class StructuredEvaluationStrategy(EvaluationStrategy):
    def __init__(self, max_length=5, debug=False):
        self.debug = debug
        self.max_length = max_length
        self.lemmatizer = WordNetLemmatizer()
        self.all_pos_tags = [None, wordnet.ADJ, wordnet.ADV, wordnet.NOUN, wordnet.VERB]
        self.ignored_punctuation = ['...', '..', ',', '``', "''"]
        super().__init__()

    def lemmatize(self, words_pos_tagged):
        lemmatized_sentence = []
        for word, tag in words_pos_tagged:
            if word in self.ignored_punctuation:
                continue
            if tag is None:
                # if there is no available tag, append the token as is
                lemmatized_sentence.append(word)
            else:
                # else use the tag (WordNet one) to lemmatize the token
                lemmatized_sentence.append(self.lemmatizer.lemmatize(word, tag))
        return lemmatized_sentence

    def lemmatize_labels(self, sample):
        label = sample['label']
        alternative_labels = sample['alternatives']
        labels = alternative_labels + [label]
        # Lemmatize the labels and alternatives for all possible pos tags
        labels_lemmatized = self.lemmatize([(l, l_pos) for l in labels for l_pos in self.all_pos_tags])
        return list(set(labels_lemmatized))

    def evaluate_strategies_fallback(self, generated_concat, labels_lemmatized, sample):
        # Look for a pattern like (before lemmatization) "The answer is"
        candidate_answer = extract_direct_answer(generated_concat, max_length=self.max_length)
        if self.debug: print("Possible direct answer: ", candidate_answer)
        direct_answer = test_label_in_answer(labels_lemmatized, candidate_answer)
        if direct_answer is not None:
            return direct_answer

        # If the previous pattern was not found, look for the next one
        # Look for a pattern like (before lemmatization) "If A is like B, then C is like",
        # with A,B,C - the analogy elements from the inference
        inference_lemmatized = self.lemmatize(pos_tag(tokenize(sample['inference'])))
        candidate_answer = extract_analogy_pattern_match_answer(generated_concat, " ".join(inference_lemmatized),
                                                                max_length=self.max_length)
        if self.debug: print("Possible analogy pattern match answer: ", candidate_answer)
        analogy_pattern_answer = test_label_in_answer(labels_lemmatized, candidate_answer)
        if analogy_pattern_answer is not None:
            return analogy_pattern_answer

        # If the previous pattern was not found, look for the next one
        # Look for the first sentence of the answer, as long as it's no longer then 5 words
        candidate_answer = extract_first_sentence_answer(generated_concat, max_length=self.max_length)
        if self.debug: print("Possible first sentence answer: ", candidate_answer)
        first_sent_answer = test_label_in_answer(labels_lemmatized, candidate_answer)
        if first_sent_answer is not None:
            return first_sent_answer
        return 0

    def evaluate(self, generated_text: str, sample: dict):

        labels_lemmatized = self.lemmatize_labels(sample)
        if self.debug:
            print("\nLemmatized labels: ")
            print(labels_lemmatized)

        generated_lemmatized = self.lemmatize(pos_tag(tokenize(generated_text)))
        if self.debug:
            print("Lemmatized generated words are: ")
            print(generated_lemmatized)

        concatenated = " ".join(generated_lemmatized)
        return self.evaluate_strategies_fallback(concatenated, labels_lemmatized, sample)


def evaluate(outputs_and_samples):
    eval_strategy_struct = StructuredEvaluationStrategy()
    eval_strategy_regex = RegexEvaluationStrategy()
    total_score_struct = 0
    total_score_regex = 0
    score_per_category = {}
    total_avg_output = 0
    for sample, generated_output in outputs_and_samples:
        category = sample['analogy_type']
        # Use the new evaluation strategy
        score_struct = eval_strategy_struct.evaluate(
            generated_output,
            sample
        )
        total_score_struct += score_struct
        # Use the regex evaluation
        scoore_regex = eval_strategy_regex.evaluate(
            generated_output,
            sample
        )
        total_score_regex += scoore_regex

        if category not in score_per_category:
            score_per_category[category] = {'correct_struct': 0, 'correct_regex': 0, 'total': 0, 'avg_length': 0, 'max_length': 0, 'min_length': 1000, 'acc_struct': 0.0, 'acc_regex': 0.0}

        score_per_category[category]['total'] += 1
        score_per_category[category]['correct_struct'] += score_struct
        score_per_category[category]['correct_regex'] += scoore_regex
        score_per_category[category]['avg_length'] += len(generated_output.split())
        if len(generated_output) > score_per_category[category]['max_length']:
            score_per_category[category]['max_length'] = len(generated_output.split())
        if len(generated_output) < score_per_category[category]['min_length']:
            score_per_category[category]['min_length'] = len(generated_output.split())
        total_avg_output += len(generated_output.split())

    final_score_struct = total_score_struct / len(outputs_and_samples) * 100
    final_score_regex = total_score_regex / len(outputs_and_samples) * 100

    final_avg_length = int(total_avg_output / len(outputs_and_samples))
    final_min_length = 1000
    final_max_length = 0
    for c in score_per_category:
        score_per_category[c]['acc_struct'] = score_per_category[c]['correct_struct'] / score_per_category[c]['total'] * 100
        score_per_category[c]['acc_regex'] = score_per_category[c]['correct_regex'] / score_per_category[c]['total'] * 100
        score_per_category[c]['avg_length'] = int(score_per_category[c]['avg_length'] / score_per_category[c]['total'])
        if score_per_category[c]['min_length'] < final_min_length:
            final_min_length = score_per_category[c]['min_length']
        if score_per_category[c]['max_length'] > final_max_length:
            final_max_length = score_per_category[c]['max_length']
    score_per_category['all'] = {
        'total': len(outputs_and_samples),
        'correct_struct': total_score_struct,
        'correct_regex': total_score_regex,
        'acc_struct': final_score_struct,
        'acc_regex': final_score_regex,
        'avg_length': final_avg_length,
        'min_length': final_min_length,
        'max_length': final_max_length
    }
    return score_per_category


def evaluate_from_file(results_file):
    results = pd.read_pickle(results_file)
    return evaluate(results)


if __name__ == '__main__':

    result_files = {
        "0shot": {
            "cot": [
                "results/0shot/cot/0shot_cot(True)_description(False)_examples(baseline)_Meta-Llama-3-8B-Instruct.pl",
                "results/0shot/cot/0shot_cot(True)_description(False)_examples(baseline)_Phi-3-mini-128k-instruct.pl",
                "results/0shot/cot/0shot_cot(True)_description(False)_examples(baseline)_Starling-LM-7B-alpha.pl"
            ],
            "nocot": [
                "results/0shot/nocot/baseline_semi_struct/0shot_cot(False)_description(False)_examples(baseline)_baseline_Meta-Llama-3-8B-Instruct.pl",
                "results/0shot/nocot/baseline_semi_struct/0shot_cot(False)_description(False)_examples(baseline)_baseline_Phi-3-mini-128k-instruct.pl",
                "results/0shot/nocot/baseline_semi_struct/0shot_cot(False)_description(False)_examples(baseline)_baseline_Starling-LM-7B-alpha.pl"
            ],
            "variations": {
                "nocot": {
                    "raw": [
                        "results/0shot/nocot/raw_just_sample/0shot_cot(False)_description(False)_examples(baseline)_Meta-Llama-3-8B-Instruct.pl",
                        "results/0shot/nocot/raw_just_sample/0shot_cot(False)_description(False)_examples(baseline)_Phi-3-mini-128k-instruct.pl",
                        "results/0shot/nocot/raw_just_sample/0shot_cot(False)_description(False)_examples(baseline)_Starling-LM-7B-alpha.pl"
                    ],
                    "strict": [
                        "results/0shot/nocot/baseline_variations/0shot_cot(False)_description(False)_examples(baseline)_baseline_Meta-Llama-3-8B-Instruct.pl",
                        "results/0shot/nocot/baseline_variations/0shot_cot(False)_description(False)_examples(baseline)_baseline_Phi-3-mini-128k-instruct.pl",
                        "results/0shot/nocot/baseline_variations/0shot_cot(False)_description(False)_examples(baseline)_baseline_Starling-LM-7B-alpha.pl"
                    ],
                    "struct_default_general_example": [
                        "results/0shot/nocot/baseline_variations/0shot_cot(False)_description(False)_examples(baseline)_STbaseline_Meta-Llama-3-8B-Instruct.pl",
                        "results/0shot/nocot/baseline_variations/0shot_cot(False)_description(False)_examples(baseline)_STbaseline_Phi-3-mini-128k-instruct.pl",
                        "results/0shot/nocot/baseline_variations/0shot_cot(False)_description(False)_examples(baseline)_STbaseline_Starling-LM-7B-alpha.pl"
                    ]
                },
                "cot_simple_instruction": [
                    "results/0shot/cot/cot_simple_instruction/0shot_cot(True)_description(False)_examples(baseline)_Meta-Llama-3-8B-Instruct.pl",
                    "results/0shot/cot/cot_simple_instruction/0shot_cot(True)_description(False)_examples(baseline)_Phi-3-mini-128k-instruct.pl",
                    "results/0shot/cot/cot_simple_instruction/0shot_cot(True)_description(False)_examples(baseline)_Starling-LM-7B-alpha.pl"
                ]
            }
        },
        "1shot": {
            "cot": [
                "results/1shot/cot/final_cot_1shot_cot(True)_description(False)_examples(baseline)_Meta-Llama-3-8B-Instruct.pl",
                "results/1shot/cot/final_cot_1shot_cot(True)_description(False)_examples(baseline)_Phi-3-mini-128k-instruct.pl",
                "results/1shot/cot/final_cot_1shot_cot(True)_description(False)_examples(baseline)_Starling-LM-7B-alpha.pl"
            ],
            "nocot": [
                "results/1shot/nocot/1shot_cot(False)_description(False)_examples(baseline)_Meta-Llama-3-8B-Instruct.pl",
                "results/1shot/nocot/1shot_cot(False)_description(False)_examples(baseline)_Phi-3-mini-128k-instruct.pl",
                "results/1shot/nocot/1shot_cot(False)_description(False)_examples(baseline)_Starling-LM-7B-alpha.pl"
            ],
            "example_types": {
                "complex": [
                    "results/1shot/cot/example_variations/complex/1shot_cot(True)_description(False)_examples(complex)_Meta-Llama-3-8B-Instruct.pl",
                    "results/1shot/cot/example_variations/complex/1shot_cot(True)_description(False)_examples(complex)_Phi-3-mini-128k-instruct.pl",
                    "results/1shot/cot/example_variations/complex/1shot_cot(True)_description(False)_examples(complex)_Starling-LM-7B-alpha.pl"
                ],
                "long": [
                    "results/1shot/cot/example_variations/long/1shot_cot(True)_description(False)_examples(long)_Meta-Llama-3-8B-Instruct.pl",
                    "results/1shot/cot/example_variations/long/1shot_cot(True)_description(False)_examples(long)_Phi-3-mini-128k-instruct.pl",
                    "results/1shot/cot/example_variations/long/1shot_cot(True)_description(False)_examples(long)_Starling-LM-7B-alpha.pl"
                ],
                "short": [
                    "results/1shot/cot/example_variations/short/1shot_cot(True)_description(False)_examples(short)_Meta-Llama-3-8B-Instruct.pl",
                    "results/1shot/cot/example_variations/short/1shot_cot(True)_description(False)_examples(short)_Phi-3-mini-128k-instruct.pl",
                    "results/1shot/cot/example_variations/short/1shot_cot(True)_description(False)_examples(short)_Starling-LM-7B-alpha.pl"
                ],
                "simple": [
                    "results/1shot/cot/example_variations/simple/1shot_cot(True)_description(False)_examples(simple)_Meta-Llama-3-8B-Instruct.pl",
                    "results/1shot/cot/example_variations/simple/1shot_cot(True)_description(False)_examples(simple)_Phi-3-mini-128k-instruct.pl",
                    "results/1shot/cot/example_variations/simple/1shot_cot(True)_description(False)_examples(simple)_Starling-LM-7B-alpha.pl"
                ]
            }
        },
        "2shot": {
            "cot": [
                "results/2shot/cot/2shot_cot(True)_description(False)_examples(baseline)_Meta-Llama-3-8B-Instruct.pl",
                "results/2shot/cot/2shot_cot(True)_description(False)_examples(baseline)_Phi-3-mini-128k-instruct.pl",
                "results/2shot/cot/2shot_cot(True)_description(False)_examples(baseline)_Starling-LM-7B-alpha.pl"
            ],
            "nocot": [
                "results/2shot/nocot/2shot_cot(False)_description(False)_examples(baseline)_Meta-Llama-3-8B-Instruct.pl",
                "results/2shot/nocot/2shot_cot(False)_description(False)_examples(baseline)_Phi-3-mini-128k-instruct.pl",
                "results/2shot/nocot/2shot_cot(False)_description(False)_examples(baseline)_Starling-LM-7B-alpha.pl"
            ]
        },
        "4shot": [
            "results/4shot/4shot_cot(False)_description(False)_examples(baseline)_Meta-Llama-3-8B-Instruct.pl",
            "results/4shot/4shot_cot(False)_description(False)_examples(baseline)_Phi-3-mini-128k-instruct.pl",
            "results/4shot/4shot_cot(False)_description(False)_examples(baseline)_Starling-LM-7B-alpha.pl"
        ],
        "8shot": [
            "results/8shot/8shot_cot(False)_description(False)_examples(baseline)_Meta-Llama-3-8B-Instruct.pl",
            "results/8shot/8shot_cot(False)_description(False)_examples(baseline)_Phi-3-mini-128k-instruct.pl",
            "results/8shot/8shot_cot(False)_description(False)_examples(baseline)_Starling-LM-7B-alpha.pl"
        ]
    }


    def print_all_results(result_files, depth=1, filter_model="", print_just_regex_acc=False, print_just_struct_acc=False):
        for key, value in result_files.items():
            print("#" * depth + " " + key)
            if isinstance(value, dict):
                print_all_results(value, depth + 1, filter_model, print_just_regex_acc, print_just_struct_acc)
            else:
                print_results(value, filter_model, print_just_regex_acc, print_just_struct_acc)

    def print_results(result_file_models, filter_model="", print_just_regex_acc=False, print_just_struct_acc=False):
        for res_file in result_file_models:
            model_name = res_file.split('_')[-1][:-3]
            if filter_model in model_name:
                results = evaluate_from_file(res_file)
                # print("For model - ", model_name)
                if print_just_struct_acc:
                    acc_struct = '%.1f' % results["all"]['acc_struct']
                    print(f"acc_struct: {acc_struct}")
                elif print_just_regex_acc:
                    acc_regex = '%.1f' % results["all"]['acc_regex']
                    print(f"acc_regex: {acc_regex}")
                else:
                    for category in results:
                        acc_struct = '%.2f' % results[category]['acc_struct']
                        acc_regex = '%.2f' % results[category]['acc_regex']
                        print(f"{category} - acc_struct: {acc_struct}  |  acc_regex: {acc_regex}  |  avg_sent_length: {results[category]['avg_length']}"
                              f"  |  min_length: {results[category]['min_length']}  |  max_length: {results[category]['max_length']}")
                print("\n")


    for model in ["Phi", "Starling", "Llama"]:
        print("--- For model: ", model)
        print_all_results(result_files, filter_model=model, print_just_struct_acc=True)
