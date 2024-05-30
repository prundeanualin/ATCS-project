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


def evaluate(outputs_and_samples, evaluation_strategy: EvaluationStrategy = StructuredEvaluationStrategy()):
    total_score = 0
    score_per_category = {}
    for sample, generated_output in outputs_and_samples:
        category = sample['analogy_type']
        score = evaluation_strategy.evaluate(
            generated_output,
            sample
        )
        total_score += score

        if category not in score_per_category:
            score_per_category[category] = {'total': 0, 'correct': 0, 'acc': 0.0}
        score_per_category[category]['total'] += 1
        score_per_category[category]['correct'] += score
    final_score = total_score / len(outputs_and_samples) * 100
    for c in score_per_category:
        score_per_category[c]['acc'] = score_per_category[c]['correct'] / score_per_category[c]['total'] * 100
    score_per_category['all'] = {'total': len(outputs_and_samples), 'correct': total_score, 'acc': final_score}
    return score_per_category


def evaluate_from_file(results_file, evaluation_strategy: EvaluationStrategy):
    results = pd.read_pickle(results_file)
    return evaluate(results, evaluation_strategy)


if __name__ == '__main__':
    results_file = "results/SCAN/0shot_cot(False)_description(False)_examples(baseline)_baseline_Meta-Llama-3-8B-Instruct.pl"
    results = pd.read_pickle(results_file)

    regex_evaluation = RegexEvaluationStrategy()
    final_evaluation = StructuredEvaluationStrategy(debug=True)
    for sample, generated_output in results[:100]:
        print("--------")
        regex_score = regex_evaluation.evaluate(
            generated_output,
            sample
        )
        final_score = final_evaluation.evaluate(
            generated_output,
            sample
        )
        if regex_score != final_score:
            print("Different For sample: ")
            print("- Inference:             " + sample['inference'])
            print(f"- Label & alternatives:  {sample['label']} - {sample['alternatives']}")
            print("- Generated:")
            print(generated_output)
            print("Regex score is: ", regex_score)
            print(f"Final score is: {final_score}")
        print("------------\n")
