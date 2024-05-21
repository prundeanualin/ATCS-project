import pandas as pd
import nltk

nltk.download('punkt')

class EvaluationStrategy:
    def __init__(self):
        self.tokenizer = None

    def evaluate(self, generated_text: str, label: str, alternative_labels: list[str]):
        raise NotImplementedError


class SimpleEvaluationStrategy(EvaluationStrategy):
    def __init__(self):
        super().__init__()

    def evaluate(self, generated_text: str, label: str, alternative_labels: list[str]):
        labels = alternative_labels + [label]
        # TODO: maybe also test other tokenization strategies
        generated_words = nltk.word_tokenize(generated_text)
        for w in generated_words:
            if w in labels:
                return 1
        return 0


def evaluate(outputs_and_samples, evaluation_strategy: EvaluationStrategy):
    total_score = 0
    for sample, generated_output in outputs_and_samples:
        score = evaluation_strategy.evaluate(
            generated_output,
            sample['label'],
            sample['alternatives']
        )
        total_score += score
    final_score = total_score / len(outputs_and_samples) * 100
    return final_score


def evaluate_from_file(results_file, evaluation_strategy: EvaluationStrategy):
    results = pd.read_pickle(results_file)
    return evaluate(results, evaluation_strategy)