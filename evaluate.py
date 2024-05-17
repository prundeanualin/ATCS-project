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
        generated_words = generated_text.split(' ')
        for w in generated_words:
            for label in labels:
                if w in label:
                    return 1
        return 0


def evaluate(items, evaluation_strategy: EvaluationStrategy):
    total_score = 0
    for item in items:
        sample = item[0]
        generated_output = item[1]
        score = evaluation_strategy.evaluate(
            generated_output,
            sample['label'],
            sample['alternatives']
        )
        total_score += score
    final_score = total_score / len(items) * 100
    return final_score
