import pickle

import torch
import numpy as np
import random
import os

SAVE_DIR = "results/SCAN"
SAVE_BATS_DIR = "results/BATS"

for d in [SAVE_DIR, SAVE_BATS_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

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


class DummyPipeline:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, prompt, **kwargs):
        return [
            {
                "generated_text": f"{prompt} <fake-assistant> This is a dummy generated text!"
            }
        ]


def save_results(results, evaluation_metrics, bats: bool, filename):
    save_d = SAVE_BATS_DIR if bats else SAVE_DIR
    detailed_save_file = os.path.join(save_d, f'{filename}.pl')
    readable_save_file = os.path.join(save_d, f'{filename}.txt')
    with open(detailed_save_file, 'wb') as f:
        pickle.dump(results, f)
    with open(readable_save_file, 'w') as f:
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
