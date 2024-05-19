import torch
import numpy as np
import random
import os
import pickle

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


def debug_print(msg):
    print("[!! DEBUG !!] " + msg)


def save_results(results, filename):
    detailed_save_file = os.path.join(SAVE_DIR, f'{filename}.pl')
    readable_save_file = os.path.join(SAVE_DIR, f'{filename}.txt')
    with open(detailed_save_file, 'ab') as f:
        pickle.dump(results, f)
    with open(readable_save_file, 'a') as f:
        for sample, output in results:
            f.write("Inference:             " + sample['inference'] + '\n')
            f.write(f"Label & alternatives:  {sample['label']} - {sample['alternatives']}" + '\n')
            f.write(output + '\n')
            f.write('\n')


class DummyPipeline:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, prompt, **kwargs):
        return [
            {
                "generated_text": f"{prompt} <fake-assistant> This is a dummy generated text!"
            }
        ]
