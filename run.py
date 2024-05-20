import argparse

from get_datasets import SCAN_EXAMPLES_FILEPATH, EXAMPLE_CATEGORIES
from prompt_templates.analogy import ANALOGY_TEMPLATE_SIMPLE_INFERENCE, ANALOGY_TEMPLATE_SIMPLE_FULL
from model import LLMObj
from evaluate import *

import numpy as np

import torch
from tqdm import tqdm
from transformers import BitsAndBytesConfig
import pickle
from datasets import ScanDataset
import os

from utils import seed_experiments, SAVE_DIR

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='microsoft/Phi-3-mini-128k-instruct', help='LLM Model')
parser.add_argument('--tokenizer', type=str, default=None, help="LLM Tokenizer. If not set, will use the HF auto tokenizer loaded from the model's name")
parser.add_argument('--quantization', type=str, default='4bit', help='LLM Quantization', choices=['None', '4bit'])
parser.add_argument('--low_cpu_mem_usage', default=True, type=bool, help='Low CPU Memory usage')
parser.add_argument('--seed', type=int, default=1234, help='Random seed to use throughout the pipeline')
parser.add_argument('--debug', type=bool, default=False, help='If running in debug mode, there will be a dummy pipeline created. No real model inference will hapen!')

parser.add_argument('--n_shot', type=int, default=0, help='Few shot example number.')
parser.add_argument('--few_shot', type=bool, default=False, help='Few shot yes/no.')
parser.add_argument('--analogy_type', type=str, default='science', help='Analogy type selection')

args = parser.parse_args()
print("CL Arguments are:")
print(args)

os.environ['HF_TOKEN'] = "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"
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

torch.set_default_device('cuda')

seed_experiments(args.seed)

# ----- Load dataset -----
examples_shot_nr = 25
dataset = ScanDataset(
    shuffle=False,
    analogy_sentence_infer=ANALOGY_TEMPLATE_SIMPLE_INFERENCE,
    analogy_sentence_full=ANALOGY_TEMPLATE_SIMPLE_FULL,
    examples_file=SCAN_EXAMPLES_FILEPATH.format(EXAMPLE_CATEGORIES[0]),
    examples_start_idx=0,
    examples_shot_nr=examples_shot_nr
)

# ----- Prepare model arguments -----
quantization = None
if args.quantization == '4bit':
    quantization = BitsAndBytesConfig(load_in_4bit=True)

model_kwargs = {
    "torch_dtype": torch.bfloat16,
    "low_cpu_mem_usage": args.low_cpu_mem_usage,
    "quantization_config": quantization
}

LLMObj_args = {
    'model': args.model,
    'model_kwargs': model_kwargs,
    'tokenizer_name': args.tokenizer,
    'dummy_pipeline': args.debug
}
print("LLMObj Arguments are:")
print(LLMObj_args)

# ----- Load the model -----
LLM = LLMObj(**LLMObj_args)


# ----- Run inference-----
if args.few_shot:

    # ----- Run Few-Shot inference-----
    
    n_relevant_prompts = 0
    for i, sample in tqdm(enumerate(dataset)):
        if sample['analogy_type'] != args.analogy_type:
            continue
        n_relevant_prompts += 1

    n_shot = args.n_shot
    n_switch = int(np.ceil(n_relevant_prompts / (examples_shot_nr / n_shot)))

    selection = -n_shot
    
    count = 0
    results = []
    for i, sample in tqdm(enumerate(dataset)):
        if sample['analogy_type'] != args.analogy_type:
            continue
        
        if count % n_switch == 0:
            selection += n_shot
        count += 1
        
        examples = dataset.current_examples[selection: selection + n_shot]
        prompt = "{}\n" * n_shot + sample['inference']
        prompt = prompt.format(*map(lambda x: x['simple'], examples))
        sample['inference'] = prompt
        output = LLM.generate(prompt)
        results.append([sample, output])

    save_file = os.path.join(SAVE_DIR, f'{args.n_shot}_shot_{args.analogy_type}_{args.model.split("/")[1]}_generated_prompts.pl')
    with open(save_file, 'wb') as f:
        pickle.dump(results, f)
    
else:

    # ----- Run Zero-shot inference -----
    results = []
    for i, sample in tqdm(enumerate(dataset)):
        output = LLM.generate(sample['inference'])
        results.append([sample, output])

    save_file = os.path.join(SAVE_DIR, f'{args.model.split("/")[1]}_generated_prompts.pl')
    with open(save_file, 'wb') as f:
        pickle.dump(results, f)

# ----- Evaluate -----

# results = pd.read_pickle(f'{args.model.split("/")[1]}_generated_prompts.pl')
acc_score = evaluate(results, SimpleEvaluationStrategy())
print(f"Score is {acc_score}%")