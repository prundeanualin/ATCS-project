import argparse

from get_datasets import *
from prompt_templates.analogy import *
from model import LLMObj
from evaluate import *

import torch
from tqdm import tqdm
from transformers import BitsAndBytesConfig
import pickle

from datasets import ScanDataset, BATSDataloader_0shot
import os

from utils import seed_experiments, SAVE_DIR, SAVE_BATS_DIR

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='microsoft/Phi-3-mini-128k-instruct', help='LLM Model')
parser.add_argument('--tokenizer', type=str, default=None, help="LLM Tokenizer. If not set, will use the HF auto tokenizer loaded from the model's name")
parser.add_argument('--quantization', type=str, default='4bit', help='LLM Quantization', choices=['None', '4bit'])
parser.add_argument('--low_cpu_mem_usage', default=True, type=bool, help='Low CPU Memory usage')
parser.add_argument('--seed', type=int, default=1234, help='Random seed to use throughout the pipeline')
parser.add_argument('--debug', type=bool, default=False, help='If running in debug mode, there will be a dummy pipeline created. No real model inference will hapen!')
parser.add_argument('--dataset', type=str, default='SCAN', help='Dataset to use for inference', choices=['SCAN', 'BATS'])
parser.add_argument('--BATS_filename', type=str, default='L01 [hypernyms - animals] sample', help='BATS filename to use for inference')
args = parser.parse_args()
print("CL Arguments are:")
print(args)

# CHANGE HF_TOKEN TO YOUR OWN TOKEN
os.environ['HF_TOKEN'] = "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"
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

torch.set_default_device('cuda')

seed_experiments(args.seed)


# ----- Load dataset -----
if args.dataset == 'SCAN':
    dataset = ScanDataset(
        shuffle=False,
        analogy_sentence_infer=ANALOGY_TEMPLATE_SIMPLE_INFERENCE,
        analogy_sentence_full=ANALOGY_TEMPLATE_SIMPLE_FULL,
        examples_file=SCAN_EXAMPLES_FILEPATH.format(EXAMPLE_CATEGORIES[0]),
        examples_start_idx=0,
        examples_shot_nr=1
    )
elif args.dataset == 'BATS':
    BATS_FILENAME = args.BATS_filename
    BATS_dataset = BATSDataloader_0shot(
        BATS_FOLDER, 
        BATS_FILENAME, 
        numberOfAnalogy = 2, 
        cot = COT_TEMPLATE,
        shuffle = False,
        promptType='by-relation', 
        promptFormat = ANALOGY_TEMPLATE_SIMPLE_INFERENCE
    )
else:
    raise ValueError("Dataset not supported")


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


# ----- Run inference for SCAN -----
results = []
if args.dataset == 'SCAN':
    for i, sample in tqdm(enumerate(dataset)):
        output = LLM.generate(sample['inference'])
        results.append([sample, output])
    save_file = os.path.join(SAVE_DIR, f'{args.model.split("/")[1]}_generated_prompts.pl')
    with open(save_file, 'wb') as f:
        pickle.dump(results, f)
elif args.dataset == 'BATS':
    for i, sample in tqdm(enumerate(BATS_dataset())):
        output = LLM.generate(sample['inference'])
        results.append([sample, output])
    model = args.model.split("/")[1]
    file_name = BATS_dataset.get_file_name()
    if not os.path.exists(f'{SAVE_BATS_DIR}/{model}'):
        os.makedirs(f'{SAVE_BATS_DIR}/{model}')
    save_file = os.path.join(f'{SAVE_BATS_DIR}/{model}', f'{file_name}_generated_prompts.pl')
    with open(save_file, 'wb') as f:
        pickle.dump(results, f)
else:
    raise ValueError("Dataset not supported")

# ----- Evaluate -----
# results = pd.read_pickle(f'{args.model.split("/")[1]}_generated_prompts.pl')
# results = pd.read_pickle(save_file)
acc_score = evaluate(results, SimpleEvaluationStrategy())
print(f"Score is {acc_score}%")