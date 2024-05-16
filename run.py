import argparse

from get_datasets import SCAN_EXAMPLES_FILEPATH, EXAMPLE_CATEGORIES
from prompt_templates.analogy import ANALOGY_TEMPLATE_SIMPLE_INFERENCE, ANALOGY_TEMPLATE_SIMPLE_FULL
from model import LLMObj
import torch
from tqdm import tqdm
from transformers import BitsAndBytesConfig
import pickle
from datasets import ScanDataset
import os

from utils import seed_experiments

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='microsoft/Phi-3-mini-128k-instruct', help='LLM Model')
parser.add_argument('--tokenizer', type=str, default='microsoft/Phi-3-mini-128k-instruct', help='LLM Tokenizer')
parser.add_argument('--quantization', type=str, default='4bit', help='LLM Quantization', choices=['None', '4bit'])
parser.add_argument('--low_cpu_mem_usage', default=True, type=bool, help='Low CPU Memory usage')
parser.add_argument('--seed', type=int, default=1234, help='Random seed to use throughout the pipeline')
args = parser.parse_args()
print("CL Arguments are:")
print(args)

os.environ['HF_TOKEN'] = "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"
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

torch.set_default_device('cuda')

seed_experiments(args.seed)

# ----- Load dataset -----
dataset = ScanDataset(
    shuffle=False,
    analogy_sentence_infer=ANALOGY_TEMPLATE_SIMPLE_INFERENCE,
    analogy_sentence_full=ANALOGY_TEMPLATE_SIMPLE_FULL,
    examples_file=SCAN_EXAMPLES_FILEPATH.format(EXAMPLE_CATEGORIES[0]),
    examples_start_idx=0,
    examples_shot_nr=1
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
    'tokenizer_name': args.tokenizer
}
print("LLMObj Arguments are:")
print(LLMObj_args)

# ----- Load the model -----
LLM = LLMObj(**LLMObj_args)

# ----- Run inference -----
generated_prompts = []
for i, sample in tqdm(enumerate(dataset)):
    output = LLM.generate(sample['inference'])
    generated_prompts.append([sample, output])

with open(f'{args.model.split("/")[1]}_generated_prompts.pl', 'wb') as f:
    pickle.dump(generated_prompts, f)
