from torch.utils.data import DataLoader

from utils import LLMObj, parse_args
import torch
from tqdm import tqdm
from transformers import BitsAndBytesConfig
import pickle
from datasets import ScanDataset
import os

args = parse_args()

model_kwargs = {"torch_dtype": torch.bfloat16}

if args.quantization != 'None':
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model_kwargs['quantization_config'] = quantization_config

if args.low_cpu_mem_usage:
    model_kwargs["low_cpu_mem_usage"] = True

LLMObj_args = {}
LLMObj_args['model'] = args.model
LLMObj_args['model_kwargs'] = model_kwargs
if args.tokenizer != 'None':
    LLMObj_args['tokenizer_name'] = args.tokenizer

print(LLMObj_args)
os.environ['HF_TOKEN'] = "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"
os.environ ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
LLM = LLMObj(**LLMObj_args)
dataloader = DataLoader(ScanDataset())
generated_prompts = []

for i, sample in tqdm(enumerate(dataloader)):
    output = LLM.generate(sample['inference'])
    generated_prompts.append([sample, output])

with open(f'{args.model.split("/")[1]}_generated_prompts.pl', 'wb') as f:
    pickle.dump(generated_prompts, f)
