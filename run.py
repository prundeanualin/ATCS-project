import argparse
import time

import numpy as np

from get_datasets import *
from prompt_templates.analogy import *
from model import LLMObj
from evaluate import *

import torch
from tqdm import tqdm
from transformers import BitsAndBytesConfig
import pickle

from datasets import ScanDataset, BATSDataloader
from datasets import BATSDataloader_0shot, BATSDataloader_fewshot

import os

from utils import seed_experiments, SAVE_DIR, SAVE_BATS_DIR, save_results

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='microsoft/Phi-3-mini-128k-instruct', help='LLM Model')
parser.add_argument('--tokenizer', type=str, default=None, help="LLM Tokenizer. If not set, will use the HF auto tokenizer loaded from the model's name")
parser.add_argument('--quantization', type=str, default='4bit', help='LLM Quantization', choices=['None', '4bit'])
parser.add_argument('--low_cpu_mem_usage', default=True, type=bool, help='Low CPU Memory usage')
parser.add_argument('--seed', type=int, default=1234, help='Random seed to use throughout the pipeline')
parser.add_argument('--dataset', type=str, default='BATS', help='Dataset to use for inference', choices=['SCAN', 'BATS', 'BATS-fewshot'])
parser.add_argument('--BATS_filename', type=str, default='L01 [hypernyms - animals]', help='BATS filename to use for inference')
parser.add_argument('--prompt_structure', type=str, default='by-relation', help='Prompt structure, e.g. if A is B (by-relation) or if A is C (by-target-word)', choices=['by-relation', 'by-target-word'])
parser.add_argument('--cot', default=False, action=argparse.BooleanOptionalAction, help='If True, will use COT template')
# parser.add_argument('--COT_template', default = False, type=lambda x: False if x.lower() == 'false' else x, help='If True, will use COT template')
parser.add_argument('--number_of_analogy', default=False, type=lambda x: False if x.lower() == 'false' else int(x), help='Number of analogy to do inference on')
parser.add_argument('--number_of_shot', default=0, type=int, help='Number of shot for BATS dataset in few-shot learning')
parser.add_argument('--explanation', default = False, type=lambda x: False if x.lower() == 'false' else x, help='If True, will include explanation in the inference')

parser.add_argument('--run_on_cpu', default=False, action=argparse.BooleanOptionalAction, help='If running on cpu, there will be a dummy pipeline created, since quantization is not supported on cpu. No real model inference will hapen!')

args = parser.parse_args()
print("CL Arguments are:")
print(args)

# CHANGE HF_TOKEN TO YOUR OWN TOKEN
os.environ['HF_TOKEN'] = "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"
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

torch.set_default_device('cuda')

seed_experiments(args.seed)


# ----- Load dataset -----
is_bats = True
if args.dataset == 'SCAN':
    is_bats=False
    dataset = ScanDataset(
        shuffle=False,
        analogy_sentence_infer=ANALOGY_TEMPLATE_SIMPLE_INFERENCE,
        analogy_sentence_full=ANALOGY_TEMPLATE_SIMPLE_FULL,
        examples_file=SCAN_EXAMPLES_FILEPATH.format(EXAMPLE_CATEGORIES[0]),
        examples_start_idx=0,
        examples_shot_nr=1
    )
elif args.dataset == 'BATS':
    dataset = BATSDataloader(
        n_shot=args.number_of_shot,
        fileName=args.BATS_filename,
        numberOfAnalogy=args.number_of_analogy,
        cot=args.cot,
        shuffle=True,
        promptType=args.prompt_structure,
        promptFormat=ANALOGY_TEMPLATE_SIMPLE_INFERENCE,
        promptFull=ANALOGY_TEMPLATE_SIMPLE_FULL,
        explanation=args.explanation
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
    'dummy_pipeline': args.run_on_cpu
}
print("LLMObj Arguments are:")
print(LLMObj_args)


# ----- Load the model -----
LLM = LLMObj(**LLMObj_args)



# ----- Run inference-----
durations = []
results = []

print("-- Running the model --")

for i, sample in enumerate(dataset):
    start = time.time()
    # if args.analogy_type and sample['analogy_type'] != args.analogy_type:
    #     continue

    # if args.dataset == 'SCAN':
    #     prompt = prepare_prompt(sample['inference'],
    #                             sample['examples'],
    #                             n_shot=args.n_shot,
    #                             cot=args.cot,
    #                             include_task_description=args.include_task_description)
    prompt = sample['inference']
    print("Prompt is: ")
    print(prompt)
    print("---------------\n")
    output = LLM.generate(prompt)

    if 'examples' in sample:
        del sample['examples']
    results.append([sample, output])

    end = time.time()
    duration = end - start
    durations.append(duration)
    print(f"Iteration index {i}/{len(dataset) - 1}: %.2f sec" % duration)

d = np.array(durations)
print("Inference duration(sec): total - %.2f, avg - %.2f, max - %.2f, min - %.2f" % (d.sum(), d.mean(), d.max(), d.min()))

# ----- Evaluate -----
print("-- Evaluating the model --")
evaluation_results = evaluate(results, SimpleEvaluationStrategy())
print("Evaluation results:")
print(evaluation_results)

# ----- Saving the results -----

model_name = args.model.split("/")[1]
n_analogy = args.number_of_analogy if args.number_of_analogy else 1
results_filename = f'{args.number_of_shot}shot_cot({args.cot})_nrA({n_analogy})_promptType({args.prompt_structure})_{args.BATS_filename}_{model_name}'
save_results(results, evaluation_results, is_bats, results_filename)