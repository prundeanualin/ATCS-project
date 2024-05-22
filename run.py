import argparse
import time

from get_datasets import SCAN_EXAMPLES_FILEPATH
from prompt_processing.templates import ANALOGY_TEMPLATE_SIMPLE_INFERENCE, ANALOGY_TEMPLATE_SIMPLE_FULL
from model import LLMObj
from evaluate import *

from transformers import BitsAndBytesConfig
from datasets import ScanDataloader
from prompt_processing.prompting import prepare_prompt

from utils import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='microsoft/Phi-3-mini-128k-instruct', help='LLM Model')
parser.add_argument('--tokenizer', type=str, default=None, help="LLM Tokenizer. If not set, will use the HF auto tokenizer loaded from the model's name")
parser.add_argument('--quantization', type=str, default='4bit', help='LLM Quantization', choices=['None', '4bit'])
parser.add_argument('--low_cpu_mem_usage', default=True, action=argparse.BooleanOptionalAction, help='Low CPU Memory usage')

parser.add_argument('--seed', type=int, default=1234, help='Random seed to use throughout the pipeline')
parser.add_argument('--save_filename_details', type=str, default=None, help='Adds more details to the save filename.')

# ---- Controlling the prompt ----
# one-shot/few-shot behaviours
parser.add_argument('--n_shot', type=int, default=0, help='Few shot number of examples.')
# Set the type of examples to use
parser.add_argument('--example_type', type=str, default='baseline', choices=['baseline', 'detailed', 'simple', 'long', 'short'], help='Few shot number of examples.')
# description of analogy resolution task
parser.add_argument('--include_task_description', default=False, action=argparse.BooleanOptionalAction, help='If true, the prompt will also include a brief description of the analogy resolution task.')
# use cot
parser.add_argument('--cot', default=False, action=argparse.BooleanOptionalAction, help='If true, the prompt will also include the CoT instruction.')

# Control the dataset iteration so that only a subsample of it is run
# You can choose by analogy type (BATS)
parser.add_argument('--analogy_type', type=str, default='', help='Analogy type selection.')
# Or you can choose by index interval (SCAN)
parser.add_argument('--data_start_idx', type=int, default=0, help='The index at which to start picking samples for inference.')
parser.add_argument('--data_end_idx', type=int, default=-1, help='The index at which to stop picking samples for inference.')

parser.add_argument('--run_on_cpu', default=False, action=argparse.BooleanOptionalAction, help='If running on cpu, there will be a dummy pipeline created, since quantization is not supported on cpu. No real model inference will hapen!')


args = parser.parse_args()
print("CL Arguments are:")
print(args)

os.environ['HF_TOKEN'] = "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"
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

torch.set_default_device('cuda')

seed_experiments(args.seed)

# ----- Load dataset -----
dataloader = ScanDataloader(
    shuffle=False,
    analogy_sentence_infer=ANALOGY_TEMPLATE_SIMPLE_INFERENCE,
    analogy_sentence_full=ANALOGY_TEMPLATE_SIMPLE_FULL,
    examples_file=SCAN_EXAMPLES_FILEPATH.format(args.example_type),
    examples_shot_nr=args.n_shot
)
data_start_idx = args.data_start_idx
data_end_idx = args.data_end_idx
if data_end_idx < 0:
    data_end_idx = len(dataloader)
# Check for valid data indices
if data_end_idx > len(dataloader):
    raise Exception("Arguments passed are invalid! data_end_idx cannot be greater than dataset size")
if data_start_idx >= data_end_idx:
    raise Exception("Arguments passed are invalid! data_start_idx must be smaller than data_end_idx")

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

for i in range(data_start_idx, data_end_idx):
    start = time.time()
    sample = dataloader[i]
    if args.analogy_type and sample['analogy_type'] != args.analogy_type:
        continue

    prompt = prepare_prompt(sample['inference'],
                            sample['examples'],
                            n_shot=args.n_shot,
                            cot=args.cot,
                            include_task_description=args.include_task_description)
    print("Prompt is: ")
    print(prompt)
    print("---------------\n")
    output = LLM.generate(prompt)

    del sample['examples']
    results.append([sample, output])

    end = time.time()
    duration = end - start
    durations.append(duration)
    print(f"Iteration index {i}/{data_end_idx - 1}: %.2f sec" % duration)

d = np.array(durations)
print("Inference duration(sec): total - %.2f, avg - %.2f, max - %.2f, min - %.2f" % (d.sum(), d.mean(), d.max(), d.min()))

# ----- Evaluate -----
print("-- Evaluating the model --")
evaluation_results = evaluate(results, SimpleEvaluationStrategy())
print("Evaluation results:")
print(evaluation_results)

# ----- Saving the results -----
results_filename = f'{args.n_shot}shot_cot({args.cot})_description({args.include_task_description})'
if args.analogy_type:
    results_filename += f'_{args.analogy_type}'
if args.data_end_idx > 0:
    results_filename += f'_dataidxs[{data_start_idx}-{data_end_idx}]'
results_filename += f'_{args.model.split("/")[1]}'
if args.save_filename_details:
    results_filename = f'{args.save_filename_details}_' + results_filename
save_results(results, evaluation_results, results_filename)