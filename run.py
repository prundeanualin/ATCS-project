import argparse
import time

from get_datasets import SCAN_EXAMPLES_FILEPATH, EXAMPLE_CATEGORIES
from prompt_templates.analogy import ANALOGY_TEMPLATE_SIMPLE_INFERENCE, ANALOGY_TEMPLATE_SIMPLE_FULL, ANALOGY_DESCRIPTION
from model import LLMObj
from evaluate import *

from transformers import BitsAndBytesConfig
from datasets import ScanDataloader

from utils import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='microsoft/Phi-3-mini-128k-instruct', help='LLM Model')
parser.add_argument('--tokenizer', type=str, default=None, help="LLM Tokenizer. If not set, will use the HF auto tokenizer loaded from the model's name")
parser.add_argument('--quantization', type=str, default='4bit', help='LLM Quantization', choices=['None', '4bit'])
parser.add_argument('--low_cpu_mem_usage', default=True, type=bool, help='Low CPU Memory usage')

parser.add_argument('--seed', type=int, default=1234, help='Random seed to use throughout the pipeline')
parser.add_argument('--save_filename_details', type=str, default=None, help='Adds more details to the save filename.')

# ---- Controlling the prompt ----
# one-shot/few-shot behaviours
parser.add_argument('--n_shot', type=int, default=0, help='Few shot number of examples.')
# description of analogy resolution task
parser.add_argument('--include_task_description', type=bool, default=False, help='If true, the prompt will also include a brief description of the analogy resolution task.')

# Control the dataset iteration so that only a subsample of it is run
# You can choose by analogy type (BATS)
parser.add_argument('--analogy_type', type=str, default='', help='Analogy type selection.')
# Or you can choose by index interval (SCAN)
parser.add_argument('--data_start_idx', type=int, default=0, help='The index at which to start picking samples for inference.')
parser.add_argument('--data_end_idx', type=int, default=-1, help='The index at which to stop picking samples for inference.')

parser.add_argument('--run_on_cpu', type=bool, default=False, help='If running on cpu, there will be a dummy pipeline created, since quantization is not supported on cpu. No real model inference will hapen!')


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
    examples_file=SCAN_EXAMPLES_FILEPATH.format(EXAMPLE_CATEGORIES[0]),
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
results_filename = f'{args.n_shot}_shot'
if args.analogy_type:
    results_filename += f'_{args.analogy_type}'
if args.data_end_idx > 0:
    results_filename += f'_[{data_start_idx}-{data_end_idx}]'
results_filename += f'_{args.model.split("/")[1]}'
if args.save_filename_details:
    results_filename = f'{args.save_filename_details}_' + results_filename

print("-- Running the model --")

for i in range(data_start_idx, data_end_idx):
    start = time.time()
    sample = dataloader[i]
    prompt = sample['inference']

    # Possibly extend the prompt with the task description and some examples
    if args.include_task_description:
        prompt = ANALOGY_DESCRIPTION + prompt
    # In case of one/few-shot, prepend the examples to the prompt
    if args.n_shot > 0:
        prompt = "{}\n" * args.n_shot + prompt
        prompt = prompt.format(*map(lambda x: x['simple'], sample['examples']))

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
acc_score = evaluate(results, SimpleEvaluationStrategy())
print(f"Score is {acc_score}%")

evaluation_metrics = {
    "acc": acc_score
}

save_results(results, evaluation_metrics, results_filename)