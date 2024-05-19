import argparse

from get_datasets import SCAN_EXAMPLES_FILEPATH, EXAMPLE_CATEGORIES
from prompt_templates.analogy import ANALOGY_TEMPLATE_SIMPLE_INFERENCE, ANALOGY_TEMPLATE_SIMPLE_FULL
from model import LLMObj
from evaluate import *

from tqdm import tqdm
from transformers import BitsAndBytesConfig
from datasets import ScanDataset

from utils import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='microsoft/Phi-3-mini-128k-instruct', help='LLM Model')
parser.add_argument('--tokenizer', type=str, default=None, help="LLM Tokenizer. If not set, will use the HF auto tokenizer loaded from the model's name")
parser.add_argument('--quantization', type=str, default='4bit', help='LLM Quantization', choices=['None', '4bit'])
parser.add_argument('--low_cpu_mem_usage', default=True, type=bool, help='Low CPU Memory usage')
parser.add_argument('--seed', type=int, default=1234, help='Random seed to use throughout the pipeline')
parser.add_argument('--debug', type=bool, default=False, help='If running in debug mode, there will only be considered a few number of entries from the dataset!')
parser.add_argument('--run_on_cpu', type=bool, default=False, help='If running on cpu, there will be a dummy pipeline created. No real model inference will hapen!')
parser.add_argument('--save_filename_details', type=str, default=None, help='Adds more details to the save filename.')
args = parser.parse_args()
print("CL Arguments are:")
print(args)

if args.debug:
    debug_print("RUNNING IN DEBUG MODE")

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
    'tokenizer_name': args.tokenizer,
    'dummy_pipeline': args.run_on_cpu
}
print("LLMObj Arguments are:")
print(LLMObj_args)

# ----- Load the model -----
LLM = LLMObj(**LLMObj_args)

# ----- Run inference -----
results = []
results_buffer_save = []

results_filename = f'{args.model.split("/")[1]}'
if args.save_filename_details:
    results_filename += f'_{args.save_filename_details}'
results_size_to_write = 100

for i, sample in tqdm(enumerate(dataset)):
    if args.debug and i >= 5:
        debug_print("Stopping at five points from the dataset")
        break
    output = LLM.generate(sample['inference'])
    results.append([sample, output])
    results_buffer_save.append([sample, output])
    if len(results_buffer_save) == results_size_to_write:
        save_results(results_buffer_save, results_filename)
        results_buffer_save = []


# ----- Evaluate -----

# results = pd.read_pickle(f'{args.model.split("/")[1]}_generated_prompts.pl')
acc_score = evaluate(results, SimpleEvaluationStrategy())
print(f"Score is {acc_score}%")