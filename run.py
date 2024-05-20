import argparse
import time

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

# Examples for one-shot and few-shot
parser.add_argument('--N_shot', type=int, default=0, help='Nr N of examples to prepare for each category to be fed in the N-shot inference.')

parser.add_argument('--seed', type=int, default=1234, help='Random seed to use throughout the pipeline')
parser.add_argument('--save_filename_details', type=str, default=None, help='Adds more details to the save filename.')

parser.add_argument('--mode', type=str, default='train', choices=['train', 'validation'], help='If running in validation mode, there will only be considered a few number of entries from the dataset!')
parser.add_argument('--run_on_cpu', type=bool, default=False, help='If running on cpu, there will be a dummy pipeline created, since quantization is not supported on cpu. No real model inference will hapen!')
args = parser.parse_args()
print("CL Arguments are:")
print(args)

debug_print(f"Running in {args.mode} mode")

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
    examples_shot_nr=args.N_shot
)
NUM_VALIDATION_SAMPLES = int(len(dataset) / 10)

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
durations = []
results_filename = f'{args.model.split("/")[1]}'
if args.save_filename_details:
    results_filename += f'_{args.save_filename_details}'

print("-- Running the model --")

for i, sample in tqdm(enumerate(dataset)):
    start = time.time()
    if args.mode == 'validation' and i == NUM_VALIDATION_SAMPLES:
        debug_print(f"Stopping at the first {NUM_VALIDATION_SAMPLES} points from the dataset")
        break
    output = LLM.generate(sample['inference'])
    results.append([sample, output])
    end = time.time()
    duration = end - start
    durations.append(duration)
    print(f"Iteration {i}/{len(dataset)}: %.2f sec" % duration)

d = np.array(durations)
print("Inference duration: avg - %.2f, max - %.2f, min - %.2f" % (d.mean(), d.max(), d.min()))

# ----- Evaluate -----
print("-- Evaluating the model --")
# results = pd.read_pickle(f'{args.model.split("/")[1]}_generated_prompts.pl')
acc_score = evaluate(results, SimpleEvaluationStrategy())
print(f"Score is {acc_score}%")

evaluation_metrics = {
    "acc": acc_score
}

save_results(results, evaluation_metrics, results_filename)