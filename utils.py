import os
import torch
from transformers import (AutoTokenizer,
                          pipeline
                          )
import textwrap
import argparse

os.environ ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
torch.set_default_device('cuda')

class LLMObj:
  def __init__(self, model, 
               model_kwargs, 
               tokenizer_name=None, 
               system_prompt="",
              #  generation_kwargs={}
               ):
    super(LLMObj, self).__init__()

    if tokenizer_name:
      tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    pipe = pipeline(
      "text-generation",
      model=model,
      tokenizer=tokenizer,
      model_kwargs=model_kwargs,
      trust_remote_code=True
    )

    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    if system_prompt != "":
      system_prompt = system_prompt
    else:
      system_prompt = "You are a friendly and helpful assistant"

    self.model = model
    self.pipe = pipe
    # self.generation_kwargs = generation_kwargs
    self.terminators = terminators
    self.chat_template = [
        {
            "role": "system",
            "content": system_prompt,
        },

        {
            "role": "user",
            "content": ""
        },
    ]

  def update_system_prompt(self, system_prompt):
    self.chat_template[0]['content'] = system_prompt

  def wrap_text(self, text, width=90):
    """Fits text to specified character width."""
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

  def generate(self, input_text, max_length=512):
    if 'Starling' in self.model:
      prompt = f"GPT4 Correct User: {input_text}<|end_of_turn|>GPT4 Correct Assistant:"

    else:
      self.chat_template[1]['content'] = input_text
      prompt = self.pipe.tokenizer.apply_chat_template(
          self.chat_template,
          tokenize=False,
          add_generation_prompt=True
          )

    outputs = self.pipe(
        prompt,
        max_new_tokens=max_length,
        pad_token_id=self.pipe.tokenizer.pad_token_id,
        eos_token_id=self.terminators,
        do_sample=False,
        temperature=0.0,
        top_p=0.9,
        # **generation_kwargs
    )

    generated_outputs = outputs[0]["generated_text"]
    text = outputs[0]["generated_text"][len(prompt):]
    wrapped_text = self.wrap_text(text)
    # display(Markdown(wrapped_text))
    return wrapped_text

  def generate_dummy(self, input_text, max_length=512):
    text = """
        This is a very long long text that will be used for dummy generation of the resulting model inference, 
        just to double check that the model is indeed performing as expected. It may not seem that long at first,
        but trust me that once you get a better look at it and realize that it's manual, then you will appreciate it's
        value better than you could imagine.
    """

    wrapped_text = self.wrap_text(text)
    # display(Markdown(wrapped_text))
    return wrapped_text
    

def parse_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      default='microsoft/Phi-3-mini-128k-instruct',
      type=str,
      help='LLM Model')
  parser.add_argument(
      '--tokenizer',
      default='microsoft/Phi-3-mini-128k-instruct',
      type=str,
      help='LLM Tokenizer')
  parser.add_argument(
      '--quantization',
      default='4bit',
      type=str,
      help='LLM Quantization',
      choices=['None', '4bit'])
  parser.add_argument(
      '--low_cpu_mem_usage',
      default=False,
      type=bool,
      help='Low CPU Memory usage')
  return parser.parse_args()