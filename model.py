from transformers import (AutoTokenizer,
                          pipeline
                          )
import textwrap

from utils import DummyPipeline


class LLMObj:
    def __init__(self, model,
                 model_kwargs,
                 tokenizer_name,
                 system_prompt="",
                 # This is used on devices without a GPU, to make sure that the rest of the code runs ok
                 dummy_pipeline=False
                 ):

        # If tokenizer name is empty, then load it based on the model's name
        if not tokenizer_name:
            tokenizer_name = model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if dummy_pipeline:
            pipe = DummyPipeline(tokenizer)
        else:
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

        generated_text = outputs[0]["generated_text"][len(prompt):]
        wrapped_text = self.wrap_text(generated_text)
        # display(Markdown(wrapped_text))
        return generated_text
