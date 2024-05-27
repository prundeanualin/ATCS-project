from prompt_processing.templates import *


def prepare_prompt(inference, examples, n_shot: int, baseline: bool, cot: bool, include_task_description: bool, special_instruction: str):
    prompt = ''

    # Possibly extend the prompt with the task description and some examples
    if include_task_description:
        prompt += ANALOGY_DESCRIPTION

    # Zero-shot
    if n_shot == 0:
        if baseline:
            prompt += BASELINE_INDICATION.format(inference)
        # Possibly add CoT instruction only if it is zero-shot
        elif cot:
            prompt += COT_INSTRUCTION.format(inference)
        else:
            prompt += inference

    # In case of one/few-shot, prepend the examples to the prompt
    else:
        for ex in examples:
            if cot:
                example_answer = ex['analogy_detailed_cot']
            else:
                example_answer = ex['analogy_complete']
            prompt += FEW_SHOT_TEMPLATE.format(ex['analogy_incomplete'], example_answer)

        # Now add the inference analogy in the same Question/Answer template
        # //TODO try to include a prompt where the model is told to respond in a specific format with the final answer so that we can use that in evaluation (maybe below)
        if special_instruction == 'cot_few_structured':
            # Add a special format for cot, so that the model is guided on a final response format
            prompt += COT_INSTRUCTION_FEW_SHOT.format(inference)
        else:
            # Add the inference analogy in the same Question/Answer template, with an empty answer
            prompt += FEW_SHOT_TEMPLATE.format(inference, '')
    return prompt
