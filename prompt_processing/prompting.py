from prompt_processing.templates import *


def prepare_prompt(inference, examples, n_shot: int, baseline: bool, cot: bool, include_task_description: bool):
    prompt = ''

    # Possibly extend the prompt with the task description and some examples
    if include_task_description:
        prompt += ANALOGY_DESCRIPTION

    # Zero-shot
    if n_shot == 0:
        # Add instruction to force short, direct answer
        if baseline:
            # prompt += STRUCTURED_BASELINE_INDICATION.format(inference)
            prompt += BASELINE_INDICATION + inference
        # Possibly add CoT instruction only if it is zero-shot
        elif cot:
            prompt += inference + " " + COT_INSTRUCTION
            # prompt += inference
        else:
            prompt += inference
    # In case of one/few-shot, prepend the examples to the prompt
    else:
        for ex in examples:
            example_answer = ex['analogy_complete']
            if cot:
                example_answer += " " + ex['analogy_detailed_cot']
            prompt += FEW_SHOT_TEMPLATE.format(ex['analogy_incomplete'], example_answer)
        # Add the inference analogy in the same Question/Answer template
        prompt += FEW_SHOT_TEMPLATE.format(inference, '')
    return prompt
