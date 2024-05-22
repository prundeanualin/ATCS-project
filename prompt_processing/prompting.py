from prompt_processing.templates import *


def prepare_prompt(inference, examples, n_shot: int, cot: bool, include_task_description: bool):
    prompt = ''

    # Possibly extend the prompt with the task description and some examples
    if include_task_description:
        prompt += ANALOGY_DESCRIPTION

    # In case of one/few-shot, prepend the examples to the prompt
    if n_shot == 0:
        # Possibly add CoT instruction only if it is zero-shot
        if cot:
            prompt += COT_INSTRUCTION
        # Finally, add the to-be-completed analogy at the end
        prompt += inference
    elif n_shot == 1:
        for ex in examples:
            example_answer = ex['analogy_complete']
            if cot:
                example_answer += " " + ex['analogy_detailed_cot']
            prompt += FEW_SHOT_TEMPLATE.format(ex['analogy_incomplete'], example_answer)
        # Add the analogy in the same Question/Answer template
        prompt += FEW_SHOT_TEMPLATE.format(inference, '')
    else:
        for ex in examples:
            example_answer = ex['analogy_complete']
            if cot:
                example_answer += ex['analogy_detailed_cot']
            prompt += example_answer + "\n"
        prompt += "\n" + inference
    return prompt
