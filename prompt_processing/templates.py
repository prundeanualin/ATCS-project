ANALOGY_TEMPLATE_SIMPLE_FULL = "If {} is like {}, then {} is like {}."
ANALOGY_TEMPLATE_SIMPLE_INFERENCE = "If {} is like {}, then {} is like..."

ANALOGY_DESCRIPTION = """
You are an expert in analogy resolution. You understand and apply relational patterns, often involving linguistic, conceptual, or functional similarities.
You will now complete analogies that look like "If A is like B, then C is like ...", where you need to find the missing answer D. For this, you need to identify the relationship
between A and C and apply this relationship on the concept in B in order to find the answer for D. Ensure that the relationship is consistent and logical. 

"""

BASELINE_INDICATION = """
Respond only with the answer! Give no explanation and no other words, apart from the answer!
"""

STRUCTURED_BASELINE_INDICATION = """
Question: If A is like B, then C is like ...
Answer: If A is like B, then C is like D

Question: {}
Answer: 
"""

FEW_SHOT_TEMPLATE = """
Question: {}
Answer: {}
"""
COT_INSTRUCTION = "Let's think step by step. "
