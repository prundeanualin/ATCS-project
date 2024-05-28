ANALOGY_TEMPLATE_SIMPLE_FULL = "If {} is like {}, then {} is like {}."
ANALOGY_TEMPLATE_SIMPLE_INFERENCE = "If {} is like {}, then {} is like..."

ANALOGY_DESCRIPTION = """
You are an expert in analogy resolution. You understand and apply relational patterns, often involving linguistic, conceptual, or functional similarities.
You will now complete analogies that look like "If A is like B, then C is like ...", where you need to find the missing answer D. For this, you need to identify the relationship
between A and C and apply this relationship on the concept in B in order to find the answer for D. Ensure that the relationship is consistent and logical. 

"""

BASELINE_INDICATION = """
Question: {}
Answer: The final answer is
"""

COT_INSTRUCTION = """
Question: {}
Answer: Let's first think this step by step and then give the final answer at the end phrased like 'The answer is: ...'.
"""

FEW_SHOT_TEMPLATE = """
Question: {}
Answer: {}
"""