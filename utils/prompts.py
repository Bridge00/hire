#Code Generator; from https://github.com/zou-group/textgrad/blob/main/evaluation/code_optimization/prompts.py
CODEGEN_SYS = """You are an AI that only responds with python code, NOT ENGLISH. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature).
Use a Python code block to write your response. For example:
```python
print('Hello world!')
```"""

#From https://arxiv.org/pdf/2410.02184#page=18.10
VANILLA_EVAL_BINARY = """
Determine the correctness of the code snippet. Output Yes or No.
Problem Statement: {PROBLEM}
Code Snippet: {CODE}
Answer(Yes or No only): 
"""

#From https://arxiv.org/pdf/2410.02184#page=18.10
CODEJUDGE_ANALYSIS = """
You will be provided with a problem statement and a code snippet that supposedly addresses the
problem in Python.
Your task is to check if the code snippet covers the required functionalities. Do not provide a
corrected version.
Evaluation Steps:
1. Read the problem statement carefully and identify the required functionalities of the
implementation. You can refer to the example to understand the problem better.
2. Read the code snippet and analyze its logic. Check if the code snippet covers all the required
functionalities of the problem.
3. Finally, conclude your evaluation.
Problem Statement: {PROBLEM}
Code Snippet: {CODE}
"""

#From https://arxiv.org/pdf/2410.02184#page=18.10
CODEJUDGE_SUMMARY = """
You will be provided with an analysis result of a code snippet.
If the analysis believes that the code snippet is correct, output: "Yes". Otherwise, output: "No".
Analysis Result: {ANALYSIS}
"""

#From https://arxiv.org/pdf/2410.02184#page=18.10
CODEJUDGE_FAULT_LOCALIZATION = """
You will be provided with a problem statement, a code snippet that supposedly addresses the problem,
and a catalog of code inconsistencies.
Evaluation Steps:
1. Read the problem statement carefully to identify the functionalities required for the
implementation.
2. Read the code snippet and compare it to the problem statement. Check if the code snippet covers
the required functionalities.
3. Output your answer in a JSON format list.
a) If the code snippet is correct, output: ["inconsistency": "None", "severity": "Negligible"].
b) If the code snippet is incorrect, output the identified inconsistencies and their severity
according to the catalog of code inconsistencies. For example: ["inconsistency": "<inconsistency1>",
"severity": "<severity1>", "inconsistency": "<inconsistency2>", "severity": "<severity2>", ...]
Problem: {PROBLEM}
Code Snippet: {CODE}
Taxonomy of Common Inconsistencies:
1. Missing dependency declarations: Negligible
2. No error messages for unexpected input cases: Negligible
3. Inefficiency, unnecessary statements: Negligible
4. Edge case not handled: Small
5. Logic error: Major
6. Function or variable not defined: Fatal
7. Code not completed: Fatal
Evaluation Form:
JSON output (a JSON list only):
"""

# Prompt for HIRE Decomposer (D)
HIRE_DECOMPOSER = """
Analyze the following Python code and decompose it into exactly {N} high-level steps.
Return the result as a valid JSON object with a single key "steps", which is a list of objects.
Each object in the "steps" list must have:
- "code_segment": The exact code snippet for that step.
- "explanation": A concise explanation of what that code does.

Code:
{CODE}
"""

HIRE_PLAN_CHECKER = """
Analyze and determine the correctness of the following high-level plan for a code solution to the following task: {PROBLEM}.
Return the result as a valid JSON object with starting with a key "correct", which is a boolean.
If "correct" is False, please provide the reasoning in a key "reasoning".
Plan:
{PLAN}
"""

