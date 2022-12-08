from __future__ import annotations
from typing import Any

'''
Input: question
Output: prompt
Note:
    ・The model's answer is the content of the model's output 「」
    ・So, it is better to end the prompt with 「 
'''
def add_prompt(question: str, **kwargs: dict[str, Any]) -> str:
    # Add your prompt
    return f'{question}答えは「'
