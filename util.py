from __future__ import annotations
import re
from typing import Any


'''
Input: question
Output: prompt
'''
def add_prompt(question: str, lang: str, **kwargs: dict[str, Any]) -> str:
    # Add your prompt
    if lang == "en":
        prompt = f"Answer this question: {question}? Answer: ["
    elif lang == "ja":
        prompt = f'{question}の答えは「'
    else:
        assert 0, lang
    return prompt


def extract_answer(output: str, lang: str) -> str:
    if lang == "en":
        return re.findall("\[(.*?)\]", output)[-1] # capture []
    elif lang == "ja":
        return re.findall("「(.*?)」", output)[-1] # capture 「」
    else:
        assert 0, lang
