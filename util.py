from __future__ import annotations
import re
from typing import Any


'''
Input: question
Output: prompt
Note:
    ・The model's answer is the content of the model's output 「」(Japanese) or [ ] (English)
    ・So, it is better to end the prompt with "「" (Japanese) or "[" (english)
'''
def add_prompt(question: str, lang: str, **kwargs: dict[str, Any]) -> str:
    # Add your prompt
    if lang == "en":
        prompt = f"Question: {question}? Answer: ["
    elif lang == "ja":
        prompt = f'質問：{question}? 回答：「'
    else:
        assert 0, lang
    return prompt


'''
Input: model prediction str
Output: answer str
Note:
    ・You can also change this part if you want.
'''
def extract_answer(output: str, lang: str) -> str:
    if lang == "en":
        return re.findall("\[(.*?)\]", output)[-1] # capture []
    elif lang == "ja":
        return re.findall("「(.*?)」", output)[-1] # capture 「」
    else:
        assert 0, lang
