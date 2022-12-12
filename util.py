from __future__ import annotations
import re
from typing import Any


'''
Input: question
Output: prompt
'''
def add_prompt(question: str, **kwargs: dict[str, Any]) -> str:
    # Add your prompt
    return f'''
Share of export from India is the maximum to the following country? [United States]
Which structure in the urinary system carries urine to the bladder? [ureters]
Who won Americas next top model season 22? [Nyle DiMarco]
{question}? [''' 


def extract_answer(output: str, english_ver: bool) -> str:
    if english_ver:
        return re.findall("\[(.*?)\]", output)[-1] # capture []
    else:
        return re.findall("「(.*?)」", output)[-1] # capture 「」
