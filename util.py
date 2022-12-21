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
        f'{question}の答えは「'
    else:
        assert 0, lang
    return prompt



        #prompt = f"Answer this question: {question}? The answer is ["
#        prompt = f'''
#        Share of export from India is the maximum to the following country? [United States]
#        Which structure in the urinary system carries urine to the bladder? [ureters]
#        Who won Americas next top model season 22? [Nyle DiMarco]
#        {question}? [''' 


        #prompt = f'質問：{question}の答えは「'    

    #prompt = f'質問：「ちょっと何言ってるか分からない」というセリフで知られる、お笑いコンビ・サンドウィッチマンのメンバーは誰?の答えは【富澤たけし】</s>質問：{question}の答えは「'
    #prompt = f'質問：「ちょっと何言ってるか分からない」というセリフで知られる、お笑いコンビ・サンドウィッチマンのメンバーは誰?の答えは「富澤たけし」</s>質問：{question}の答えは「'
    #prompt = f'質問：『ちょっと何言ってるか分からない』というセリフで知られる、お笑いコンビ・サンドウィッチマンのメンバーは誰?の答えは「富澤たけし」</s>質問：{question}の答えは「'    
    #prompt = f'質問：質問：スペイン領バレアレス諸島に属し、ダンスミュージックが盛んなことから『パーティ・アイランド』とも呼ばれる島はどこ?の答えは「イビサ島」</s>質問：『ちょっと何言ってるか分からない』というセリフで知られる、お笑いコンビ・サンドウィッチマンのメンバーは誰?の答えは「富澤たけし」</s>質問：{question}の答えは「'




def extract_answer(output: str, lang: str) -> str:
    if lang == "en":
        return re.findall("\[(.*?)\]", output)[-1] # capture []
    elif lang == "ja":
        return re.findall("「(.*?)」", output)[-1] # capture 「」
    else:
        assert 0, lang
