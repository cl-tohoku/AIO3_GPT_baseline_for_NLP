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
    #prompt = f'{question}の答えは「'
    #prompt = f'質問：{question}の答えは「'    

    #prompt = f'質問：「ちょっと何言ってるか分からない」というセリフで知られる、お笑いコンビ・サンドウィッチマンのメンバーは誰?の答えは【富澤たけし】</s>質問：{question}の答えは「'
    #prompt = f'質問：「ちょっと何言ってるか分からない」というセリフで知られる、お笑いコンビ・サンドウィッチマンのメンバーは誰?の答えは「富澤たけし」</s>質問：{question}の答えは「'
    #prompt = f'質問：『ちょっと何言ってるか分からない』というセリフで知られる、お笑いコンビ・サンドウィッチマンのメンバーは誰?の答えは「富澤たけし」</s>質問：{question}の答えは「'    
    #prompt = f'質問：質問：スペイン領バレアレス諸島に属し、ダンスミュージックが盛んなことから『パーティ・アイランド』とも呼ばれる島はどこ?の答えは「イビサ島」</s>質問：『ちょっと何言ってるか分からない』というセリフで知られる、お笑いコンビ・サンドウィッチマンのメンバーは誰?の答えは「富澤たけし」</s>質問：{question}の答えは「'
    prompt = f'質問：クォークと反クォークからなり、湯川秀樹が理論的に存在を予言し、ノーベル物理学賞受賞にいたった粒子は何?の答えは「中間子」</s>質問：質問：スペイン領バレアレス諸島に属し、『パーティ・アイランド』とも呼ばれる島はどこ?の答えは「イビサ島」</s>質問：『ちょっと何言ってるか分からない』というセリフで知られる、お笑いコンビ・サンドウィッチマンのメンバーは誰?の答えは「富澤たけし」</s>質問：{question}の答えは「'


    return prompt
