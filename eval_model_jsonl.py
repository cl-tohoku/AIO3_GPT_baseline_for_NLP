import pandas as pd
import re
import torch
from transformers import T5Tokenizer, AutoModelForCausalLM, AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
import os
from util import add_prompt

def main(args):
    ### load pretrained model (https://huggingface.co/rinna/japanese-gpt-1b) ###
    path = 'models/japanese-gpt.pt'
    if os.path.exists(path):
        print("model loading via offline...")
        tokenizer = T5Tokenizer.from_pretrained(
            "./models/japanese-gpt_tokeizer")
        model = AutoModelForCausalLM.from_pretrained(
            "./models/japanese-gpt.pt")
    else:
        print("model loading via online...")
        tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt-1b")
        model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-1b")
        if args.save_model:
            print("saving model to local.")
            tokenizer.save_pretrained("./models/japanese-gpt_tokeizer")
            model.save_pretrained("./models/japanese-gpt.pt")

    if torch.cuda.is_available():
        model = model.to("cuda")
    print("INFO: The model was loaded.")


    ### load data ###
    data = pd.read_json(args.input_file, lines=True)
    if args.sample > 0:
        data = data.sample(n=args.sample)

    texts = list(data["original_question"])
    answers = list(data["answers"])
    model_answers = []
    max_length = 100
    for text in tqdm(texts, total=len(texts)):
        text = add_prompt(text)
        token_ids = tokenizer.encode(
            text, add_special_tokens=False, return_tensors="pt")
        model.eval()
        with torch.no_grad():
            output_ids = model.generate(
                token_ids.to(model.device),
                max_length=len(token_ids[0])+max_length,
                min_length=1,
                do_sample=False,  # これをFalseにするとランダム性が消える？
                top_k=500,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bad_word_ids=[[tokenizer.unk_token_id]]
            )

        output = tokenizer.decode(output_ids.tolist()[0])
        try:
            model_ans = re.findall("「(.*?)」", output)[-1] # capture 「」
        except IndexError:
            model_ans = output
            print("longer output:", output)
        model_answers.append(model_ans)
    assert len(model_answers) == len(answers), f"モデル出力と実際のデータ数が一致していません。"
    correct = [1 if result in gold else 0 for result,
               gold in zip(model_answers, answers)]
    correct_num = correct.count(1)
    acc = correct_num/len(model_answers)
    print(f"問題数:{len(answers)}")
    print(f"正解数:{correct_num}")
    print(f"正解率:{acc:.3f}")
    df_ans = pd.DataFrame([texts, answers, model_answers, correct], index=[
        "問題", "答え", "モデル出力", "自動評価"]).T
    # df_ans.to_csv('work/model_ans_ver2.csv')
    df_ans.to_csv(args.output_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""
    日本語GPTモデルによるQAのzero-shot推論のサンプルコード。
    """)
    parser.add_argument("input_file",
                        type=str,
                        help="json lines形式で1行1問で書かれている評価データセット。"
                        )
    parser.add_argument("--output_file",
                        type=str,
                        default="work/model_answer.csv",
                        help="GPTモデルの出力結果を格納するファイル。")
    parser.add_argument("--sample",
                        default=-1,
                        type=int,
                        help="モデルに解かせる問題数。指定がない場合は全データに対して推論を行う。")
    parser.add_argument("--save_model",
                        action="store_true",
                        help="If true, save japanese GPT model in local environment")
    args = parser.parse_args()

    main(args)
