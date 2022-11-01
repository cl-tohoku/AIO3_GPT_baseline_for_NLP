import pandas as pd
import re
import torch
from transformers import T5Tokenizer, AutoModelForCausalLM, AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
import os


def main(args):
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
    data = pd.read_json(args.input_file, lines=True)
    questions = list(data["question"])
    model_answers = []
    max_length = 100
    for question in tqdm(questions):
        question = question + "答えは「"
        token_ids = tokenizer.encode(
            question, add_special_tokens=False, return_tensors="pt")
        model.eval()
        with torch.no_grad():
            output_ids = model.generate(
                token_ids.to(model.device),
                max_length=len(token_ids[0])+max_length,
                min_length=1,
                do_sample=False, 
                top_k=500,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bad_word_ids=[[tokenizer.unk_token_id]]
            )

        output = tokenizer.decode(output_ids.tolist()[0])
        try:
            model_ans = re.findall("「(.*?)」", output)[-1]
        except IndexError:
            model_ans = output
            print("longer output:", output)
        model_answers.append(model_ans)
    assert len(model_answers) == len(questions), f"モデル出力と実際のデータ数が一致していません。"
    data["prediction"] = model_answers
    data.drop(columns=["question"],inplace=True)
    data.to_json(args.output_file, orient='records', force_ascii=False, lines=True)

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
                        default="work/model_answer.jsonl",
                        help="GPTモデルの出力結果を格納するファイル。")
    parser.add_argument("--save_model",
                        action="store_true",
                        help="日本語GPTモデルを自分のパソコンに保存する")
    args = parser.parse_args()

    main(args)
