import pandas as pd
import torch
from transformers import T5Tokenizer, AutoModelForCausalLM, AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
import os
from util import add_prompt, extract_answer


def main(args):
    ### load pretrained model ###
    model_path = './models/japanese-gpt.pt'
    tokenizer_path = './models/japanese-gpt_tokenizer'
    if args.english_ver:
        model_path = './models/english-gpt.pt'
        tokenizer_path = './models/english-gpt_tokenizer'

    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        print("model loading via offline...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        print("model loading via online...")
        if args.english_ver:
            # https://huggingface.co/gpt2-xl
            tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
            model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
        else:
            # https://huggingface.co/rinna/japanese-gpt-1b
            tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-1b")
            model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-1b")
        if args.save_model:
            print("saving model to local.")
            tokenizer.save_pretrained(tokenizer_path)
            model.save_pretrained(model_path)

    if torch.cuda.is_available():
        model = model.to("cuda")
    print("INFO: The model was loaded.")


    ### load data ###
    data = pd.read_json(args.input_file, lines=True)
    if args.sample > 0:
        data = data.sample(n=args.sample, random_state=42)

    texts = list(data["question"])
    answers = list(data["answers"])

    model_answers = []
    max_length = 10 

    ### predict answer ###
    for text in tqdm(texts, total=len(texts)):
        text = add_prompt(text) # add your own prompt to question
        token_ids = tokenizer.encode(
            text, add_special_tokens=False, return_tensors="pt")
        model.eval()
        with torch.no_grad():
            if args.english_ver:
                output_ids = model.generate(
                    token_ids.to(model.device),
                    max_length=len(token_ids[0])+max_length,
                    min_length=1,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
            else:
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
        print('debug: ', output)
        try:
            model_ans = extract_answer(output, args.english_ver)
        except IndexError:
            model_ans = output
            print("longer output:", output)
        model_answers.append(model_ans)
    assert len(model_answers) == len(answers), f"Model output does not match input data count."
    correct = [1 if result in gold else 0 for result,
               gold in zip(model_answers, answers)]
    correct_num = correct.count(1)
    acc = correct_num/len(model_answers)
    print(f"Number of questions:{len(answers)}")
    print(f"Number of correct answers:{correct_num}")
    print(f"Accuracy rate:{acc:.3f}")
    df_ans = pd.DataFrame([texts, answers, model_answers, correct], index=[
        "question", "answers", "prediction", "correct"]).T
    df_ans.to_csv(args.output_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""
    Sample code for zero-shot inference of QA with Japanese GPT model for development data.
    """)
    parser.add_argument("input_file",
                        type=str,
                        default="data/dev.jsonl",
                        help="Evaluation data set written in json lines format with one question per line.")
    parser.add_argument("--output_file",
                        type=str,
                        default="work/model_answer.csv",
                        help="Where to save GPT model output.")
    parser.add_argument("--sample",
                        default=-1,
                        type=int,
                        help="Number of questions to be solved by the model. If not specified, inference is performed on all data.")
    parser.add_argument("--save_model",
                        action="store_true",
                        help="If true, save GPT model in local environment.")
    parser.add_argument("--english_ver",
                        action="store_true",
                        help="If true, use english GPT model.")
    args = parser.parse_args()

    main(args)
