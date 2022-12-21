import pandas as pd
import torch
from transformers import T5Tokenizer, AutoModelForCausalLM, AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
import os

from util import add_prompt, extract_answer
import accelerate
import logging


logger = logging.getLogger(__name__)
fmt = "%(asctime)s %(levelname)s %(name)s : %(message)s"
logging.basicConfig(level=logging.INFO, format=fmt)


def main(args):
    ### load pretrained model ###
    if args.lang == "en":
        model_path = './models/english-gpt.pt'
        tokenizer_path = './models/english-gpt_tokenizer'
    elif args.lang == "ja":
        model_path = './models/japanese-gpt.pt'
        tokenizer_path = './models/japanese-gpt_tokenizer'
    else:
        assert 0, args.lang

        
    if not args.force_load_model and os.path.exists(model_path) and os.path.exists(tokenizer_path):
        logger.info("model loading via offline...")
        t_tkn = tokenizer_path
        t_mdl = model_path
    else:
        logger.info("model loading via online...")
        if args.lang == "en":
            # https://huggingface.co/gpt2-xl
            t_tkn = "gpt2-xl" #"EleutherAI/gpt-neo-1.3B" #"neulab/gpt2-finetuned-wikitext103" #"gpt2-xl"
            t_mdl = "gpt2-xl" #"EleutherAI/gpt-neo-1.3B" #"neulab/gpt2-finetuned-wikitext103" #"gpt2-xl"
        elif args.lang == "ja":
            # https://huggingface.co/rinna/japanese-gpt-1b
            t_tkn = "rinna/japanese-gpt-1b"
            t_mdl = "rinna/japanese-gpt-1b"
        else:
            assert 0, args.lang
    tokenizer = AutoTokenizer.from_pretrained(t_tkn)
    model = AutoModelForCausalLM.from_pretrained(
        t_mdl,
        torch_dtype=torch.float16,
        #load_in_8bit=True,
        device_map="auto",
    )

    if args.save_model:
        logger.info("saving model to local.")
        tokenizer.save_pretrained(tokenizer_path)
        model.save_pretrained(model_path, max_shard_size="500MB")

    if torch.cuda.is_available():
        model = model.to("cuda")
    logger.info("The model was loaded.")


    ### load data ###
    logger.info("loading data...")
    data = pd.read_json(args.input_file, lines=True)
    if args.sample > 0:

        data = data.sample(n=args.sample, random_state=42)

    qid = list(data["qid"])
    texts = list(data["question"])
    if "answers" in data:
        answers = list(data["answers"])
    else:
        answers = None
    predictions = []
    output_list = []    
    max_length = 100 

    logger.info("start estimation...")
    ### predict answer ###
    for text in tqdm(texts, total=len(texts)):
        text = add_prompt(text, args.lang) # add your own prompt to question
        token_ids = tokenizer.encode(
            text, add_special_tokens=False, return_tensors="pt")
        model.eval()
        with torch.no_grad():
            if args.lang == "en":
                output_ids = model.generate(
                    token_ids.to(model.device),
                    max_length=len(token_ids[0])+max_length,
                    min_length=1,
                    do_sample=False,
                    top_k=500,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    #bos_token_id=tokenizer.bos_token_id,
                    #eos_token_id=tokenizer.eos_token_id,
                )
            elif args.lang == "ja":
                output_ids = model.generate(
                    token_ids.to(model.device),
                    max_length=len(token_ids[0])+max_length,
                    min_length=1,
                    do_sample=False,
                    #top_k=500,
                    #top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            else:
                assert 0, args.lang
        output = tokenizer.decode(output_ids.tolist()[0])
        logger.debug(f"debug: {answers[len(predictions)]} ||| {output}")
        print(f"debug: {answers[len(predictions)]} ||| {output}")
        try:
            model_ans = extract_answer(output, args.lang)
        except IndexError:
            model_ans = output
            logger.warning(f"longer output: {output}")
        output_list.append(output)
        predictions.append(model_ans)
    if answers is not None:
        assert len(predictions) == len(answers), f"Model output does not match input data count."
        correct = [1 if result in gold else 0 for result,
                   gold in zip(predictions, answers)]
        correct_num = correct.count(1)
        acc = correct_num/len(predictions)
        print(f"Number of questions:{len(answers)}")
        print(f"Number of correct answers:{correct_num}")
        print(f"Accuracy rate:{acc:.3f}")
        df_ans = pd.DataFrame([qid, texts, answers, predictions, correct, output_list], index=[
            "qid", "question", "answers", "prediction", "correct", "output"]).T
    else:
        df_ans = pd.DataFrame([qid, texts, predictions, output_list], index=[
            "qid", "question", "prediction", "output"]).T

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
    parser.add_argument("--force_load_model",
                        action="store_true",
                        help="If true, load model and tokenizer from online.")    
    parser.add_argument("--lang",
                        type=str,
                        default="ja",
                        help="Language [ja,en]")
    args = parser.parse_args()

    main(args)
