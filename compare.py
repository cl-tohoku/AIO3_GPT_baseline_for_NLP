import pandas as pd
import json

def main(file_lst):
    all_answers = []
    
    with open(file_lst[0]) as gold_file:
        gold_answers = []
        for line in gold_file:
            json_obj = json.loads(line)
            gold_answers.append(json_obj["answers"])
        all_answers.append(gold_answers)
            
    for file in file_lst:
        with open(file) as input_file:
            answers = []
            indices = []
            for line in input_file:
                json_obj = json.loads(line)
                answers.append(json_obj["prediction"])
                indices.append(json_obj["question"])
            all_answers.append(answers)
    
    print(len(all_answers), len(indices), len(file_lst))
    df = pd.DataFrame(data=all_answers, index=["gold"] + file_lst, columns=indices)
    df = df.T
    df.to_csv('outputs/output.tsv', sep='\t', index=True, index_label='col0')


if __name__ == "__main__":
    main([
        "outputs/LecNLP_test_ja_prediction1.jsonl",
        "outputs/LecNLP_test_ja_prediction2.jsonl"
    ])
