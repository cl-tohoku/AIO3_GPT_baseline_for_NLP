# Introduction
This repository holds a baseline model for exercises in Natural Language Processing 2023.

# Contents
## Environment construction
- First, clone this repository with the following command.
```bash
$ git clone -b js_fix1 https://github.com/cl-tohoku/AIO3_GPT_baseline_for_NLP.git
$ cd AIO3_GPT_baseline_for_NLP
```

- Execute the following command for setup.
```bash
$ bash setup.sh
```

- Run python virtual env.
```bash
$ . .venv/bin/activate
```


### Development & Test Data

The file is in json lines format, consisting mainly of the elements shown below.
- `qid`: question id
- `question`: question
- `answers`: answers list
```json

{
  "qid": "AIO02-0002", 
  "question": "氷った海に穴を開けて漁をすることから、漢字で「氷の下の魚」と書くタラ科の魚は何?",
  "answers": ["コマイ"]
}
```

## Zero-shot inference
By executing the following code, you can perform zero-shot inference

### Evaluate Test Data
```bash
# Example
$ python eval_model_jsonl.py data/LecNLP_test_ja.jsonl --output_file outputs/LecNLP_test_ja_prediction.jsonl --lang ja
$ python eval_model_jsonl.py data/LecNLP_test_en.jsonl --output_file outputs/LecNLP_test_en_prediction.jsonl --lang en

# Example: Evaluate only 100 samples from test data (determined as the --sample option for faster development)
$ python eval_model_jsonl.py data/LecNLP_test_ja.jsonl --output_file outputs/LecNLP_test_ja_prediction.jsonl --lang ja --sample 100
```
#### Japanese: originally rinna Corporation's [Japanese GPT model](https://huggingface.co/rinna/japanese-gpt-1b).
#### English:  originally [gpt-2-large model](https://huggingface.co/gpt2-large).




You can see the results in detail with the following command.
```bash
$ jq -s '.' outputs/LecNLP_test_ja_prediction.jsonl | less
$ jq -s '.' outputs/LecNLP_test_en_prediction.jsonl | less
```


## Exercise Contents
Add sentences to the question text to create your own prompt. Modify add_prompt function in util.py.
```python
# util.py

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
        prompt = f"Answer this question: {question}? Answer: ["
    elif lang == "ja":
        prompt = f'{question}の答えは「'
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
```
