# Introduction
This repository holds a baseline model for exercises in Natural Language Processing 2023.

# Contents
## Environment construction
- First, clone this repository with the following command.
```bash
$ git clone https://github.com/cl-tohoku/AIO3_GPT_baseline_for_NLP.git
$ cd AIO3_GPT_baseline_for_NLP
```

- Execute the following command.
```bash
$ bash setup.sh
```


### Development Data

The file is in json lines format, consisting mainly of the elements shown below.
- `qid`: question id
- `number`: integer type question id
- `question`: question
- `answers`: answers list
```json

{
  "qid": "AIO02-0002", 
  "competition": "第2回AI王", 
  "section": "開発データ問題",
  "number": 2, 
  "question": "氷った海に穴を開けて漁をすることから、漢字で「氷の下の魚」と書くタラ科の魚は何?",
  "answers": ["コマイ"]
  }

```
### Test Data
The test data is in JSON Lines (jsonl) format as shown below, containing only the question ID (qid) and the question text (question).
```json
{"qid": "AIO02-1001", "question": "全長は約10.9km。アメリカの国道1号線の一部である、フロリダ・キーズの島々を結ぶ橋の名前は何?"}

{"qid": "AIO02-1002", "question": "コロイド溶液に光を通した時、光の散乱によって道筋が見える、という現象を、発見者にちなんで何現象という?"}
```

## Zero-shot inference using Japanese GPT model
By executing the following code, you can perform zero-shot inference using rinna Corporation's [Japanese GPT model](https://huggingface.co/rinna/japanese-gpt-1b).
### Development Data
```bash
# Example
$ python eval_model_jsonl.py data/dev.jsonl --output_file work/model_answer.csv --save_model --sample 1000
```

You can see the results in detail with the following command.
```bash
$ less work/model_answer.csv
```

### Test Data
By executing the following code, you can output model answers for test data.
```bash
# Example
$ python eval_model_jsonl_unlabel.py data/test.jsonl --output_file work/model_answer.jsonl
```

## Exercise Contents
Add sentences to the question text to create your own prompt. Modify add_prompt function in util.py.
```python
# util.py

'''
Input: question
Output: prompt
Note:
    ・The model's answer is the content of the model's output 「」
    ・So, it is better to end the prompt with 「 
'''
def add_prompt(question: str, **kwargs: dict[str, Any]) -> str:
    # Add your prompt
    return f'{question}答えは「'

```
