# AIO3_GPT_baseline

- [AI王 〜クイズAI日本一決定戦〜](https://www.nlp.ecei.tohoku.ac.jp/projects/aio/)
- 昨年度の概要は [こちら](https://sites.google.com/view/project-aio/competition2)


## 目次

- [AIO3_GPT_baseline](#aio3_gpt_baseline)
  - [目次](#目次)
  - [環境構築](#環境構築)
    - [Dockerコンテナの起動](#dockerコンテナの起動)
  - [データセット](#データセット)
    - [開発用データ](#開発用データ)
    - [テスト用データ](#テスト用データ)
  - [日本語GPTモデルによるzero-shot推論](#日本語gptモデルによるzero-shot推論)
    - [開発用データ](#開発用データ-1)
    - [テスト用データ](#テスト用データ-1)
    - [最終提出](#最終提出)

## 環境構築
- まず、以下のコマンドで本リポジトリをクローンしてください。
```bash
$ git clone https://github.com/cl-tohoku/AIO3_GPT_baseline.git
$ cd AIO3_GPT_baseline
```

- 以下のコマンドでdataディレクトリとworkディレクトリ、modelsディレクトリを作成してください。
```bash
$ mkdir data
$ mkdir work
$ mkdir models
```
### Dockerコンテナの起動
- 以下のコマンドによってDockerコンテナを起動します
```bash
$ docker image build --tag aio3_gpt:latest .
$ docker container run --name gpt_baseline \
  --rm \
  --interactive \
  --tty \
  --gpus all \
  --mount type=bind,src=$(pwd),dst=/code/AIO3_GPT_baseline \
  aio3_gpt:latest \
  bash
```


## データセット

- 訓練データには、クイズ大会[「abc/EQIDEN」](http://abc-dive.com/questions/) の過去問題に対して Wikipedia の記事段落の付与を自動で行ったものを使用しています。
- 開発・評価用クイズ問題には、[株式会社キュービック](http://www.qbik.co.jp/) および [クイズ法人カプリティオ](http://capriccio.tokyo/) へ依頼して作成されたものを使用しています。

- 以上のデータセットの詳細については、[AI王 〜クイズAI日本一決定戦〜](https://www.nlp.ecei.tohoku.ac.jp/projects/aio/)の公式サイト、および下記論文をご覧下さい。

> __JAQKET: クイズを題材にした日本語QAデータセット__
> - https://www.nlp.ecei.tohoku.ac.jp/projects/jaqket/
> - 鈴木正敏, 鈴木潤, 松田耕史, ⻄田京介, 井之上直也. JAQKET:クイズを題材にした日本語QAデータセットの構築. 言語処理学会第26回年次大会(NLP2020) [\[PDF\]](https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P2-24.pdf)

### 開発用データ

主に以下に示した要素からなるjson lines形式のファイルになっています。
- `qid`: 問題インデックス
- `number`: 整数型の問題インデックス
- `question`: 質問
- `answers`: 答えのリスト
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
### テスト用データ
第3回コンペティションのリーダーボード投稿用テストデータは下記よりダウンロードできます。
- [リーダーボード投稿用テストデータ](https://www.google.com/url?q=https%3A%2F%2Fjaqket.s3.ap-northeast-1.amazonaws.com%2Fdata%2Faio_03%2Faio_03_test_unlabeled.jsonl&sa=D&sntz=1&usg=AOvVaw2VL7kspkyoOnakZZr6FUDR)

テストデータは，質問 ID (  qid  ) と問題文 (  question  ) のみを含んだ下記のような JSON Lines (jsonl) 形式になっています。
```json
{"qid": "AIO02-1001", "question": "全長は約10.9km。アメリカの国道1号線の一部である、フロリダ・キーズの島々を結ぶ橋の名前は何?"}

{"qid": "AIO02-1002", "question": "コロイド溶液に光を通した時、光の散乱によって道筋が見える、という現象を、発見者にちなんで何現象という?"}
```

## 日本語GPTモデルによるzero-shot推論
以下のコードを実行することでrinna株式会社の[日本語GPTモデル](https://huggingface.co/rinna/japanese-gpt-1b)によるzero-shot推論を行うことができます。
### 開発用データ
```bash
#実行例
$ python eval_model_jsonl.py path/to/eval_file.jsonl --output_file work/model_answer.csv
```
### テスト用データ
以下のコードを実行することでリーダーボードに投稿できる形式の解答ファイルを出力できます。
```bash
#実行例
$ python eval_model_jsonl_unlabel.py path/to/eval_file.jsonl --output_file work/model_answer.jsonl --save_model
```


__Accuracy__
- 第二回の開発データ1000問を予測した際の正解率 (Exact Match)

| データ     |  Acc |
| :--------- | ---: |
| 評価セット | 31.6 |

### 最終提出
- 最終提出の際はDockerイメージを提出する必要があります。その際、以下のコマンドで実行可能な推論スクリプト`submission.sh`を含む必要があります。
```bash
bash ./submission.sh <input_file> <output_file>
```
