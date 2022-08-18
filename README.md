# AIO3_GPT_baseline

- [AI王 〜クイズAI日本一決定戦〜](https://www.nlp.ecei.tohoku.ac.jp/projects/aio/)
- 昨年度の概要は [こちら](https://sites.google.com/view/project-aio/competition2)


## 目次

- [AIO3_GPT_baseline](#aio3_gpt_baseline)
  - [目次](#目次)
  - [環境構築](#環境構築)
  - [データセット](#データセット)
    - [開発用データ](#開発用データ)
  - [日本語GPTモデルによるzero-shot推論](#日本語gptモデルによるzero-shot推論)

## 環境構築
- cuda バージョンに合わせて、以下より torch をインストールして下さい。
  - [https://pytorch.org](https://pytorch.org)

- その他のライブラリについては以下のコマンドを実行してインストールして下さい。

```bash
$ pip install -r requirements.txt
```

- 以下のコマンドでdataディレクトリとworkディレクトリ、modelsディレクトリを作成してください。
```bash
$ mkdir data
$ mkdir work
$ mkdir models
```
## データセット

- 訓練データには、クイズ大会[「abc/EQIDEN」](http://abc-dive.com/questions/) の過去問題に対して Wikipedia の記事段落の付与を自動で行ったものを使用しています。
- 開発・評価用クイズ問題には、[株式会社キュービック](http://www.qbik.co.jp/) および [クイズ法人カプリティオ](http://capriccio.tokyo/) へ依頼して作成されたものを使用しています。

- 以上のデータセットの詳細については、[AI王 〜クイズAI日本一決定戦〜](https://www.nlp.ecei.tohoku.ac.jp/projects/aio/)の公式サイト、および下記論文をご覧下さい。

> __JAQKET: クイズを題材にした日本語QAデータセット__
> - https://www.nlp.ecei.tohoku.ac.jp/projects/jaqket/
> - 鈴木正敏, 鈴木潤, 松田耕史, ⻄田京介, 井之上直也. JAQKET:クイズを題材にした日本語QAデータセットの構築. 言語処理学会第26回年次大会(NLP2020) [\[PDF\]](https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P2-24.pdf)

### 開発用データ

主に以下に示した要素からなるjson lines形式のファイル。
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
## 日本語GPTモデルによるzero-shot推論
以下のコードを実行することでrinna株式会社の[日本語GPTモデル](https://huggingface.co/rinna/japanese-gpt-1b)によるzero-shot推論を行うことができる。
```bash
#実行例
$ python eval_model_jsonl.py path/to/eval_file.jsonl --output_file work/model_answer.csv
```

__Accuracy__
- 第二回の開発データ1000問を予測した際の正解率 (Exact Match)

| データ     |  Acc |
| :--------- | ---: |
| 評価セット | 31.6 |
