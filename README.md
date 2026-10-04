# Kaggle solutions

小さなコンペから、検証・提出・振り返りを積み重ねるための記録です。

| Competition | Validation | Kaggle score | Status |
| --- | --- | --- | --- |
| [Titanic](https://www.kaggle.com/competitions/titanic) | 実行中 | 未提出 | 初回モデルを検証中 |

## 進め方

1. 公式の説明・評価指標・データを確認する。
2. 簡単なベースラインと、事前に決めた検証分割を用意する。
3. 特徴量やモデルを少数比較し、検証結果から提出するモデルを選ぶ。
4. Kaggleへ提出し、実際のスコア・提出日時・予測ファイルのハッシュを記録する。
5. コード、再現手順、検証結果、改善点をこのリポジトリに保存する。

提出スコアを見て同じテストセットへ繰り返し合わせ込まず、改善はまずローカル検証で判断します。
学習データ、モデル本体、APIキーはGitに含めません。

## 構成

- `titanic/train.py`: 特徴量作成・交差検証・学習・提出ファイル作成
- `titanic/solution.ipynb`: 同じPythonソースから生成したKaggle用Notebook
- `titanic/results/`: 検証指標と提出履歴
- `scripts/build_notebook.py`: Notebookの再生成

## 再現

Python 3.12を使用します。

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Kaggleの公式[Dataページ](https://www.kaggle.com/competitions/titanic/data)から `train.csv` と `test.csv` を取得し、`titanic/data/` に置きます。
自分で認証済みのKaggle CLIを用意している場合は次でも取得できます。

```sh
kaggle competitions download -c titanic -p titanic/data
unzip titanic/data/titanic.zip -d titanic/data
python titanic/train.py --data-dir titanic/data --output-dir titanic/artifacts
```

Kaggleで実行する場合は、TitanicをInputに追加したCPU Notebookに `titanic/solution.ipynb` をImportし、Save & Run Allを実行します。
依存パッケージの実際のバージョンは `metrics.json` に記録されます。Kaggle側の環境とローカルの固定バージョンが異なる場合は、完全に同じ予測にならないことがあります。

```sh
python scripts/build_notebook.py
```

提出ファイルは `submission.csv`、検証結果は `metrics.json` に出力されます。
提出前には418行・PassengerIdの一意性・列名・0/1予測をコード内で検証します。

## 次の候補

Titanicを一区切りにした後、Spaceship Titanicなどの表形式コンペへ進みます。
それぞれで検証方法と評価指標を確認し、既存の結果を残して次へ進めます。
