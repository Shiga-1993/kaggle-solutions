# Kaggle solutions

小さなコンペから、検証・提出・振り返りを積み重ねるための記録です。

毎朝 **08:00 America/Chicago** に、1件の取り組みやすいタスクを進める日次運用を設定しました。
コード・検証・正式スコアを記録し、実際に進めた作業をコミットします。
運用の詳細は [Daily workflow](DAILY_WORKFLOW.md) にあります。

| Competition | Validation | Kaggle score | Status |
| --- | --- | --- | --- |
| [Titanic](https://www.kaggle.com/competitions/titanic) | 提出モデル: Stratified 83.84% / Ticket-group 81.03% | **0.77033 (77.03%)** | 1件提出・2回のモデル比較を記録済み |

初回の正式スコアは控えめでした。2回目のRandom Forestはチケット別CVが81.48%へ改善しましたが、事前の追加提出基準に届かず、提出を見送りました。
検証スコアと正式スコアを区別し、採用しなかった実験も残しています。詳細は [Titanicの記録](titanic/README.md) を参照してください。

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
- `titanic/train_v2.py`, `titanic/solution_v2.ipynb`: 2回目の比較実験
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
記録した正式スコアに対応する実行環境は [environment.txt](titanic/results/environment.txt) と [metrics.json](titanic/results/metrics.json) にあります。
依存パッケージの実際のバージョンは `metrics.json` に記録されます。Kaggle側の環境とローカルの固定バージョンが異なる場合は、完全に同じ予測にならないことがあります。

```sh
python scripts/build_notebook.py
```

提出ファイルは `submission.csv`、検証結果は `metrics.json` に出力されます。
提出前には418行・PassengerIdの一意性・列名・0/1予測をコード内で検証します。
2回目の実験を再現する場合は `python scripts/build_notebook.py --iteration 2` を実行し、`solution_v2.ipynb` を使います。

## 次の候補

Titanicを一区切りにした後、Spaceship Titanicなどの表形式コンペへ進みます。
それぞれで検証方法と評価指標を確認し、既存の結果を残して次へ進めます。
