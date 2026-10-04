# Kaggle solutions

Kaggleコンペの解法、実装、検証結果をまとめたリポジトリです。

| Competition | Model | Stratified CV | Ticket-group CV | Kaggle score |
| --- | --- | --- | --- | --- |
| [Titanic](titanic/README.md) | CatBoost | 83.84% | 81.03% | **0.77033** |

CVは学習データの交差検証、Kaggle scoreは正式提出のPublic scoreです。
モデルの比較結果と検証条件は各コンペのREADMEに記載しています。

## 再現

Python 3.12のローカル環境:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

[Titanicの公式データ](https://www.kaggle.com/competitions/titanic/data)の `train.csv` と `test.csv` を `titanic/data/` に配置し、実行します。

```sh
python titanic/train.py --data-dir titanic/data --output-dir titanic/artifacts
```

`submission.csv` に予測、`metrics.json` に検証結果と実行環境を出力します。
提出ファイルは418行、PassengerIdの一意性、列名、0/1予測を検証します。

Kaggle上ではTitanicをInputに追加し、[solution.ipynb](titanic/solution.ipynb) をCPU NotebookにImportして実行します。
正式スコアを記録した実行環境は [environment.txt](titanic/results/environment.txt) を参照してください。
ローカルとKaggleで依存パッケージのバージョンが異なるため、予測が完全に一致しない場合があります。

## ファイル

- [train.py](titanic/train.py): Titanicの特徴量作成、CatBoostの交差検証、予測
- [train_v2.py](titanic/train_v2.py): CatBoost、Random Forest、Extra Treesの比較
- [results](titanic/results/): 検証指標、実行環境、正式提出スコア
- [build_notebook.py](scripts/build_notebook.py): 学習スクリプトからNotebookを生成

```sh
python scripts/build_notebook.py --iteration 2
```

生データと学習済みモデルはリポジトリに含めていません。
