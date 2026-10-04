# Titanic

公式競技: https://www.kaggle.com/competitions/titanic

## 初回アプローチ

- 891人の公式学習データのみを使用。418人のテストデータの正解は利用しない。
- 性別、客室等級、年齢、運賃に、家族人数、単独乗船、1人あたり運賃、敬称、客室デッキ、チケット接頭辞を追加。
- CatBoostの深さ4・5・6を、共通の5分割Stratified CV（seed=42）で比較。
- 正解率、次にLoglossの順でモデルを選択。しきい値は事前に0.5と固定。
- 同一チケットが学習・検証へまたがらない5分割CVも、診断として実行。
- 選んだ設定を全学習データで3つのseedにより学習し、確率を平均して提出。

外部の乗客記録、テスト正解、目的変数から作る家族・チケット特徴量は使用しません。
すべての特徴量はその乗客の行だけから作成し、欠損は固定値で処理します。

## 検証の限界

3設定の選択に同じCVを使用しているため、その最良CV値には選択による楽観性があります。
チケット別CVは独立した未使用holdoutではありません。
全データで再学習する3-seed平均モデルは、別途CV評価していません。
CVとKaggleの評価データには差があるため、両スコアを区別して記録します。

## 成果物

実行結果と正式提出スコアは `results/` に保存します。
生データと提出CSVは `.gitignore` で除外します。

## 初回結果

深さ6が選択され、通常CVは83.84%、チケット別CVは81.03%でした。
2026-10-04の正式提出は **0.77033 (77.03%)** でした。
正式スコアはCVより低く、初回の改善幅は控えめです。

## 2回目の比較

初回の差を受け、名前の長さを特徴量から除き、正則化を強めたCatBoost、Random Forest、Extra Treesを比較します。
共通のStratified CVとTicket-group CVの正解率を平均して選びます。
初回から平均CVが0.002以上、Ticket-group CVが0.005以上改善した場合だけ追加提出します。
この比較ルールは2回目の実行と正式スコアを見る前に固定しています。

```sh
python scripts/build_notebook.py --iteration 2
python titanic/train_v2.py --data-dir titanic/data --output-dir titanic/artifacts/v2
```

Kaggleでは `solution_v2.ipynb` をImportして実行します。

## 比較結果

| 実験 | 通常CV | チケット別CV | 正式Kaggleスコア |
| --- | --- | --- | --- |
| 初回 CatBoost depth6 | 83.84% | 81.03% | **77.03%** |
| 2回目 正則化CatBoost | 83.50% | 81.03% | 未提出 |
| 2回目 Random Forest | 83.84% | 81.48% | 未提出 |
| 2回目 Extra Trees | 81.93% | 80.02% | 未提出 |

Random Forestの平均CVは初回から0.224ポイント改善し、チケット別CVは0.449ポイント改善しました。
チケット別CVの改善が事前基準の0.5ポイントに届かなかったため、追加提出は行いませんでした。
このモデルのKaggleスコアは未測定です。

提出したモデルでは性別だけの学習データベースライン78.68%を通常CVで上回りましたが、正式スコアは77.03%でした。
通常CVの楽観性と、評価データとの差を今後の課題として残します。

- [初回Kaggle Notebook Version 2](https://www.kaggle.com/code/mvfrsshiga/titanic-validated-catboost-baseline?scriptVersionId=355219298)
- [2回目Kaggle Notebook Version 3](https://www.kaggle.com/code/mvfrsshiga/titanic-validated-catboost-baseline?scriptVersionId=355221248)
- [正式提出履歴](results/submissions.json)
- [実行履歴と失敗・修正](results/run_history.json)
- [初回検証指標](results/metrics.json)
- [2回目検証指標](results/metrics_v2.json)

![初回の正式提出スコア](results/kaggle-submission-v1.png)
