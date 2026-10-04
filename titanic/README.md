# Titanic

[Titanic — Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) の生存予測。評価指標はAccuracyです。
学習データ891人、テストデータ418人を使用しました。

## 特徴量とモデル

性別、客室等級、年齢、運賃、同乗家族数、乗船港に、家族人数、単独乗船、1人あたり運賃、敬称、名前の長さ、客室デッキ、チケット接頭辞などを追加しました。
特徴量は各乗客の行から作成し、数値の欠損は `-1`、カテゴリの欠損は `Unknown` で補完します。

CatBoostの深さ4・5・6を5分割Stratified CV（seed=42）で比較し、Accuracyが最大の深さ6を選択しました。
同点の場合はLoglossで比較します。同一チケットの乗客を同じfoldにまとめた5分割CVも計算しました。
全学習データでseed 42・137・2026の3モデルを学習し、予測確率を平均してしきい値0.5で分類します。

追加比較では名前の長さを除き、正則化を強めたCatBoost、Random Forest、Extra Treesを評価しました。
Random ForestとExtra TreesにはカテゴリのOne-hot encodingを適用し、木の数400、最大深さ7、葉の最小サンプル数3を使用しています。

## 結果

| Model | Stratified CV | Ticket-group CV | Kaggle Public score |
| --- | --- | --- | --- |
| CatBoost depth 6 | 83.84% | 81.03% | **0.77033** |
| CatBoost（正則化強化） | 83.50% | 81.03% | 未提出 |
| Random Forest | 83.84% | 81.48% | 未提出 |
| Extra Trees | 81.93% | 80.02% | 未提出 |

正式提出は2026-10-04のCatBoostモデルです。
Random Forestは2種類のCVの平均が0.224ポイント、Ticket-group CVが0.449ポイント改善しました。
追加提出の条件（平均CVで0.2ポイント以上、Ticket-group CVで0.5ポイント以上の改善）を満たさず、Kaggleスコアは未測定です。

モデル選択に同じCVを使用しているため、最良CV値には選択による楽観性があります。
Ticket-group CVは独立したholdoutではなく、最終的な3-seed平均モデルは別途CV評価していません。

## 再現

リポジトリ直下で依存パッケージをインストールし、公式データを `titanic/data/` に配置します。

```sh
pip install -r requirements.txt
python titanic/train.py --data-dir titanic/data --output-dir titanic/artifacts
python titanic/train_v2.py --data-dir titanic/data --output-dir titanic/artifacts/v2
```

予測は `submission.csv`、検証結果は初回が `metrics.json`、モデル比較が `metrics_v2.json` に出力されます。
KaggleではTitanicをInputに追加し、[solution.ipynb](solution.ipynb) または [solution_v2.ipynb](solution_v2.ipynb) をCPU Notebookで実行します。

- [CatBoostの検証指標](results/metrics.json)
- [モデル比較の検証指標](results/metrics_v2.json)
- [正式提出スコア](results/submissions.json)
- [Kaggle実行環境](results/environment.txt)

![CatBoostの正式提出スコア](results/kaggle-submission-v1.png)
