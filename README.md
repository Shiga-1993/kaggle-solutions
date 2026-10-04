# Kaggle solutions

Solutions, code and validation results for Kaggle competitions.

| Competition | Model | Stratified CV | Ticket-group CV | Kaggle score |
| --- | --- | --- | --- | --- |
| [Titanic](titanic/README.md) | CatBoost | 83.84% | 81.03% | **0.77033** |

CV refers to cross-validation on the training data; Kaggle score is the Public score of a submitted prediction file.
Each competition's README describes the model comparisons and validation setup.

## Reproduction

Set up a local Python 3.12 environment:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download `train.csv` and `test.csv` from the [official Titanic data page](https://www.kaggle.com/competitions/titanic/data), place them in `titanic/data/`, and run:

```sh
python titanic/train.py --data-dir titanic/data --output-dir titanic/artifacts
```

The script writes predictions to `submission.csv` and validation results and package versions to `metrics.json`.
It checks that the submission has 418 rows, unique PassengerIds, the expected column names and binary predictions.

On Kaggle, add Titanic as an input, import [solution.ipynb](titanic/solution.ipynb) into a CPU Notebook, and run all cells.
See [environment.txt](titanic/results/environment.txt) for the environment used to produce the recorded Public score.
Local and Kaggle package versions differ, so predictions may not match exactly.

## Files

- [train.py](titanic/train.py): Titanic feature engineering, CatBoost cross-validation and predictions
- [train_v2.py](titanic/train_v2.py): Comparison of CatBoost, Random Forest and Extra Trees
- [results](titanic/results/): Validation metrics, package versions and submission scores
- [build_notebook.py](scripts/build_notebook.py): Generates notebooks from the training scripts

To rebuild both notebooks:

```sh
python scripts/build_notebook.py --iteration 2
```

Raw competition data and trained models are excluded from this repository.
