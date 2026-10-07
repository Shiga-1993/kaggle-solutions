# Kaggle solutions

Solutions, code and validation results for Kaggle competitions.

| Competition | Model | Metric | Local validation | Additional validation | Kaggle score |
| --- | --- | --- | --- | --- | --- |
| [Titanic](titanic/README.md) | CatBoost | Accuracy ↑ | 83.84% | Ticket-group CV: 81.03% | **0.77033** |
| [Titanic](titanic/README.md) | Random Forest, compact features, three-seed ensemble | Accuracy ↑ | 84.18% | Ticket-group CV: 82.49% | 0.76315 |
| [House Prices](house-prices/README.md) | 50/50 Ridge + CatBoost | Log RMSE ↓ | 0.12544 | Sale-year diagnostic: 0.12699 | **0.11820** |
| [Rental Listing Inquiries](rental-listing-inquiries/README.md) | CatBoost | Log Loss ↓ | April → May: 0.57453 | April–May → June: 0.54584 | Public **0.54681**, Private **0.54702** |
| [Sberbank Russian Housing Market](sberbank-russian-housing-market/README.md) | CatBoost | RMSLE ↓ | 2014-H2: 0.43490 | 2015-H1: 0.39340 | Public **0.33140**, Private **0.32980** |

CV refers to cross-validation on the training data; Kaggle score is the official score of a submitted prediction file.
Higher accuracy is better; lower log RMSE, RMSLE and Log Loss are better. Scores from different competition metrics are not directly comparable.
Each competition's README describes the model comparisons and validation setup.
The latest Titanic experiment improved CV accuracy but reduced the Public score; the original CatBoost submission remains the best measured submission.
The CatBoost row contains the original single-model-per-fold CV, while the compact Random Forest row evaluates the three-seed ensemble used for its submission.

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
- [train_v3.py](titanic/train_v3.py): Compact passenger features, fold-fitted preprocessing and ensemble validation
- [results](titanic/results/): Validation metrics, package versions and submission scores
- [build_notebook.py](scripts/build_notebook.py): Generates notebooks from the training scripts

To rebuild all three notebooks:

```sh
python scripts/build_notebook.py --iteration 3
```

Raw competition data and trained models are excluded from this repository.

## House Prices reproduction

Download the four official files from the [House Prices data page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) into `house-prices/data/`, then run:

```sh
python house-prices/train.py --data-dir house-prices/data --output-dir house-prices/artifacts
python scripts/build_house_prices_notebook.py
```

See the [House Prices solution](house-prices/README.md) for feature construction, model settings, validation limitations and the measured score.
The [standalone notebook](house-prices/solution.ipynb) runs on CPU with the official competition input.

## Rental Listing Inquiries reproduction

Download the three official files from the [Rental Listing Inquiries data page](https://www.kaggle.com/competitions/two-sigma-connect-rental-listing-inquiries/data) into `rental-listing-inquiries/data/`, then run:

```sh
python rental-listing-inquiries/train.py --data-dir rental-listing-inquiries/data --output-dir rental-listing-inquiries/artifacts
python scripts/build_rental_listing_notebook.py
```

See the [Rental Listing Inquiries solution](rental-listing-inquiries/README.md) for the month-separated validation, model comparison, limitations and measured late-submission scores. The [standalone notebook](rental-listing-inquiries/solution.ipynb) contains the same training code and runs on CPU.

## Sberbank Russian Housing Market reproduction

Download the five official files from the [Sberbank data page](https://www.kaggle.com/competitions/sberbank-russian-housing-market/data) into `sberbank-russian-housing-market/data/`, then run:

```sh
python sberbank-russian-housing-market/train.py --data-dir sberbank-russian-housing-market/data --output-dir sberbank-russian-housing-market/artifacts
python scripts/build_sberbank_notebook.py
```

See the [Sberbank solution](sberbank-russian-housing-market/README.md) for chronological model selection, the later-period audit, transaction-type diagnostics and measured late-submission scores. The [standalone notebook](sberbank-russian-housing-market/solution.ipynb) contains the same training code and runs on CPU.
