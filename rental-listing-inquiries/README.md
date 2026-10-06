# Two Sigma Connect: Rental Listing Inquiries

Predict whether a New York City rental listing receives **high**, **medium** or **low** inquiry interest. The [competition](https://www.kaggle.com/competitions/two-sigma-connect-rental-listing-inquiries) evaluates multiclass Log Loss; lower is better.

## Data

Use only `train.json`, `test.json` and `sample_submission.csv` from the [official competition data page](https://www.kaggle.com/competitions/two-sigma-connect-rental-listing-inquiries/data). There are 49,352 labeled listings and 74,659 test listings, all from April–June 2016. The training labels contain 34,284 low, 11,229 medium and 3,839 high-interest listings.

The input file hashes are recorded in [results/metrics.json](results/metrics.json). No external data, test labels, manually annotated test rows or downloaded photographs are used. Raw competition data is excluded from this repository and remains subject to the [competition rules](https://www.kaggle.com/competitions/two-sigma-connect-rental-listing-inquiries/rules).

## Method

1. Build row-level features from log rent, rent per room, bedroom/bathroom counts, latitude/longitude, studio status, photo and amenity counts, description length and creation time. Normalize manager ID, building ID and display address as categorical features. Add nine amenity indicators, including no fee, doorman, laundry and pets.
2. Fit candidate models on **April** (16,411 rows) and select the lowest **May** Log Loss (15,797 rows). Compare Logistic Regression, CatBoost and a fixed equal probability blend, alongside a training-class-prior baseline.
3. Refit only the selected recipe on April–May (32,208 rows) and evaluate it once on the untouched **June** partition (17,144 rows). Do not adjust features, weights or parameters from this result.
4. Refit the selected recipe on all labeled listings and output `listing_id,high,medium,low`, preserving the official template's row order. Verify finite probabilities, row sums, unique IDs, row count and column order before submission.

Logistic Regression uses training-fitted median imputation and standardization, one-hot categories (`min_frequency=5`) and amenity TF-IDF (up to 1,500 unigram/bigram features). CatBoost uses native categorical features and amenity indicators. Its fixed settings are 800 trees, depth 6, learning rate 0.05, L2 regularization 6 and seed 42. There is no hyperparameter search or early stopping.

`listing_id` is used only to align predictions; it is not a model feature. The full description and street-address text are not tokenized. CatBoost's categorical statistics are learned from the relevant training partition.

## Measured results

### Model selection: April → May

| Model | Log Loss ↓ | Accuracy |
| --- | ---: | ---: |
| Training-class-prior baseline | 0.79250 | 69.15% |
| Logistic Regression | 0.63602 | 71.10% |
| CatBoost | **0.57453** | **74.14%** |
| Equal Logistic Regression/CatBoost blend | 0.57929 | 73.59% |

CatBoost was selected before inspecting June labels or a new leaderboard score.

### Independent audit: April–May → June

| Model | Log Loss ↓ | Accuracy |
| --- | ---: | ---: |
| Training-class-prior baseline | 0.78257 | 70.00% |
| Selected CatBoost | **0.54584** | **75.58%** |

The June Log Loss improves by about 30.3% over the class-prior baseline. Per-class negative log probability is 1.60568 for high, 1.08593 for medium and 0.25674 for low interest. The rare high-interest class remains the hardest.

### Official Kaggle submission

| Date | Status | Public Log Loss ↓ | Private Log Loss ↓ |
| --- | --- | ---: | ---: |
| 2026-10-06 | Complete (after deadline) | **0.54681** | **0.54702** |

This is a late practice submission to a competition that ended in 2017. Both scores above were displayed by Kaggle; they are distinct from local validation results. One locally selected prediction file was submitted.

Prediction SHA-256: `8c1e700ca65601e7fbe44cf17a353723d4935de3f4ad99c01b470be80208b3c5`.

See [full validation metrics](results/metrics.json), [submission record](results/submissions.json) and [official score evidence](results/kaggle-submission-v1.png).

## Validation limitations

May results are used for model selection and are optimistic for that choice. June provides one later-month audit, not repeated cross-validation. Managers and buildings can appear in several months; this is not an evaluation restricted to entirely new managers or properties.

June Log Loss is 0.55342 for known managers (15,435 rows) and 0.47740 for unseen or missing managers (1,709 rows). It is 0.59001 for known buildings (12,047 rows) and 0.44143 for unseen or missing buildings (5,097 rows). These groups differ in difficulty; their loss values do not establish an effect of category familiarity.

Test listings span the same April–June period, so a forward June split is not an exact leaderboard proxy. The final model uses June labels after the independent audit. Feature importance and probabilities describe predictive associations, not causal effects on demand. No tuning follows the official scores.

## Reproduction

From the repository root, create a Python 3.12 environment and install the pinned dependencies:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the three official files after joining the competition and place them in `rental-listing-inquiries/data/`. Either extracted files or the original `train.json.zip`, `test.json.zip` and `sample_submission.csv.zip` archives are supported.

```sh
python rental-listing-inquiries/train.py --data-dir rental-listing-inquiries/data --output-dir rental-listing-inquiries/artifacts
python scripts/build_rental_listing_notebook.py
```

The script saves `submission.csv` and `metrics.json` in the ignored output directory. The measured local run used Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.7.2 and CatBoost 1.2.8, with two CPU threads. Its model comparison, audit and final fit took about 81 seconds, excluding data download.

The [standalone notebook](solution.ipynb) runs on CPU with the official competition input attached. The [public Kaggle notebook](https://www.kaggle.com/code/mvfrsshiga/rental-listing-inquiries-temporal-catboost), version 1, completed successfully on CPU in 186.7 seconds and reproduced the CatBoost May and June Log Loss values. Its output CSV has a different hash (`275d761a1b94874295175cce4c5f8cf0ae5bbbd4262734a6894b5b6c1e813c87`) and was not submitted. A different package/runtime environment can produce different predictions; the official scores above belong to the CSV identified by its hash.

The repository source is released under the [MIT License](LICENSE), as specified by the competition's public code-sharing rules. The Kaggle-hosted copy is also published under Kaggle's Apache 2.0 license. These source licenses do not redistribute competition data.
