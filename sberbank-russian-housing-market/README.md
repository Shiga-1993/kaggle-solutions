# Sberbank Russian Housing Market

Predict apartment sale prices in Russia using housing attributes, location information and macroeconomic indicators. The [competition](https://www.kaggle.com/competitions/sberbank-russian-housing-market) evaluates Root Mean Squared Logarithmic Error (RMSLE); lower is better.

## Data

Use `train.csv`, `test.csv`, `macro.csv`, `sample_submission.csv` and `data_dictionary.txt` from the [official data page](https://www.kaggle.com/competitions/sberbank-russian-housing-market/data). The 30,471 labeled transactions span August 2011–June 2015; the 7,662 test transactions span July 2015–May 2016. The macro file has 2,484 dated observations. File hashes are recorded in [results/metrics.json](results/metrics.json).

Only official training labels are used. No external data, test answers, label corrections or target-based row removal are used. All training transactions are retained. Raw data is excluded from this repository and remains subject to the [competition rules](https://www.kaggle.com/competitions/sberbank-russian-housing-market/rules).

## Method

1. Fit the target as `log1p(price_doc)`. Use all official housing covariates except transaction ID, timestamp text and target. Treat text fields, nearest-location IDs, material and property condition as categories.
2. Mark nonpositive areas, living/kitchen areas larger than total area and implausible construction years as missing, preserving inconsistency flags. Add area ratios, area per room, building age, relative floor, log areas and transaction calendar features. Flag floors above the reported building height. These are fixed checks on each row, without target information.
3. Join six official macro indicators by exact transaction date: oil price, consumer price index, USD/RUB, EUR/RUB, mortgage rate and unemployment. This produces 315 features, including 24 categorical fields.
4. Train on transactions before July 2014 and select the lowest July–December 2014 RMSLE among Ridge, CatBoost and a fixed equal blend in log space.
5. Refit only the selected recipe on transactions through December 2014 and audit it once on January–June 2015. Refit the same recipe on all training rows for the final test predictions. Preserve the official template's ID order and verify finite positive prices, unique IDs, row count and columns.

Ridge uses training-fitted median imputation, missing indicators, standard scaling and one-hot categories with `min_frequency=10`. Its regularization is `alpha=10`. CatBoost uses 1,000 trees, depth 6, learning rate 0.05, L2 regularization 6 and seed 42, with two CPU threads. Both models' log predictions are bounded by the target range of their current fitting partition. There is no parameter search or early stopping.

## Measured results

### Local model selection: training before July 2014 → July–December 2014

There are 20,483 fitting rows and 6,749 selection rows.

| Model | RMSLE ↓ | Median absolute percentage error |
| --- | ---: | ---: |
| Training-mean log-price baseline | 0.60721 | 27.88% |
| Ridge | 0.45805 | 13.83% |
| CatBoost | **0.43490** | 16.33% |
| Equal Ridge/CatBoost blend | 0.43813 | 14.23% |

CatBoost was selected on RMSLE before evaluating 2015 labels or a new leaderboard score. Ridge's lower median percentage error does not make it better on the competition metric, which is more sensitive to large log errors.

### Independent local audit: training through 2014 → January–June 2015

There are 27,232 fitting rows and 3,239 audit rows.

| Model | RMSLE ↓ | Median absolute percentage error |
| --- | ---: | ---: |
| Training-mean log-price baseline | 0.60238 | 31.21% |
| Selected CatBoost | **0.39340** | **13.05%** |

CatBoost improves audit RMSLE by about 34.7% over the baseline. Its mean signed log error is +0.01961. Investment transactions remain harder than owner-occupied transactions:

| Transaction type | Audit rows | RMSLE ↓ | Median absolute percentage error |
| --- | ---: | ---: | ---: |
| Investment | 1,755 | 0.50701 | 17.13% |
| OwnerOccupier | 1,484 | 0.18382 | 7.70% |

These groups differ in their price and property distributions; the comparison is descriptive. Monthly audit results and predictive feature importance are included in the full metrics file.

### Official Kaggle submission

| Date | Status | Public RMSLE ↓ | Private RMSLE ↓ |
| --- | --- | ---: | ---: |
| 2026-10-07 | Complete (after deadline) | **0.33140** | **0.32980** |

One prediction file from Kaggle notebook **version 1**, run `356113165`, was submitted to this competition, which ended in 2017. Both official scores were displayed by Kaggle. No model changes or additional submissions followed these scores.

Submitted prediction SHA-256: `e6e9936f14a04c89bb5b0b4b413c0a87fc81afccfe15563cffb11a5a6458970f`.

The Kaggle run selected CatBoost with 2014-H2 RMSLE **0.43480** and reproduced the 2015-H1 audit RMSLE **0.39340**. The local final CSV has a different hash, `9a51753cd049c7245f346595d7d47442cce7b0b9b4da05df64fc05c4e77c19da`, and was not submitted. Package and runtime differences can change predictions; the official scores belong to the Kaggle output hash above.

See [local metrics](results/metrics.json), [Kaggle runtime results](results/kaggle-runtime.json), [submission record](results/submissions.json) and [official score evidence](results/kaggle-submission-v1.png).

## Validation limitations

The 2014-H2 scores are used for model selection and are optimistic for that choice. The 2015-H1 audit is one later-period split, rather than repeated cross-validation; repeated properties may cross time boundaries. The final fit includes audit labels after this evaluation.

The official test period is later and can have different economic conditions and transaction types. A lower leaderboard score than audit RMSLE does not establish a higher rank or identical task difficulty. No leaderboard-based price scaling is applied.

The provided macro series are joined by date, but their original publication vintages are not verified. This is a retrospective competition forecast, not an assessment of accuracy using only information available in real time. Feature importance measures predictive associations, not causal price effects.

## Reproduction

From the repository root, create a Python 3.12 environment and install the pinned dependencies:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the five official files after joining the competition and place them in `sberbank-russian-housing-market/data/`. Either extracted files or the corresponding individual `.zip` archives are supported.

```sh
python sberbank-russian-housing-market/train.py --data-dir sberbank-russian-housing-market/data --output-dir sberbank-russian-housing-market/artifacts
python scripts/build_sberbank_notebook.py
```

The script writes `submission.csv` and `metrics.json` to the ignored output directory. The measured local run used Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.7.2 and CatBoost 1.2.8. The model comparison, audit and final fit took about 118 seconds, excluding data download.

The [standalone notebook](solution.ipynb) contains the same training code and runs on CPU with the official competition input attached. Kaggle version 1 completed successfully in 314.6 seconds, including startup. It used Python 3.13.15, NumPy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1 and CatBoost 1.2.10. Reproduction does not guarantee byte-identical predictions across environments.

The repository source is released under the [MIT License](LICENSE), as specified by the competition's public code-sharing rules. These source licenses do not redistribute competition data.
