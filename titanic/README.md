# Titanic

Survival prediction for [Titanic — Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic), evaluated by accuracy.
The official dataset contains 891 training passengers and 418 test passengers.

## Features and models

Features include sex, passenger class, age, fare, family counts and embarkation port, along with family size, traveling alone, fare per person, title, name length, cabin deck and ticket prefix.
Features are derived from each passenger's row. Missing numeric values are filled with `-1`; missing categorical values are filled with `Unknown`.

CatBoost depths 4, 5 and 6 were compared using five-fold stratified cross-validation (seed=42). Depth 6 achieved the highest accuracy and was selected.
Log loss breaks accuracy ties. Five-fold ticket-group cross-validation also evaluates the selected model, keeping passengers with the same ticket in the same fold.
Three models are fitted on the full training data with seeds 42, 137 and 2026. Their probabilities are averaged and classified at a threshold of 0.5.

A second comparison removed name length and evaluated CatBoost with stronger regularization, Random Forest and Extra Trees.
Random Forest and Extra Trees use one-hot encoding for categorical features, 400 trees, a maximum depth of 7 and a minimum of 3 samples per leaf.

## Results

| Model | Stratified CV | Ticket-group CV | Kaggle Public score |
| --- | --- | --- | --- |
| CatBoost depth 6 | 83.84% | 81.03% | **0.77033** |
| CatBoost (stronger regularization) | 83.50% | 81.03% | Not submitted |
| Random Forest | 83.84% | 81.48% | Not submitted |
| Extra Trees | 81.93% | 80.02% | Not submitted |

The CatBoost model was submitted on 2026-10-04.
Random Forest improved mean accuracy across the two CV schemes by 0.224 percentage points and ticket-group CV accuracy by 0.449 percentage points.
It did not meet the submission criteria of at least 0.2 percentage points improvement in mean CV and 0.5 percentage points in ticket-group CV, so its Kaggle score has not been measured.

Candidate selection reuses the same CV folds, which can make the best CV score optimistic.
Ticket-group CV is not an independent holdout, and the final three-seed ensemble has not been separately evaluated by cross-validation.

## Reproduction

From the repository root, install the dependencies and place the official data in `titanic/data/`:

```sh
pip install -r requirements.txt
python titanic/train.py --data-dir titanic/data --output-dir titanic/artifacts
python titanic/train_v2.py --data-dir titanic/data --output-dir titanic/artifacts/v2
```

Predictions are written to `submission.csv`. Validation results are saved as `metrics.json` for the first model and `metrics_v2.json` for the model comparison.
On Kaggle, add Titanic as an input and run [solution.ipynb](solution.ipynb) or [solution_v2.ipynb](solution_v2.ipynb) in a CPU Notebook.

- [CatBoost validation metrics](results/metrics.json)
- [Model comparison metrics](results/metrics_v2.json)
- [Submission score](results/submissions.json)
- [Kaggle environment](results/environment.txt)

![CatBoost submission score](results/kaggle-submission-v1.png)
