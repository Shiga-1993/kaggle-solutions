"""Build the standalone California Housing notebook."""

import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
folder = root / "california-housing"
source = (folder / "train.py").read_text()
explanation = """# California housing values: random audit and spatial validation

Predict `MedHouseVal` using only the official synthetic Playground Season 3, Episode 1 data. Evaluate RMSE on the untransformed target.

1. Reserve 20% of training rows as an untouched random audit (seed 42). Split the remaining rows into fitting and selection partitions (seed 43), giving approximately 60/20/20 proportions.
2. Compare Ridge, CatBoost on the eight original covariates, CatBoost with fixed room/population ratios and log features, and an equal blend of the two CatBoost variants. Select the lowest selection RMSE.
3. Refit the chosen fixed recipe on the development partition and audit it once. Run a separate fixed half-degree spatial-cell holdout as a regional-transfer diagnostic; do not tune from either result.
4. Fit all official training labels and predict the test set in the official template's ID order.

CatBoost uses 1,200 trees, depth 6, learning rate 0.05, L2 regularization 6, seed 42 and two CPU threads. IDs are excluded. No original-dataset labels, external data, test labels, target encoding or leaderboard-based prediction scaling are used. Ridge preprocessing is fitted only on its fitting partition. Predictions are bounded by that partition's observed target range.

Random validation allows recurring coordinates and measures interpolation within the synthetic competition distribution. The spatial split uses disjoint half-degree coordinate cells and measures a different task; its fitting rows may include random-audit labels. Synthetic targets do not represent current property prices.

Source, reproduction instructions and measured results: https://github.com/Shiga-1993/kaggle-solutions/tree/main/california-housing.

Competition source: Walter Reade and Ashley Chow. Regression with a Tabular California Housing Dataset. Kaggle, 2023. https://www.kaggle.com/competitions/playground-series-s3e1.

The repository source is available under the MIT License. The Kaggle-hosted copy also carries Kaggle's Apache 2.0 license. Competition data is excluded from the repository and remains subject to the competition rules. The submitted CSV is identified by hash in the repository; runtime differences can change predictions.
"""
notebook = {
    "cells": [
        {"cell_type": "markdown", "id": "method", "metadata": {}, "source": explanation.splitlines(keepends=True)},
        {"cell_type": "code", "id": "training", "metadata": {}, "execution_count": None, "outputs": [],
         "source": (source.replace('if __name__ == "__main__":', "if False:") +
                    '\nmetrics = run("/kaggle/input/playground-series-s3e1", "/kaggle/working")\n').splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
(folder / "solution.ipynb").write_text(json.dumps(notebook, indent=2) + "\n")
print("Created california-housing/solution.ipynb")
