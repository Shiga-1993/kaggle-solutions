"""Build the standalone Sberbank housing price notebook."""

import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
folder = root / "sberbank-russian-housing-market"
source = (folder / "train.py").read_text()
explanation = """# Sberbank housing prices: chronological validation

Predict apartment sale prices using the official housing covariates and six date-aligned macroeconomic indicators. Fit the target as `log1p(price_doc)` and evaluate RMSLE.

1. Keep all transactions. Flag inconsistent areas and floors; mark nonpositive areas and implausible build years as missing. Add area ratios, calendar features and log areas. Exclude the transaction ID from the model.
2. Fit Ridge and CatBoost on transactions before July 2014. Compare their forecasts and a fixed equal blend in log space on July–December 2014.
3. Refit only the selected recipe on data through December 2014. Audit it once on January–June 2015 without further tuning.
4. Refit the chosen recipe on all training rows and predict the official test set in the submission template's order.

Ridge preprocessing is fitted only on the relevant training partition. CatBoost uses native categories, 1,000 trees, depth 6, learning rate 0.05 and seed 42. No external data, test labels, label-based row corrections or leaderboard-based price scaling are used.

The supplied macro series are joined by transaction date; their original publication vintages are not verified. The result is a retrospective competition forecast, not a claim of real-time forecasting accuracy. The later test period can have different economic conditions and transaction types.

Source, reproduction instructions and measured submission results: https://github.com/Shiga-1993/kaggle-solutions/tree/main/sberbank-russian-housing-market.

The source is available under the MIT License. The Kaggle-hosted copy is also published under Kaggle's Apache 2.0 license. Competition data remains subject to the competition rules. The official scores in the repository identify the submitted CSV by hash; a different runtime can generate different predictions.
"""
notebook = {
    "cells": [
        {"cell_type": "markdown", "id": "method", "metadata": {}, "source": explanation.splitlines(keepends=True)},
        {"cell_type": "code", "id": "training", "metadata": {}, "execution_count": None, "outputs": [],
         "source": (source.replace('if __name__ == "__main__":', "if False:") +
                    '\nmetrics = run("/kaggle/input/sberbank-russian-housing-market", "/kaggle/working")\n').splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
(folder / "solution.ipynb").write_text(json.dumps(notebook, indent=2) + "\n")
print("Created sberbank-russian-housing-market/solution.ipynb")
