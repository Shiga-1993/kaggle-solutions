"""Build a standalone Kaggle notebook from the House Prices training script."""

import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / "house-prices/train.py").read_text()
notebook = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# House Prices: log-price regression\n",
            "Predict Ames house sale prices using official training labels only.\n",
            "Use row-level area, age and amenity features, with preprocessing fitted inside each training fold.\n",
            "Compare Ridge, Elastic Net, CatBoost and an equal Ridge/CatBoost blend using fixed five-fold CV.\n",
            "Check a sale-year diagnostic and average the selected fold models' log-price predictions.\n",
            "The reproducible source and measured submission results are available at https://github.com/Shiga-1993/kaggle-solutions/tree/main/house-prices.\n",
            "The submitted file is generated with the package versions recorded in the repository; a different runtime can produce different predictions.\n"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": [source.replace('if __name__ == "__main__":', "if False:") +
                    '\nmetrics = run("/kaggle/input/house-prices-advanced-regression-techniques", "/kaggle/working")\n']},
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
path = root / "house-prices/solution.ipynb"
path.write_text(json.dumps(notebook, indent=2) + "\n")
print("Created house-prices/solution.ipynb")
