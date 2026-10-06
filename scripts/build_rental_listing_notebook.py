"""Build a standalone CPU notebook for rental listing interest prediction."""

import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / "rental-listing-inquiries/train.py").read_text()
notebook = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# Rental listing inquiries: temporal validation\n",
            "Predict high, medium and low interest in NYC rental listings using only official competition training labels.\n",
            "Compare a training-class-prior baseline, Logistic Regression, CatBoost and a fixed equal blend.\n",
            "Fit candidate models on April 2016, select on May log loss, then audit only the selected recipe on June.\n",
            "Use row-level rent, room, location, listing-content and amenity features. Fit category encoding, imputation, scaling and TF-IDF only on the relevant training partition.\n",
            "After the audit, refit the selected recipe on all training months and produce probabilities in the official submission template order.\n",
            "No external data, test labels, manual test annotations or downloaded photos are used.\n",
            "Source, reproduction instructions and measured submission results: https://github.com/Shiga-1993/kaggle-solutions/tree/main/rental-listing-inquiries.\n",
            "The submitted CSV is generated with the locally recorded package versions; a different runtime can produce different predictions.\n",
            "The source code is shared under the MIT License, as specified by the competition's code-sharing rules.\n"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": [source.replace('if __name__ == "__main__":', "if False:") +
                    '\nmetrics = run("/kaggle/input/two-sigma-connect-rental-listing-inquiries", "/kaggle/working")\n']},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
path = root / "rental-listing-inquiries/solution.ipynb"
path.write_text(json.dumps(notebook, indent=2) + "\n")
print("Created rental-listing-inquiries/solution.ipynb")
