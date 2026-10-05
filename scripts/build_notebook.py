"""Build Kaggle notebooks from the Titanic training scripts."""
import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--iteration", type=int, choices=[1, 2, 3], default=1)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
source = (root / "titanic/train.py").read_text()
notebook = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# Titanic: CatBoost\n",
            "Passenger features include family size, titles, cabin decks and ticket prefixes.\n",
            "Compare depths 4, 5 and 6 using 5-fold stratified CV and evaluate ticket-group CV.\n",
            "Average predictions from three random seeds and classify at a probability threshold of 0.5.\n"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": [source.replace("if __name__ == \"__main__\":", "if False:") +
                    '\nmetrics = run("/kaggle/input/titanic", "/kaggle/working")\n']},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
(root / "titanic/solution.ipynb").write_text(json.dumps(notebook, indent=2) + "\n")
print("Created titanic/solution.ipynb")
if args.iteration >= 2:
    v2_source = (root / "titanic/train_v2.py").read_text()
    notebook["cells"] = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# Titanic: tree model comparison\n",
            "Compare regularized CatBoost, Random Forest and Extra Trees.\n",
            "Select by mean accuracy across stratified and ticket-group cross-validation.\n"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": ['from pathlib import Path\nPath("/kaggle/working/train.py").write_text(' + repr(source) + ')\n']},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": [v2_source.replace('if __name__ == "__main__":', "if False:") +
                    '\nmetrics = run_v2("/kaggle/input/titanic", "/kaggle/working")\n']},
    ]
    (root / "titanic/solution_v2.ipynb").write_text(json.dumps(notebook, indent=2)+"\n")
    print("Created titanic/solution_v2.ipynb")
if args.iteration >= 3:
    v3_source = (root / "titanic/train_v3.py").read_text()
    notebook["cells"] = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# Titanic: compact features and ensemble validation\n",
            "Normalize passenger titles and fit imputation, scaling and categorical encoding inside each fold.\n",
            "Compare Logistic Regression, shallow Gradient Boosting and Random Forest with fixed parameters.\n",
            "Validate the selected three-seed ensemble against the original CatBoost ensemble using both stratified and ticket-group CV.\n"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": ['from pathlib import Path\nPath("/kaggle/working/train.py").write_text(' + repr(source) + ')\n']},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": [v3_source.replace('if __name__ == "__main__":', "if False:") +
                    '\nmetrics = run_v3("/kaggle/input/titanic", "/kaggle/working")\n']},
    ]
    (root / "titanic/solution_v3.ipynb").write_text(json.dumps(notebook, indent=2) + "\n")
    print("Created titanic/solution_v3.ipynb")
