"""Package the checked-in Python source without maintaining a second pipeline."""
import hashlib
import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--iteration", type=int, choices=[1, 2], default=1)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
source = (root / "titanic/train.py").read_text()
source_hash = hashlib.sha256(source.encode()).hexdigest()
notebook = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# Titanic: reproducible CatBoost baseline\n",
            "Official train labels only; row-level family/title/cabin features; three fixed candidates.\n",
            "5-fold stratified validation and ticket-group diagnostic; one submission selected before leaderboard feedback.\n",
            f"Source: Shiga-1993/kaggle-solutions, titanic/train.py (SHA256 {source_hash}).\n"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": [source.replace("if __name__ == \"__main__\":", "if False:") +
                    '\nmetrics = run("/kaggle/input/titanic", "/kaggle/working")\n']},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
(root / "titanic/solution.ipynb").write_text(json.dumps(notebook, indent=2) + "\n")
print("Notebook built from source SHA256", source_hash)
if args.iteration == 2:
    v2_source = (root / "titanic/train_v2.py").read_text()
    v2_hash = hashlib.sha256(v2_source.encode()).hexdigest()
    notebook["cells"] = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# Titanic iteration 2: regularization and tree comparison\n",
            "Select using the mean of stratified and ticket-group CV, before seeing the new leaderboard result.\n",
            f"train.py SHA256 {source_hash}; train_v2.py SHA256 {v2_hash}.\n"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": ['from pathlib import Path\nPath("/kaggle/working/train.py").write_text(' + repr(source) + ')\n']},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": [v2_source.replace('if __name__ == "__main__":', "if False:") +
                    '\nmetrics = run_v2("/kaggle/input/titanic", "/kaggle/working")\n']},
    ]
    (root / "titanic/solution_v2.ipynb").write_text(json.dumps(notebook, indent=2)+"\n")
    print("Iteration 2 source SHA256", v2_hash)
