"""Package the checked-in Python source without maintaining a second pipeline."""
import hashlib
import json
from pathlib import Path

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
