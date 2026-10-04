"""Titanic: row-level features, honest CV, and a reproducible submission.

Only the official competition training labels are used. No external passenger
records, test labels, target encodings, or leaderboard-driven model selection.
"""

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


CATEGORICAL = ["Sex", "Embarked", "Title", "Deck", "TicketPrefix", "ClassSex"]
CONFIGS = {
    "catboost_depth4": {"depth": 4, "iterations": 1000, "learning_rate": 0.035, "l2_leaf_reg": 5},
    "catboost_depth5": {"depth": 5, "iterations": 1000, "learning_rate": 0.035, "l2_leaf_reg": 5},
    "catboost_depth6": {"depth": 6, "iterations": 1000, "learning_rate": 0.035, "l2_leaf_reg": 8},
}


def features(df):
    """Deterministic per-row transforms; never inspect Survived or other rows."""
    x = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].copy()
    x["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    x["IsAlone"] = (x["FamilySize"] == 1).astype(int)
    x["FarePerPerson"] = df["Fare"] / x["FamilySize"]
    x["LogFare"] = np.log1p(df["Fare"])
    x["AgeMissing"] = df["Age"].isna().astype(int)
    x["IsChild"] = (df["Age"] < 16).astype(int)
    x["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.", expand=False).str.strip()
    x["NameLength"] = df["Name"].str.len()
    cabin = df["Cabin"].fillna("")
    x["Deck"] = cabin.str.extract(r"([A-Za-z])", expand=False).fillna("Unknown")
    x["CabinCount"] = cabin.str.split().str.len()
    x["TicketPrefix"] = df["Ticket"].str.replace(r"[0-9. /]", "", regex=True).str.upper().replace("", "NUMERIC")
    x["ClassSex"] = df["Pclass"].astype(str) + "_" + df["Sex"]
    for col in CATEGORICAL:
        x[col] = x[col].fillna("Unknown").astype(str)
    # A fixed missing sentinel does not estimate statistics from validation data.
    for col in x.columns.difference(CATEGORICAL):
        x[col] = x[col].fillna(-1).astype(float)
    assert "Survived" not in x and "PassengerId" not in x
    assert not x.isna().any().any()
    return x


def model(config, seed):
    return CatBoostClassifier(**config, loss_function="Logloss", random_seed=seed,
                              verbose=False, thread_count=2, allow_writing_files=False)


def evaluate(x, y, splits, config):
    oof = np.zeros(len(y))
    folds = []
    for fold, (fit_idx, val_idx) in enumerate(splits):
        estimator = model(config, 42 + fold)
        estimator.fit(x.iloc[fit_idx], y.iloc[fit_idx], cat_features=CATEGORICAL)
        oof[val_idx] = estimator.predict_proba(x.iloc[val_idx])[:, 1]
        folds.append(float(accuracy_score(y.iloc[val_idx], oof[val_idx] >= 0.5)))
    return {"accuracy": float(accuracy_score(y, oof >= 0.5)),
            "log_loss": float(log_loss(y, oof)), "fold_accuracy": folds}, oof


def run(data_dir, output_dir):
    started = time.time()
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    if not (data_dir / "train.csv").exists() and str(data_dir).startswith("/kaggle/input/"):
        # Kaggle may mount competitions under /kaggle/input/competitions/.
        candidates = [p.parent for p in Path("/kaggle/input").rglob("train.csv")
                      if (p.parent / "test.csv").exists() and p.parent.name == "titanic"]
        assert len(candidates) == 1, f"Expected one official Titanic input: {candidates}"
        data_dir = candidates[0]
    print(f"DATA_DIRECTORY={data_dir}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, test = pd.read_csv(data_dir / "train.csv"), pd.read_csv(data_dir / "test.csv")
    assert len(train) == 891 and len(test) == 418
    assert "Survived" in train and "Survived" not in test
    assert train.PassengerId.is_unique and test.PassengerId.is_unique
    assert set(train.Survived.unique()) == {0, 1}
    x, xt, y = features(train), features(test), train.Survived
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(x, y))
    baseline = float(accuracy_score(y, train.Sex.eq("female").astype(int)))
    scores = {}
    for name, config in CONFIGS.items():
        scores[name], _ = evaluate(x, y, splits, config)
        print(json.dumps({"candidate": name, **scores[name]}), flush=True)
    selected = max(scores, key=lambda k: (scores[k]["accuracy"], -scores[k]["log_loss"]))
    # Diagnostic only: prevent shared tickets crossing folds. Model selection
    # uses the predeclared stratified folds above, not this or the leaderboard.
    groups = train.Ticket.fillna("MISSING").astype(str)
    group_splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(x, y, groups))
    for fit_idx, val_idx in group_splits:
        assert set(groups.iloc[fit_idx]).isdisjoint(set(groups.iloc[val_idx]))
    group_score, _ = evaluate(x, y, group_splits, CONFIGS[selected])
    probabilities = []
    for seed in [42, 137, 2026]:
        estimator = model(CONFIGS[selected], seed)
        estimator.fit(x, y, cat_features=CATEGORICAL)
        probabilities.append(estimator.predict_proba(xt)[:, 1])
    predictions = (np.mean(probabilities, axis=0) >= 0.5).astype(int)
    submission = pd.DataFrame({"PassengerId": test.PassengerId, "Survived": predictions})
    assert submission.columns.tolist() == ["PassengerId", "Survived"]
    assert len(submission) == 418 and submission.PassengerId.is_unique
    assert submission.Survived.isin([0, 1]).all()
    assert submission.PassengerId.equals(test.PassengerId)
    submission_path = output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    metrics = {
        "competition": "titanic", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(data_dir),
        "train_rows": len(train), "test_rows": len(test),
        "data_sha256": {name: hashlib.sha256((data_dir/name).read_bytes()).hexdigest()
                        for name in ["train.csv", "test.csv"]},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__, "catboost": catboost.__version__},
        "validation": "5-fold stratified CV; shuffle=True, seed=42; threshold=0.5",
        "baseline_female_survives_accuracy": baseline,
        "candidate_scores": scores, "selected_model": selected, "selected_config": CONFIGS[selected],
        "ticket_group_cv_diagnostic": group_score,
        "validation_caveat": "Candidate selection on the same folds is optimistic; ticket-group CV is a diagnostic, not an independent untouched holdout. The final three-seed refit ensemble is not separately CV-scored.",
        "features": x.columns.tolist(), "final_seeds": [42, 137, 2026],
        "submission_sha256": hashlib.sha256(submission_path.read_bytes()).hexdigest(),
        "predicted_survivors": int(predictions.sum()), "runtime_seconds": time.time()-started,
        "kaggle_public_score": None,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("FINAL_METRICS=" + json.dumps(metrics), flush=True)
    print("Submission verified: 418 unique PassengerIds; binary predictions; exact schema.", flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/kaggle/input/titanic")
    parser.add_argument("--output-dir", default="/kaggle/working")
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)
