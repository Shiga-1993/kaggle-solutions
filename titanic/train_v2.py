"""Compare CatBoost, Random Forest and Extra Trees for Titanic."""
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from train import CATEGORICAL, features

NAMES = ["catboost_regularized", "random_forest", "extra_trees"]
BASELINE = {"stratified": 0.8383838383838383, "ticket_group": 0.8103254769921436}


def estimator(name, seed, columns):
    if name == "catboost_regularized":
        return CatBoostClassifier(depth=4, iterations=600, learning_rate=0.035,
                                 l2_leaf_reg=15, loss_function="Logloss", random_seed=seed,
                                 thread_count=2, verbose=False, allow_writing_files=False)
    preprocessing = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("numeric", "passthrough", [c for c in columns if c not in CATEGORICAL]),
    ])
    cls = RandomForestClassifier if name == "random_forest" else ExtraTreesClassifier
    trees = cls(n_estimators=400, max_depth=7, min_samples_leaf=3, max_features=0.8,
                random_state=seed, n_jobs=2)
    return make_pipeline(preprocessing, trees)


def fit(model, name, x, y):
    if name == "catboost_regularized":
        model.fit(x, y, cat_features=CATEGORICAL)
    else:
        model.fit(x, y)
    return model


def cross_validate(name, x, y, splits):
    predictions = np.zeros(len(y))
    scores = []
    for fold, (tr, va) in enumerate(splits):
        model = fit(estimator(name, 42 + fold, x.columns), name, x.iloc[tr], y.iloc[tr])
        predictions[va] = model.predict_proba(x.iloc[va])[:, 1]
        scores.append(float(accuracy_score(y.iloc[va], predictions[va] >= 0.5)))
    return {"accuracy": float(accuracy_score(y, predictions >= 0.5)),
            "log_loss": float(log_loss(y, predictions)), "fold_accuracy": scores}


def run_v2(data_dir, output_dir):
    start = time.time()
    data_dir, out = Path(data_dir), Path(output_dir)
    if not (data_dir / "train.csv").exists():
        candidates = [p.parent for p in Path("/kaggle/input").rglob("train.csv")
                      if p.parent.name == "titanic" and (p.parent / "test.csv").exists()]
        assert len(candidates) == 1
        data_dir = candidates[0]
    out.mkdir(parents=True, exist_ok=True)
    train, test = pd.read_csv(data_dir/"train.csv"), pd.read_csv(data_dir/"test.csv")
    assert len(train) == 891 and len(test) == 418 and "Survived" not in test
    # Remove name length: it has no clear survival mechanism and can overfit.
    x, xt, y = features(train).drop(columns="NameLength"), features(test).drop(columns="NameLength"), train.Survived
    groups = train.Ticket.fillna("MISSING").astype(str)
    stratified = list(StratifiedKFold(5, shuffle=True, random_state=42).split(x, y))
    grouped = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(x, y, groups))
    for tr, va in grouped:
        assert set(groups.iloc[tr]).isdisjoint(set(groups.iloc[va]))
    scores = {}
    for name in NAMES:
        scores[name] = {"stratified": cross_validate(name, x, y, stratified),
                        "ticket_group": cross_validate(name, x, y, grouped)}
        scores[name]["selection_score"] = (scores[name]["stratified"]["accuracy"] + scores[name]["ticket_group"]["accuracy"]) / 2
        print("CANDIDATE=" + json.dumps({"name": name, **scores[name]}), flush=True)
    selected = max(NAMES, key=lambda name: scores[name]["selection_score"])
    baseline_score = (BASELINE["stratified"] + BASELINE["ticket_group"])/2
    # Require minimum improvements in both validation criteria.
    worth_submitting = (scores[selected]["selection_score"] >= baseline_score + 0.002
                        and scores[selected]["ticket_group"]["accuracy"] >= BASELINE["ticket_group"] + 0.005)
    probabilities = []
    for seed in [42, 137, 2026]:
        model = fit(estimator(selected, seed, x.columns), selected, x, y)
        probabilities.append(model.predict_proba(xt)[:, 1])
    predictions = (np.mean(probabilities, axis=0) >= 0.5).astype(int)
    submission = pd.DataFrame({"PassengerId": test.PassengerId, "Survived": predictions})
    assert len(submission) == 418 and submission.PassengerId.is_unique
    assert submission.Survived.isin([0, 1]).all()
    assert submission.PassengerId.equals(test.PassengerId)
    path = out/"submission.csv"
    submission.to_csv(path, index=False)
    metrics = {
        "competition": "titanic", "iteration": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_sha256": {f: hashlib.sha256((data_dir/f).read_bytes()).hexdigest() for f in ["train.csv", "test.csv"]},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "catboost": catboost.__version__},
        "candidate_scores": scores, "selected_model": selected,
        "selection_rule": "Maximize mean of fixed stratified CV and ticket-group CV accuracy; threshold=0.5.",
        "baseline_cv": BASELINE, "baseline_selection_score": baseline_score,
        "submission_gate": "mean CV improves >=0.002 and ticket-group CV improves >=0.005 versus v1",
        "worth_submitting": bool(worth_submitting), "features": x.columns.tolist(), "final_seeds": [42, 137, 2026],
        "submission_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "predicted_survivors": int(predictions.sum()), "runtime_seconds": time.time()-start,
        "caveat": "Both CV schemes have been reused for candidate selection, so they are optimistic. The final three-seed refit ensemble is not separately CV-scored. No external or test labels are used.",
    }
    (out/"metrics_v2.json").write_text(json.dumps(metrics, indent=2)+"\n")
    print("FINAL_METRICS_V2=" + json.dumps(metrics), flush=True)
    print("Submission verified: 418 unique PassengerIds; binary predictions; exact schema.", flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/kaggle/input/titanic")
    parser.add_argument("--output-dir", default="/kaggle/working")
    args = parser.parse_args()
    run_v2(args.data_dir, args.output_dir)
