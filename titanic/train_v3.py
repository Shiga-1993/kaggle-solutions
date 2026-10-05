"""Compare compact passenger features with fold-fitted preprocessing."""

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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train import CATEGORICAL, CONFIGS, features, model


CATEGORIES = ["Pclass", "Sex", "Embarked", "Title", "Deck", "FamilyBand", "ClassSex"]
NAMES = ["logistic_regression", "gradient_boosting", "random_forest"]
SEEDS = [42, 137, 2026]
HISTORICAL_BASELINE = {"stratified": 0.8383838383838383, "ticket_group": 0.8103254769921436}
PARAMETERS = {
    "logistic_regression": {"C": 1.0, "max_iter": 2000},
    "gradient_boosting": {"n_estimators": 200, "learning_rate": 0.05,
                          "max_depth": 2, "min_samples_leaf": 8, "subsample": 0.8},
    "random_forest": {"n_estimators": 400, "max_depth": 6,
                      "min_samples_leaf": 4, "max_features": 0.8, "n_jobs": 2},
}


def compact_features(df):
    """Create row-level features without fitting statistics or using labels."""
    x = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]].copy()
    title = df.Name.str.extract(r",\s*([^.]*)\.", expand=False).str.strip()
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    x["Title"] = title.where(title.isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
    x["Deck"] = df.Cabin.fillna("").str.extract(r"([A-Za-z])", expand=False).fillna("Unknown")
    family_size = df.SibSp + df.Parch + 1
    x["FamilyBand"] = np.select([family_size.eq(1), family_size.le(4)], ["alone", "small"], default="large")
    x["ClassSex"] = df.Pclass.astype(str) + "_" + df.Sex
    x["LogFare"] = np.log1p(df.Fare)
    x["LogFarePerPerson"] = np.log1p(df.Fare / family_size)
    x["AgeMissing"] = df.Age.isna().astype(int)
    x["IsChild"] = df.Age.lt(16).astype(int)
    for column in CATEGORIES:
        x[column] = x[column].fillna("Unknown").astype(str)
    assert "Survived" not in x and "PassengerId" not in x
    return x


def estimator(name, seed, columns):
    if name == "catboost_v1":
        return model(CONFIGS["catboost_depth6"], seed)
    numeric = [column for column in columns if column not in CATEGORIES]
    preprocessing = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORIES),
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
    ])
    cls = {"logistic_regression": LogisticRegression,
           "gradient_boosting": GradientBoostingClassifier,
           "random_forest": RandomForestClassifier}[name]
    return make_pipeline(preprocessing, cls(**PARAMETERS[name], random_state=seed))


def fit(estimator_, name, x, y):
    if name == "catboost_v1":
        estimator_.fit(x, y, cat_features=CATEGORICAL)
    else:
        estimator_.fit(x, y)
    return estimator_


def cross_validate(name, x, y, splits, ensemble=False):
    oof = np.full(len(y), np.nan)
    fold_accuracy = []
    for fold, (training, validation) in enumerate(splits):
        probabilities = []
        for seed in SEEDS if ensemble else [42 + fold]:
            fitted = fit(estimator(name, seed, x.columns), name, x.iloc[training], y.iloc[training])
            probabilities.append(fitted.predict_proba(x.iloc[validation])[:, 1])
        oof[validation] = np.mean(probabilities, axis=0)
        fold_accuracy.append(float(accuracy_score(y.iloc[validation], oof[validation] >= 0.5)))
    assert np.isfinite(oof).all()
    return {"accuracy": float(accuracy_score(y, oof >= 0.5)),
            "log_loss": float(log_loss(y, oof)), "fold_accuracy": fold_accuracy}, oof


def group_bootstrap_difference(y, candidate, baseline, groups):
    """Resample complete tickets to describe uncertainty in paired CV accuracy."""
    codes, tickets = pd.factorize(groups, sort=True)
    difference = ((candidate >= 0.5) == y.to_numpy()).astype(int) - ((baseline >= 0.5) == y.to_numpy()).astype(int)
    totals = np.bincount(codes, weights=difference)
    sizes = np.bincount(codes)
    rng = np.random.default_rng(20261005)
    deltas = []
    for _ in range(2000):
        sampled = rng.integers(0, len(tickets), size=len(tickets))
        deltas.append(float(totals[sampled].sum() / sizes[sampled].sum()))
    return {"accuracy_difference": float(difference.mean()),
            "ticket_bootstrap_95_percent_interval": np.quantile(deltas, [0.025, 0.975]).tolist(),
            "resamples": 2000, "seed": 20261005,
            "limitation": "Conditional on fitted out-of-fold predictions; excludes training and model-selection uncertainty."}


def run_v3(data_dir, output_dir):
    started = time.time()
    data_dir, out = Path(data_dir), Path(output_dir)
    if not (data_dir / "train.csv").exists() and str(data_dir).startswith("/kaggle/input/"):
        candidates = [p.parent for p in Path("/kaggle/input").rglob("train.csv")
                      if p.parent.name == "titanic" and (p.parent / "test.csv").exists()]
        assert len(candidates) == 1
        data_dir = candidates[0]
    train, test = pd.read_csv(data_dir / "train.csv"), pd.read_csv(data_dir / "test.csv")
    assert len(train) == 891 and len(test) == 418
    assert "Survived" in train and "Survived" not in test
    assert train.PassengerId.is_unique and test.PassengerId.is_unique
    assert set(train.Survived.unique()) == {0, 1}
    x, xt, y = compact_features(train), compact_features(test), train.Survived
    original_x = features(train)
    groups = train.Ticket.fillna("MISSING").astype(str)
    splits = {
        "stratified": list(StratifiedKFold(5, shuffle=True, random_state=42).split(x, y)),
        "ticket_group": list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(x, y, groups)),
    }
    for training, validation in splits["ticket_group"]:
        assert set(groups.iloc[training]).isdisjoint(set(groups.iloc[validation]))
    scores = {}
    for name in NAMES:
        scores[name] = {scheme: cross_validate(name, x, y, folds)[0] for scheme, folds in splits.items()}
        scores[name]["selection_score"] = np.mean([scores[name][s]["accuracy"] for s in splits]).item()
        print("CANDIDATE=" + json.dumps({"name": name, **scores[name]}), flush=True)
    selected = max(NAMES, key=lambda n: (scores[n]["selection_score"],
                                        -np.mean([scores[n][s]["log_loss"] for s in splits])))
    ensemble_scores, paired_differences, baseline_oof = {}, {}, {}
    for name in ["catboost_v1", selected]:
        ensemble_scores[name] = {}
        for scheme, folds in splits.items():
            score, oof = cross_validate(name, original_x if name == "catboost_v1" else x, y, folds, ensemble=True)
            ensemble_scores[name][scheme] = score
            if name == "catboost_v1":
                baseline_oof[scheme] = oof
            else:
                paired_differences[scheme] = group_bootstrap_difference(y, oof, baseline_oof[scheme], groups)
            print("ENSEMBLE=" + json.dumps({"name": name, "scheme": scheme, **score}), flush=True)
    selected_mean = np.mean([ensemble_scores[selected][s]["accuracy"] for s in splits]).item()
    baseline_mean = np.mean([ensemble_scores["catboost_v1"][s]["accuracy"] for s in splits]).item()
    historical_mean = np.mean(list(HISTORICAL_BASELINE.values())).item()
    gate_checks = {
        "mean_improvement_over_ensemble_baseline": selected_mean >= baseline_mean + 0.002,
        "group_improvement_over_ensemble_baseline": ensemble_scores[selected]["ticket_group"]["accuracy"] >= ensemble_scores["catboost_v1"]["ticket_group"]["accuracy"] + 0.005,
        "mean_improvement_over_historical_v1": selected_mean >= historical_mean + 0.002,
        "group_improvement_over_historical_v1": ensemble_scores[selected]["ticket_group"]["accuracy"] >= HISTORICAL_BASELINE["ticket_group"] + 0.005,
    }
    probabilities = []
    for seed in SEEDS:
        fitted = fit(estimator(selected, seed, x.columns), selected, x, y)
        probabilities.append(fitted.predict_proba(xt)[:, 1])
    predictions = (np.mean(probabilities, axis=0) >= 0.5).astype(int)
    submission = pd.DataFrame({"PassengerId": test.PassengerId, "Survived": predictions})
    assert submission.columns.tolist() == ["PassengerId", "Survived"]
    assert len(submission) == 418 and submission.PassengerId.is_unique
    assert submission.PassengerId.equals(test.PassengerId) and submission.Survived.isin([0, 1]).all()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "submission.csv"
    submission.to_csv(path, index=False)
    metrics = {
        "competition": "titanic", "iteration": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "https://www.kaggle.com/competitions/titanic/data",
        "data_sha256": {f: hashlib.sha256((data_dir / f).read_bytes()).hexdigest() for f in ["train.csv", "test.csv"]},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "sklearn": sklearn.__version__, "catboost": catboost.__version__},
        "features": x.columns.tolist(), "parameters": PARAMETERS,
        "preprocessing": "Row-level compact features; median imputation, scaling and one-hot encoding fitted only on each training fold.",
        "validation": "Fixed 5-fold stratified and ticket-group CV, shuffle=True, seed=42; probability threshold=0.5.",
        "candidate_scores": scores, "selected_model": selected,
        "selection_rule": "Maximize mean accuracy across the two CV schemes; break ties by mean log loss, before any new Kaggle score.",
        "ensemble_cv": ensemble_scores, "final_seeds": SEEDS, "paired_differences": paired_differences,
        "historical_v1_cv": HISTORICAL_BASELINE,
        "submission_gate": "Three-seed ensemble must improve mean CV by >=0.002 and ticket-group CV by >=0.005 against both the re-evaluated v1 ensemble and historical v1 results.",
        "submission_gate_checks": {k: bool(v) for k, v in gate_checks.items()},
        "worth_submitting": bool(all(gate_checks.values())),
        "submission_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "predicted_survivors": int(predictions.sum()), "runtime_seconds": time.time() - started,
        "kaggle_public_score": None,
        "caveat": "The CV folds have been reused across experiments and are not independent holdouts. Ensemble CV evaluates the same three-seed procedure as the final refit. Bootstrap intervals exclude model-selection and training uncertainty. No external or test labels are used.",
    }
    (out / "metrics_v3.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("FINAL_METRICS_V3=" + json.dumps(metrics), flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/kaggle/input/titanic")
    parser.add_argument("--output-dir", default="/kaggle/working")
    args = parser.parse_args()
    run_v3(args.data_dir, args.output_dir)
