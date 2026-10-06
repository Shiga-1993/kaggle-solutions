"""Predict rental listing interest with month-separated model selection and audit."""

import argparse
import hashlib
import io
import json
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import catboost
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


COMPETITION = "two-sigma-connect-rental-listing-inquiries"
CLASSES = ["high", "medium", "low"]
CLASS_IDS = np.arange(3)
CATEGORICAL = ["manager_id", "building_id", "display_address"]
AMENITIES = {
    "no_fee": r"no[ -]fee",
    "doorman": r"doorman",
    "elevator": r"elevator",
    "laundry": r"laundry|washer|dryer",
    "dishwasher": r"dishwasher",
    "hardwood": r"hardwood",
    "fitness": r"fitness|gym",
    "outdoor": r"outdoor|roof|garden|terrace|balcony",
    "pets": r"pets|cats|dogs",
}
PARAMETERS = {
    "logistic": {"C": 1.0, "solver": "lbfgs", "max_iter": 2000, "tol": 0.001},
    "catboost": {"iterations": 800, "depth": 6, "learning_rate": 0.05,
                 "l2_leaf_reg": 6, "loss_function": "MultiClass", "random_seed": 42,
                 "thread_count": 2, "verbose": False, "allow_writing_files": False},
    "blend": {"logistic_weight": 0.5, "catboost_weight": 0.5},
}


def input_bytes(folder, name):
    """Read official plain files or the original competition ZIP members."""
    path = folder / name
    if path.is_file():
        return path.read_bytes()
    with ZipFile(folder / (name + ".zip")) as archive:
        members = [p for p in archive.infolist() if Path(p.filename).name == name and not p.is_dir()]
        assert len(members) == 1, f"Expected one {name} member"
        return archive.read(members[0])


def features(df):
    """Build row-level features without target labels or fitted data statistics."""
    x = df[["bathrooms", "bedrooms", "latitude", "longitude"]].copy()
    price = df.price.clip(lower=0)
    x["log_price"] = np.log1p(price)
    x["log_price_per_bedroom"] = np.log1p(price / df.bedrooms.clip(lower=1))
    x["log_price_per_room"] = np.log1p(price / (df.bedrooms + df.bathrooms).clip(lower=1))
    x["bathrooms_per_bedroom"] = df.bathrooms / df.bedrooms.clip(lower=1)
    x["is_studio"] = df.bedrooms.eq(0).astype(int)
    x["photo_count"] = df.photos.map(len)
    x["amenity_count"] = df.features.map(len)
    description = df.description.fillna("").astype(str).map(lambda s: re.sub(r"<[^>]+>", " ", s))
    x["description_length"] = description.str.len()
    x["description_word_count"] = description.str.split().map(len)
    created = pd.to_datetime(df.created)
    x["created_hour"] = created.dt.hour
    x["created_weekday"] = created.dt.dayofweek
    x["created_day"] = (created - pd.Timestamp("2016-04-01")).dt.total_seconds() / 86400
    for column in CATEGORICAL:
        x[column] = df[column].fillna("Unknown").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        x[column] = x[column].replace({"0": "Unknown", "": "Unknown"})
    amenity_text = df.features.map(lambda values: " ".join(str(s).lower() for s in values))
    x["amenity_text"] = amenity_text
    for name, pattern in AMENITIES.items():
        x["has_" + name] = amenity_text.str.contains(pattern, regex=True).astype(int)
    assert "listing_id" not in x and "interest_level" not in x
    return x


def estimator(name, columns):
    if name == "catboost":
        return CatBoostClassifier(**PARAMETERS[name])
    numeric = [c for c in columns if c not in CATEGORICAL + ["amenity_text"]]
    preprocessing = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=5), CATEGORICAL),
        ("amenities", TfidfVectorizer(min_df=3, max_features=1500, ngram_range=(1, 2), sublinear_tf=True), "amenity_text"),
    ], sparse_threshold=1.0)
    return make_pipeline(preprocessing, LogisticRegression(**PARAMETERS[name]))


def probabilities(values):
    values = np.asarray(values, dtype=float)
    assert values.ndim == 2 and values.shape[1] == 3
    assert np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
    assert np.allclose(values.sum(axis=1), 1, atol=1e-8)
    values = np.clip(values, 1e-15, 1 - 1e-15)
    return values / values.sum(axis=1, keepdims=True)


def fit_predict(name, x, y, targets):
    if name == "blend":
        first, _ = fit_predict("logistic", x, y, targets)
        second, model = fit_predict("catboost", x, y, targets)
        return probabilities(0.5 * first + 0.5 * second), model
    model = estimator(name, x.columns)
    if name == "catboost":
        model.fit(x.drop(columns="amenity_text"), y, cat_features=CATEGORICAL)
        values = model.predict_proba(targets.drop(columns="amenity_text"))
    else:
        model.fit(x, y)
        values = model.predict_proba(targets)
    assert np.array_equal(model.classes_, CLASS_IDS), "Probability class order changed"
    return probabilities(values), model


def prior_predictions(y, n):
    priors = np.bincount(y, minlength=3) / len(y)
    return probabilities(np.tile(priors, (n, 1)))


def scores(y, prediction):
    return {"log_loss": float(log_loss(y, prediction, labels=CLASS_IDS)),
            "accuracy": float(accuracy_score(y, np.argmax(prediction, axis=1))),
            "class_log_loss": {name: float(-np.log(prediction[y == i, i]).mean())
                               for i, name in enumerate(CLASSES)}}


def subset_scores(y, prediction, mask):
    count = int(mask.sum())
    return {"rows": count,
            "log_loss": float(log_loss(y[mask], prediction[mask], labels=CLASS_IDS)) if count else None}


def run(data_dir, output_dir):
    started = time.time()
    folder, out = Path(data_dir), Path(output_dir)
    if str(folder).startswith("/kaggle/input/") and not folder.exists():
        candidates = [p for p in Path("/kaggle/input").rglob(COMPETITION) if p.is_dir()]
        assert len(candidates) == 1
        folder = candidates[0]
    contents = {name: input_bytes(folder, name) for name in ["train.json", "test.json", "sample_submission.csv"]}
    train = pd.read_json(io.BytesIO(contents["train.json"])).reset_index(drop=True)
    test = pd.read_json(io.BytesIO(contents["test.json"])).reset_index(drop=True)
    sample = pd.read_csv(io.BytesIO(contents["sample_submission.csv"]))
    assert len(train) == 49352 and len(test) == 74659
    assert train.listing_id.is_unique and test.listing_id.is_unique and sample.listing_id.is_unique
    assert set(train.listing_id).isdisjoint(test.listing_id)
    assert set(sample.listing_id) == set(test.listing_id)
    assert set(sample.columns) == set(["listing_id"] + CLASSES)
    assert "interest_level" in train and "interest_level" not in test
    x, xt = features(train), features(test)
    assert x.columns.equals(xt.columns)
    y = train.interest_level.map({name: i for i, name in enumerate(CLASSES)}).to_numpy()
    assert np.isin(y, CLASS_IDS).all()
    months = pd.to_datetime(train.created).dt.to_period("M").astype(str)
    assert set(months) == {"2016-04", "2016-05", "2016-06"}
    april = np.flatnonzero(months.eq("2016-04"))
    may = np.flatnonzero(months.eq("2016-05"))
    june = np.flatnonzero(months.eq("2016-06"))
    selection_predictions, candidates = {}, {}
    baseline_selection = scores(y[may], prior_predictions(y[april], len(may)))
    for name in ["logistic", "catboost"]:
        predictions, _ = fit_predict(name, x.iloc[april], y[april], x.iloc[may])
        selection_predictions[name] = predictions
        candidates[name] = scores(y[may], predictions)
        print("SELECTION=" + json.dumps({"model": name, **candidates[name]}), flush=True)
    selection_predictions["blend"] = probabilities(0.5 * selection_predictions["logistic"] + 0.5 * selection_predictions["catboost"])
    candidates["blend"] = scores(y[may], selection_predictions["blend"])
    selected = min(candidates, key=lambda name: candidates[name]["log_loss"])
    print("SELECTED=" + selected, flush=True)
    # Choose once on May; June is not used to adjust features, weights or parameters.
    past = np.concatenate([april, may])
    audit_predictions, audit_model = fit_predict(selected, x.iloc[past], y[past], x.iloc[june])
    baseline_audit = scores(y[june], prior_predictions(y[past], len(june)))
    audit = scores(y[june], audit_predictions)
    audit_subsets = {}
    for column in ["manager_id", "building_id"]:
        seen = x.iloc[june][column].isin(set(x.iloc[past][column]) - {"Unknown"}).to_numpy()
        audit_subsets[column] = {"seen": subset_scores(y[june], audit_predictions, seen),
                                 "unseen_or_missing": subset_scores(y[june], audit_predictions, ~seen)}
    importance = None
    if selected in ["catboost", "blend"]:
        importance = dict(sorted(zip(audit_model.feature_names_, audit_model.feature_importances_.tolist()),
                                 key=lambda pair: pair[1], reverse=True))
    print("JUNE_AUDIT=" + json.dumps(audit), flush=True)
    worth_submitting = (candidates[selected]["log_loss"] <= 0.65 and audit["log_loss"] <= 0.68
                        and candidates[selected]["log_loss"] < 0.9 * baseline_selection["log_loss"]
                        and audit["log_loss"] < 0.9 * baseline_audit["log_loss"])
    prediction, _ = fit_predict(selected, x, y, xt)
    submission = pd.DataFrame(prediction, columns=CLASSES)
    submission.insert(0, "listing_id", test.listing_id)
    submission = submission.set_index("listing_id").loc[sample.listing_id].reset_index()[sample.columns]
    assert len(submission) == 74659 and submission.listing_id.equals(sample.listing_id)
    assert submission.columns.equals(sample.columns)
    probabilities(submission[CLASSES].to_numpy())
    out.mkdir(parents=True, exist_ok=True)
    path = out / "submission.csv"
    submission.to_csv(path, index=False)
    metrics = {
        "competition": COMPETITION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": f"https://www.kaggle.com/competitions/{COMPETITION}/data",
        "data_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()},
        "train_rows": len(train), "test_rows": len(test), "class_order": CLASSES,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "sklearn": sklearn.__version__, "catboost": catboost.__version__},
        "class_counts": train.interest_level.value_counts().to_dict(),
        "train_month_counts": months.value_counts().sort_index().to_dict(),
        "test_month_counts": pd.to_datetime(test.created).dt.to_period("M").astype(str).value_counts().sort_index().to_dict(),
        "features": x.columns.tolist(), "categorical_features": CATEGORICAL, "amenity_patterns": AMENITIES,
        "parameters": PARAMETERS,
        "linear_preprocessing": "Train-fitted median imputation and StandardScaler; one-hot categories with min_frequency=5 and unknown categories ignored; amenity TF-IDF min_df=3, max_features=1500, ngram_range=(1,2), sublinear_tf=True.",
        "model_selection": {"train_month": "2016-04", "validation_month": "2016-05", "training_rows": len(april),
                            "validation_rows": len(may), "baseline": baseline_selection, "candidates": candidates,
                            "rule": "Lowest May log loss among three fixed candidates before evaluating June or viewing a new leaderboard score."},
        "selected_model": selected,
        "holdout": {"training_months": ["2016-04", "2016-05"], "validation_month": "2016-06", "training_rows": len(past),
                    "validation_rows": len(june), "baseline": baseline_audit, "selected_model": audit,
                    "identity_subsets": audit_subsets},
        "holdout_model_feature_importance": importance,
        "worth_submitting": bool(worth_submitting),
        "submission_gate": "May log loss <=0.65, June log loss <=0.68, and at least 10% improvement over each training-prior baseline.",
        "prediction_method": "Refit the May-selected fixed recipe on all April-June training rows, seed 42; output probabilities in the official template order.",
        "submission_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest() if "__file__" in globals() else None,
        "runtime_seconds": time.time() - started,
        "limitations": "May is reused for model selection; June evaluates only the selected recipe, without further tuning. Listings can share managers and buildings across months; identity subset scores expose this difference. Competition test listings span the same April-June period, so the forward June audit is not an exact leaderboard proxy. The final model is refitted on June labels after the audit. No external data, test labels, manual test annotation or downloaded photos are used. Probabilities are predictive associations, not causal demand estimates.",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("FINAL=" + json.dumps({"selected": selected, "may_log_loss": candidates[selected]["log_loss"],
                               "june_log_loss": audit["log_loss"], "worth_submitting": bool(worth_submitting),
                               "submission_sha256": metrics["submission_sha256"]}), flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=f"/kaggle/input/{COMPETITION}")
    parser.add_argument("--output-dir", default="/kaggle/working")
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)
