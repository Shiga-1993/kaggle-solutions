"""Predict sale prices with chronological model selection and an untouched audit."""

import argparse
import hashlib
import io
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import catboost
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


COMPETITION = "sberbank-russian-housing-market"
MACRO_COLUMNS = ["oil_urals", "cpi", "usdrub", "eurrub", "mortgage_rate", "unemployment"]
PARAMETERS = {
    "ridge": {"alpha": 10.0, "solver": "lsqr", "tol": 0.0001, "max_iter": 2000},
    "catboost": {"iterations": 1000, "depth": 6, "learning_rate": 0.05,
                 "l2_leaf_reg": 6, "loss_function": "RMSE", "random_seed": 42,
                 "thread_count": 2, "verbose": False, "allow_writing_files": False},
    "blend": {"ridge_weight": 0.5, "catboost_weight": 0.5},
}


def input_bytes(folder, name):
    path = folder / name
    if path.is_file():
        return path.read_bytes()
    with ZipFile(folder / (name + ".zip")) as archive:
        members = [p for p in archive.infolist() if Path(p.filename).name == name and not p.is_dir()]
        assert len(members) == 1, f"Expected one {name} member"
        return archive.read(members[0])


def features(df, macro):
    """Apply fixed row-level consistency checks; retain every transaction."""
    x = df.drop(columns=[c for c in ["id", "timestamp", "price_doc"] if c in df]).copy()
    dates = pd.to_datetime(df.timestamp)
    for column in ["full_sq", "life_sq", "kitch_sq"]:
        x[column + "_nonpositive"] = x[column].le(0).astype(int)
        x[column] = x[column].where(x[column] > 0)
    for column in ["life_sq", "kitch_sq"]:
        invalid = x[column].gt(x.full_sq)
        x[column + "_greater_than_full"] = invalid.astype(int)
        x[column] = x[column].mask(invalid)
    x["floor_above_max"] = x.floor.gt(x.max_floor).astype(int)
    x["max_floor"] = x.max_floor.where(x.max_floor > 0)
    x["floor"] = x.floor.where(x.floor >= 0)
    bad_year = x.build_year.lt(1800) | x.build_year.gt(dates.dt.year + 5)
    x["build_year_invalid"] = bad_year.astype(int)
    x["build_year"] = x.build_year.mask(bad_year)
    x["building_age"] = dates.dt.year - x.build_year
    x["life_share"] = x.life_sq / x.full_sq
    x["kitchen_share"] = x.kitch_sq / x.full_sq
    x["area_per_room"] = x.full_sq / x.num_room.where(x.num_room > 0)
    x["relative_floor"] = x.floor / x.max_floor
    for column in ["full_sq", "life_sq", "kitch_sq", "area_per_room"]:
        x["log_" + column] = np.log1p(x[column])
    x["transaction_year"] = dates.dt.year
    x["transaction_month"] = dates.dt.month
    x["transaction_weekday"] = dates.dt.dayofweek
    x["transaction_days"] = (dates - pd.Timestamp("2011-01-01")).dt.days
    assert macro.timestamp.is_unique and df.timestamp.isin(macro.timestamp).all()
    aligned = macro.set_index("timestamp").loc[df.timestamp, MACRO_COLUMNS].reset_index(drop=True)
    for column in MACRO_COLUMNS:
        x["macro_" + column] = pd.to_numeric(aligned[column], errors="coerce").to_numpy()
    categorical = [c for c in x if x[c].dtype == object or c.startswith("ID_") or c in ["material", "state"]]
    for column in categorical:
        x[column] = x[column].fillna("Unknown").astype(str)
    numeric = [c for c in x if c not in categorical]
    x[numeric] = x[numeric].replace([np.inf, -np.inf], np.nan)
    assert "price_doc" not in x and "id" not in x and "timestamp" not in x
    return x, categorical


def fit_predict(name, x, y, target, categorical):
    if name == "blend":
        a, _ = fit_predict("ridge", x, y, target, categorical)
        b, model = fit_predict("catboost", x, y, target, categorical)
        return 0.5 * a + 0.5 * b, model
    if name == "catboost":
        model = CatBoostRegressor(**PARAMETERS[name])
        model.fit(x, y, cat_features=categorical)
    else:
        numeric = [c for c in x if c not in categorical]
        preprocessing = ColumnTransformer([
            ("numeric", make_pipeline(SimpleImputer(strategy="median", add_indicator=True,
                                                   keep_empty_features=True), StandardScaler()), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=10), categorical),
        ], sparse_threshold=1.0)
        model = make_pipeline(preprocessing, Ridge(**PARAMETERS[name]))
        model.fit(x, y)
    prediction = np.asarray(model.predict(target), dtype=float)
    assert prediction.shape == (len(target),) and np.isfinite(prediction).all()
    # Bounds use only the current fitting partition, including for Ridge extrapolation.
    return np.clip(prediction, float(y.min()), float(y.max())), model


def scores(y, prediction):
    error = prediction - y
    return {"rmsle": float(np.sqrt(np.mean(error ** 2))),
            "mean_log_error": float(np.mean(error)),
            "median_absolute_percentage_error": float(np.median(np.abs(np.expm1(prediction) - np.expm1(y)) / np.expm1(y)))}


def diagnostic_groups(frame, y, prediction, column):
    output = {}
    for value in sorted(frame[column].fillna("Unknown").astype(str).unique()):
        mask = frame[column].fillna("Unknown").astype(str).eq(value).to_numpy()
        output[value] = {"rows": int(mask.sum()), **scores(y[mask], prediction[mask])}
    return output


def run(data_dir, output_dir):
    started = time.time()
    folder, out = Path(data_dir), Path(output_dir)
    if str(folder).startswith("/kaggle/input/") and not folder.exists():
        matches = [p for p in Path("/kaggle/input").rglob(COMPETITION) if p.is_dir()]
        assert len(matches) == 1
        folder = matches[0]
    names = ["train.csv", "test.csv", "macro.csv", "sample_submission.csv", "data_dictionary.txt"]
    contents = {name: input_bytes(folder, name) for name in names}
    train, test, macro, sample = [pd.read_csv(io.BytesIO(contents[name])) for name in names[:4]]
    assert len(train) == 30471 and len(test) == 7662
    assert train.id.is_unique and test.id.is_unique and sample.id.is_unique
    assert set(train.id).isdisjoint(test.id) and set(test.id) == set(sample.id)
    assert sample.columns.tolist() == ["id", "price_doc"]
    assert "price_doc" in train and "price_doc" not in test
    assert np.isfinite(train.price_doc).all() and train.price_doc.gt(0).all()
    dates = pd.to_datetime(train.timestamp)
    x, categorical = features(train, macro)
    xt, test_categorical = features(test, macro)
    assert x.columns.equals(xt.columns) and categorical == test_categorical
    y = np.log1p(train.price_doc.to_numpy(dtype=float))
    early = np.flatnonzero(dates < "2014-07-01")
    selection = np.flatnonzero((dates >= "2014-07-01") & (dates < "2015-01-01"))
    audit = np.flatnonzero(dates >= "2015-01-01")
    assert [len(early), len(selection), len(audit)] == [20483, 6749, 3239]
    assert dates.iloc[early].max() < dates.iloc[selection].min() < dates.iloc[audit].min()
    selection_prediction, candidates = {}, {}
    selection_baseline = scores(y[selection], np.full(len(selection), y[early].mean()))
    for name in ["ridge", "catboost"]:
        pred, _ = fit_predict(name, x.iloc[early], y[early], x.iloc[selection], categorical)
        selection_prediction[name] = pred
        candidates[name] = scores(y[selection], pred)
        print("SELECTION=" + json.dumps({"model": name, **candidates[name]}), flush=True)
    selection_prediction["blend"] = 0.5 * selection_prediction["ridge"] + 0.5 * selection_prediction["catboost"]
    candidates["blend"] = scores(y[selection], selection_prediction["blend"])
    selected = min(candidates, key=lambda name: candidates[name]["rmsle"])
    print("SELECTED=" + selected, flush=True)
    # The fixed recipe is selected before evaluating any 2015 labels or leaderboard score.
    past = np.concatenate([early, selection])
    audit_prediction, audit_model = fit_predict(selected, x.iloc[past], y[past], x.iloc[audit], categorical)
    audit_baseline = scores(y[audit], np.full(len(audit), y[past].mean()))
    audit_score = scores(y[audit], audit_prediction)
    print("AUDIT=" + json.dumps(audit_score), flush=True)
    by_type = diagnostic_groups(train.iloc[audit], y[audit], audit_prediction, "product_type")
    audit_month = train.iloc[audit].assign(month=train.timestamp.iloc[audit].str[:7].to_numpy())
    by_month = diagnostic_groups(audit_month, y[audit], audit_prediction, "month")
    importance = None
    if selected in ["catboost", "blend"]:
        importance = dict(sorted(zip(audit_model.feature_names_, audit_model.feature_importances_.tolist()),
                                 key=lambda pair: pair[1], reverse=True))
    worth_submitting = (candidates[selected]["rmsle"] <= 0.45 and audit_score["rmsle"] <= 0.45
                        and candidates[selected]["rmsle"] < 0.8 * selection_baseline["rmsle"]
                        and audit_score["rmsle"] < 0.8 * audit_baseline["rmsle"])
    final_log_prediction, _ = fit_predict(selected, x, y, xt, categorical)
    prediction = np.expm1(final_log_prediction)
    assert np.isfinite(prediction).all() and (prediction > 0).all()
    submission = pd.DataFrame({"id": test.id, "price_doc": prediction})
    submission = submission.set_index("id").loc[sample.id].reset_index()[sample.columns]
    assert len(submission) == 7662 and submission.id.equals(sample.id)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "submission.csv"
    submission.to_csv(csv_path, index=False)
    metrics = {
        "competition": COMPETITION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric": "Root Mean Squared Logarithmic Error", "target_transform": "log1p(price_doc)",
        "data_source": f"https://www.kaggle.com/competitions/{COMPETITION}/data",
        "data_sha256": {name: hashlib.sha256(payload).hexdigest() for name, payload in contents.items()},
        "train_rows": len(train), "test_rows": len(test), "feature_count": len(x.columns),
        "train_date_range": [str(dates.min().date()), str(dates.max().date())],
        "test_date_range": [test.timestamp.min(), test.timestamp.max()],
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "sklearn": sklearn.__version__, "catboost": catboost.__version__},
        "features": x.columns.tolist(), "categorical_features": categorical, "macro_features": MACRO_COLUMNS,
        "parameters": PARAMETERS,
        "feature_method": "All official housing covariates except transaction ID, timestamp string and target; fixed row-level area/year checks and flags; area ratios and logs; transaction calendar features; six official macro indicators joined by exact timestamp. No label-based row filtering or handcrafted target encoding.",
        "linear_preprocessing": "Training-fitted median imputation with missing indicators, standard scaling, one-hot categories with min_frequency=10 and unknown categories ignored. Ridge alpha=10. Log predictions bounded by the fitting partition's target range for every model.",
        "model_selection": {"training_rows": len(early), "validation_rows": len(selection),
                            "train_end": "2014-06-30", "validation_start": "2014-07-01", "validation_end": "2014-12-31",
                            "baseline": selection_baseline, "candidates": candidates,
                            "rule": "Lowest 2014-H2 RMSLE among three fixed candidates before evaluating 2015-H1 or viewing a new leaderboard score."},
        "selected_model": selected,
        "holdout": {"training_rows": len(past), "validation_rows": len(audit), "train_end": "2014-12-31",
                    "validation_start": "2015-01-01", "validation_end": "2015-06-30",
                    "baseline": audit_baseline, "selected_model": audit_score,
                    "by_product_type": by_type, "by_month": by_month},
        "holdout_model_feature_importance": importance,
        "worth_submitting": bool(worth_submitting),
        "submission_gate": "Selection and audit RMSLE <=0.45 and at least 20% improvement over each training-log-mean baseline.",
        "prediction_method": "Refit the selected fixed recipe on every official training row; inverse log1p predictions in the official template order.",
        "submission_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest() if "__file__" in globals() else None,
        "runtime_seconds": time.time() - started,
        "limitations": "The 2014-H2 selection score is reused to choose a model. Only the chosen recipe is evaluated once on 2015-H1; no tuning follows the audit. Macro indicators are supplied historical series joined by date, not verified real-time publication vintages, so this is a retrospective competition forecast. Macroeconomic and transaction-type distributions can shift in the later 2015-2016 test period. Repeated properties may cross time boundaries. All training labels and rows are retained. The final fit includes 2015-H1 labels after the audit. No external data, test labels, target-based row corrections or leaderboard-based price scaling are used.",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("FINAL=" + json.dumps({"selected": selected, "selection_rmsle": candidates[selected]["rmsle"],
                               "audit_rmsle": audit_score["rmsle"], "worth_submitting": bool(worth_submitting),
                               "submission_sha256": metrics["submission_sha256"]}), flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=f"/kaggle/input/{COMPETITION}")
    parser.add_argument("--output-dir", default="/kaggle/working")
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)
