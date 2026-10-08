"""Predict synthetic California housing values with an untouched audit."""

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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


COMPETITION = "playground-series-s3e1"
TARGET = "MedHouseVal"
COVARIATES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population",
              "AveOccup", "Latitude", "Longitude"]
CATBOOST_PARAMETERS = dict(iterations=1200, depth=6, learning_rate=0.05,
                          l2_leaf_reg=6, loss_function="RMSE", random_seed=42,
                          thread_count=2, verbose=False, allow_writing_files=False)


def input_bytes(folder, name):
    path = folder / name
    if path.is_file():
        return path.read_bytes()
    with ZipFile(folder / (name + ".zip")) as archive:
        members = [p for p in archive.infolist() if Path(p.filename).name == name and not p.is_dir()]
        assert len(members) == 1
        return archive.read(members[0])


def features(frame, engineered):
    """Use fixed row-level transformations; never use ID or target."""
    x = frame[COVARIATES].copy().astype(float)
    if engineered:
        for column in ["MedInc", "AveRooms", "AveBedrms", "Population", "AveOccup"]:
            x["log_" + column] = np.log1p(x[column].where(x[column] >= 0))
        x["bedroom_share"] = x.AveBedrms / x.AveRooms.where(x.AveRooms > 0)
        x["rooms_per_person"] = x.AveRooms / x.AveOccup.where(x.AveOccup > 0)
        x["estimated_households"] = x.Population / x.AveOccup.where(x.AveOccup > 0)
        x["nonbedroom_rooms"] = x.AveRooms - x.AveBedrms
    return x.replace([np.inf, -np.inf], np.nan)


def fit_predict(name, train, y, target):
    if name == "blend":
        a, _ = fit_predict("catboost_raw", train, y, target)
        b, _ = fit_predict("catboost_features", train, y, target)
        return 0.5 * a + 0.5 * b, None
    x = features(train, engineered=name == "catboost_features")
    xt = features(target, engineered=name == "catboost_features")
    if name == "ridge":
        model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                              StandardScaler(), Ridge(alpha=10.0))
    else:
        model = CatBoostRegressor(**CATBOOST_PARAMETERS)
    model.fit(x, y)
    prediction = np.asarray(model.predict(xt), dtype=float)
    assert np.isfinite(prediction).all()
    return np.clip(prediction, float(y.min()), float(y.max())), model


def scores(y, prediction):
    error = prediction - y
    return {"rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))),
            "mean_error": float(np.mean(error))}


def groups_diagnostic(frame, y, prediction):
    masks = {"latitude_below_36_5": frame.Latitude.to_numpy() < 36.5,
             "latitude_at_least_36_5": frame.Latitude.to_numpy() >= 36.5,
             "target_below_5": y < 5,
             "target_at_least_5": y >= 5}
    return {name: {"rows": int(mask.sum()), **scores(y[mask], prediction[mask])}
            for name, mask in masks.items() if mask.any()}


def indices_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def run(data_dir, output_dir):
    started = time.time()
    folder, out = Path(data_dir), Path(output_dir)
    if str(folder).startswith("/kaggle/input/") and not folder.exists():
        matches = [p for p in Path("/kaggle/input").rglob(COMPETITION) if p.is_dir()]
        assert len(matches) == 1
        folder = matches[0]
    names = ["train.csv", "test.csv", "sample_submission.csv"]
    contents = {name: input_bytes(folder, name) for name in names}
    train, test, sample = [pd.read_csv(io.BytesIO(contents[name])) for name in names]
    assert len(train) == 37137 and len(test) == 24759
    assert train.columns.tolist() == ["id", *COVARIATES, TARGET]
    assert test.columns.tolist() == ["id", *COVARIATES]
    assert sample.columns.tolist() == ["id", TARGET]
    assert train.id.is_unique and test.id.is_unique and sample.id.is_unique
    assert set(train.id).isdisjoint(test.id) and set(test.id) == set(sample.id)
    assert not train.duplicated(COVARIATES).any()
    y = train[TARGET].to_numpy(dtype=float)
    assert np.isfinite(y).all() and (y > 0).all()
    all_indices = np.arange(len(train))
    development, audit = train_test_split(all_indices, test_size=0.2, random_state=42)
    fitting, selection = train_test_split(development, test_size=0.25, random_state=43)
    assert set(fitting).isdisjoint(selection) and set(development).isdisjoint(audit)
    assert len(fitting) + len(selection) + len(audit) == len(train)
    baseline_selection = scores(y[selection], np.full(len(selection), y[fitting].mean()))
    prediction, candidates = {}, {}
    for name in ["ridge", "catboost_raw", "catboost_features"]:
        prediction[name], _ = fit_predict(name, train.iloc[fitting], y[fitting], train.iloc[selection])
        candidates[name] = scores(y[selection], prediction[name])
        print("SELECTION=" + json.dumps({"model": name, **candidates[name]}), flush=True)
    prediction["blend"] = 0.5 * prediction["catboost_raw"] + 0.5 * prediction["catboost_features"]
    candidates["blend"] = scores(y[selection], prediction["blend"])
    selected = min(candidates, key=lambda name: candidates[name]["rmse"])
    print("SELECTED=" + selected, flush=True)
    # Evaluate only the selected fixed recipe on the untouched random audit.
    audit_prediction, _ = fit_predict(selected, train.iloc[development], y[development], train.iloc[audit])
    audit_scores = scores(y[audit], audit_prediction)
    audit_baseline = scores(y[audit], np.full(len(audit), y[development].mean()))
    print("AUDIT=" + json.dumps(audit_scores), flush=True)
    coordinate_keys = train.Latitude.astype(str) + "," + train.Longitude.astype(str)
    known_coordinates = coordinate_keys.iloc[audit].isin(coordinate_keys.iloc[development]).to_numpy()
    audit_coordinate_groups = {
        name: {"rows": int(mask.sum()), **scores(y[audit][mask], audit_prediction[mask])}
        for name, mask in {"seen_coordinates": known_coordinates, "unseen_coordinates": ~known_coordinates}.items()
        if mask.any()
    }
    # A fixed spatial stress test follows selection and never changes the recipe.
    spatial_keys = (np.floor(train.Latitude / 0.5).astype(int).astype(str) + ":" +
                    np.floor(train.Longitude / 0.5).astype(int).astype(str))
    spatial_fit, spatial_holdout = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=44)
                                        .split(train, groups=spatial_keys))
    assert set(spatial_keys.iloc[spatial_fit]).isdisjoint(spatial_keys.iloc[spatial_holdout])
    spatial_prediction, _ = fit_predict(selected, train.iloc[spatial_fit], y[spatial_fit], train.iloc[spatial_holdout])
    spatial_scores = scores(y[spatial_holdout], spatial_prediction)
    spatial_baseline = scores(y[spatial_holdout], np.full(len(spatial_holdout), y[spatial_fit].mean()))
    print("SPATIAL=" + json.dumps(spatial_scores), flush=True)
    worth_submitting = (candidates[selected]["rmse"] < 0.8 * baseline_selection["rmse"] and
                        audit_scores["rmse"] < 0.8 * audit_baseline["rmse"])
    final_prediction, _ = fit_predict(selected, train, y, test)
    submission = pd.DataFrame({"id": test.id, TARGET: final_prediction})
    submission = submission.set_index("id").loc[sample.id].reset_index()[sample.columns]
    assert len(submission) == 24759 and submission.id.equals(sample.id)
    assert np.isfinite(submission[TARGET]).all() and submission[TARGET].gt(0).all()
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "submission.csv"
    submission.to_csv(csv_path, index=False)
    metrics = {
        "competition": COMPETITION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric": "Root Mean Squared Error", "target_transform": "None",
        "data_source": f"https://www.kaggle.com/competitions/{COMPETITION}/data",
        "data_provenance": "Official synthetic competition data generated from a model trained on the California Housing dataset. Only official competition labels are used; no original dataset or other external data is added.",
        "data_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()},
        "train_rows": len(train), "test_rows": len(test),
        "data_diagnostics": {"missing_training_values": int(train.isna().sum().sum()),
                             "identical_training_covariates": int(train.duplicated(COVARIATES).sum()),
                             "repeated_coordinate_rows": int(train.duplicated(["Latitude", "Longitude"]).sum()),
                             "target_range": [float(y.min()), float(y.max())]},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "sklearn": sklearn.__version__, "catboost": catboost.__version__},
        "raw_features": COVARIATES, "engineered_features": features(train, True).columns.tolist(),
        "parameters": {"ridge_alpha": 10.0, "catboost": CATBOOST_PARAMETERS,
                       "blend_weights": {"catboost_raw": 0.5, "catboost_features": 0.5}},
        "feature_method": "Eight original numeric covariates; the engineered variant adds five log1p features, bedroom share, rooms per person, estimated households and nonbedroom rooms. IDs are excluded. No target encoding, external labels or row removal. Predictions are bounded by the current fitting partition's target range.",
        "model_selection": {"training_rows": len(fitting), "validation_rows": len(selection),
                            "seed": 43, "baseline": baseline_selection, "candidates": candidates,
                            "training_indices_sha256": indices_hash(fitting), "validation_indices_sha256": indices_hash(selection),
                            "rule": "Lowest selection RMSE among four fixed candidates before opening the random audit or viewing a new leaderboard score."},
        "selected_model": selected,
        "holdout": {"training_rows": len(development), "validation_rows": len(audit), "seed": 42,
                    "training_indices_sha256": indices_hash(development), "validation_indices_sha256": indices_hash(audit),
                    "baseline": audit_baseline, "selected_model": audit_scores,
                    "by_group": groups_diagnostic(train.iloc[audit], y[audit], audit_prediction),
                    "by_coordinate_familiarity": audit_coordinate_groups},
        "spatial_diagnostic": {"grid_width_degrees": 0.5, "seed": 44,
                               "training_rows": len(spatial_fit), "validation_rows": len(spatial_holdout),
                               "training_cells": int(spatial_keys.iloc[spatial_fit].nunique()),
                               "validation_cells": int(spatial_keys.iloc[spatial_holdout].nunique()),
                               "training_indices_sha256": indices_hash(spatial_fit),
                               "validation_indices_sha256": indices_hash(spatial_holdout),
                               "baseline": spatial_baseline, "selected_model": spatial_scores,
                               "note": "Disjoint half-degree coordinate cells. A stress test for regional transfer, not a model-selection criterion or an independent second audit; its fitting rows can include random-audit labels. No tuning follows this diagnostic."},
        "worth_submitting": bool(worth_submitting),
        "submission_gate": "At least 20% RMSE improvement over the fitting-mean baseline in both selection and untouched random audit.",
        "submission_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest() if "__file__" in globals() else None,
        "runtime_seconds": time.time() - started,
        "limitations": "Random validation measures interpolation within the synthetic competition distribution. Coordinates can appear in both fitting and validation rows despite no identical full covariate vectors. The fixed spatial test measures a different generalization task and has adjacent regions, unequal cell sizes and possible latent dependence. These synthetic targets are not current real property valuations. Final fitting uses every training label after evaluation. No tuning or prediction scaling follows audit or official scores.",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("FINAL=" + json.dumps({"selected": selected, "selection_rmse": candidates[selected]["rmse"],
                               "audit_rmse": audit_scores["rmse"], "spatial_rmse": spatial_scores["rmse"],
                               "worth_submitting": bool(worth_submitting), "submission_sha256": metrics["submission_sha256"]}), flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=f"/kaggle/input/{COMPETITION}")
    parser.add_argument("--output-dir", default="/kaggle/working")
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)
