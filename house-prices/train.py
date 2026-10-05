"""Predict Ames house sale prices with log-price regression and fixed CV."""

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
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


COMPETITION = "house-prices-advanced-regression-techniques"
NAMES = ["ridge", "elastic_net", "catboost", "ridge_catboost_blend"]
LOG_COLUMNS = [
    "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
    "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea",
    "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea",
    "MiscVal", "TotalSF", "TotalPorchSF",
]
PARAMETERS = {
    "ridge": {"alpha": 20.0},
    "elastic_net": {"alpha": 0.0007, "l1_ratio": 0.8, "max_iter": 20000, "tol": 0.0001},
    "catboost": {"iterations": 1500, "depth": 5, "learning_rate": 0.035, "l2_leaf_reg": 8,
                 "loss_function": "RMSE", "thread_count": 2, "verbose": False,
                 "allow_writing_files": False},
    "ridge_catboost_blend": {"ridge_weight": 0.5, "catboost_weight": 0.5},
}


def features(df):
    """Use each house's attributes, with no target or fitted dataset statistics."""
    x = df.drop(columns=["Id", "SalePrice"], errors="ignore").copy()
    x["MSSubClass"] = df.MSSubClass.astype(str)
    x["MoSold"] = df.MoSold.astype(str)
    x["HouseAge"] = (df.YrSold - df.YearBuilt).clip(lower=0)
    x["YearsSinceRemodel"] = (df.YrSold - df.YearRemodAdd).clip(lower=0)
    x["TotalSF"] = df.GrLivArea + df.TotalBsmtSF.fillna(0)
    x["TotalBath"] = df.FullBath + 0.5 * df.HalfBath + df.BsmtFullBath.fillna(0) + 0.5 * df.BsmtHalfBath.fillna(0)
    x["TotalPorchSF"] = df[["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]].sum(axis=1)
    x["HasGarage"] = df.GarageArea.fillna(0).gt(0).astype(int)
    x["HasBasement"] = df.TotalBsmtSF.fillna(0).gt(0).astype(int)
    x["HasSecondFloor"] = df["2ndFlrSF"].gt(0).astype(int)
    x["HasFireplace"] = df.Fireplaces.gt(0).astype(int)
    for column in LOG_COLUMNS:
        x[column] = np.log1p(x[column].clip(lower=0))
    categorical = x.select_dtypes(include="object").columns.tolist()
    for column in categorical:
        x[column] = x[column].fillna("Missing").astype(str)
    assert "Id" not in x and "SalePrice" not in x
    return x, categorical


def estimator(name, seed, columns, categorical):
    if name == "catboost":
        return CatBoostRegressor(**PARAMETERS[name], random_seed=seed)
    numeric = [c for c in columns if c not in categorical]
    preprocessing = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("numeric", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numeric),
    ])
    cls = Ridge if name == "ridge" else ElasticNet
    return make_pipeline(preprocessing, cls(**PARAMETERS[name]))


def predict(name, x, y, xt, categorical, seed):
    if name == "ridge_catboost_blend":
        return 0.5 * predict("ridge", x, y, xt, categorical, seed) + 0.5 * predict("catboost", x, y, xt, categorical, seed)
    fitted = estimator(name, seed, x.columns, categorical)
    if name == "catboost":
        fitted.fit(x, y, cat_features=categorical)
    else:
        fitted.fit(x, y)
    return fitted.predict(xt)


def rmse(y, predictions):
    return float(np.sqrt(mean_squared_error(y, predictions)))


def summarize(y, oof, splits):
    assert np.isfinite(oof).all()
    return {"log_rmse": rmse(y, oof),
            "fold_log_rmse": [rmse(y.iloc[val], oof[val]) for _, val in splits]}


def run(data_dir, output_dir):
    started = time.time()
    data_dir, out = Path(data_dir), Path(output_dir)
    if not (data_dir / "train.csv").exists() and str(data_dir).startswith("/kaggle/input/"):
        candidates = [p.parent for p in Path("/kaggle/input").rglob("train.csv")
                      if p.parent.name == COMPETITION and (p.parent / "test.csv").exists()]
        assert len(candidates) == 1
        data_dir = candidates[0]
    train, test = pd.read_csv(data_dir / "train.csv"), pd.read_csv(data_dir / "test.csv")
    assert len(train) == 1460 and len(test) == 1459
    assert "SalePrice" in train and "SalePrice" not in test
    assert train.Id.is_unique and test.Id.is_unique
    assert set(train.Id).isdisjoint(set(test.Id)) and train.SalePrice.gt(0).all()
    x, categorical = features(train)
    xt, test_categorical = features(test)
    assert x.columns.equals(xt.columns) and categorical == test_categorical
    y = np.log(train.SalePrice)
    splits = list(KFold(5, shuffle=True, random_state=42).split(x))
    oof = {name: np.full(len(y), np.nan) for name in NAMES}
    baseline = np.full(len(y), np.nan)
    for fold, (training, validation) in enumerate(splits):
        baseline[validation] = y.iloc[training].mean()
        for name in ["ridge", "elastic_net", "catboost"]:
            oof[name][validation] = predict(name, x.iloc[training], y.iloc[training], x.iloc[validation], categorical, 42 + fold)
        oof["ridge_catboost_blend"][validation] = 0.5 * oof["ridge"][validation] + 0.5 * oof["catboost"][validation]
        print(f"Completed fold {fold + 1}/5", flush=True)
    scores = {name: summarize(y, predictions, splits) for name, predictions in oof.items()}
    baseline_score = summarize(y, baseline, splits)
    for name, score in scores.items():
        print("CANDIDATE=" + json.dumps({"name": name, **score}), flush=True)
    selected = min(NAMES, key=lambda name: scores[name]["log_rmse"])
    # This time split is a diagnostic: its labels also participated in CV selection.
    past = np.flatnonzero(train.YrSold.to_numpy() <= 2008)
    future = np.flatnonzero(train.YrSold.to_numpy() >= 2009)
    assert train.YrSold.iloc[past].max() < train.YrSold.iloc[future].min()
    time_predictions = predict(selected, x.iloc[past], y.iloc[past], x.iloc[future], categorical, 42)
    temporal = {"training_years": [2006, 2007, 2008], "validation_years": [2009, 2010],
                "training_rows": len(past), "validation_rows": len(future),
                "selected_model_log_rmse": rmse(y.iloc[future], time_predictions),
                "mean_log_price_baseline_rmse": rmse(y.iloc[future], np.full(len(future), y.iloc[past].mean()))}
    worth_submitting = (scores[selected]["log_rmse"] <= 0.16
                        and scores[selected]["log_rmse"] < 0.8 * baseline_score["log_rmse"]
                        and temporal["selected_model_log_rmse"] <= 0.20)
    # Average fold models in log space; CV directly evaluates each member's recipe.
    log_predictions = []
    for fold, (training, _) in enumerate(splits):
        log_predictions.append(predict(selected, x.iloc[training], y.iloc[training], xt, categorical, 42 + fold))
    prices = np.exp(np.mean(log_predictions, axis=0))
    assert np.isfinite(prices).all() and (prices > 0).all()
    submission = pd.DataFrame({"Id": test.Id, "SalePrice": prices})
    assert submission.columns.tolist() == ["Id", "SalePrice"] and len(submission) == 1459
    assert submission.Id.is_unique and submission.Id.equals(test.Id)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "submission.csv"
    submission.to_csv(path, index=False)
    metrics = {
        "competition": COMPETITION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": f"https://www.kaggle.com/competitions/{COMPETITION}/data",
        "data_sha256": {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
                        for name in ["train.csv", "test.csv", "data_description.txt", "sample_submission.csv"]},
        "train_rows": len(train), "test_rows": len(test),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "sklearn": sklearn.__version__, "catboost": catboost.__version__},
        "target_transform": "Natural log of SalePrice; inverse transform exp, matching RMSE on log prices.",
        "validation": "5-fold KFold, shuffle=True, seed=42; all 1460 training rows retained.",
        "features": x.columns.tolist(), "categorical_features": categorical,
        "log1p_features": LOG_COLUMNS, "parameters": PARAMETERS,
        "baseline": baseline_score, "candidate_scores": scores, "selected_model": selected,
        "selection_rule": "Lowest pooled out-of-fold log RMSE among four fixed candidates, before viewing a new leaderboard score.",
        "temporal_diagnostic": temporal, "worth_submitting": bool(worth_submitting),
        "submission_gate": "CV log RMSE <=0.16, at least 20% improvement over mean-log-price baseline, and temporal diagnostic log RMSE <=0.20.",
        "prediction_method": "Mean of five fold-model log predictions, followed by exp; seeds 42-46.",
        "submission_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "prediction_summary": {"minimum": float(prices.min()), "median": float(np.median(prices)), "maximum": float(prices.max())},
        "runtime_seconds": time.time() - started,
        "caveat": "CV is reused for model selection and may be optimistic. Temporal validation is a diagnostic, not an untouched holdout. OOF RMSE evaluates individual fold recipes, not an independently validated five-fold inference ensemble. No rows are removed and no external or test labels are used.",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("FINAL_METRICS=" + json.dumps(metrics), flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=f"/kaggle/input/{COMPETITION}")
    parser.add_argument("--output-dir", default="/kaggle/working")
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)
