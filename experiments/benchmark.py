"""Small pre-holdout benchmark. September/October are excluded throughout.

Run from the repository with .venv/Scripts/python experiments/benchmark.py.
These experiments are exploratory and are not the production prediction code.
"""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def benchmark_metadata():
    return {
        "input_sha256": hashlib.sha256((ROOT / "data" / "train_test.csv").read_bytes()).hexdigest(),
        "split": {
            "training_start_inclusive": "2025-01-01",
            "training_end_inclusive": "2025-06-30",
            "selection_start_inclusive": "2025-07-01",
            "selection_end_inclusive": "2025-08-31",
            "excluded_holdout_start_inclusive": "2025-09-01",
            "excluded_holdout_end_inclusive": "2025-10-31",
        },
        "seed": SEED,
    }


def metrics(y, pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "wape": float(np.sum(np.abs(np.asarray(y) - pred)) / np.sum(np.abs(y))),
        "r2": float(r2_score(y, pred)),
    }


def features(df, reduced=False, linear=False, no_quote=False):
    cat = ["pickup", "delivery", "equipment"]
    num = ["distance", "weight"]
    if not reduced:
        num += ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "market_index"]
        if not no_quote:
            num += ["quote_signal"]
    x = df[cat + num].copy()
    x[cat] = x[cat].fillna("__MISSING__").astype(str)
    x["weight_missing"] = (x.weight.isna() | (x.weight <= 0)).astype(int)
    x["weight"] = x.weight.where(x.weight > 0)
    date = pd.to_datetime(df.date)
    x["day_of_week"] = date.dt.dayofweek
    x["day_of_month"] = date.dt.day
    x["day_of_year"] = date.dt.dayofyear
    x["time_days"] = (date - pd.Timestamp("2025-01-01")).dt.days
    x["weekend"] = (date.dt.dayofweek >= 5).astype(int)
    for period, values in [(7, date.dt.dayofweek), (365.25, date.dt.dayofyear), (30.4375, date.dt.day)]:
        x[f"sin_{period}"] = np.sin(2 * np.pi * values / period)
        x[f"cos_{period}"] = np.cos(2 * np.pi * values / period)
    x["log_distance"] = np.log1p(x.distance)
    x["sqrt_distance"] = np.sqrt(x.distance)
    x["lane"] = x.pickup + "->" + x.delivery
    cat += ["lane"]
    if not reduced:
        if not no_quote:
            x["quote_total"] = x.distance * x.quote_signal
        x["market_missing"] = x.market_index.isna().astype(int)
        x["delta_lat"] = x.delivery_lat - x.pickup_lat
        x["delta_lon"] = x.delivery_lon - x.pickup_lon
    if linear:
        # Explicit interactions express per-mile equipment/weight/date effects.
        numeric_names = [c for c in x if c not in cat]
        for name in numeric_names:
            if name != "distance":
                x[f"distance_x_{name}"] = x.distance * x[name]
        for kind in ["Dry Van", "Reefer", "Flatbed"]:
            x[f"distance_x_{kind}"] = x.distance * (x.equipment == kind)
            if not reduced and not no_quote:
                x[f"quote_total_x_{kind}"] = x.quote_total * (x.equipment == kind)
    return x, cat


def no_quote_benchmark():
    """Additional requested comparisons retaining market/coordinates, excluding quote."""
    df = pd.read_csv(ROOT / "data" / "train_test.csv")
    df = df.loc[df.date < "2025-09-01"].copy()
    train = df.loc[df.date < "2025-07-01"].copy()
    select = df.loc[df.date >= "2025-07-01"].copy()
    y = select.posted_rate.to_numpy()
    result = {**benchmark_metadata(), "scope": "Jan-Jun training; Jul-Aug selection; September/October excluded", "models": {}}
    for reduced, target, linear in [(False, "raw", True), (False, "log_rpm", False), (True, "log_rpm", False)]:
        xtrain, cat = features(train, reduced, linear=linear, no_quote=True)
        xtest, _ = features(select, reduced, linear=linear, no_quote=True)
        nums = [c for c in xtrain if c not in cat]
        proc = ColumnTransformer([
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), nums),
            ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cat),
        ])
        model = Pipeline([("preprocess", proc), ("ridge", Ridge(alpha=20.0, solver="lsqr"))])
        yt = train.posted_rate if target == "raw" else np.log(train.posted_rate / train.distance)
        model.fit(xtrain, yt)
        pred = model.predict(xtest)
        if target == "log_rpm":
            pred = np.exp(pred) * select.distance.to_numpy()
        pred = np.maximum(1.0, pred)
        name = f"ridge_{'reduced' if reduced else 'full_noquote'}_{target}"
        entry = metrics(y, pred)
        chart = pd.read_csv(ROOT / "data" / "december_chart_inputs.csv")
        if reduced:
            xc, _ = features(chart, True, linear=linear, no_quote=True)
            pc = model.predict(xc)
            if target == "log_rpm":
                pc = np.exp(pc) * chart.distance.to_numpy()
            entry["december_chart_minmax_sanity"] = [float(np.min(pc)), float(np.max(pc))]
        result["models"][name] = entry
        print(name, entry, flush=True)
    for target in ["raw", "log_rpm"]:
        xtrain, cat = features(train, no_quote=True)
        xtest, _ = features(select, no_quote=True)
        nums = [c for c in xtrain if c not in cat]
        medians = xtrain[nums].median()
        xtrain[nums] = xtrain[nums].fillna(medians)
        xtest[nums] = xtest[nums].fillna(medians)
        yt = train.posted_rate if target == "raw" else np.log(train.posted_rate / train.distance)
        params = dict(iterations=700, depth=6, learning_rate=0.055, l2_leaf_reg=5,
                      loss_function="RMSE", random_seed=SEED, thread_count=4,
                      verbose=False, allow_writing_files=False)
        model = CatBoostRegressor(**params)
        model.fit(xtrain, yt, cat_features=cat)
        pred = model.predict(xtest)
        if target == "log_rpm":
            pred = np.exp(pred) * select.distance.to_numpy()
        pred = np.maximum(1.0, pred)
        name = f"catboost_full_noquote_{target}"
        entry = metrics(y, pred)
        entry["params"] = params
        entry["importance_top10"] = sorted(zip(xtrain.columns, model.feature_importances_), key=lambda p: p[1], reverse=True)[:10]
        result["models"][name] = entry
        print(name, {k:v for k,v in entry.items() if k not in ["params", "importance_top10"]}, flush=True)
        (ROOT / "experiments" / "noquote_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    primary_path = ROOT / "experiments" / "results.json"
    combined = json.loads(primary_path.read_text(encoding="utf-8")) if primary_path.exists() else {}
    combined.update(benchmark_metadata())
    combined.setdefault("models", {}).update(result["models"])
    primary_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    return result


def main():
    started = time.time()
    df = pd.read_csv(ROOT / "data" / "train_test.csv")
    # Do not inspect or tune on root agent's untouched final holdout.
    df = df.loc[df.date < "2025-09-01"].copy()
    train = df.loc[df.date < "2025-07-01"].copy()
    select = df.loc[df.date >= "2025-07-01"].copy()
    y = select.posted_rate.to_numpy()
    result = {
        **benchmark_metadata(),
        "scope": "Jan-Jun 2025 training; Jul-Aug 2025 model selection; Sep-Oct excluded",
        "train_rows": len(train), "selection_rows": len(select),
        "seed": SEED, "models": {},
    }
    result["models"]["distance_times_quote_signal"] = metrics(y, select.distance.to_numpy() * select.quote_signal.to_numpy())
    print("baseline", result["models"]["distance_times_quote_signal"], flush=True)
    for reduced in [False, True]:
        xtrain, cat = features(train, reduced, linear=True)
        xtest, _ = features(select, reduced, linear=True)
        nums = [c for c in xtrain if c not in cat]
        proc = ColumnTransformer([
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), nums),
            ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cat),
        ])
        model = Pipeline([("preprocess", proc), ("ridge", Ridge(alpha=20.0, solver="lsqr"))])
        model.fit(xtrain, train.posted_rate)
        pred = np.maximum(1.0, model.predict(xtest))
        name = "ridge_reduced" if reduced else "ridge_full"
        result["models"][name] = metrics(y, pred)
        print(name, result["models"][name], flush=True)
    for reduced, target in [(False, "raw"), (False, "log"), (False, "rpm_residual"), (True, "raw"), (True, "log_rpm")]:
        xtrain, cat = features(train, reduced)
        xtest, _ = features(select, reduced)
        nums = [c for c in xtrain if c not in cat]
        # Each missing-value replacement uses only Jan-Jun training rows.
        medians = xtrain[nums].median()
        xtrain[nums] = xtrain[nums].fillna(medians)
        xtest[nums] = xtest[nums].fillna(medians)
        if target == "raw":
            yt = train.posted_rate.to_numpy()
        elif target == "log":
            yt = np.log(train.posted_rate.to_numpy())
        elif target == "rpm_residual":
            yt = train.posted_rate.to_numpy() / train.distance.to_numpy() - train.quote_signal.to_numpy()
        elif target == "log_rpm":
            yt = np.log(train.posted_rate.to_numpy() / train.distance.to_numpy())
        params = dict(iterations=700, depth=6, learning_rate=0.055, l2_leaf_reg=5,
                      loss_function="RMSE", random_seed=SEED, thread_count=4,
                      verbose=False, allow_writing_files=False)
        model = CatBoostRegressor(**params)
        model.fit(xtrain, yt, cat_features=cat)
        pred = model.predict(xtest)
        if target == "log":
            pred = np.exp(pred)
        elif target == "rpm_residual":
            pred = (pred + select.quote_signal.to_numpy()) * select.distance.to_numpy()
        elif target == "log_rpm":
            pred = np.exp(pred) * select.distance.to_numpy()
        pred = np.maximum(1.0, pred)
        name = f"catboost_{'reduced' if reduced else 'full'}_{target}"
        entry = metrics(y, pred)
        entry["params"] = params
        entry["importance_top10"] = sorted(zip(xtrain.columns, model.feature_importances_), key=lambda p: p[1], reverse=True)[:10]
        # Robust descriptive view supplements the full score, which retains every label.
        high = y > 2.5 * select.distance.to_numpy() * select.quote_signal.to_numpy()
        entry["high_rate_rows"] = int(high.sum())
        entry["ordinary_rows_metrics"] = metrics(y[~high], pred[~high])
        result["models"][name] = entry
        print(name, {k:v for k,v in entry.items() if k not in ["params", "importance_top10"]}, flush=True)
        (ROOT / "experiments" / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["elapsed_seconds"] = time.time() - started
    (ROOT / "experiments" / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import sys
    if "--no-quote" in sys.argv:
        no_quote_benchmark()
    else:
        main()
