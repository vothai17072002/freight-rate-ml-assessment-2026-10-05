"""Lock model selection before a chronological holdout; refit only after scoring."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import metrics
from .model import FreightModel, CAT_PARAMS, SEED

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "ridge_full_noquote_raw": dict(algorithm="ridge", target="raw", reduced=False, include_quote=False),
    "catboost_full_noquote_raw": dict(algorithm="catboost", target="raw", reduced=False, include_quote=False),
    "catboost_full_noquote_log_rpm": dict(algorithm="catboost", target="log_rpm", reduced=False, include_quote=False),
    "ridge_reduced": dict(algorithm="ridge", target="raw", reduced=True, include_quote=False),
    "catboost_reduced_raw": dict(algorithm="catboost", target="raw", reduced=True, include_quote=False),
    "catboost_reduced_log_rpm": dict(algorithm="catboost", target="log_rpm", reduced=True, include_quote=False),
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def split_description(train, test):
    return {"train_start": str(train.date.min()), "train_end": str(train.date.max()),
            "train_rows": len(train), "validation_start": str(test.date.min()),
            "validation_end": str(test.date.max()), "validation_rows": len(test)}


def baseline(train, frame):
    rpm = train.posted_rate / train.distance
    by_equipment = rpm.groupby(train.equipment).median()
    return frame.distance.to_numpy() * frame.equipment.map(by_equipment).fillna(rpm.median()).to_numpy()


def make_charts(result, holdout, preds, importance, reports):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                        "axes.spines.top": False, "axes.spines.right": False,
                        "figure.facecolor": "white", "axes.facecolor": "white"})
    candidates = sorted(result["candidate_results"], key=lambda x: x["rmse"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    labels = [c["name"].replace("catboost_", "Cat ").replace("ridge_", "Ridge ").replace("_", " ") for c in candidates]
    color = ["#0072B2" if c["eligible"] else "#A7B5BE" for c in candidates]
    for ax, field in zip(axes, ["rmse", "mae"]):
        ax.barh(labels, [c[field] for c in candidates], color=color)
        ax.invert_yaxis()
        ax.set_xlabel(field.upper() + " (USD)")
        ax.grid(axis="x", alpha=0.18)
    fig.suptitle("Model selection: July-August 2025\nBlue: production candidates with quote excluded; gray: comparison only", fontsize=12)
    fig.savefig(reports / "model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    y = holdout.posted_rate.to_numpy()
    axes[0].scatter(y, preds, s=5, alpha=0.2, c="#0072B2", rasterized=True)
    bound = max(y.max(), preds.max())
    axes[0].plot([1, bound], [1, bound], "--", color="#D55E00", linewidth=1)
    axes[0].set(xscale="log", yscale="log", xlabel="Observed price (USD; log scale)", ylabel="Predicted price (USD; log scale)")
    errors = preds - y
    axes[1].hist(errors, bins=80, color="#0072B2", alpha=.8)
    axes[1].axvline(0, color="#D55E00", linestyle="--")
    axes[1].set(xlabel="Prediction minus observed (USD)", ylabel="Loads")
    fig.suptitle("Locked September-October holdout: all labels retained", fontsize=12)
    fig.savefig(reports / "holdout_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if importance:
        top = importance[:12][::-1]
        fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
        ax.barh([x[0] for x in top], [x[1] for x in top], color="#009E73")
        ax.set(xlabel="CatBoost prediction value change importance (%)", title="Final primary model: feature importance")
        fig.savefig(reports / "feature_importance.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def run(data_dir=Path("data"), output_dir=Path("artifacts"), reports_dir=Path("reports"), recompute=False):
    data_dir, output_dir, reports_dir = Path(data_dir), Path(output_dir), Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    result_path = ROOT / "experiments/results.json"
    if recompute or not result_path.exists():
        if data_dir.resolve() != (ROOT / "data").resolve():
            raise ValueError("Benchmark recomputation expects the repository data/ folder.")
        for extra in [[], ["--no-quote"]]:
            subprocess.run([sys.executable, str(ROOT / "experiments/benchmark.py"), *extra], cwd=ROOT, check=True)
    selection = json.loads(result_path.read_text(encoding="utf-8"))
    train_hash = hashlib.sha256((data_dir / "train_test.csv").read_bytes()).hexdigest()
    if selection.get("input_sha256") != train_hash:
        raise ValueError("Recorded selection results do not match train_test.csv. Recompute selection.")
    available = selection["models"]
    for name in SPECS:
        if name not in available:
            raise ValueError(f"Missing candidate {name}; recompute selection.")
    # Quote exclusion is a declared feature policy BEFORE the holdout is scored.
    primary_names = [n for n, s in SPECS.items() if not s["reduced"]]
    reduced_names = [n for n, s in SPECS.items() if s["reduced"]]
    primary_name = min(primary_names, key=lambda n: (available[n]["rmse"], available[n]["mae"]))
    reduced_name = min(reduced_names, key=lambda n: (available[n]["rmse"], available[n]["mae"]))
    lock = {"primary": {"name": primary_name, "spec": SPECS[primary_name]},
            "reduced": {"name": reduced_name, "spec": SPECS[reduced_name]},
            "params": CAT_PARAMS, "input_sha256": train_hash,
            "selection_rule": "Minimize July-August dollar RMSE within each quote-excluded input policy; MAE breaks ties. No September-October tuning."}
    write_json(output_dir / "selected_config.json", lock)
    print("LOCKED", primary_name, reduced_name, flush=True)

    dev = pd.read_csv(data_dir / "train_test.csv")
    if dev.load_id.isna().any() or dev.load_id.duplicated().any():
        raise ValueError("Development IDs must be unique and non-missing.")
    if (dev.date < "2025-01-01").any() or (dev.date > "2025-10-31").any():
        raise ValueError("This assessment's split expects January-October 2025 development rows.")
    sel_train = dev.loc[dev.date < "2025-07-01"].copy()
    sel_test = dev.loc[(dev.date >= "2025-07-01") & (dev.date < "2025-09-01")].copy()
    history = dev.loc[dev.date < "2025-09-01"].copy()
    holdout = dev.loc[dev.date >= "2025-09-01"].copy()
    if min(len(sel_train), len(sel_test), len(history), len(holdout)) == 0:
        raise ValueError("Chronological split has an empty partition.")

    primary = FreightModel(**SPECS[primary_name]).fit(history)
    pred = primary.predict(holdout)
    reduced = FreightModel(**SPECS[reduced_name]).fit(history)
    reduced_pred = reduced.predict(holdout)
    pred_base = baseline(history, holdout)
    holdout_results = {"full": metrics(holdout.posted_rate, pred),
                       "reduced": metrics(holdout.posted_rate, reduced_pred),
                       "baseline": metrics(holdout.posted_rate, pred_base)}
    print("HOLDOUT", json.dumps(holdout_results), flush=True)
    evidence = holdout[["load_id", "date", "equipment", "posted_rate"]].copy()
    evidence["predicted_rate"] = pred
    evidence["reduced_predicted_rate"] = reduced_pred
    evidence["baseline_predicted_rate"] = pred_base
    evidence.to_csv(reports_dir / "holdout_predictions.csv", index=False)
    groups = {"month": holdout.date.str[:7], "equipment": holdout.equipment,
              "weight_quality": pd.Series(np.where(holdout.weight.isna() | (holdout.weight <= 0), "missing_or_invalid", "valid"), index=holdout.index)}
    known = set(history.pickup) | set(history.delivery)
    groups["city_coverage"] = pd.Series(np.where(holdout.pickup.isin(known) & holdout.delivery.isin(known), "seen_cities", "unseen_city"), index=holdout.index)
    subgroup = {}
    for kind, group in groups.items():
        subgroup[kind] = []
        for value in sorted(group.unique()):
            mask = (group == value).to_numpy()
            subgroup[kind].append({"group": str(value), **metrics(holdout.posted_rate.to_numpy()[mask], pred[mask])})

    # Configurations and preprocessing rules remain frozen after holdout evaluation.
    primary = FreightModel(**SPECS[primary_name]).fit(dev)
    reduced = FreightModel(**SPECS[reduced_name]).fit(dev)
    joblib.dump(primary, output_dir / "full_model.joblib", compress=3)
    joblib.dump(reduced, output_dir / "reduced_model.joblib", compress=3)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": {"name": primary_name, "target": primary.target, "params": CAT_PARAMS,
                           "spec": SPECS[primary_name],
                           "feature_policy": "Use shipment, location, calendar and available market_index; exclude load_id, target and unstable quote_signal."},
        "reduced_model": {"name": reduced_name, "target": reduced.target, "params": CAT_PARAMS,
                          "spec": SPECS[reduced_name], "feature_policy": "Only pickup, delivery, distance, equipment, weight and date. No invented future market/quote values or coordinates."},
        "data_summary": {"train_rows": len(dev), "train_start": dev.date.min(), "train_end": dev.date.max(),
                         "selection_train_rows": len(sel_train), "selection_validation_rows": len(sel_test), "holdout_rows": len(holdout)},
        "split": {"selection": split_description(sel_train, sel_test), "holdout": split_description(history, holdout),
                  "final": {"train_rows": len(dev), "train_start": dev.date.min(), "train_end": dev.date.max()},
                  "policy": "Selection uses January-June fit and July-August scores; configurations frozen before September-October evaluation; final refit uses all labeled rows."},
        "candidate_results": [{"name": n, "eligible": n in SPECS, **{k: v for k, v in m.items() if k in ["rmse", "mae", "wape", "r2"]}} for n, m in available.items()],
        "selection_results": {"primary": available[primary_name], "reduced": available[reduced_name]},
        "holdout_results": holdout_results, "subgroup_metrics": subgroup,
        "december": {"model": reduced_name, "feature_policy": "Dedicated six-input model; predictive scenario with unknown realized future market conditions.",
                     "input_rows": 31, "start": "2025-12-01", "end": "2025-12-31",
                     "limits": "No December labels exist locally. Calendar features encode cycles but a tree model does not extrapolate a new time trend. This is not a measured December accuracy claim."},
        "reproducibility": {"random_seed": SEED, "thread_count": 4, "python": sys.version,
                            "source_train_sha256": train_hash,
                            "dependencies": {n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scikit-learn", "catboost", "matplotlib", "joblib", "reportlab"]},
                            "selection_rule": lock["selection_rule"],
                            "model_hashes": {n: hashlib.sha256((output_dir / n).read_bytes()).hexdigest() for n in ["full_model.joblib", "reduced_model.joblib"]}},
    }
    make_charts(result, holdout, pred, primary.importance(), reports_dir)
    pd.DataFrame(primary.importance(), columns=["feature", "importance"]).to_csv(reports_dir / "feature_importance.csv", index=False)
    write_json(output_dir / "metrics.json", result)
    print(f"Saved final models and metrics in {output_dir}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--recompute-selection", action="store_true")
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.reports_dir, args.recompute_selection)


if __name__ == "__main__":
    main()
