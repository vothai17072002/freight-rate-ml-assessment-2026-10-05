"""Predict held-out loads and the fixed December input, preserving input IDs."""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def run(data_dir=Path("data"), model_dir=Path("artifacts"),
        output=Path("validation_predictions.csv"),
        december_output=Path("submission/december_chart_predictions.csv")):
    data_dir, model_dir = Path(data_dir), Path(model_dir)
    validation = pd.read_csv(data_dir / "validation.csv")
    template = pd.read_csv(data_dir / "validation_predictions_template.csv")
    if validation.load_id.isna().any() or validation.load_id.duplicated().any():
        raise ValueError("Validation IDs must be unique and non-missing.")
    if template.load_id.isna().any() or template.load_id.duplicated().any():
        raise ValueError("Template IDs must be unique and non-missing.")
    if set(template.load_id) != set(validation.load_id):
        raise ValueError("Template IDs do not match validation IDs.")
    # Load only trusted artifacts produced by this repository's training code.
    model = joblib.load(model_dir / "full_model.joblib")
    reduced = joblib.load(model_dir / "reduced_model.joblib")
    prices = pd.Series(model.predict(validation), index=validation.load_id)
    result = template[["load_id"]].copy()
    result["predicted_rate"] = result.load_id.map(prices).round(2)
    if not np.isfinite(result.predicted_rate).all() or (result.predicted_rate <= 0).any():
        raise ValueError("Invalid final prices.")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    december = pd.read_csv(data_dir / "december_chart_inputs.csv")
    december["predicted_rate"] = np.round(reduced.predict(december), 2)
    december_output = Path(december_output)
    december_output.parent.mkdir(parents=True, exist_ok=True)
    december.to_csv(december_output, index=False)
    print(f"Wrote {len(result):,} predictions: {output}")
    print(f"Wrote {len(december)} December predictions: {december_output}")
    return result, december


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("validation_predictions.csv"))
    parser.add_argument("--december-output", type=Path, default=Path("submission/december_chart_predictions.csv"))
    args = parser.parse_args()
    run(args.data_dir, args.model_dir, args.output, args.december_output)


if __name__ == "__main__":
    main()
