"""Post-selection cold-city stress on January-August; never tune on this result.

Eight cities are selected without consulting their targets, by ascending
SHA-256(city name). Every January-June row touching one is removed before fit.
Only July-August rows touching those cities are scored. This controlled stress
does not measure accuracy for the actual November-December unseen cities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import pandas as pd

from src.evaluation import metrics
from src.model import FreightModel


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(data_dir: Path, config_path: Path, output_path: Path) -> dict:
    input_path = data_dir / "train_test.csv"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_hash = digest(input_path)
    if input_hash != config["input_sha256"]:
        raise ValueError("Frozen model selection and stress input hashes differ.")
    spec = dict(config["primary"]["spec"])
    if spec.get("reduced") or spec.get("include_quote", True):
        raise ValueError("Stress expects the selected full-input quote-excluded model.")
    params = dict(config["params"])
    params["thread_count"] = min(int(params.get("thread_count", 4)), 4)
    frame = pd.read_csv(input_path)
    january_june = frame.loc[(frame.date >= "2025-01-01") & (frame.date < "2025-07-01")].copy()
    july_august = frame.loc[(frame.date >= "2025-07-01") & (frame.date < "2025-09-01")].copy()
    candidate_cities = set(january_june.pickup) | set(january_june.delivery)
    if len(candidate_cities) < 8:
        raise ValueError("At least eight training cities are required.")
    heldout_cities = sorted(
        candidate_cities,
        key=lambda city: (hashlib.sha256(city.encode("utf-8")).hexdigest(), city),
    )[:8]
    excluded = january_june.pickup.isin(heldout_cities) | january_june.delivery.isin(heldout_cities)
    training = january_june.loc[~excluded].copy()
    stress = july_august.loc[
        july_august.pickup.isin(heldout_cities) | july_august.delivery.isin(heldout_cities)
    ].copy()
    if training.empty or stress.empty:
        raise ValueError("Cold-city stress contains an empty fit or scoring partition.")
    assert not ((set(training.pickup) | set(training.delivery)) & set(heldout_cities))
    assert training.date.max() < stress.date.min()
    start = time.perf_counter()
    model = FreightModel(**spec, params=params).fit(training)
    predictions = model.predict(stress)
    score = metrics(stress.posted_rate, predictions)
    report = {
        "description": "Post-selection controlled cold-city stress. No model or feature selection is performed using these scores.",
        "limits": "This measures July-August errors for eight deliberately excluded January-June cities. It is not actual November-December unseen-city accuracy and does not score September-October.",
        "input_sha256": input_hash,
        "selected_config_sha256": digest(config_path),
        "city_selection_rule": "Ascending SHA256 of UTF-8 city name, with name as tie-breaker; take the first eight cities in the January-June pickup/delivery union. No labels used to choose cities.",
        "heldout_cities_in_hash_order": heldout_cities,
        "candidate_city_count": len(candidate_cities),
        "model": {"name": config["primary"]["name"], "spec": spec, "params": params},
        "training": {
            "original_january_june_rows": len(january_june),
            "excluded_rows_touching_heldout_cities": int(excluded.sum()),
            "retained_rows": len(training),
            "date_min": training.date.min(),
            "date_max": training.date.max(),
            "retained_city_count": len(set(training.pickup) | set(training.delivery)),
            "heldout_city_overlap": 0,
        },
        "stress": {
            "original_july_august_rows": len(july_august),
            "rows_touching_heldout_cities": len(stress),
            "date_min": stress.date.min(),
            "date_max": stress.date.max(),
            "rows_with_both_endpoints_heldout": int((stress.pickup.isin(heldout_cities) & stress.delivery.isin(heldout_cities)).sum()),
        },
        "metrics": score,
        "preprocessing_policy": "Feature generation and numerical imputation use src.model.FreightModel; imputation medians are fitted on retained January-June training rows only. quote_signal is excluded.",
        "elapsed_seconds": round(time.perf_counter() - start, 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"heldout_cities": heldout_cities, "training_rows": len(training), "stress_rows": len(stress), "metrics": score}), flush=True)
    print(f"Wrote {output_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("artifacts/selected_config.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/cold_city_stress.json"))
    args = parser.parse_args()
    run(args.data_dir, args.config, args.output)


if __name__ == "__main__":
    main()
