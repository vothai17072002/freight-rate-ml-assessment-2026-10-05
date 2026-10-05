"""One command for audit, training, predictions, official scorer and report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .predict import run as predict
from .train import run as train, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-training", action="store_true", help="Use the trusted supplied trained model artifacts.")
    parser.add_argument("--recompute-selection", action="store_true", help="Rerun the entire bounded selection benchmark before training.")
    args = parser.parse_args()
    if args.skip_training and args.recompute_selection:
        parser.error("--skip-training and --recompute-selection cannot be combined")
    subprocess.run([sys.executable, "-m", "src.data_audit", "--data-dir", "data", "--output-dir", "reports"], cwd=ROOT, check=True)
    if not args.skip_training:
        train(ROOT / "data", ROOT / "artifacts", ROOT / "reports", args.recompute_selection)
    output, december = predict(ROOT / "data", ROOT / "artifacts", ROOT / "validation_predictions.csv", ROOT / "submission/december_chart_predictions.csv")
    metrics_path = ROOT / "artifacts/metrics.json"
    results = json.loads(metrics_path.read_text(encoding="utf-8"))
    results["validation_output"] = {"rows": len(output), "columns": list(output.columns),
                                    "minimum": float(output.predicted_rate.min()), "maximum": float(output.predicted_rate.max()),
                                    "mean": float(output.predicted_rate.mean()),
                                    "file": "validation_predictions.csv", "ground_truth_available": False}
    results["december"].update({"minimum": float(december.predicted_rate.min()), "maximum": float(december.predicted_rate.max()),
                              "mean": float(december.predicted_rate.mean()), "file": "submission/december_chart_predictions.csv"})
    stress_path = ROOT / "reports/cold_city_stress.json"
    if stress_path.exists():
        results["cold_city_stress"] = json.loads(stress_path.read_text(encoding="utf-8"))
    write_json(metrics_path, results)
    scorer = subprocess.run([sys.executable, "score.py", "--predictions", "validation_predictions.csv",
                            "--december-predictions", "submission/december_chart_predictions.csv",
                            "--output-dir", "reports/scorer_results"], cwd=ROOT, check=True,
                           text=True, capture_output=True)
    (ROOT / "reports/scorer_output.txt").write_text(scorer.stdout, encoding="utf-8")
    print(scorer.stdout, end="")
    subprocess.run([sys.executable, "-m", "src.build_report", "--metrics", "artifacts/metrics.json",
                    "--audit", "reports/data_audit.json", "--output", "reports/freight_rate_report.pdf",
                    "--render-dir", "reports/rendered", "--qa-json", "reports/pdf_qa.json"], cwd=ROOT, check=True)
    print("Completed predictions, official chart and report. Loom recording remains a candidate action.")


if __name__ == "__main__":
    main()
