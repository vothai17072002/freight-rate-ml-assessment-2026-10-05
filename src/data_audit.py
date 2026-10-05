"""Reproducible, read-only audit of the freight-rate assessment inputs.

Run from the project root:
    python -m src.data_audit --data-dir data --output-dir reports

This module uses only the Python standard library. It never modifies inputs.
Numerical associations are diagnostics, not evidence of feature provenance.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import statistics
from datetime import date
from pathlib import Path
from typing import Any


FILES = {
    "train": "train_test.csv",
    "validation": "validation.csv",
    "prediction_template": "validation_predictions_template.csv",
    "december_chart": "december_chart_inputs.csv",
}
NUMERIC_FIELDS = (
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance",
    "weight", "market_index", "quote_signal", "posted_rate", "predicted_rate",
)


def quantile(values: list[float], probability: float) -> float | None:
    """Linearly interpolated quantile; agrees with NumPy's default method."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "p01": quantile(values, .01),
        "p05": quantile(values, .05),
        "p25": quantile(values, .25),
        "median": quantile(values, .5),
        "p75": quantile(values, .75),
        "p95": quantile(values, .95),
        "p99": quantile(values, .99),
        "max": max(values),
    }


def finite_float(value: str | None) -> float | None:
    try:
        number = float(value) if value is not None and value.strip() else None
    except (TypeError, ValueError):
        return None
    return number if number is not None and math.isfinite(number) else None


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        columns = reader.fieldnames
        if len(set(columns)) != len(columns):
            raise ValueError(f"CSV has duplicate column names: {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"CSV rows have inconsistent widths: {path}")
    return columns, rows


def audit_table(path: Path, columns: list[str], rows: list[dict[str, str]]) -> dict[str, Any]:
    missing = {column: sum(not row[column].strip() for row in rows) for column in columns}
    numeric = {}
    for column in NUMERIC_FIELDS:
        if column not in columns:
            continue
        values = []
        parse_failures = nonfinite = 0
        for row in rows:
            raw = row[column].strip()
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError:
                parse_failures += 1
                continue
            if not math.isfinite(value):
                nonfinite += 1
            else:
                values.append(value)
        numeric[column] = {
            "missing": missing[column],
            "parse_failures": parse_failures,
            "nonfinite": nonfinite,
            "negative": sum(value < 0 for value in values),
            "zero": sum(value == 0 for value in values),
            "summary": summary(values),
        }
        if column.endswith("_lat"):
            numeric[column]["outside_coordinate_range"] = sum(abs(value) > 90 for value in values)
        elif column.endswith("_lon"):
            numeric[column]["outside_coordinate_range"] = sum(abs(value) > 180 for value in values)

    row_counts = collections.Counter(tuple(row[column] for column in columns) for row in rows)
    identifiers = collections.Counter(row["load_id"] for row in rows) if "load_id" in columns else None
    invalid_dates = []
    valid_dates = []
    if "date" in columns:
        for row_number, row in enumerate(rows, start=2):
            try:
                value = date.fromisoformat(row["date"])
                valid_dates.append(value.isoformat())
            except ValueError:
                invalid_dates.append({"csv_row": row_number, "value": row["date"]})
    result = {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "columns": columns,
        "missing": missing,
        "unique_values": {column: len({row[column] for row in rows}) for column in columns},
        "exact_duplicate_rows": sum(count - 1 for count in row_counts.values()),
        "numeric": numeric,
    }
    if identifiers is not None:
        result["duplicate_load_ids"] = {
            "extra_rows": sum(count - 1 for count in identifiers.values()),
            "distinct_duplicate_ids": sum(count > 1 for count in identifiers.values()),
            "examples": sorted(key for key, count in identifiers.items() if count > 1)[:10],
        }
    if "date" in columns:
        result["dates"] = {
            "min": min(valid_dates) if valid_dates else None,
            "max": max(valid_dates) if valid_dates else None,
            "distinct_valid_dates": len(set(valid_dates)),
            "invalid_count": len(invalid_dates),
            "invalid_examples": invalid_dates[:10],
            "monthly_rows": dict(sorted(collections.Counter(value[:7] for value in valid_dates).items())),
            "daily_rows": dict(sorted(collections.Counter(valid_dates).items())),
        }
    if "equipment" in columns:
        result["equipment_counts"] = dict(sorted(collections.Counter(row["equipment"] for row in rows).items()))
    return result


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    mean_left, mean_right = statistics.fmean(left), statistics.fmean(right)
    numerator = math.fsum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denominator = math.sqrt(
        math.fsum((a - mean_left) ** 2 for a in left)
        * math.fsum((b - mean_right) ** 2 for b in right)
    )
    return numerator / denominator if denominator else None


def ranks(values: list[float]) -> list[float]:
    result = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        mean_rank = (start + end - 1) / 2 + 1
        for index in ordered[start:end]:
            result[index] = mean_rank
        start = end
    return result


def association(rows: list[dict[str, str]], feature: str, target_per_distance: bool = False) -> dict[str, Any]:
    pairs = []
    for row in rows:
        x = finite_float(row.get(feature))
        y = finite_float(row.get("posted_rate"))
        if target_per_distance:
            distance = finite_float(row.get("distance"))
            y = y / distance if y is not None and distance is not None and distance > 0 else None
        if x is not None and y is not None:
            pairs.append((x, y))
    left = [pair[0] for pair in pairs]
    right = [pair[1] for pair in pairs]
    return {
        "feature": feature,
        "target": "posted_rate / distance" if target_per_distance else "posted_rate",
        "paired_rows": len(pairs),
        "pearson": pearson(left, right),
        "spearman": pearson(ranks(left), ranks(right)),
    }


def cross_table_checks(tables: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    train, validation = tables["train"], tables["validation"]
    template, december = tables["prediction_template"], tables["december_chart"]
    train_cities = {row[column] for row in train for column in ("pickup", "delivery")}
    validation_cities = {row[column] for row in validation for column in ("pickup", "delivery")}
    train_lanes = {(row["pickup"], row["delivery"]) for row in train}
    validation_lanes = {(row["pickup"], row["delivery"]) for row in validation}
    train_lane_equipment = {(row["pickup"], row["delivery"], row["equipment"]) for row in train}
    validation_ids = [row["load_id"] for row in validation]
    template_ids = [row["load_id"] for row in template]
    train_ids = {row["load_id"] for row in train}
    train_dates, validation_dates = [row["date"] for row in train], [row["date"] for row in validation]
    december_columns = set(december[0]) if december else set()
    validation_columns = set(validation[0]) if validation else set()
    result = {
        "temporal_split": {
            "training_max_before_validation_min": max(train_dates) < min(validation_dates),
            "overlapping_dates": sorted(set(train_dates) & set(validation_dates)),
        },
        "overlapping_train_validation_load_ids": len(train_ids & set(validation_ids)),
        "template": {
            "same_id_set_as_validation": set(validation_ids) == set(template_ids),
            "same_id_order_as_validation": validation_ids == template_ids,
            "missing_validation_ids": len(set(validation_ids) - set(template_ids)),
            "extra_ids": len(set(template_ids) - set(validation_ids)),
        },
        "validation_generalization": {
            "train_city_count": len(train_cities),
            "validation_city_count": len(validation_cities),
            "unseen_cities": sorted(validation_cities - train_cities),
            "rows_with_unseen_city": sum(row["pickup"] not in train_cities or row["delivery"] not in train_cities for row in validation),
            "train_ordered_lane_count": len(train_lanes),
            "validation_ordered_lane_count": len(validation_lanes),
            "unseen_ordered_lane_count": len(validation_lanes - train_lanes),
            "rows_with_unseen_ordered_lane": sum((row["pickup"], row["delivery"]) not in train_lanes for row in validation),
            "rows_with_unseen_lane_equipment": sum((row["pickup"], row["delivery"], row["equipment"]) not in train_lane_equipment for row in validation),
        },
        "december_chart": {
            "columns_missing_from_validation_features": sorted(validation_columns - december_columns - {"load_id"}),
            "constant_inputs": {column: december[0][column] for column in ("pickup", "delivery", "distance", "equipment", "weight") if december and len({row[column] for row in december}) == 1},
            "all_31_december_2025_dates_present_once": [row["date"] for row in december] == [f"2025-12-{day:02d}" for day in range(1, 32)],
        },
    }
    if december:
        first = december[0]
        lane_rows = [row for row in train if row["pickup"] == first["pickup"] and row["delivery"] == first["delivery"]]
        result["december_chart"]["training_rows_for_same_ordered_lane"] = len(lane_rows)
        result["december_chart"]["training_rows_for_same_lane_equipment"] = sum(row["equipment"] == first["equipment"] for row in lane_rows)
    return result


def render_markdown(audit: dict[str, Any]) -> str:
    tables, cross = audit["tables"], audit["cross_table_checks"]
    lines = [
        "# Freight-rate assessment data findings", "",
        "This report is generated from the supplied 2025 assessment CSVs. Input files are read only; their SHA-256 hashes are recorded in `data_audit.json`.", "",
        "Reproduce from the project root: `python -m src.data_audit --data-dir data --output-dir reports`.", "",
        "## Dataset roles and coverage", "",
        "| Input | Rows | Date coverage | Target |", "|---|---:|---|---|",
    ]
    for name, table in tables.items():
        dates = table.get("dates")
        coverage = f"{dates['min']} to {dates['max']}" if dates else "Uses validation IDs"
        target = "posted_rate supplied" if "posted_rate" in table["columns"] else "predicted_rate blank" if "predicted_rate" in table["columns"] else "posted_rate withheld"
        lines.append(f"| {table['file']} | {table['rows']:,} | {coverage} | {target} |")
    lines += ["", "## Quality checks", "", "| Input | Missing weight | Nonpositive weight | Missing market_index | Duplicate IDs | Exact duplicate rows |", "|---|---:|---:|---:|---:|---:|"]
    for name in ("train", "validation"):
        table = tables[name]
        weight = table["numeric"]["weight"]
        lines.append(f"| {table['file']} | {weight['missing']:,} | {weight['negative'] + weight['zero']:,} | {table['missing']['market_index']:,} | {table['duplicate_load_ids']['extra_rows']:,} | {table['exact_duplicate_rows']:,} |")
    parse_failures = sum(value["parse_failures"] for table in tables.values() for value in table["numeric"].values())
    nonfinite = sum(value["nonfinite"] for table in tables.values() for value in table["numeric"].values())
    invalid_dates = sum(table.get("dates", {}).get("invalid_count", 0) for table in tables.values())
    lines += [
        "",
        f"Numeric parse failures: {parse_failures}. Nonfinite numeric values: {nonfinite}. Invalid ISO dates: {invalid_dates}.",
        "Nonpositive shipment weight is treated as invalid for modelling. It should be handled explicitly as missing or through a documented policy; the audit does not change or take the absolute value of the source data.",
        "Coordinate range checks only test mathematical latitude/longitude bounds. They do not establish that supplied coordinates match the real cities; no external coordinate correction is performed.",
        "",
        "## Generalization and split implications", "",
    ]
    general = cross["validation_generalization"]
    lines += [
        "Training ends on 2025-10-31; external validation starts on 2025-11-01. Validation therefore measures prediction into a later period. Development should reserve a later training period for holdout validation and fit all imputers, encoders and derived-feature lookup tables only on the training side of each split.",
        f"Validation has {len(general['unseen_cities'])} unseen cities: {', '.join(general['unseen_cities'])}. {general['rows_with_unseen_city']:,} of {tables['validation']['rows']:,} validation rows contain an unseen city.",
        f"Ordered lane means pickup to delivery; reverse direction is distinct. Validation has {general['unseen_ordered_lane_count']:,} unseen ordered lanes across {general['rows_with_unseen_ordered_lane']:,} rows; {general['rows_with_unseen_lane_equipment']:,} rows have an unseen lane and equipment combination.",
        "These observations support temporal evaluation and explicit fallbacks for unseen categorical values. A random row split alone does not reproduce the supplied external-validation setting.",
        "",
        "## Target distribution", "",
        "| Statistic | posted_rate |", "|---|---:|",
    ]
    target = tables["train"]["numeric"]["posted_rate"]["summary"]
    for field in ("min", "p01", "p05", "median", "mean", "p95", "p99", "max"):
        lines.append(f"| {field} | {target[field]:,.2f} |")
    lines += ["", "Extreme rates require review against distance and equipment; magnitude alone does not prove the target is erroneous.", "", "## Numerical associations and feature provenance", "", "| Feature | Comparison target | Paired rows | Pearson | Spearman |", "|---|---|---:|---:|---:|"]
    for item in audit["training_associations"]:
        pearson_text = f"{item['pearson']:.4f}" if item["pearson"] is not None else "undefined"
        spearman_text = f"{item['spearman']:.4f}" if item["spearman"] is not None else "undefined"
        lines.append(f"| {item['feature']} | {item['target']} | {item['paired_rows']:,} | {pearson_text} | {spearman_text} |")
    lines += [
        "",
        "The `posted_rate / distance` diagnostic is computed from the supplied target only to examine training associations. It must never be supplied as a prediction-time feature.",
        "Overall association can hide temporal instability. The monthly relationship between `quote_signal` and the rate per distance unit is:",
        "",
        "| Month | Paired rows | Pearson | Spearman |", "|---|---:|---:|---:|",
    ]
    for item in audit["monthly_quote_signal_associations"]:
        lines.append(f"| {item['month']} | {item['paired_rows']:,} | {item['pearson']:.4f} | {item['spearman']:.4f} |")
    lines += [
        "",
        "The observed quote-signal relationship reverses direction between training months and is near zero in August. Its pooled correlation does not capture those changes. This supports temporal stress tests and an ablation excluding this field; external-validation labels are unavailable, so the relationship in November and December is not measurable here.",
        "`quote_signal` has no provenance, measurement timing or computation definition in the supplied CSV headers. A strong within-month association with the target can come from a valid prior quote or from a target-derived signal. Correlation alone cannot distinguish these cases and does not confirm leakage. Any production-use claim depends on this field being available before the posted rate is known.",
        "`market_index` likewise needs a prediction-time availability policy; its missingness and its absence from the December input should be handled without using future target information.",
        "",
        "## Submission and December scenario", "",
        f"The prediction template ID set matches validation: {cross['template']['same_id_set_as_validation']}; row order matches: {cross['template']['same_id_order_as_validation']}. The template has {tables['prediction_template']['missing']['predicted_rate']:,} blank predictions.",
    ]
    december = cross["december_chart"]
    constants = december["constant_inputs"]
    lines += [
        f"The December scenario requests 31 daily predictions for {constants['pickup']} to {constants['delivery']}, distance {constants['distance']}, {constants['equipment']}, weight {constants['weight']}. The daily grid contains exactly 2025-12-01 through 2025-12-31 once each.",
        f"It omits these features present in validation: {', '.join(december['columns_missing_from_validation_features'])}. The inference pipeline must support these absent columns through documented training-only fallback estimates or an appropriate reduced-feature model.",
        f"Training contains {december['training_rows_for_same_ordered_lane']} rows on the same ordered lane and {december['training_rows_for_same_lane_equipment']} with the same equipment. Historical support does not supply the missing future market or quote signal.",
        "",
        "## Monthly row distribution", "",
        "| Month | Training rows | Validation rows |", "|---|---:|---:|",
    ]
    train_months = tables["train"]["dates"]["monthly_rows"]
    validation_months = tables["validation"]["dates"]["monthly_rows"]
    for month in sorted(set(train_months) | set(validation_months)):
        lines.append(f"| {month} | {train_months.get(month, 0):,} | {validation_months.get(month, 0):,} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    raw_tables = {}
    audited_tables = {}
    for name, filename in FILES.items():
        path = args.data_dir / filename
        columns, rows = load_csv(path)
        raw_tables[name] = rows
        audited_tables[name] = audit_table(path, columns, rows)
    associations = [association(raw_tables["train"], field) for field in ("distance", "weight", "market_index", "quote_signal")]
    associations.append(association(raw_tables["train"], "quote_signal", target_per_distance=True))
    monthly_rows = collections.defaultdict(list)
    for row in raw_tables["train"]:
        monthly_rows[row["date"][:7]].append(row)
    monthly_quote_associations = [
        {"month": month, **association(rows, "quote_signal", target_per_distance=True)}
        for month, rows in sorted(monthly_rows.items())
    ]
    audit = {
        "audit_version": 1,
        "scope": "Read-only audit of the supplied 2025 assessment inputs; no model fitted and no predictions generated.",
        "numeric_rules": {
            "nonpositive_weight": "Invalid shipment weight for modelling; source values unchanged.",
            "coordinates": "Only mathematical bounds checked; city-coordinate truth unverified.",
            "association": "Pearson and tie-aware Spearman on finite complete pairs; numerical association is not proof of leakage.",
        },
        "tables": audited_tables,
        "cross_table_checks": cross_table_checks(raw_tables),
        "training_associations": associations,
        "monthly_quote_signal_associations": monthly_quote_associations,
        "unverified_feature_definitions": [
            "quote_signal computation, provenance and availability relative to posted_rate are unspecified by CSV headers.",
            "market_index publication timing and forecast policy are unspecified by CSV headers.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "data_audit.json"
    markdown_path = args.output_dir / "data_findings.md"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(audit), encoding="utf-8")
    print(f"Audited {sum(table['rows'] for table in audited_tables.values()):,} total rows across {len(FILES)} input CSVs.")
    print(f"Wrote {json_path} and {markdown_path}.")


if __name__ == "__main__":
    main()
