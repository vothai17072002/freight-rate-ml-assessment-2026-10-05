"""Deterministic features; never use IDs or labels as predictors."""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date"]
FULL_COLUMNS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "market_index", "quote_signal"]


def features(frame: pd.DataFrame, reduced: bool = False, linear: bool = False, include_quote: bool = True):
    full = [c for c in FULL_COLUMNS if include_quote or c != "quote_signal"]
    required = BASE_COLUMNS + ([] if reduced else full)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing model input columns: {missing}")
    cat = ["pickup", "delivery", "equipment"]
    num = ["distance", "weight"] + ([] if reduced else full)
    x = frame[cat + num].copy()
    for c in cat:
        x[c] = x[c].fillna("__MISSING__").astype(str)
    for c in num:
        x[c] = pd.to_numeric(x[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x["distance"].isna().any() or (x["distance"] <= 0).any():
        raise ValueError("Distance must be finite and strictly positive.")
    x["weight_missing"] = (x.weight.isna() | (x.weight <= 0)).astype(int)
    x["weight"] = x.weight.where(x.weight > 0)
    date = pd.to_datetime(frame.date, errors="raise")
    if date.isna().any():
        raise ValueError("Dates cannot be missing.")
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
    cat.append("lane")
    if not reduced:
        for c in ["market_index"] + (["quote_signal"] if include_quote else []):
            x[c] = x[c].where(x[c] > 0)
        if include_quote:
            x["quote_total"] = x.distance * x.quote_signal
        x["market_missing"] = x.market_index.isna().astype(int)
        x["delta_lat"] = x.delivery_lat - x.pickup_lat
        x["delta_lon"] = x.delivery_lon - x.pickup_lon
    if linear:
        numeric_names = [c for c in x if c not in cat]
        for name in numeric_names:
            if name != "distance":
                x[f"distance_x_{name}"] = x.distance * x[name]
        for kind in ["Dry Van", "Reefer", "Flatbed"]:
            x[f"distance_x_{kind}"] = x.distance * (x.equipment == kind)
            if not reduced and include_quote:
                x[f"quote_total_x_{kind}"] = x.quote_total * (x.equipment == kind)
    return x, cat
