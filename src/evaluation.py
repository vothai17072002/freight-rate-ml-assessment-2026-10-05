"""All metrics use actual dollars and retain every observed target."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {"rows": len(y), "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
            "wape": float(np.abs(y - pred).sum() / np.abs(y).sum()),
            "r2": float(r2_score(y, pred)) if len(y) > 1 else None}
