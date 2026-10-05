"""Small reusable model wrapper with training-only preprocessing."""
from __future__ import annotations

import numpy as np
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import features

SEED = 42
CAT_PARAMS = dict(iterations=700, depth=6, learning_rate=0.055,
                  l2_leaf_reg=5, loss_function="RMSE", random_seed=SEED,
                  thread_count=4, verbose=False, allow_writing_files=False)


class FreightModel:
    def __init__(self, algorithm="catboost", target="raw", reduced=False, params=None, include_quote=True):
        self.algorithm = algorithm
        self.target = target
        self.reduced = reduced
        self.params = params or {}
        self.include_quote = include_quote

    def fit(self, frame):
        x, cat = features(frame, reduced=self.reduced, linear=self.algorithm == "ridge", include_quote=self.include_quote)
        y = frame.posted_rate.to_numpy(dtype=float)
        if not np.isfinite(y).all() or (y <= 0).any():
            raise ValueError("Training prices must be finite and positive.")
        self.feature_names_ = list(x.columns)
        self.training_rows_ = len(frame)
        if self.algorithm == "ridge":
            if self.target != "raw":
                raise ValueError("The production Ridge baseline supports the raw dollar target only.")
            nums = [c for c in x if c not in cat]
            self.model_ = Pipeline([
                ("preprocess", ColumnTransformer([
                    ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                                           ("scale", StandardScaler())]), nums),
                    ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cat)])),
                ("ridge", Ridge(alpha=self.params.get("alpha", 20.0), solver="lsqr"))])
            self.model_.fit(x, y)
        elif self.algorithm == "catboost":
            nums = [c for c in x if c not in cat]
            self.medians_ = x[nums].median().fillna(0.0)
            x[nums] = x[nums].fillna(self.medians_)
            if self.target == "log":
                y = np.log(y)
            elif self.target == "log_rpm":
                y = np.log(y / frame.distance.to_numpy())
            elif self.target == "rpm_residual":
                if self.reduced or not self.include_quote:
                    raise ValueError("Quote residual needs full inputs.")
                y = y / frame.distance.to_numpy() - frame.quote_signal.to_numpy()
            elif self.target != "raw":
                raise ValueError(f"Unsupported target: {self.target}")
            self.model_ = CatBoostRegressor(**(CAT_PARAMS | self.params))
            self.model_.fit(x, y, cat_features=cat)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        return self

    def predict(self, frame):
        x, _ = features(frame, reduced=self.reduced, linear=self.algorithm == "ridge", include_quote=self.include_quote)
        if list(x.columns) != self.feature_names_:
            raise ValueError("Feature order changed.")
        if self.algorithm == "catboost":
            x[self.medians_.index] = x[self.medians_.index].fillna(self.medians_)
        pred = self.model_.predict(x)
        if self.algorithm == "catboost":
            if self.target == "log":
                pred = np.exp(pred)
            elif self.target == "log_rpm":
                pred = np.exp(pred) * frame.distance.to_numpy()
            elif self.target == "rpm_residual":
                pred = (pred + frame.quote_signal.to_numpy()) * frame.distance.to_numpy()
        if not np.isfinite(pred).all():
            raise ValueError("Model produced non-finite prices.")
        # Only a feasibility floor, never calibrated on final validation labels.
        return np.maximum(1.0, pred)

    def importance(self):
        if self.algorithm != "catboost":
            return []
        return sorted(zip(self.feature_names_, self.model_.feature_importances_.tolist()),
                      key=lambda pair: pair[1], reverse=True)
