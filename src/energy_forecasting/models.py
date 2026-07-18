"""Model zoo: naive baselines plus the two candidate learners.

The baselines are first-class sklearn-style estimators so they run through
the same backtest loop as real models — a candidate only earns selection by
beating them consistently across folds (docx model selection rule).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energy_forecasting.config import ForecastConfig


class ColumnEchoBaseline(BaseEstimator, RegressorMixin):
    """Predicts a single feature column verbatim.

    ``column='load_now'`` is the last-value baseline (persistence);
    ``column='seasonal_ref'`` is the seasonal naive (same time yesterday,
    aligned to the target timestamp). ``fallback_column`` covers rows where
    the primary column is missing.
    """

    def __init__(self, column: str, fallback_column: str = "load_now"):
        self.column = column
        self.fallback_column = fallback_column

    def fit(self, X: pd.DataFrame, y=None):  # noqa: ARG002 - baselines learn nothing
        if self.column not in X.columns:
            raise ValueError(f"Baseline column {self.column!r} not in features")
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = X[self.column].astype(float)
        if self.fallback_column in X.columns and self.fallback_column != self.column:
            pred = pred.fillna(X[self.fallback_column].astype(float))
        return pred.to_numpy()


def make_last_value() -> ColumnEchoBaseline:
    return ColumnEchoBaseline(column="load_now")


def make_seasonal_naive() -> ColumnEchoBaseline:
    return ColumnEchoBaseline(column="seasonal_ref", fallback_column="load_now")


def make_ridge(cfg: ForecastConfig) -> Pipeline:
    """Interpretable linear baseline; scaling is fit inside the fold only."""
    _ = cfg  # Ridge itself is deterministic; kept for a uniform factory signature
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def make_hgb(cfg: ForecastConfig) -> HistGradientBoostingRegressor:
    """Primary nonlinear candidate: captures interactions at CPU-friendly cost."""
    return HistGradientBoostingRegressor(
        loss="absolute_error",  # aligned with MAE-centric evaluation
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=False,  # folds are small; keep runs deterministic
        random_state=cfg.seed,
    )


def model_factories(cfg: ForecastConfig) -> "dict":
    """Ordered registry used by the backtest runner (baselines first)."""
    return {
        "last_value": lambda: make_last_value(),
        "seasonal_naive": lambda: make_seasonal_naive(),
        "ridge": lambda: make_ridge(cfg),
        "hgb": lambda: make_hgb(cfg),
    }
