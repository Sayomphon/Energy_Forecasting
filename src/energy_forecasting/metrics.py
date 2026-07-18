"""Forecast metrics and slice-based error analysis.

Peak error is computed against a threshold that must come from *training*
data only (docx: "กำหนด peak จาก train เท่านั้น") — the threshold is an input
here, never derived from the evaluation targets themselves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(err**2)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted absolute percentage error: sum|err| / sum|actual|."""
    y_true = np.asarray(y_true)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - np.asarray(y_pred))) / denom)


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean signed error (pred - actual): positive = systematic over-forecast."""
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def peak_mae(y_true: np.ndarray, y_pred: np.ndarray, peak_threshold: float) -> float:
    """MAE restricted to actuals at/above the train-derived peak threshold."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true >= peak_threshold
    if not mask.any():
        return float("nan")
    return mae(y_true[mask], y_pred[mask])


def evaluate_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    peak_threshold: float,
) -> "dict":
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "peak_mae": peak_mae(y_true, y_pred, peak_threshold),
        "n_obs": int(len(np.asarray(y_true))),
        "peak_threshold": float(peak_threshold),
    }


def slice_errors(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_quartiles: "list",
) -> pd.DataFrame:
    """Per-slice MAE/bias: hour of day, weekday/weekend, and load quartile.

    ``train_quartiles`` are the [q25, q50, q75] cut points computed on the
    training target so evaluation slices don't peek at their own labels.
    """
    frame = pd.DataFrame(
        {
            "ts": pd.Series(timestamps).reset_index(drop=True),
            "y": np.asarray(y_true),
            "pred": np.asarray(y_pred),
        }
    )
    frame["abs_err"] = (frame["y"] - frame["pred"]).abs()
    frame["signed_err"] = frame["pred"] - frame["y"]
    frame["hour"] = frame["ts"].dt.hour
    frame["day_type"] = np.where(frame["ts"].dt.dayofweek >= 5, "weekend", "weekday")
    q25, q50, q75 = train_quartiles
    frame["load_quartile"] = pd.cut(
        frame["y"],
        bins=[-np.inf, q25, q50, q75, np.inf],
        labels=["q1_low", "q2", "q3", "q4_peak"],
    )

    out = []
    for dim in ("hour", "day_type", "load_quartile"):
        grouped = (
            frame.groupby(dim, observed=True)
            .agg(mae=("abs_err", "mean"), bias=("signed_err", "mean"), n=("y", "size"))
            .reset_index()
            .rename(columns={dim: "slice_value"})
        )
        grouped.insert(0, "slice_dim", dim)
        out.append(grouped)
    return pd.concat(out, ignore_index=True)


def residual_summary(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    peak_threshold: float,
    max_acf_lag: int = 12,
) -> "dict":
    """Compact residual diagnostics for residual_summary.json."""
    resid = np.asarray(y_true) - np.asarray(y_pred)
    resid_s = pd.Series(resid)
    acf = {
        f"lag_{k}": float(resid_s.autocorr(lag=k)) for k in (1, 2, 3, 6, max_acf_lag)
    }
    return {
        "n_obs": int(len(resid)),
        "period_start": str(pd.Series(timestamps).min()),
        "period_end": str(pd.Series(timestamps).max()),
        "mean_residual": float(np.mean(resid)),
        "std_residual": float(np.std(resid)),
        "metrics": evaluate_all(y_true, y_pred, peak_threshold),
        "residual_autocorrelation": acf,
    }
