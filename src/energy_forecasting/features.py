"""Leakage-safe feature engineering.

Every feature built here answers one question explicitly: *what is known at
prediction time t?* Rules enforced throughout (docx sections 3 and 6):

- Readings stamped at t are known at t (``load_now``, sensors, weather).
- ``lag_k`` uses the value at t - k steps.
- Rolling stats always ``shift(1)`` **before** aggregating, so the window is
  strictly in the past — never centered, never including t's own future.
- The seasonal reference is aligned to the *target* time: for a forecast of
  t + horizon, "same time yesterday" is t + horizon - 24h, which is still in
  the past when horizon < 24h.
- Calendar features derive from t only.

Training and inference share this exact module — the inference bundle calls
``add_features`` on raw history, eliminating training-serving skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_forecasting.config import ForecastConfig

# Raw sensor columns that are observed at time t and may be used as-is.
# rv1/rv2 are random noise kept deliberately as negative controls: if a model
# ranks them highly important, it is fitting noise.
CONTEMPORANEOUS_EXCLUDE = ("date",)


def steps_per_day(cfg: ForecastConfig) -> int:
    return (24 * 60) // cfg.sampling_minutes


def add_features(df: pd.DataFrame, cfg: ForecastConfig) -> tuple[pd.DataFrame, list]:
    """Attach all model features to a time-sorted frame.

    Args:
        df: Output of :func:`energy_forecasting.data.prepare_base` (sorted,
            de-duplicated, may or may not contain the label column).
        cfg: Temporal configuration shared with inference.

    Returns:
        ``(frame, feature_cols)`` where ``frame`` keeps timestamp/label columns
        and rows lacking full history are dropped, and ``feature_cols`` is the
        deterministic ordered list used by both training and inference.
    """
    if not df[cfg.timestamp_col].is_monotonic_increasing:
        raise ValueError("Frame must be sorted by timestamp before building features.")

    out = df.copy()
    src = out[cfg.target_source_col]
    feature_cols = []

    # --- Recent load -------------------------------------------------------
    out["load_now"] = src  # reading stamped at t: known at t
    feature_cols.append("load_now")
    for k in cfg.lags:
        col = f"lag_{k}"
        out[col] = src.shift(k)
        feature_cols.append(col)

    # Seasonal reference aligned to the target time (t + horizon - 24h).
    seasonal_shift = steps_per_day(cfg) - cfg.horizon_steps
    out["seasonal_ref"] = src.shift(seasonal_shift)
    feature_cols.append("seasonal_ref")

    # --- Past-only rolling stats ------------------------------------------
    past = src.shift(1)  # strictly before t
    for w in cfg.rolling_windows:
        rolled = past.rolling(w, min_periods=w)
        for stat in ("mean", "std", "min", "max"):
            col = f"roll_{stat}_{w}"
            out[col] = getattr(rolled, stat)()
            feature_cols.append(col)

    # --- Calendar ----------------------------------------------------------
    ts = out[cfg.timestamp_col]
    hour = ts.dt.hour + ts.dt.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    dow = ts.dt.dayofweek
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["is_weekend"] = (dow >= 5).astype(int)
    feature_cols += ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]

    # --- Indoor/outdoor state observed at t --------------------------------
    indoor_t = [c for c in out.columns if c.startswith("T") and c[1:].isdigit()]
    indoor_rh = [c for c in out.columns if c.startswith("RH_") and c[3:].isdigit()]
    if indoor_t:
        out["indoor_T_mean"] = out[indoor_t].mean(axis=1)
        feature_cols.append("indoor_T_mean")
        if "T_out" in out.columns:
            out["diff_T_in_out"] = out["indoor_T_mean"] - out["T_out"]
            feature_cols.append("diff_T_in_out")
    if indoor_rh:
        out["indoor_RH_mean"] = out[indoor_rh].mean(axis=1)
        feature_cols.append("indoor_RH_mean")
        if "RH_out" in out.columns:
            out["diff_RH_in_out"] = out["indoor_RH_mean"] - out["RH_out"]
            feature_cols.append("diff_RH_in_out")

    # Raw contemporaneous sensors (weather, per-room readings, lights,
    # negative controls). The raw target column is excluded: it already
    # enters as load_now.
    skip = set(feature_cols) | {cfg.target_source_col, cfg.target_col, *CONTEMPORANEOUS_EXCLUDE}
    for col in out.columns:
        if col in skip or col == cfg.timestamp_col:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            feature_cols.append(col)

    # --- Drop rows lacking full history ------------------------------------
    lag_like = [c for c in feature_cols if c.startswith(("lag_", "roll_", "seasonal_"))]
    before = len(out)
    out = out.dropna(subset=lag_like).reset_index(drop=True)
    out.attrs["rows_dropped_insufficient_history"] = before - len(out)

    return out, feature_cols


def availability_audit(feature_cols: list, cfg: ForecastConfig) -> pd.DataFrame:
    """Human-readable audit: which timestamps feed each feature.

    Evidence table for the leakage-analysis notebook section — every feature
    must map to data at or before t.
    """
    rows = []
    for col in feature_cols:
        if col == "load_now":
            window = "t"
        elif col.startswith("lag_"):
            k = int(col.split("_")[1])
            window = f"t - {k * cfg.sampling_minutes} min"
        elif col == "seasonal_ref":
            shift = steps_per_day(cfg) - cfg.horizon_steps
            window = f"t - {shift * cfg.sampling_minutes} min (target time - 24h)"
        elif col.startswith("roll_"):
            w = int(col.split("_")[-1])
            window = f"[t - {w * cfg.sampling_minutes} min, t - {cfg.sampling_minutes} min]"
        elif col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"):
            window = "t (calendar only)"
        else:
            window = "t (sensor reading stamped at t)"
        rows.append({"feature": col, "data_used_from": window, "uses_future": False})
    return pd.DataFrame(rows)
