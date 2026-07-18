"""Packaged inference: bundle persistence and the forecast contract.

Security notes
--------------
- Bundles are joblib pickles. Loading a pickle executes code, so
  ``load_bundle`` verifies a sha256 recorded at save time before
  deserialising, and refuses files without a matching sidecar unless the
  caller explicitly opts out. Only load bundles you produced.
- ``predict_one`` validates its inputs (history schema, timezone-aware
  ordering, freshness) and degrades to a documented fallback instead of
  silently feeding NaN into the model.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from energy_forecasting.config import ForecastConfig
from energy_forecasting.data import parse_timestamps
from energy_forecasting.features import add_features

logger = logging.getLogger(__name__)

BUNDLE_NAME = "forecast_bundle.joblib"
CHECKSUM_SUFFIX = ".sha256"


@dataclass
class ForecastBundle:
    """Everything inference needs, saved and loaded as one unit."""

    model: object
    feature_columns: "list"
    config: ForecastConfig
    model_version: str
    trained_at: str  # ISO timestamp recorded by the training run
    train_data_sha256: str = ""
    metrics: "dict" = None


def _sha256_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_bundle(bundle: ForecastBundle, artifacts_dir: Path) -> Path:
    """Persist the bundle plus a sha256 sidecar for integrity verification."""
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / BUNDLE_NAME
    joblib.dump(
        {
            "model": bundle.model,
            "feature_columns": list(bundle.feature_columns),
            "config": bundle.config.to_dict(),
            "model_version": bundle.model_version,
            "trained_at": bundle.trained_at,
            "train_data_sha256": bundle.train_data_sha256,
            "metrics": bundle.metrics or {},
        },
        path,
    )
    digest = _sha256_bytes(path)
    path.with_suffix(path.suffix + CHECKSUM_SUFFIX).write_text(digest + "\n")
    logger.info("Saved bundle to %s (sha256=%s)", path, digest[:12])
    return path


def load_bundle(path: Path, allow_unverified: bool = False) -> ForecastBundle:
    """Load a bundle after verifying its sha256 sidecar.

    Raises if the sidecar is missing/mismatched, unless ``allow_unverified``
    is explicitly set (never do this for files from an untrusted source —
    joblib deserialisation can execute arbitrary code).
    """
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + CHECKSUM_SUFFIX)
    if sidecar.exists():
        expected = sidecar.read_text().strip()
        actual = _sha256_bytes(path)
        if actual != expected:
            raise ValueError(
                f"Bundle checksum mismatch for {path}: expected {expected}, got {actual}. "
                "Refusing to deserialise a modified artifact."
            )
    elif not allow_unverified:
        raise FileNotFoundError(
            f"No checksum sidecar next to {path}. Pass allow_unverified=True only "
            "if you fully trust this file."
        )
    raw = joblib.load(path)  # noqa: S301 - integrity verified above
    return ForecastBundle(
        model=raw["model"],
        feature_columns=raw["feature_columns"],
        config=ForecastConfig(**{
            k: tuple(v) if k in ("lags", "rolling_windows") else v
            for k, v in raw["config"].items()
        }),
        model_version=raw["model_version"],
        trained_at=raw["trained_at"],
        train_data_sha256=raw.get("train_data_sha256", ""),
        metrics=raw.get("metrics", {}),
    )


def predict_one(
    bundle: ForecastBundle,
    history: pd.DataFrame,
    prediction_time: "pd.Timestamp | str",
) -> "dict":
    """Produce one forecast following the inference contract (docx section 9).

    Args:
        bundle: A loaded :class:`ForecastBundle`.
        history: Raw readings (same schema as training data) with timestamps
            at or before ``prediction_time``. Needs ``config.min_history_steps``
            rows for a model prediction; less history triggers the fallback.
        prediction_time: The "now" the forecast is issued at (time t).

    Returns:
        Contract dict: prediction_time, target_time, forecast value,
        quality_flag, fallback_used, model_version.
    """
    cfg = bundle.config
    # Parse through the same helper as the rest of the pipeline so a caller may
    # pass a Timestamp, an ISO string, or the source's no-space format — timestamp
    # parsing lives in exactly one place.
    t = parse_timestamps(pd.Series([prediction_time])).iloc[0]
    if pd.isna(t):
        raise ValueError(f"Unparseable prediction_time: {prediction_time!r}")

    def _contract(value: float, flag: str, fallback: bool) -> "dict":
        return {
            "prediction_time": t.isoformat(),
            "target_time": (t + pd.Timedelta(minutes=cfg.horizon_minutes)).isoformat(),
            "forecast_appliances_wh": round(float(value), 2),
            "quality_flag": flag,
            "fallback_used": fallback,
            "model_version": bundle.model_version,
        }

    required = {cfg.timestamp_col, cfg.target_source_col}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"History is missing required columns: {sorted(missing)}")

    hist = history.copy()
    hist[cfg.timestamp_col] = parse_timestamps(hist[cfg.timestamp_col])
    hist = (
        hist[hist[cfg.timestamp_col] <= t]  # hard guard: nothing after t
        .sort_values(cfg.timestamp_col)
        .drop_duplicates(subset=cfg.timestamp_col, keep="last")
        .reset_index(drop=True)
    )
    if hist.empty:
        raise ValueError("No history at or before prediction_time.")

    last_ts = hist[cfg.timestamp_col].iloc[-1]
    last_value = float(hist[cfg.target_source_col].iloc[-1])
    staleness_min = (t - last_ts).total_seconds() / 60.0

    # Degraded paths: stale feed or not enough history for the lag features.
    if staleness_min > cfg.max_staleness_minutes:
        logger.warning("History stale by %.0f min; using last-value fallback", staleness_min)
        return _contract(last_value, "stale_history_fallback", True)
    if len(hist) < cfg.min_history_steps:
        logger.warning(
            "Only %d rows of history (<%d); using last-value fallback",
            len(hist), cfg.min_history_steps,
        )
        return _contract(last_value, "insufficient_history_fallback", True)

    feat, _ = add_features(hist, cfg)
    if feat.empty:
        return _contract(last_value, "insufficient_history_fallback", True)
    row = feat.iloc[[-1]].reindex(columns=bundle.feature_columns)
    if row.isna().any(axis=None):
        # A sensor column present in training is missing/NaN now.
        return _contract(last_value, "missing_features_fallback", True)

    value = float(bundle.model.predict(row)[0])
    return _contract(value, "ok", False)
