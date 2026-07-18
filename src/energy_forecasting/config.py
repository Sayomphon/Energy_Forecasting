"""Central configuration for the forecasting pipeline.

Every temporal assumption (horizon, lags, windows, sampling interval) lives
here so that training and inference are guaranteed to share one definition.
The config is frozen: experiments that change it must create a new instance,
which keeps run lineage explicit.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ForecastConfig:
    """Temporal and modelling configuration.

    Attributes:
        horizon_steps: Forecast horizon in sampling steps (6 x 10 min = 1 hour).
        sampling_minutes: Expected spacing between consecutive records.
        lags: Past-only lag steps for the target series. 144 = same time 24h ago.
        rolling_windows: Window sizes (in steps) for shift(1)-then-rolling stats.
        seed: Random seed for model fitting. Splits are deterministic from
            timestamps and never depend on this seed.
        timestamp_col: Name of the datetime column in the raw data.
        target_source_col: Raw column the forecast target is derived from.
        target_col: Name of the derived label column (value at t + horizon).
        train_frac / val_frac: Chronological split fractions; test gets the rest.
        n_backtest_folds: Expanding-window folds inside the train+val region.
        peak_quantile: Train-only quantile that defines "peak load" for peak MAE.
        max_staleness_minutes: At inference, history older than this triggers
            the fallback path instead of a model prediction.
    """

    horizon_steps: int = 6
    sampling_minutes: int = 10
    lags: tuple = (1, 6, 12, 144)
    rolling_windows: tuple = (6, 36, 144)
    seed: int = 42
    timestamp_col: str = "date"
    target_source_col: str = "Appliances"
    target_col: str = "target_1h"
    train_frac: float = 0.70
    val_frac: float = 0.15
    n_backtest_folds: int = 3
    peak_quantile: float = 0.90
    max_staleness_minutes: int = 30
    model_version: str = "energy-1h-v1"

    @property
    def horizon_minutes(self) -> int:
        return self.horizon_steps * self.sampling_minutes

    @property
    def min_history_steps(self) -> int:
        """Steps of history required before the first feature row is valid.

        Rolling windows consume ``window`` values *after* a shift(1), so they
        need ``window + 1`` steps; lags need ``lag`` steps.
        """
        return max(max(self.lags), max(self.rolling_windows) + 1)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save_json(self, path: Path, extra: dict = None) -> None:
        """Persist the config (plus optional metadata) as feature_config.json."""
        payload = self.to_dict()
        payload["horizon_minutes"] = self.horizon_minutes
        payload["min_history_steps"] = self.min_history_steps
        if extra:
            payload.update(extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    @classmethod
    def from_json(cls, path: Path) -> "ForecastConfig":
        raw = json.loads(Path(path).read_text())
        field_names = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in raw.items() if k in field_names}
        for tuple_field in ("lags", "rolling_windows"):
            if tuple_field in kwargs:
                kwargs[tuple_field] = tuple(kwargs[tuple_field])
        return cls(**kwargs)


# Repository-relative default paths. Callers may override; nothing below is
# created implicitly except by explicit save/fetch calls.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

UCI_DATASET_ID = 374
UCI_DATASET_NAME = "Appliances Energy Prediction"
UCI_DATASET_URL = "https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction"
UCI_LICENSE = "CC BY 4.0"
