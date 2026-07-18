"""Shared fixtures: a synthetic 10-minute series with a known daily pattern.

Synthetic data lets every temporal test assert exact values instead of
eyeballing real-data statistics.
"""

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.config import ForecastConfig


@pytest.fixture()
def cfg() -> ForecastConfig:
    return ForecastConfig()


@pytest.fixture()
def raw_frame() -> pd.DataFrame:
    """14 days of 10-minute readings (2016 rows) with sensors and controls."""
    rng = np.random.default_rng(0)
    n = 14 * 144
    ts = pd.date_range("2024-01-01 00:00", periods=n, freq="10min")
    hour = ts.hour + ts.minute / 60.0
    daily = 60 + 40 * np.sin(2 * np.pi * (hour - 7) / 24.0) ** 2
    load = daily + rng.normal(0, 5, n)
    return pd.DataFrame(
        {
            "date": ts,
            "Appliances": np.clip(load, 10, None).round(1),
            "lights": rng.integers(0, 40, n).astype(float),
            "T1": 20 + rng.normal(0, 1, n),
            "RH_1": 40 + rng.normal(0, 3, n),
            "T_out": 5 + rng.normal(0, 2, n),
            "RH_out": 80 + rng.normal(0, 5, n),
            "rv1": rng.uniform(0, 50, n),
            "rv2": rng.uniform(0, 50, n),
        }
    )
