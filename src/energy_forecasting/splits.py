"""Chronological splitting and expanding-window backtesting.

No shuffling anywhere. Splits are positional over a time-sorted frame, so
they are deterministic from the data alone (never from a random seed), and
each fold's train block strictly precedes its validation block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from energy_forecasting.config import ForecastConfig


@dataclass(frozen=True)
class SplitIndices:
    """Positional index arrays for one chronological train/val/test split."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def chronological_split(n_rows: int, cfg: ForecastConfig) -> SplitIndices:
    """Split ``[0, n_rows)`` into contiguous train / validation / test blocks."""
    if n_rows < 10:
        raise ValueError(f"Not enough rows to split: {n_rows}")
    train_end = int(n_rows * cfg.train_frac)
    val_end = int(n_rows * (cfg.train_frac + cfg.val_frac))
    return SplitIndices(
        train=np.arange(0, train_end),
        val=np.arange(train_end, val_end),
        test=np.arange(val_end, n_rows),
    )


def expanding_window_folds(n_rows_train_val: int, cfg: ForecastConfig) -> list:
    """Expanding-window folds inside the train+val region.

    The region is cut into ``n_backtest_folds + 1`` equal blocks: fold i
    trains on blocks ``[0..i]`` and validates on block ``i + 1``. Training
    history grows while validation always lies strictly in the future,
    mirroring how the model would be retrained over time.

    Returns a list of ``(train_idx, val_idx)`` positional array pairs.
    """
    k = cfg.n_backtest_folds
    if k < 1:
        raise ValueError("n_backtest_folds must be >= 1")
    block = n_rows_train_val // (k + 1)
    if block < 1:
        raise ValueError(f"Region of {n_rows_train_val} rows too small for {k} folds")
    folds = []
    for i in range(1, k + 1):
        train_idx = np.arange(0, i * block)
        val_end = (i + 1) * block if i < k else n_rows_train_val
        val_idx = np.arange(i * block, val_end)
        folds.append((train_idx, val_idx))
    return folds


def assert_temporal_order(
    timestamps: pd.Series,
    split: SplitIndices,
) -> None:
    """Guard invariant: max(train time) < min(val time) < min(test time)."""
    ts = pd.Series(timestamps).reset_index(drop=True)
    t_train_max = ts.iloc[split.train].max()
    t_val_min = ts.iloc[split.val].min()
    t_val_max = ts.iloc[split.val].max()
    t_test_min = ts.iloc[split.test].min()
    if not (t_train_max < t_val_min):
        raise AssertionError(f"Train leaks into validation: {t_train_max} >= {t_val_min}")
    if not (t_val_max < t_test_min):
        raise AssertionError(f"Validation leaks into test: {t_val_max} >= {t_test_min}")


def backtest_calendar(
    timestamps: pd.Series,
    folds: list,
) -> pd.DataFrame:
    """Summarise each fold's coverage for reporting (docx section 07)."""
    ts = pd.Series(timestamps).reset_index(drop=True)
    rows = []
    for i, (train_idx, val_idx) in enumerate(folds):
        rows.append(
            {
                "fold": i,
                "train_start": ts.iloc[train_idx[0]],
                "train_end": ts.iloc[train_idx[-1]],
                "val_start": ts.iloc[val_idx[0]],
                "val_end": ts.iloc[val_idx[-1]],
                "n_train": len(train_idx),
                "n_val": len(val_idx),
            }
        )
    return pd.DataFrame(rows)
