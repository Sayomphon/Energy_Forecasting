"""Split-ordering invariants and metric correctness."""

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.config import ForecastConfig
from energy_forecasting.metrics import (
    bias,
    evaluate_all,
    mae,
    peak_mae,
    rmse,
    slice_errors,
    wape,
)
from energy_forecasting.splits import (
    assert_temporal_order,
    backtest_calendar,
    chronological_split,
    expanding_window_folds,
)


class TestChronologicalSplit:
    def test_blocks_are_contiguous_and_ordered(self, cfg):
        split = chronological_split(1000, cfg)
        assert split.train[-1] + 1 == split.val[0]
        assert split.val[-1] + 1 == split.test[0]
        assert split.test[-1] == 999
        assert len(split.train) == 700 and len(split.val) == 150

    def test_temporal_order_guard_passes_on_sorted_time(self, cfg):
        ts = pd.Series(pd.date_range("2024-01-01", periods=1000, freq="10min"))
        split = chronological_split(1000, cfg)
        assert_temporal_order(ts, split)  # must not raise

    def test_temporal_order_guard_catches_shuffled_time(self, cfg):
        ts = (
            pd.Series(pd.date_range("2024-01-01", periods=1000, freq="10min"))
            .sample(frac=1.0, random_state=0)
            .reset_index(drop=True)
        )
        split = chronological_split(1000, cfg)
        with pytest.raises(AssertionError):
            assert_temporal_order(ts, split)

    def test_too_few_rows_raises(self, cfg):
        with pytest.raises(ValueError):
            chronological_split(5, cfg)


class TestExpandingWindowFolds:
    def test_train_grows_and_val_always_follows(self, cfg):
        folds = expanding_window_folds(1200, cfg)
        assert len(folds) == cfg.n_backtest_folds
        prev_train_len = 0
        for train_idx, val_idx in folds:
            assert len(train_idx) > prev_train_len
            assert train_idx[-1] + 1 == val_idx[0]  # val starts right after train
            prev_train_len = len(train_idx)

    def test_last_fold_reaches_region_end(self, cfg):
        folds = expanding_window_folds(1201, cfg)
        assert folds[-1][1][-1] == 1200

    def test_folds_never_overlap_train_and_val(self, cfg):
        for train_idx, val_idx in expanding_window_folds(1000, cfg):
            assert set(train_idx).isdisjoint(set(val_idx))

    def test_calendar_report(self, cfg):
        ts = pd.Series(pd.date_range("2024-01-01", periods=1200, freq="10min"))
        cal = backtest_calendar(ts, expanding_window_folds(1200, cfg))
        assert (cal["train_end"] < cal["val_start"]).all()


class TestMetrics:
    def test_known_values(self):
        y, p = np.array([10.0, 20.0, 30.0]), np.array([12.0, 18.0, 30.0])
        assert mae(y, p) == pytest.approx(4 / 3)
        assert rmse(y, p) == pytest.approx(np.sqrt(8 / 3))
        assert wape(y, p) == pytest.approx(4 / 60)
        assert bias(y, p) == pytest.approx(0.0)

    def test_bias_sign_convention(self):
        y, p = np.array([10.0, 10.0]), np.array([15.0, 15.0])
        assert bias(y, p) == 5.0  # over-forecast is positive

    def test_peak_mae_uses_threshold(self):
        y = np.array([10.0, 100.0, 200.0])
        p = np.array([0.0, 90.0, 190.0])
        assert peak_mae(y, p, peak_threshold=100.0) == pytest.approx(10.0)

    def test_peak_mae_nan_when_no_peaks(self):
        assert np.isnan(peak_mae(np.array([1.0]), np.array([1.0]), 100.0))

    def test_wape_zero_denominator(self):
        assert np.isnan(wape(np.array([0.0]), np.array([1.0])))

    def test_evaluate_all_keys(self):
        out = evaluate_all(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 1.5)
        assert {"mae", "rmse", "wape", "bias", "peak_mae", "n_obs"} <= set(out)

    def test_slice_errors_covers_dimensions(self):
        ts = pd.Series(pd.date_range("2024-01-01", periods=288, freq="10min"))
        y = np.linspace(10, 100, 288)
        p = y + 5
        out = slice_errors(ts, y, p, train_quartiles=[30.0, 55.0, 80.0])
        assert set(out["slice_dim"]) == {"hour", "day_type", "load_quartile"}
        assert out[out["slice_dim"] == "hour"]["n"].sum() == 288


def test_config_min_history(cfg):
    # max lag 144 vs max window 144 + 1 shift step
    assert cfg.min_history_steps == 145


def test_config_json_roundtrip(tmp_path):
    cfg = ForecastConfig()
    path = tmp_path / "feature_config.json"
    cfg.save_json(path, extra={"feature_columns": ["a", "b"]})
    loaded = ForecastConfig.from_json(path)
    assert loaded == cfg
