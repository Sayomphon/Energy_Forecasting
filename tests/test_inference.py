"""Bundle roundtrip, integrity verification, and fallback behaviour."""

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.backtest import run_backtest, select_model
from energy_forecasting.data import prepare_base
from energy_forecasting.features import add_features
from energy_forecasting.inference import (
    ForecastBundle,
    load_bundle,
    predict_one,
    save_bundle,
)
from energy_forecasting.models import make_hgb, make_last_value, model_factories
from energy_forecasting.splits import expanding_window_folds


@pytest.fixture()
def trained_bundle(raw_frame, cfg):
    base = prepare_base(raw_frame, cfg)
    frame, cols = add_features(base, cfg)
    model = make_hgb(cfg)
    model.fit(frame[cols], frame[cfg.target_col])
    return ForecastBundle(
        model=model,
        feature_columns=cols,
        config=cfg,
        model_version=cfg.model_version,
        trained_at="2026-07-18T00:00:00+00:00",
    )


class TestBundleRoundtrip:
    def test_reloaded_bundle_predicts_identically(self, trained_bundle, raw_frame, cfg, tmp_path):
        base = prepare_base(raw_frame, cfg)
        frame, cols = add_features(base, cfg)
        sample = frame[cols].tail(12)

        before = trained_bundle.model.predict(sample)
        path = save_bundle(trained_bundle, tmp_path)
        loaded = load_bundle(path)
        after = loaded.model.predict(sample[loaded.feature_columns])

        np.testing.assert_allclose(before, after)
        assert loaded.config == cfg
        assert loaded.model_version == cfg.model_version

    def test_tampered_bundle_is_rejected(self, trained_bundle, tmp_path):
        path = save_bundle(trained_bundle, tmp_path)
        with open(path, "ab") as f:
            f.write(b"tampered")
        with pytest.raises(ValueError, match="checksum mismatch"):
            load_bundle(path)

    def test_missing_sidecar_is_rejected_by_default(self, trained_bundle, tmp_path):
        path = save_bundle(trained_bundle, tmp_path)
        path.with_suffix(path.suffix + ".sha256").unlink()
        with pytest.raises(FileNotFoundError):
            load_bundle(path)


class TestPredictOne:
    def test_ok_prediction_follows_contract(self, trained_bundle, raw_frame, cfg):
        t = raw_frame["date"].iloc[-1]
        out = predict_one(trained_bundle, raw_frame, t)
        assert out["quality_flag"] == "ok"
        assert out["fallback_used"] is False
        assert out["model_version"] == cfg.model_version
        assert pd.Timestamp(out["target_time"]) - pd.Timestamp(out["prediction_time"]) == (
            pd.Timedelta(minutes=60)
        )
        assert out["forecast_appliances_wh"] > 0

    def test_future_rows_in_history_are_ignored(self, trained_bundle, raw_frame):
        """Feeding data after t must not change the forecast at t."""
        t = raw_frame["date"].iloc[-200]
        hist_only = raw_frame[raw_frame["date"] <= t]
        with_future = raw_frame.copy()
        a = predict_one(trained_bundle, hist_only, t)
        b = predict_one(trained_bundle, with_future, t)
        assert a["forecast_appliances_wh"] == b["forecast_appliances_wh"]

    def test_stale_history_falls_back_to_last_value(self, trained_bundle, raw_frame, cfg):
        last_ts = raw_frame["date"].iloc[-1]
        t = last_ts + pd.Timedelta(minutes=cfg.max_staleness_minutes + 10)
        out = predict_one(trained_bundle, raw_frame, t)
        assert out["fallback_used"] is True
        assert out["quality_flag"] == "stale_history_fallback"
        assert out["forecast_appliances_wh"] == pytest.approx(
            raw_frame["Appliances"].iloc[-1], abs=0.01
        )

    def test_insufficient_history_falls_back(self, trained_bundle, raw_frame, cfg):
        short = raw_frame.tail(cfg.min_history_steps - 10)
        t = short["date"].iloc[-1]
        out = predict_one(trained_bundle, short, t)
        assert out["fallback_used"] is True
        assert out["quality_flag"] == "insufficient_history_fallback"

    def test_missing_required_column_raises(self, trained_bundle, raw_frame):
        t = raw_frame["date"].iloc[-1]
        with pytest.raises(ValueError, match="missing required columns"):
            predict_one(trained_bundle, raw_frame.drop(columns=["Appliances"]), t)

    def test_prediction_time_accepts_iso_string(self, trained_bundle, raw_frame):
        t = raw_frame["date"].iloc[-1]
        from_ts = predict_one(trained_bundle, raw_frame, t)
        from_iso = predict_one(trained_bundle, raw_frame, t.isoformat())
        assert from_iso["forecast_appliances_wh"] == from_ts["forecast_appliances_wh"]

    def test_prediction_time_accepts_uci_no_space_string(self, trained_bundle, raw_frame):
        """Regression: raw UCI timestamps ('2016-05-2718:00:00') must parse."""
        t = raw_frame["date"].iloc[-1]
        mangled = t.strftime("%Y-%m-%d%H:%M:%S")
        out = predict_one(trained_bundle, raw_frame, mangled)
        assert pd.Timestamp(out["prediction_time"]) == t

    def test_unparseable_prediction_time_raises(self, trained_bundle, raw_frame):
        with pytest.raises(ValueError, match="Unparseable prediction_time"):
            predict_one(trained_bundle, raw_frame, "not-a-timestamp")


class TestBacktestAndSelection:
    def test_backtest_runs_all_models_and_folds(self, raw_frame, cfg):
        base = prepare_base(raw_frame, cfg)
        frame, cols = add_features(base, cfg)
        n = int(len(frame) * 0.85)
        folds = expanding_window_folds(n, cfg)
        results = run_backtest(
            frame[cols].iloc[:n],
            frame[cfg.target_col].iloc[:n],
            frame["date"].iloc[:n],
            folds,
            model_factories(cfg),
            cfg,
        )
        assert len(results) == 4 * cfg.n_backtest_folds
        assert (results["val_start"] > results["train_end"]).all()
        assert results["mae"].notna().all()

    def test_selection_returns_audit_trail(self, raw_frame, cfg):
        base = prepare_base(raw_frame, cfg)
        frame, cols = add_features(base, cfg)
        n = int(len(frame) * 0.85)
        folds = expanding_window_folds(n, cfg)
        results = run_backtest(
            frame[cols].iloc[:n],
            frame[cfg.target_col].iloc[:n],
            frame["date"].iloc[:n],
            folds,
            model_factories(cfg),
            cfg,
        )
        decision = select_model(results)
        assert decision["selected"] in {"last_value", "seasonal_naive", "ridge", "hgb"}
        assert "reason" in decision and "audit" in decision


def test_last_value_baseline_echoes_load_now(raw_frame, cfg):
    base = prepare_base(raw_frame, cfg)
    frame, cols = add_features(base, cfg)
    model = make_last_value().fit(frame[cols])
    np.testing.assert_allclose(model.predict(frame[cols]), frame["load_now"].to_numpy())
