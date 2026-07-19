"""Feature-set v2 (lean) toggle — controlled-experiment plumbing.

The v2 set drops every raw weather/indoor sensor (and the indoor aggregates
built from them) while keeping recent-load dynamics, calendar features, and the
rv1/rv2 negative controls. These tests pin that contract down: v2 removes
exactly the sensor block, keeps the core, leaves v1 bit-for-bit unchanged,
stays leakage-safe, and round-trips through serialisation.
See docs/V2_FEATURE_SET_PLAN.md §6.
"""

import pandas as pd
import pytest

from energy_forecasting.config import ForecastConfig
from energy_forecasting.data import prepare_base
from energy_forecasting.features import NEGATIVE_CONTROL_COLS, add_features


@pytest.fixture()
def cfg_v2() -> ForecastConfig:
    return ForecastConfig(feature_set="v2")


def _feature_cols(raw_frame, cfg) -> list:
    base = prepare_base(raw_frame, cfg)
    _, cols = add_features(base, cfg)
    return cols


class TestV2FeatureSet:
    def test_v2_drops_sensor_features(self, raw_frame, cfg_v2):
        """Every raw sensor and sensor-derived aggregate is gone in v2."""
        cols = _feature_cols(raw_frame, cfg_v2)
        assert not any(c.startswith(("indoor_", "diff_")) for c in cols), (
            "indoor/outdoor aggregates must not survive into v2"
        )
        for banned in ("lights", "T1", "RH_1", "T_out", "RH_out"):
            assert banned not in cols, f"{banned} is a raw sensor and must be dropped in v2"

    def test_v2_keeps_core_features(self, raw_frame, cfg_v2):
        """Recent-load dynamics, seasonal ref, rolling stats, calendar remain."""
        cols = _feature_cols(raw_frame, cfg_v2)
        expected_core = {
            "load_now",
            "lag_1",
            "lag_6",
            "lag_12",
            "lag_144",
            "seasonal_ref",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "is_weekend",
        }
        assert expected_core <= set(cols)
        # All rolling-stat columns survive: 4 stats x 3 windows.
        assert sum(c.startswith("roll_") for c in cols) == 12

    def test_v2_keeps_negative_controls(self, raw_frame, cfg_v2):
        """rv1/rv2 stay in v2 so the negative-control check remains valid."""
        cols = _feature_cols(raw_frame, cfg_v2)
        for rv in NEGATIVE_CONTROL_COLS:
            assert rv in cols, f"{rv} negative control must be kept in v2 backtests"

    def test_v2_is_strict_subset_of_v1(self, raw_frame, cfg, cfg_v2):
        """v2 only removes columns from v1; it never introduces new ones."""
        cols_v1 = _feature_cols(raw_frame, cfg)
        cols_v2 = _feature_cols(raw_frame, cfg_v2)
        assert set(cols_v2) < set(cols_v1)


class TestV1Unchanged:
    def test_v1_default_feature_cols_unchanged(self, raw_frame, cfg):
        """Regression guard: the default (v1) set is exactly as before.

        On the synthetic fixture the full ordered set is the 23 core+calendar
        columns, the 4 indoor aggregates, then the raw contemporaneous loop
        (lights, T1, RH_1, T_out, RH_out, rv1, rv2) in source-column order.
        Order matters: it is the column order both training and inference rely
        on, so this asserts the exact list, not just its membership.
        """
        cols = _feature_cols(raw_frame, cfg)
        expected = [
            "load_now",
            "lag_1",
            "lag_6",
            "lag_12",
            "lag_144",
            "seasonal_ref",
            "roll_mean_6",
            "roll_std_6",
            "roll_min_6",
            "roll_max_6",
            "roll_mean_36",
            "roll_std_36",
            "roll_min_36",
            "roll_max_36",
            "roll_mean_144",
            "roll_std_144",
            "roll_min_144",
            "roll_max_144",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "is_weekend",
            "indoor_T_mean",
            "diff_T_in_out",
            "indoor_RH_mean",
            "diff_RH_in_out",
            "lights",
            "T1",
            "RH_1",
            "T_out",
            "RH_out",
            "rv1",
            "rv2",
        ]
        assert cols == expected


class TestLeakageSafetyV2:
    def test_v2_features_do_not_depend_on_the_future(self, raw_frame, cfg_v2):
        """Anti-leakage (mutate-the-future) must still hold for the lean set.

        Dropping features must not weaken the availability guarantee: features
        stamped at or before t stay identical when every future row is corrupted.
        """
        base = prepare_base(raw_frame, cfg_v2)
        frame_a, cols = add_features(base, cfg_v2)

        cut = 1200
        corrupted = base.copy()
        num_cols = [c for c in corrupted.columns if c not in ("date", "target_1h")]
        corrupted.loc[cut + 1 :, num_cols] = 9999.0
        frame_b, _ = add_features(corrupted, cfg_v2)

        t_cut = base.loc[cut, "date"]
        a = frame_a[frame_a["date"] <= t_cut][cols]
        b = frame_b[frame_b["date"] <= t_cut][cols]
        pd.testing.assert_frame_equal(a, b)


class TestConfigRoundtrip:
    def test_config_feature_set_roundtrip(self, tmp_path, cfg_v2):
        """v2 config serialises and reloads identically (feature_set survives)."""
        path = tmp_path / "feature_config.json"
        cfg_v2.save_json(path)
        loaded = ForecastConfig.from_json(path)
        assert loaded == cfg_v2
        assert loaded.feature_set == "v2"
        assert loaded.include_sensor_features is False

    def test_default_config_is_v1(self, cfg):
        """The reproducible default stays v1 with sensors included."""
        assert cfg.feature_set == "v1"
        assert cfg.include_sensor_features is True

    def test_invalid_feature_set_rejected(self):
        """Guardrail: an unknown feature_set fails fast at construction."""
        with pytest.raises(ValueError, match="feature_set must be"):
            ForecastConfig(feature_set="v3")
