"""Temporal-correctness tests — the heart of this project.

Each test encodes one leakage trap from the plan document: wrong target
shift, future-looking rolling windows, unsorted input, or features that
change when future rows change.
"""

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.data import parse_timestamps, prepare_base, validate_contract
from energy_forecasting.features import add_features, availability_audit, steps_per_day


class TestTargetShift:
    def test_target_is_exactly_six_steps_ahead(self, raw_frame, cfg):
        base = prepare_base(raw_frame, cfg)
        i = 100
        t = base.loc[i, "date"]
        future = raw_frame.set_index("date").loc[t + pd.Timedelta(minutes=60), "Appliances"]
        assert base.loc[i, "target_1h"] == future

    def test_tail_rows_without_label_are_dropped(self, raw_frame, cfg):
        base = prepare_base(raw_frame, cfg)
        assert len(base) == len(raw_frame) - cfg.horizon_steps
        assert not base["target_1h"].isna().any()

    def test_unsorted_input_is_handled(self, raw_frame, cfg):
        shuffled = raw_frame.sample(frac=1.0, random_state=1)
        base = prepare_base(shuffled, cfg)
        sorted_base = prepare_base(raw_frame, cfg)
        pd.testing.assert_frame_equal(base, sorted_base)

    def test_duplicate_timestamps_are_deduplicated(self, raw_frame, cfg):
        doubled = pd.concat([raw_frame, raw_frame.iloc[:50]]).reset_index(drop=True)
        base = prepare_base(doubled, cfg)
        assert not base["date"].duplicated().any()


class TestPastOnlyFeatures:
    def test_lag_values_match_manual_lookup(self, raw_frame, cfg):
        base = prepare_base(raw_frame, cfg)
        frame, _ = add_features(base, cfg)
        row = frame.iloc[500]
        src = base.set_index("date")["Appliances"]
        for k in cfg.lags:
            expected = src.loc[row["date"] - pd.Timedelta(minutes=10 * k)]
            assert row[f"lag_{k}"] == expected, f"lag_{k} mismatch"

    def test_rolling_mean_uses_strictly_past_window(self, raw_frame, cfg):
        """docx unit example: roll over [t-w, t-1], never including t."""
        base = prepare_base(raw_frame, cfg)
        frame, _ = add_features(base, cfg)
        offset = len(base) - len(frame)  # rows dropped for missing history
        i_frame, i_base = 100, 100 + offset
        expected = base["Appliances"].iloc[i_base - 6 : i_base].mean()
        assert frame["roll_mean_6"].iloc[i_frame] == pytest.approx(expected)

    def test_features_do_not_depend_on_the_future(self, raw_frame, cfg):
        """Mutate everything after time T; features at T must not change."""
        base = prepare_base(raw_frame, cfg)
        frame_a, cols = add_features(base, cfg)

        cut = 1200
        corrupted = base.copy()
        num_cols = [c for c in corrupted.columns if c not in ("date", "target_1h")]
        corrupted.loc[cut + 1 :, num_cols] = 9999.0
        frame_b, _ = add_features(corrupted, cfg)

        t_cut = base.loc[cut, "date"]
        a = frame_a[frame_a["date"] <= t_cut][cols]
        b = frame_b[frame_b["date"] <= t_cut][cols]
        pd.testing.assert_frame_equal(a, b)

    def test_seasonal_ref_aligns_with_target_time_minus_24h(self, raw_frame, cfg):
        base = prepare_base(raw_frame, cfg)
        frame, _ = add_features(base, cfg)
        row = frame.iloc[400]
        target_time = row["date"] + pd.Timedelta(minutes=cfg.horizon_minutes)
        expected = (
            base.set_index("date")["Appliances"].loc[target_time - pd.Timedelta(hours=24)]
        )
        assert row["seasonal_ref"] == expected

    def test_no_nans_in_feature_matrix(self, raw_frame, cfg):
        base = prepare_base(raw_frame, cfg)
        frame, cols = add_features(base, cfg)
        lag_like = [c for c in cols if c.startswith(("lag_", "roll_", "seasonal_"))]
        assert not frame[lag_like].isna().any().any()

    def test_availability_audit_covers_every_feature(self, raw_frame, cfg):
        base = prepare_base(raw_frame, cfg)
        _, cols = add_features(base, cfg)
        audit = availability_audit(cols, cfg)
        assert set(audit["feature"]) == set(cols)
        assert not audit["uses_future"].any()

    def test_steps_per_day(self, cfg):
        assert steps_per_day(cfg) == 144


class TestTimestampParsing:
    def test_standard_format(self):
        out = parse_timestamps(pd.Series(["2016-01-11 17:00:00", "2016-01-11 17:10:00"]))
        assert out.iloc[0] == pd.Timestamp("2016-01-11 17:00:00")

    def test_uci_no_space_format(self):
        """The real ucimlrepo export: date and time concatenated without a space."""
        out = parse_timestamps(pd.Series(["2016-01-1117:00:00", "2016-05-2718:00:00"]))
        assert out.iloc[0] == pd.Timestamp("2016-01-11 17:00:00")
        assert out.iloc[1] == pd.Timestamp("2016-05-27 18:00:00")

    def test_garbage_becomes_nat_not_crash(self):
        out = parse_timestamps(pd.Series(["not-a-date", "2016-01-11 17:00:00"]))
        assert out.isna().iloc[0]

    def test_already_datetime_passthrough(self):
        ts = pd.Series(pd.date_range("2024-01-01", periods=3, freq="10min"))
        pd.testing.assert_series_equal(parse_timestamps(ts), ts)

    def test_contract_accepts_uci_no_space_format(self, raw_frame, cfg):
        mangled = raw_frame.copy()
        mangled["date"] = mangled["date"].dt.strftime("%Y-%m-%d%H:%M:%S")
        report = validate_contract(mangled, cfg)
        assert report.passed, report.failures


class TestDataContract:
    def test_clean_frame_passes(self, raw_frame, cfg):
        report = validate_contract(raw_frame, cfg)
        assert report.passed, report.failures
        assert report.modal_interval_minutes == 10.0

    def test_negative_target_fails(self, raw_frame, cfg):
        bad = raw_frame.copy()
        bad.loc[10, "Appliances"] = -5.0
        report = validate_contract(bad, cfg)
        assert not report.passed
        assert any("negative" in f for f in report.failures)

    def test_missing_column_fails(self, raw_frame, cfg):
        report = validate_contract(raw_frame.drop(columns=["Appliances"]), cfg)
        assert not report.passed

    def test_gaps_are_reported_but_tolerated(self, raw_frame, cfg):
        gapped = raw_frame.drop(index=range(300, 320)).reset_index(drop=True)
        report = validate_contract(gapped, cfg)
        assert report.passed
        assert report.n_gaps == 1

    def test_negative_controls_present(self, raw_frame, cfg):
        base = prepare_base(raw_frame, cfg)
        _, cols = add_features(base, cfg)
        assert "rv1" in cols and "rv2" in cols


class TestNoLabelInFeatures:
    def test_target_column_never_a_feature(self, raw_frame, cfg):
        base = prepare_base(raw_frame, cfg)
        _, cols = add_features(base, cfg)
        assert cfg.target_col not in cols

    def test_raw_source_column_not_duplicated(self, raw_frame, cfg):
        base = prepare_base(raw_frame, cfg)
        _, cols = add_features(base, cfg)
        # The reading at t enters once, as load_now — not twice.
        assert cfg.target_source_col not in cols
        assert cols.count("load_now") == 1


def test_features_correlate_with_future_less_than_perfectly(raw_frame, cfg):
    """Smoke check: if any feature equals the label, something leaked."""
    base = prepare_base(raw_frame, cfg)
    frame, cols = add_features(base, cfg)
    y = frame[cfg.target_col]
    for col in cols:
        corr = np.corrcoef(frame[col], y)[0, 1]
        assert not np.isclose(abs(corr), 1.0), f"{col} is a perfect copy of the label"
