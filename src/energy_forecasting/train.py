"""End-to-end training entry point.

Usage:
    python -m energy_forecasting.train --fetch     # first run (downloads once)
    python -m energy_forecasting.train             # reruns use the local cache

Order of operations mirrors the notebook blueprint: contract -> features ->
chronological split -> expanding-window backtest -> selection -> one-shot
test evaluation -> packaged artifacts. The test set is touched exactly once,
after model selection is frozen.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from energy_forecasting import __version__
from energy_forecasting.backtest import (
    ablation_delta,
    run_backtest,
    select_model,
    summarize_backtest,
)
from energy_forecasting.config import ARTIFACTS_DIR, DATA_DIR, ForecastConfig
from energy_forecasting.data import (
    LINEAGE_NAME,
    fetch_raw,
    load_raw,
    prepare_base,
    validate_contract,
)
from energy_forecasting.features import add_features
from energy_forecasting.inference import ForecastBundle, save_bundle
from energy_forecasting.metrics import evaluate_all, residual_summary, slice_errors
from energy_forecasting.models import model_factories
from energy_forecasting.splits import (
    assert_temporal_order,
    backtest_calendar,
    chronological_split,
    expanding_window_folds,
)

logger = logging.getLogger("energy_forecasting.train")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the 1h-ahead energy forecaster.")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    p.add_argument("--fetch", action="store_true", help="Download the dataset if not cached.")
    p.add_argument("--no-ablation", action="store_true", help="Skip feature-ablation runs.")
    p.add_argument(
        "--feature-set",
        choices=("v1", "v2"),
        default="v1",
        help="v1 = full sensor set (default); v2 = lean recent-load + calendar set.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # v2 (lean) artifacts carry their own model_version so a promoted bundle is
    # unambiguous; v1 keeps the original identifier for reproducibility.
    model_version = "energy-1h-v2" if args.feature_set == "v2" else "energy-1h-v1"
    cfg = ForecastConfig(feature_set=args.feature_set, model_version=model_version)
    logger.info("Feature set: %s (model_version=%s)", cfg.feature_set, cfg.model_version)
    artifacts = Path(args.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    # --- 1. Data + contract -------------------------------------------------
    if args.fetch:
        fetch_raw(args.data_dir)
    raw = load_raw(args.data_dir)
    report = validate_contract(raw, cfg)
    (artifacts / "data_contract_report.json").write_text(json.dumps(report.to_dict(), indent=2))
    if not report.passed:
        logger.error("Data contract FAILED: %s", report.failures)
        return 1
    logger.info(
        "Contract OK: %d rows, %s -> %s, %d gaps, %d duplicate timestamps",
        report.n_rows,
        report.time_min,
        report.time_max,
        report.n_gaps,
        report.n_duplicate_timestamps,
    )

    # --- 2. Target + features ----------------------------------------------
    base = prepare_base(raw, cfg)
    frame, feature_cols = add_features(base, cfg)
    logger.info(
        "Features: %d columns, %d rows (%d dropped for insufficient history)",
        len(feature_cols),
        len(frame),
        frame.attrs["rows_dropped_insufficient_history"],
    )

    X = frame[feature_cols]
    y = frame[cfg.target_col]
    ts = frame[cfg.timestamp_col]

    # --- 3. Chronological split + backtest folds ----------------------------
    split = chronological_split(len(frame), cfg)
    assert_temporal_order(ts, split)
    n_trainval = len(split.train) + len(split.val)
    folds = expanding_window_folds(n_trainval, cfg)
    calendar = backtest_calendar(ts.iloc[:n_trainval], folds)
    calendar.to_csv(artifacts / "backtest_calendar.csv", index=False)

    # --- 4. Backtest all models ---------------------------------------------
    factories = model_factories(cfg)
    results = run_backtest(
        X.iloc[:n_trainval],
        y.iloc[:n_trainval],
        ts.iloc[:n_trainval],
        folds,
        factories,
        cfg,
    )
    results.to_csv(artifacts / "backtest_metrics.csv", index=False)
    summary = summarize_backtest(results)
    summary.to_csv(artifacts / "backtest_summary.csv", index=False)
    logger.info("Backtest summary:\n%s", summary.to_string(index=False))

    # --- 5. Selection (baselines are the bar) -------------------------------
    selection = select_model(results)
    (artifacts / "model_selection.json").write_text(json.dumps(selection, indent=2))
    selected = selection["selected"]
    logger.info("Selected model: %s (%s)", selected, selection["reason"])

    # --- 6. Optional ablation on the selected candidate ---------------------
    if not args.no_ablation and selected in ("ridge", "hgb"):
        groups = {
            "rolling_stats": [c for c in feature_cols if c.startswith("roll_")],
            "calendar": ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"],
            "weather_sensors": [
                c
                for c in feature_cols
                if c not in {"load_now", "seasonal_ref"}
                and not c.startswith(("lag_", "roll_", "hour_", "dow_", "is_weekend", "rv"))
            ],
            "negative_controls": [c for c in feature_cols if c in ("rv1", "rv2")],
        }
        # In the lean v2 set the weather/indoor group is empty (already dropped);
        # skip empty groups so ablation never reports a trivial zero-delta row.
        groups = {name: cols for name, cols in groups.items() if cols}
        ablation = ablation_delta(
            X.iloc[:n_trainval],
            y.iloc[:n_trainval],
            folds,
            factories[selected],
            cfg,
            groups,
        )
        ablation.to_csv(artifacts / "ablation_results.csv", index=False)
        logger.info("Ablation:\n%s", ablation.to_string(index=False))

    # --- 7. Final fit on train+val, one-shot test evaluation ----------------
    final_model = factories[selected]()
    final_model.fit(X.iloc[:n_trainval], y.iloc[:n_trainval])
    peak_thr = float(y.iloc[:n_trainval].quantile(cfg.peak_quantile))
    test_pred = final_model.predict(X.iloc[split.test])
    y_test = y.iloc[split.test].to_numpy()
    ts_test = ts.iloc[split.test]

    test_metrics = evaluate_all(y_test, test_pred, peak_thr)
    logger.info("FINAL TEST metrics (%s): %s", selected, test_metrics)

    quartiles = [float(y.iloc[:n_trainval].quantile(q)) for q in (0.25, 0.50, 0.75)]
    slices = slice_errors(ts_test, y_test, test_pred, quartiles)
    slices.to_csv(artifacts / "test_slice_errors.csv", index=False)
    resid = residual_summary(ts_test, y_test, test_pred, peak_thr)
    (artifacts / "residual_summary.json").write_text(json.dumps(resid, indent=2))

    # --- 8. Package ---------------------------------------------------------
    lineage_path = args.data_dir / LINEAGE_NAME
    data_sha = ""
    if lineage_path.exists():
        data_sha = json.loads(lineage_path.read_text()).get("sha256", "")

    bundle = ForecastBundle(
        model=final_model,
        feature_columns=feature_cols,
        config=cfg,
        model_version=cfg.model_version,
        trained_at=datetime.now(timezone.utc).isoformat(),
        train_data_sha256=data_sha,
        metrics={"selected_model": selected, "test": test_metrics},
    )
    bundle_path = save_bundle(bundle, artifacts)
    cfg.save_json(
        artifacts / "feature_config.json",
        extra={
            "feature_columns": feature_cols,
            "package_version": __version__,
            "selected_model": selected,
            "train_data_sha256": data_sha,
        },
    )
    (artifacts / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    logger.info("Artifacts written to %s (bundle: %s)", artifacts, bundle_path.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
