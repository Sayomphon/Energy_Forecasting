"""Controlled experiment: full (v1) vs lean (v2) feature set.

Runs the *identical* pipeline — same rows, same chronological split, same
expanding-window folds, same model zoo, same selection rule — for both feature
sets and reports the backtest deltas. Only the feature columns differ, so any
change in MAE is attributable to the feature set alone (docs/V2_FEATURE_SET_PLAN.md).

Why a separate script and not train.py: the main entry point stays a single,
uncluttered path (contract -> features -> split -> backtest -> select -> test).
This experiment fans out over two feature sets on the train+val region only.

Methodology guardrails (§3):
- The comparison lives entirely on the train+val region. The held-out test set
  is never touched here; it is opened once, by train.py, on the promoted winner.
- Promotion is decided by criteria fixed *before* running (§3.2), not by picking
  whichever set looks best after the fact. Reporting an honest "no promotion"
  is as valid an outcome as a win.

Run:
    OMP_NUM_THREADS=1 python scripts/compare_feature_sets.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from energy_forecasting.backtest import run_backtest, select_model, summarize_backtest
from energy_forecasting.config import ARTIFACTS_DIR, DATA_DIR, ForecastConfig
from energy_forecasting.data import load_raw, prepare_base, validate_contract
from energy_forecasting.features import add_features
from energy_forecasting.models import model_factories
from energy_forecasting.splits import chronological_split, expanding_window_folds

logger = logging.getLogger("compare_feature_sets")

# Promotion criteria, fixed before running (docs/V2_FEATURE_SET_PLAN.md §3.2).
# v2 is promoted only if BOTH hold on the backtest folds.
MAE_IMPROVEMENT_MIN_WH = 1.0  # v2 mae_mean must beat v1 by at least this much
PEAK_MAE_REGRESS_MAX_FRAC = 0.05  # v2 peak_mae_mean may worsen by at most 5%

FEATURE_SETS = ("v1", "v2")


def backtest_feature_set(raw: pd.DataFrame, feature_set: str) -> dict:
    """Backtest every model on one feature set over the train+val region."""
    cfg = ForecastConfig(feature_set=feature_set)
    report = validate_contract(raw, cfg)
    if not report.passed:
        raise SystemExit(f"Data contract failed for {feature_set}: {report.failures}")

    base = prepare_base(raw, cfg)
    frame, feature_cols = add_features(base, cfg)
    X = frame[feature_cols]
    y = frame[cfg.target_col]
    ts = frame[cfg.timestamp_col]

    split = chronological_split(len(frame), cfg)
    n_trainval = len(split.train) + len(split.val)
    folds = expanding_window_folds(n_trainval, cfg)

    results = run_backtest(
        X.iloc[:n_trainval],
        y.iloc[:n_trainval],
        ts.iloc[:n_trainval],
        folds,
        model_factories(cfg),
        cfg,
    )
    summary = summarize_backtest(results)
    selection = select_model(results)
    return {
        "feature_set": feature_set,
        "n_features": len(feature_cols),
        "n_rows": len(frame),
        "n_trainval": n_trainval,
        "n_folds": len(folds),
        "summary": summary,
        "selected": selection["selected"],
        "selection_reason": selection["reason"],
    }


def _row(summary: pd.DataFrame, model: str) -> pd.Series:
    return summary.set_index("model").loc[model]


def decide(v1: dict, v2: dict) -> dict:
    """Apply the pre-registered promotion criteria to the selected models."""
    # Compare the model each set would actually deploy (its selected winner).
    m1, m2 = v1["selected"], v2["selected"]
    r1, r2 = _row(v1["summary"], m1), _row(v2["summary"], m2)

    mae_gain = float(r1["mae_mean"] - r2["mae_mean"])  # positive = v2 better
    peak_regress_frac = float((r2["peak_mae_mean"] - r1["peak_mae_mean"]) / r1["peak_mae_mean"])

    passes_mae = mae_gain >= MAE_IMPROVEMENT_MIN_WH
    passes_peak = peak_regress_frac <= PEAK_MAE_REGRESS_MAX_FRAC
    promote = passes_mae and passes_peak

    return {
        "v1_selected": m1,
        "v2_selected": m2,
        "v1_mae_mean": float(r1["mae_mean"]),
        "v2_mae_mean": float(r2["mae_mean"]),
        "mae_gain_wh": mae_gain,
        "v1_peak_mae_mean": float(r1["peak_mae_mean"]),
        "v2_peak_mae_mean": float(r2["peak_mae_mean"]),
        "peak_regress_frac": peak_regress_frac,
        "passes_mae_criterion": passes_mae,
        "passes_peak_criterion": passes_peak,
        "promote_v2": promote,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Compare v1 (full) vs v2 (lean) feature sets.")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    raw = load_raw(args.data_dir)
    runs = {fs: backtest_feature_set(raw, fs) for fs in FEATURE_SETS}

    # Fairness invariant: identical rows/folds, differing only in feature columns.
    if runs["v1"]["n_rows"] != runs["v2"]["n_rows"]:
        raise SystemExit("Row counts differ between v1 and v2 — comparison is not controlled.")

    for fs in FEATURE_SETS:
        r = runs[fs]
        logger.info(
            "[%s] %d features, %d rows, %d folds, selected=%s",
            fs,
            r["n_features"],
            r["n_rows"],
            r["n_folds"],
            r["selected"],
        )
        logger.info("[%s] backtest summary:\n%s", fs, r["summary"].to_string(index=False))

    verdict = decide(runs["v1"], runs["v2"])

    # Persist a tidy comparison table (both sets, all models) for the docs.
    combined = pd.concat(
        [r["summary"].assign(feature_set=fs) for fs, r in runs.items()],
        ignore_index=True,
    )
    cols = ["feature_set", *[c for c in combined.columns if c != "feature_set"]]
    out_path = Path(args.artifacts_dir) / "feature_set_comparison.csv"
    combined[cols].to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)

    bar = "=" * 68
    print(f"\n{bar}\nPROMOTION DECISION (criteria fixed before running, §3.2)\n{bar}")
    print(
        f"  v1 selected: {verdict['v1_selected']:<6}  mae_mean={verdict['v1_mae_mean']:.3f}  "
        f"peak_mae_mean={verdict['v1_peak_mae_mean']:.3f}"
    )
    print(
        f"  v2 selected: {verdict['v2_selected']:<6}  mae_mean={verdict['v2_mae_mean']:.3f}  "
        f"peak_mae_mean={verdict['v2_peak_mae_mean']:.3f}"
    )
    mae_status = "PASS" if verdict["passes_mae_criterion"] else "FAIL"
    peak_status = "PASS" if verdict["passes_peak_criterion"] else "FAIL"
    print(
        f"\n  Criterion 1 (MAE improves >= {MAE_IMPROVEMENT_MIN_WH} Wh): "
        f"gain={verdict['mae_gain_wh']:+.3f} Wh  -> {mae_status}"
    )
    print(
        f"  Criterion 2 (peak MAE regress <= {PEAK_MAE_REGRESS_MAX_FRAC:.0%}): "
        f"regress={verdict['peak_regress_frac']:+.2%}  -> {peak_status}"
    )
    decision = (
        "PROMOTE v2 (retrain + open test on v2)"
        if verdict["promote_v2"]
        else "KEEP v1 (report honest negative)"
    )
    print(f"\n  ==> DECISION: {decision}\n{bar}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
