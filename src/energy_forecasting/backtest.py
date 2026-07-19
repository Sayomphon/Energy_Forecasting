"""Expanding-window backtest runner.

Runs every registered model over every fold with identical data handling:
fit on the fold's past, predict the fold's future, score with the fold's
train-only peak threshold. Latency and model size are recorded because the
selection rule weighs operability, not accuracy alone.
"""

from __future__ import annotations

import io
import logging
import time

import joblib
import numpy as np
import pandas as pd

from energy_forecasting.config import ForecastConfig
from energy_forecasting.metrics import evaluate_all

logger = logging.getLogger(__name__)


def _model_size_kb(model) -> float:
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return round(buf.getbuffer().nbytes / 1024, 1)


def run_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    folds: list,
    factories: dict,
    cfg: ForecastConfig,
) -> pd.DataFrame:
    """Score every model on every expanding-window fold.

    Args:
        X: Feature matrix (train+val region, time-sorted, positional index).
        y: Target aligned with X.
        timestamps: Timestamps aligned with X (for reporting).
        folds: ``(train_idx, val_idx)`` pairs from ``expanding_window_folds``.
        factories: name -> zero-arg factory returning a fresh unfitted model.
        cfg: For the train-only peak quantile.

    Returns:
        One row per (model, fold) with metrics, latency, and size.
    """
    ts = pd.Series(timestamps).reset_index(drop=True)
    rows = []
    for name, factory in factories.items():
        for fold_id, (train_idx, val_idx) in enumerate(folds):
            model = factory()
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

            # Peak threshold from this fold's training targets only.
            peak_thr = float(y_tr.quantile(cfg.peak_quantile))

            t0 = time.perf_counter()
            model.fit(X_tr, y_tr)
            fit_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            pred = model.predict(X_va)
            predict_ms_per_1k = (time.perf_counter() - t0) / max(len(X_va), 1) * 1000 * 1000

            row = {
                "model": name,
                "fold": fold_id,
                "train_end": ts.iloc[train_idx[-1]],
                "val_start": ts.iloc[val_idx[0]],
                "val_end": ts.iloc[val_idx[-1]],
                "fit_seconds": round(fit_s, 3),
                "predict_ms_per_1k_rows": round(predict_ms_per_1k, 2),
                "model_size_kb": _model_size_kb(model),
            }
            row.update(evaluate_all(y_va.to_numpy(), pred, peak_thr))
            rows.append(row)
            logger.info(
                "%s fold %d: MAE=%.2f WAPE=%.3f peak_MAE=%.2f",
                name,
                fold_id,
                row["mae"],
                row["wape"],
                row["peak_mae"],
            )
    return pd.DataFrame(rows)


def summarize_backtest(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean/std per model across folds (stability view)."""
    return (
        results.groupby("model")
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            wape_mean=("wape", "mean"),
            peak_mae_mean=("peak_mae", "mean"),
            peak_mae_std=("peak_mae", "std"),
            bias_mean=("bias", "mean"),
            fit_seconds_mean=("fit_seconds", "mean"),
            predict_ms_per_1k_rows=("predict_ms_per_1k_rows", "mean"),
            model_size_kb=("model_size_kb", "mean"),
        )
        .sort_values("mae_mean")
        .reset_index()
    )


def select_model(
    results: pd.DataFrame,
    baseline_names: tuple = ("last_value", "seasonal_naive"),
) -> dict:
    """Apply the docx selection rule.

    A candidate is eligible only if it beats **every** baseline on MAE in a
    majority of folds. Among eligible candidates, pick the lowest mean MAE,
    breaking near-ties (<2% MAE difference) by lower peak MAE.

    Returns a dict with the winner and the audit trail of the decision.
    """
    folds = sorted(results["fold"].unique())
    candidates = [m for m in results["model"].unique() if m not in baseline_names]
    audit = []
    eligible = []

    for cand in candidates:
        wins = {b: 0 for b in baseline_names}
        for fold in folds:
            fold_rows = results[results["fold"] == fold].set_index("model")
            for b in baseline_names:
                if fold_rows.loc[cand, "mae"] < fold_rows.loc[b, "mae"]:
                    wins[b] += 1
        beats_all = all(w > len(folds) / 2 for w in wins.values())
        audit.append({"model": cand, "fold_wins_vs_baselines": wins, "eligible": beats_all})
        if beats_all:
            eligible.append(cand)

    if not eligible:
        # Honest failure mode: no candidate justified its complexity.
        fallback = (
            results[results["model"].isin(baseline_names)].groupby("model")["mae"].mean().idxmin()
        )
        return {"selected": fallback, "reason": "no candidate beat baselines", "audit": audit}

    summary = summarize_backtest(results[results["model"].isin(eligible)])
    best = summary.iloc[0]
    selected = best["model"]
    reason = f"lowest mean MAE ({best['mae_mean']:.2f}) among eligible candidates"
    if len(summary) > 1:
        runner = summary.iloc[1]
        if (runner["mae_mean"] - best["mae_mean"]) / best["mae_mean"] < 0.02 and (
            runner["peak_mae_mean"] < best["peak_mae_mean"]
        ):
            selected = runner["model"]
            reason = (
                f"MAE within 2% of best ({runner['mae_mean']:.2f} vs {best['mae_mean']:.2f}) "
                f"but lower peak MAE ({runner['peak_mae_mean']:.2f} vs {best['peak_mae_mean']:.2f})"
            )
    return {"selected": str(selected), "reason": reason, "audit": audit}


def ablation_delta(
    X: pd.DataFrame,
    y: pd.Series,
    folds: list,
    factory,
    cfg: ForecastConfig,
    feature_groups: dict,
) -> pd.DataFrame:
    """Feature-ablation runs: drop one group, measure the MAE delta.

    ``feature_groups`` maps group name -> list of columns to remove. A near-zero
    (or negative) delta for the negative-control group is the expected result.
    """

    def _mean_mae(cols) -> float:
        maes = []
        for train_idx, val_idx in folds:
            model = factory()
            model.fit(X[cols].iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X[cols].iloc[val_idx])
            maes.append(float(np.mean(np.abs(y.iloc[val_idx].to_numpy() - pred))))
        return float(np.mean(maes))

    full_cols = list(X.columns)
    base_mae = _mean_mae(full_cols)
    rows = [{"ablation": "full_feature_set", "mean_mae": base_mae, "delta_vs_full": 0.0}]
    for group, cols in feature_groups.items():
        kept = [c for c in full_cols if c not in set(cols)]
        m = _mean_mae(kept)
        rows.append({"ablation": f"drop_{group}", "mean_mae": m, "delta_vs_full": m - base_mae})
    _ = cfg
    return pd.DataFrame(rows)
