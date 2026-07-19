# Smart Building Energy Forecasting — One-Hour-Ahead Load Forecast

[![CI](https://github.com/Sayomphon/Energy_Forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/Sayomphon/Energy_Forecasting/actions/workflows/ci.yml)

Time-series ML that predicts appliance energy use **60 minutes ahead** from
10-minute smart-home readings, engineered the way forecasting has to be
engineered: **leakage-safe features, chronological backtesting, naive baselines
as the bar, and a packaged inference contract with graceful fallback.**

## Problem

Buildings and factories plan energy ahead of time to avoid peak demand charges
and keep comfort/production stable. A 1-hour-ahead forecast of appliance load
gives facility operators time to pre-position HVAC and load control, raise peak
alerts, and plan demand — an explainable horizon that still opens the door to
demand response later.

- **Prediction unit**: appliance energy (Wh) at `t + 60 min`, for every timestamp `t`
- **Data**: [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
  — 19,735 records, 10-min sampling, ~4.5 months, one house in Belgium (CC BY 4.0)
- **Human-in-the-loop**: forecasts carry a quality flag; operators confirm any setpoint change

## Approach

```
10-min readings → time validation → target = shift(-6) → past-only features
     → chronological split (70/15/15) → expanding-window backtest (3 folds)
     → baselines vs Ridge vs HistGradientBoosting → one-shot test evaluation
     → bundle + sha256 → inference contract with freshness fallback
```

What makes it defensible:

1. **No leakage, by test not by convention** — every feature is audited for
   "known at time t"; unit tests *mutate the future* and assert features at `t`
   don't change ([tests/test_temporal_features.py](tests/test_temporal_features.py)).
2. **Baselines are first-class** — last-value and target-time-aligned seasonal
   naive run through the same backtest loop; a candidate is only eligible if it
   beats **every** baseline in a majority of folds.
3. **Stability over a single score** — 3 expanding-window folds report mean ± std,
   plus latency and model size; peak error uses a train-only q90 threshold.
4. **The test block is opened once**, after the selection rule is frozen.
5. **One feature implementation** — training and inference call the same
   `add_features`, eliminating training-serving skew.
6. **Negative controls** — the dataset's random columns `rv1`/`rv2` stay in;
   if they rank important, the model is fitting noise.

## Result

The shipped model is **HistGradientBoosting on the lean `v2` feature set**
(`energy-1h-v2`, 25 features). It was chosen over the full `v1` set (54 features)
by a controlled experiment whose promotion criteria were fixed *before* the run.

Backtest over 3 expanding-window folds (validation MAE in Wh, mean ± std), HGB
on each feature set:

| Feature set | Features | MAE | WAPE | Peak MAE (≥ train q90) | Fit time |
|---|---|---|---|---|---|
| **`v2` lean** ✅ shipped | 25 | **40.68 ± 4.18** | 0.419 | 237.9 | 0.7 s |
| `v1` full | 54 | 43.96 ± 8.09 | 0.453 | 239.0 | 1.1 s |

On both sets HGB beat every baseline in **3/3 folds** (Ridge 2/3; last-value
55.4, seasonal-naive 61.4 Wh) — audit trail in `artifacts/model_selection.json`,
head-to-head in `artifacts/feature_set_comparison.csv`. One-shot held-out test
(final ~3 weeks, n = 2,938): **MAE 33.6 · WAPE 0.347 · bias −15.7 · peak
MAE 220.0** (`v1` was MAE 34.0 / peak 220.4).

The `v1` → `v2` experiment, and honest findings straight from the artifacts:

- **Dropping the sensor block was tested, not assumed.** `v1` ablation flagged
  that removing all weather/indoor sensors *improves* MAE by ~3.3 Wh. Since that
  could be an artefact of ablating a fully-fit model, `v2` was **retrained from
  scratch** on the lean set and backtested on identical folds. It cleared both
  pre-set bars — MAE −3.27 Wh and peak MAE −0.45% (no regression) — so `v2` was
  promoted. It also **roughly halves fold-to-fold variance** (MAE std 8.09 →
  4.18): fewer noisy inputs, a steadier model. (`scripts/compare_feature_sets.py`)
- **Peaks are still the hard part**: peak MAE (220 Wh) dwarfs overall MAE
  (34 Wh), and test bias is −16 Wh — the model systematically under-forecasts
  spikes. Next step: quantile regression / a dedicated peak-event classifier.
- **Negative controls behave**: `rv1`/`rv2` stay in even in `v2`; dropping them
  shifts MAE by ~0.3 Wh (noise level) — the model is not mining the random columns.
- **Residual ACF at lag 1 = 0.64**: short-term structure remains that lagged
  features don't fully capture.

Reproduce: `python -m energy_forecasting.train --fetch --feature-set v2` (shipped
model), `--feature-set v1` for the baseline, or `python
scripts/compare_feature_sets.py` for the head-to-head.

## Key engineering decisions

| Decision | Why |
|---|---|
| Feature-based models (Ridge, HistGradientBoosting), no LSTM | ~19k rows + 10 h timebox: temporal correctness and backtesting prove more than sequence-model complexity |
| `shift(1)` before every rolling window | rolling stats must never see the value at `t` itself, and never the future |
| Seasonal reference aligned to **target** time (`t + 1h − 24h`) | "same time yesterday" must be measured against the time being forecast, not the time of forecasting |
| Peak threshold from train quantile only | evaluation slices must not peek at their own labels |
| MAE loss in the gradient booster | aligned with the MAE/WAPE-centric evaluation |
| sha256 sidecar verified before unpickling the bundle | joblib deserialisation executes code — never load unverified artifacts |
| Fallback instead of NaN at inference | stale/short history returns a flagged last-value forecast, never a silent guess |

## Repository layout

```
├── src/energy_forecasting/   # installable package: all temporal logic lives here
│   ├── config.py             # frozen ForecastConfig — single source of truth
│   ├── data.py               # cached ingestion + lineage + data contract
│   ├── features.py           # past-only features + availability audit
│   ├── splits.py             # chronological split + expanding-window folds
│   ├── models.py             # baselines (first-class) + Ridge + HGB
│   ├── metrics.py            # MAE/RMSE/WAPE/bias/peak-MAE + slices
│   ├── backtest.py           # fold runner + selection rule + ablation
│   ├── inference.py          # bundle save/load (sha256) + forecast contract
│   └── train.py              # end-to-end CLI
├── notebooks/energy_1h_forecast.ipynb   # 19-section narrative notebook
├── tests/                    # 64 tests — temporal correctness is CI-enforced
├── artifacts/                # generated: bundle, configs, metrics, reports
├── docs/model_card.md
└── PROJECT_LOG.md            # build log (Thai)
```

## Quickstart

```bash
pip install -e ".[dev]"
pytest                                     # 64 tests
python -m energy_forecasting.train --fetch # downloads UCI data once (~12 MB), trains, packages
```

Programmatic inference:

```python
from energy_forecasting.inference import load_bundle, predict_one
bundle = load_bundle("artifacts/forecast_bundle.joblib")   # sha256-verified
predict_one(bundle, history_df, "2026-07-18T10:00:00")
# {'prediction_time': ..., 'target_time': ..., 'forecast_appliances_wh': ...,
#  'quality_flag': 'ok', 'fallback_used': False, 'model_version': 'energy-1h-v2'}
```

## Limitations

- **Domain shift**: one Belgian house ≠ Thai buildings/factories — different
  seasons, holidays, occupancy, and BMS. Requires local re-validation and
  retraining before any real use.
- **Point forecasts only** (no uncertainty intervals yet — stretch: quantile /
  conformal regression).
- Weather is *observed at t*, not a forecast feed; adequate at 1 h, not day-ahead.
- Proven result = forecast accuracy on held-out time under stated availability
  assumptions — **not** energy-cost savings or HVAC optimisation.

## Attribution & license

- Dataset: Candanedo, L. (2017). *Appliances Energy Prediction* [Dataset].
  UCI Machine Learning Repository. Licensed **CC BY 4.0**. Data is fetched from
  the source at build time and is not redistributed in this repository.
- Code: Apache License 2.0 (see [LICENSE](LICENSE)).
