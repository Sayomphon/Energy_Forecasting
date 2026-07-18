"""One-hour-ahead appliance energy load forecasting.

Leakage-safe time-series pipeline for the UCI Appliances Energy Prediction
dataset: data contract validation, past-only feature engineering,
chronological splits with expanding-window backtesting, naive baselines,
and a packaged inference bundle with freshness-aware fallback.
"""

from energy_forecasting.config import ForecastConfig

__version__ = "1.0.0"

__all__ = ["ForecastConfig", "__version__"]
