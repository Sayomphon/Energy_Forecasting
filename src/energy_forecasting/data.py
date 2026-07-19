"""Data ingestion, caching, and the data contract.

Download is deliberately separated from training (docx section 4: failure
handling): ``fetch_raw`` hits the network once and caches a CSV with a sha256
sidecar; everything downstream reads only the cache, so reruns survive an
unstable network and lineage stays verifiable.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from energy_forecasting.config import (
    UCI_DATASET_ID,
    UCI_DATASET_NAME,
    UCI_DATASET_URL,
    UCI_LICENSE,
    ForecastConfig,
)

logger = logging.getLogger(__name__)

RAW_CSV_NAME = "appliances_energy_raw.csv"
LINEAGE_NAME = "data_lineage.json"

# The ucimlrepo export of this dataset drops the space between date and time
# ("2016-01-1117:00:00"). The raw cache stays faithful to the source; parsing
# is normalised here, in code, where it is versioned and testable.
_TIMESTAMP_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d%H:%M:%S")


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse timestamps strictly, tolerating the known no-space UCI format.

    Tries each explicit format on the whole column (all-or-nothing, so mixed
    or ambiguous formats are never silently half-parsed). Returns NaT-filled
    series only if no format matches, which the data contract then rejects.
    """
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.Series(values)
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return pd.to_datetime(values, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(values, errors="coerce")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_raw(data_dir: Path, force: bool = False) -> Path:
    """Download the UCI dataset once and cache it locally with lineage metadata.

    Returns the path to the cached CSV. Requires network + ``ucimlrepo`` only
    on the first call (or with ``force=True``).
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / RAW_CSV_NAME
    if csv_path.exists() and not force:
        logger.info("Using cached raw data at %s", csv_path)
        return csv_path

    from ucimlrepo import fetch_ucirepo  # imported lazily: only needed online

    logger.info("Fetching UCI dataset id=%s (%s)", UCI_DATASET_ID, UCI_DATASET_NAME)
    ds = fetch_ucirepo(id=UCI_DATASET_ID)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)

    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    lineage = {
        "source": UCI_DATASET_URL,
        "dataset_id": UCI_DATASET_ID,
        "dataset_name": UCI_DATASET_NAME,
        "license": UCI_LICENSE,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "sha256": _sha256(csv_path),
    }
    (data_dir / LINEAGE_NAME).write_text(json.dumps(lineage, indent=2))
    logger.info("Cached %s rows to %s (sha256=%s)", df.shape[0], csv_path, lineage["sha256"][:12])
    return csv_path


def load_raw(data_dir: Path, verify_hash: bool = True) -> pd.DataFrame:
    """Load the cached CSV, verifying integrity against the lineage sidecar."""
    data_dir = Path(data_dir)
    csv_path = data_dir / RAW_CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No cached dataset at {csv_path}. Run fetch_raw() first "
            "(network + ucimlrepo required once)."
        )
    lineage_path = data_dir / LINEAGE_NAME
    if verify_hash and lineage_path.exists():
        expected = json.loads(lineage_path.read_text()).get("sha256")
        actual = _sha256(csv_path)
        if expected and actual != expected:
            raise ValueError(
                f"Raw data hash mismatch: expected {expected}, got {actual}. "
                "The cache was modified; re-run fetch_raw(force=True)."
            )
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------


@dataclass
class DataContractReport:
    """Outcome of the pre-training data contract checks (docx section 3)."""

    n_rows: int = 0
    time_min: str = ""
    time_max: str = ""
    n_duplicate_timestamps: int = 0
    n_gaps: int = 0
    gap_examples: list = field(default_factory=list)
    n_negative_target: int = 0
    modal_interval_minutes: float = 0.0
    passed: bool = False
    failures: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "time_min": self.time_min,
            "time_max": self.time_max,
            "n_duplicate_timestamps": self.n_duplicate_timestamps,
            "n_gaps": self.n_gaps,
            "gap_examples": self.gap_examples,
            "n_negative_target": self.n_negative_target,
            "modal_interval_minutes": self.modal_interval_minutes,
            "passed": self.passed,
            "failures": self.failures,
        }


def validate_contract(df: pd.DataFrame, cfg: ForecastConfig) -> DataContractReport:
    """Check schema/time assumptions before any modelling.

    Hard failures (missing columns, unparseable time, negative target) set
    ``passed=False``; gaps and duplicates are reported but tolerated because
    the feature builder handles them explicitly.
    """
    report = DataContractReport()

    for col in (cfg.timestamp_col, cfg.target_source_col):
        if col not in df.columns:
            report.failures.append(f"missing required column: {col}")
    if report.failures:
        return report

    ts = parse_timestamps(df[cfg.timestamp_col])
    n_unparseable = int(ts.isna().sum())
    if n_unparseable:
        report.failures.append(f"{n_unparseable} unparseable timestamps")

    ts_sorted = ts.sort_values()
    report.n_rows = len(df)
    report.time_min = str(ts_sorted.iloc[0])
    report.time_max = str(ts_sorted.iloc[-1])
    report.n_duplicate_timestamps = int(ts_sorted.duplicated().sum())

    deltas = ts_sorted.diff().dropna()
    if len(deltas):
        modal = deltas.mode().iloc[0]
        report.modal_interval_minutes = modal.total_seconds() / 60.0
        expected = pd.Timedelta(minutes=cfg.sampling_minutes)
        gaps = deltas[deltas > expected]
        report.n_gaps = int(len(gaps))
        report.gap_examples = [
            f"{ts_sorted.iloc[i - 1]} -> {ts_sorted.iloc[i]} ({d})"
            for i, d in list(zip(gaps.index[:5], gaps.head(5)))
        ]
        if report.modal_interval_minutes != float(cfg.sampling_minutes):
            report.failures.append(
                f"modal interval {report.modal_interval_minutes} min "
                f"!= expected {cfg.sampling_minutes} min"
            )

    target = pd.to_numeric(df[cfg.target_source_col], errors="coerce")
    report.n_negative_target = int((target < 0).sum())
    if report.n_negative_target:
        report.failures.append(f"{report.n_negative_target} negative target values")
    if target.isna().any():
        report.failures.append(f"{int(target.isna().sum())} non-numeric target values")

    report.passed = not report.failures
    return report


def prepare_base(df: pd.DataFrame, cfg: ForecastConfig) -> pd.DataFrame:
    """Sort by time, drop duplicate timestamps, and attach the shifted label.

    The label is built *after* sorting (docx data contract) and the last
    ``horizon_steps`` rows — which have no future label — are dropped.
    """
    out = df.copy()
    out[cfg.timestamp_col] = parse_timestamps(out[cfg.timestamp_col])
    out = (
        out.sort_values(cfg.timestamp_col)
        .drop_duplicates(subset=cfg.timestamp_col, keep="first")
        .reset_index(drop=True)
    )
    out[cfg.target_col] = out[cfg.target_source_col].shift(-cfg.horizon_steps)
    out = out.dropna(subset=[cfg.target_col]).reset_index(drop=True)

    assert out[cfg.timestamp_col].is_monotonic_increasing
    assert not out[cfg.timestamp_col].duplicated().any()
    return out
