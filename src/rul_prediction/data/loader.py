"""Load and describe raw NASA C-MAPSS text files into pandas DataFrames."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NUM_SENSORS = 21
NUM_SETTINGS = 3

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, NUM_SENSORS + 1)]
SETTING_COLUMNS = [f"setting_{i}" for i in range(1, NUM_SETTINGS + 1)]

DATA_COLUMNS = ["engine_id", "cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS]

# Number of training and test engines per dataset (official C-MAPSS magnitudes).
EXPECTED_ENGINE_COUNTS = {
    "FD001": {"train": 100, "test": 100},
    "FD002": {"train": 260, "test": 259},
    "FD003": {"train": 100, "test": 100},
    "FD004": {"train": 249, "test": 248},
}


def sensor_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c in SENSOR_COLUMNS]


def _read_raw(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    frame.columns = DATA_COLUMNS[: frame.shape[1]]
    return frame


def load_train(dataset: str, data_dir: str | Path = "data/raw") -> pd.DataFrame:
    path = Path(data_dir) / f"train_{dataset}.txt"
    return _read_raw(path)


def load_test(dataset: str, data_dir: str | Path = "data/raw") -> pd.DataFrame:
    path = Path(data_dir) / f"test_{dataset}.txt"
    return _read_raw(path)


def load_rul(dataset: str, data_dir: str | Path = "data/raw") -> np.ndarray:
    """Load true RUL values for the official test set (int array per engine)."""
    path = Path(data_dir) / f"RUL_{dataset}.txt"
    return pd.read_csv(path, header=None, sep=r"\s+", engine="python")[0].to_numpy(
        dtype=int
    )


def summarize(frame: pd.DataFrame, dataset: str, kind: str) -> dict:
    """Compute basic dataset-level descriptive statistics (no feature removal)."""
    n_engines = frame["engine_id"].nunique()
    lifetime = frame.groupby("engine_id")["cycle"].max()
    return {
        "dataset": dataset,
        "kind": kind,
        "train_engines": n_engines,
        "rows": int(len(frame)),
        "operating_settings": NUM_SETTINGS,
        "sensors": NUM_SENSORS,
        "min_lifetime": int(lifetime.min()),
        "max_lifetime": int(lifetime.max()),
        "mean_lifetime": float(lifetime.mean()),
        "median_lifetime": float(lifetime.median()),
        "std_lifetime": float(lifetime.std()),
    }


def data_summary_lines(summary: dict) -> list[str]:
    return [
        f"Dataset: {summary['dataset']}",
        f"{summary['kind'].title()} engines: {summary['train_engines']}",
        f"{summary['kind'].title()} rows: {summary['rows']}",
        f"Operating settings: {summary['operating_settings']}",
        f"Sensors: {summary['sensors']}",
        f"Minimum training lifetime: {summary.get('min_lifetime')}",
        f"Maximum training lifetime: {summary.get('max_lifetime')}",
        f"Mean lifetime: {summary.get('mean_lifetime')}",
        f"Median lifetime: {summary.get('median_lifetime')}",
        f"Lifetime standard deviation: {summary.get('std_lifetime')}",
    ]