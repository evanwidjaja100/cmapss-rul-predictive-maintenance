"""Leakage-safe RUL construction and feature scaling.

All fitted quantities (e.g. the scaler) must be fit on TRAINING ENGINES ONLY;
the same transformer is then applied to train, validation and test features.
"""

from __future__ import annotations

from pathlib import Path

from joblib import dump, load

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]


def add_rul(frame, max_rul: int | None = None, clip: bool = True):
    """Add `rul = max_cycle(engine) - cycle`, optionally clipped at `max_rul`."""
    frame = frame.copy()
    frame["rul"] = frame.groupby("engine_id")["cycle"].transform("max") - frame["cycle"]
    if clip and max_rul is not None:
        frame["rul"] = frame["rul"].clip(upper=max_rul)
    return frame


def fit_scaler(frame, feature_cols):
    """Fit a StandardScaler on the given (training-only) rows. Never fit on the full set."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(frame[feature_cols].to_numpy(dtype=float))
    return scaler


def transform(frame, feature_cols, scaler) -> object:
    """Return scaled feature array for a frame using the fitted scaler."""
    return scaler.transform(frame[feature_cols].to_numpy(dtype=float))


def save_scaler(scaler, path: str | Path) -> None:
    dump(scaler, path)


def load_scaler(path: str | Path) -> object:
    return load(path)