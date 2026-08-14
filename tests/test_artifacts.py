"""Data-contract tests on the persisted processed artifacts (skipped when absent)."""

from pathlib import Path

import numpy as np
import pytest

PROCESSED = Path("data/processed")


@pytest.mark.skipif(not (PROCESSED / "FD001_train_sequences.npz").exists(),
                    reason="processed artifacts not present")
def test_sequence_features_are_scaled():
    """Regression guard: windows must carry SCALED sensor values (std ~ 1),
    never raw magnitudes (a silent unscaled-features bug collapsed NNs)."""
    d = np.load(PROCESSED / "FD001_train_sequences.npz")
    X = d["X"]
    per_feature_std = X.reshape(-1, X.shape[-1]).std(axis=0)
    assert np.all(per_feature_std < 10), per_feature_std
    assert X.shape[1] == 30 and X.shape[2] == 21
    assert X.dtype == np.float32


@pytest.mark.skipif(not (PROCESSED / "FD001_train_sequences.npz").exists(),
                    reason="processed artifacts not present")
def test_targets_are_clipped_rul():
    d = np.load(PROCESSED / "FD001_train_sequences.npz")
    assert d["y"].max() <= 125.0
    assert d["y"].min() >= 0.0