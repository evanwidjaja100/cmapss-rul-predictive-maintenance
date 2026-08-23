"""Variable-history feature engineering for Methodology M1 classical models.

Features are computed from OBSERVED cycles only (rows up to and including the
cutoff cycle). Shorter histories are handled gracefully: rolling means fall
back to the latest value when insufficient history exists, so no synthetic
sensor history is invented. No future cycles, no full-lifetime information and
no true RUL ever enter the feature vector.
"""

from __future__ import annotations

import numpy as np

RECENT_MEANS = (5, 10)
FEATURE_GROUPS = ("latest", "mean", "std", "min", "max", "slope", "delta", "last5_mean", "last10_mean")
N_EXTRA_FEATURES = 2  # engine_age, history_length


def _ls_slope(observed: np.ndarray) -> np.ndarray:
    n = len(observed)
    t = np.arange(n, dtype=float)
    t_c = t - t.mean()
    denom = np.sum(t_c**2)
    if denom == 0:  # n == 1 -> slope undefined, use zeros
        return np.zeros(observed.shape[1], dtype=float)
    mean = observed.mean(axis=0)
    return np.sum(t_c[:, None] * (observed - mean[None, :]), axis=0) / denom


def extract_m1_features(observed: np.ndarray, cutoff_cycle: int):
    """Return (features, names) for ONE engine's observed history.

    Parameters
    ----------
    observed : (n_observed, n_features) scaled rows for cycles 1..n_observed.
    cutoff_cycle : 1-based cycle at which the prediction is made (engine age).

    Returns
    -------
    features : float array (n_feat,) ; names : list[str] of length n_feat.
    n_feat = n_sensors * len(FEATURE_GROUPS) + N_EXTRA_FEATURES.
    """
    observed = np.asarray(observed, dtype=float)
    n_observed = observed.shape[0]
    assert n_observed >= 1
    last = observed[-1]
    parts = [
        last,
        observed.mean(axis=0),
        observed.std(axis=0),
        observed.min(axis=0),
        observed.max(axis=0),
        _ls_slope(observed),
        last - observed[0],
    ]
    for k in RECENT_MEANS:
        parts.append(observed[-k:].mean(axis=0) if n_observed >= k else last)

    features = np.concatenate([p.reshape(-1) for p in parts])
    features = np.concatenate([features, [float(cutoff_cycle), float(n_observed)]])
    names = [f"sensor_{j + 1}_{group}" for group in FEATURE_GROUPS for j in range(observed.shape[1])]
    names += ["engine_age", "history_length"]
    assert len(features) == len(names) == observed.shape[1] * len(FEATURE_GROUPS) + N_EXTRA_FEATURES
    return features, names