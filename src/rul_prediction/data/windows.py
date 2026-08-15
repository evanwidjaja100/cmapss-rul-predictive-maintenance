"""Single shared implementation of short-history handling (left-padding).

Used identically by training, validation, calibration, serving and official
test inference so that the representation seen at inference time always
appears during training (Methodology V2, Issue 6 fix).

Convention: rows are sorted by cycle; row index ``cycle - 1``. Padding uses
zeros (scaled sensor means are ~0); a binary mask marks observed timesteps
(1 = observed, 0 = padded) for masking-aware sequence models.
"""

from __future__ import annotations

import numpy as np


def build_window(scaled: np.ndarray, cutoff_cycle: int, window: int):
    """Left-padded window ending at ``cutoff_cycle`` for ONE engine.

    Parameters
    ----------
    scaled : float array (n_cycles, n_features) for one engine, sorted by cycle.
    cutoff_cycle : 1-based cycle at which the prediction is made (1..n_cycles).
    window : requested history length in cycles.

    Returns
    -------
    (window_array, n_observed, padded) :
        window_array  (window, n_features) float32; leading (window - n_observed)
                      rows are zero padding, observed rows are the last
                      ``n_observed`` cycles up to and including the cutoff.
        n_observed   observed cycles INSIDE the window (1..window).
        padded       True when the engine had fewer than ``window`` cycles.
    """
    assert 1 <= cutoff_cycle <= len(scaled), f"cutoff {cutoff_cycle} out of range for {len(scaled)} cycles"
    observed = scaled[:cutoff_cycle]  # cycles 1..cutoff -> rows 0..cutoff-1; no future cycles
    n_observed = min(len(observed), window)  # observed cycles INSIDE the window (1..window)
    if len(observed) >= window:
        return observed[len(observed) - window :].astype(np.float32), n_observed, False
    pad = np.zeros((window - len(observed), observed.shape[1]), dtype=np.float32)
    return np.concatenate([pad, observed], axis=0).astype(np.float32), n_observed, True


def window_mask(n_observed: int, window: int) -> np.ndarray:
    """Binary mask (1 = observed timestep, 0 = padded) for a window of given length."""
    mask = np.zeros(window, dtype=np.float32)
    mask[window - n_observed :] = 1.0
    return mask