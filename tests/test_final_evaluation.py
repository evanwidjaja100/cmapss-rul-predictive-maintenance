"""Unit tests for the final-evaluation harness (test-window padding logic)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

FIN_EVAL = Path(__file__).resolve().parents[1] / "scripts" / "final_evaluation.py"


@pytest.fixture(scope="module")
def fe_mod():
    spec = importlib.util.spec_from_file_location("final_evaluation", FIN_EVAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_padding_short_unit_keeps_tail(fe_mod):
    rng = np.random.default_rng(0)
    data = rng.normal(size=(151, 21)).astype(np.float32)
    unit_ids = np.concatenate([np.full(31, 1), np.full(120, 2)])
    X, n_cycles, padded = fe_mod._test_windows(data, unit_ids, window=90)
    short = data[:31]
    assert X.shape[0] == 2
    assert n_cycles.tolist() == [31, 120]
    assert padded.tolist() == [True, False]
    assert X[0].shape == (90, 21)
    np.testing.assert_allclose(X[0, 90 - 31 :], short, atol=1e-6)
    np.testing.assert_allclose(X[0, : 90 - 31], 0.0, atol=1e-6)
    assert X[1].shape == (90, 21)


def test_long_unit_takes_last_window(fe_mod):
    rng = np.random.default_rng(1)
    long = rng.normal(size=(150, 21)).astype(np.float32)
    X, n_cycles, padded = fe_mod._test_windows(long, np.full(150, 7), window=90)
    assert X.shape[0] == 1 and n_cycles[0] == 150 and not padded[0]
    np.testing.assert_allclose(X[0], long[60:], atol=1e-6)