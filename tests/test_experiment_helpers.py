"""Unit tests for validation-only experiment infrastructure helpers."""

import importlib.util
from pathlib import Path

import pytest

RUN_EXPERIMENT = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py"


@pytest.fixture(scope="module")
def re_mod():
    spec = importlib.util.spec_from_file_location("run_experiment", RUN_EXPERIMENT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cap_from_variant(re_mod):
    assert re_mod._cap_from_variant("w30_c125_all") == 125
    assert re_mod._cap_from_variant("w90_c45_all") == 45
    assert re_mod._cap_from_variant("w30_cnone_all") is None