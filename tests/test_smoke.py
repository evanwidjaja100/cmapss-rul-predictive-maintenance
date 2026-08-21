"""Phase 0 smoke tests: package imports and basic metadata."""

import rul_prediction
import pytest

pytestmark = pytest.mark.unit


def test_package_imports():
    assert rul_prediction.__version__


def test_package_name():
    assert rul_prediction.__name__ == "rul_prediction"