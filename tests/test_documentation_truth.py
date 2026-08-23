"""Documentation truth guards (Phase 6).

Ensures stable docs contain no permanent mutable numeric current-count claims.
Historical counts must be tied to date/commit context.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.static_contract

ROOT = Path(__file__).resolve().parents[1]


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_readme_no_permanent_mutable_counts():
    txt = _read(ROOT / "README.md")
    # Fail if README contains a bare "164 passed" or "149 passed, 10 skipped" style permanent claim without historical qualifier
    # We allow such strings inside historical sections if preceded by "Snapshot" or "historical" or "2026-08"
    # Simple guard: README should not contain a line with "passed" and a number that is not inside a historical note
    # For now, we enforce that README contains no line with "164 passed" or "149 passed" at all (since those are baseline historical numbers moved to CHANGELOG/report)
    # and no "current: \d+ passed" pattern
    assert "164 passed" not in txt, "README must not contain permanent 164 passed claim; move to dated historical report"
    assert "149 passed" not in txt, "README must not contain permanent 149 passed claim"
    assert "159 passed" not in txt
    # mutable current-count claim pattern: "current:" + number + "passed" (case insensitive)
    assert not re.search(r"current\s*:\s*\d+\s*passed", txt, flags=re.I), "README must not contain unlabeled current:N passed"
    # also guard "Measured (2026-08-18)" without snapshot wording is stale; should be historical snapshot wording
    if "Measured (2026-08-18)" in txt:
        assert "Snapshot" in txt or "historical" in txt.lower()


def test_project_spec_no_permanent_mutable_counts():
    txt = _read(ROOT / "PROJECT_SPEC.md")
    assert "164 full local" not in txt, "PROJECT_SPEC must not contain permanent 164 full local count"
    assert "149 passed" not in txt
    assert not re.search(r"current\s*:\s*\d+\s*passed", txt, flags=re.I)
    # Should describe tiers, not numeric current counts
    assert "static_contract" in txt or "tracked_artifacts" in txt, "PROJECT_SPEC should describe test tiers"


def test_changelog_historical_counts_are_dated():
    txt = _read(ROOT / "CHANGELOG.md")
    # Historical counts in CHANGELOG should be tied to date/commit; check that the snapshot line mentions commit
    if "164 passed" in txt:
        # Find surrounding context 200 chars
        for m in re.finditer(r"164 passed", txt):
            ctx = txt[max(0, m.start() - 400): m.end() + 400]
            assert "23cc934" in ctx or "2026-08-18" in ctx, "Historical 164 passed must be tied to date/commit"


def test_config_provenance_manifest_wording_matches_code():
    """Ensure docs mention correct config fields that exist in code."""
    # Check that README mentions correct tier commands
    readme = _read(ROOT / "README.md")
    assert 'pytest -m static_contract' in readme
    assert 'pytest -m "not needs_artifacts"' in readme or "not needs_artifacts" in readme
    # Check that FD004 config docs mention structured optimizer if code does
    cfg_text = _read(ROOT / "configs/final_model_m3_fd004.yaml")
    assert "optimizer:" in cfg_text
    # provenance docs should mention source_tree_hash is via git ls-files, not filesystem
    # We check that reproducibility docstring contains expected phrase
    import inspect
    from rul_prediction import reproducibility

    assert "git ls-files" in inspect.getsource(reproducibility.tracked_source_tree_details)
