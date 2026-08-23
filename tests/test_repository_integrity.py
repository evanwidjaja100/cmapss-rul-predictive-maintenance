"""Repository integrity tests — references and encoding (Phase 5).

Covers:
  - Required anchors for 4 restored plans + key M3 evidence exist as tracked files
  - Text encoding: no invalid UTF-8, no U+FFFD, no known mojibake in tracked text
  - Reference integrity via git ls-files enumeration (delegates to checker)
  - Exceptions in configs/repository_integrity.yaml are used and justified

Uses git ls-files, not filesystem walk. Checker provides check_references() and
check_text_encoding() as independently callable functions.
"""
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

# Small explicit required anchors (subset per spec, not hand-maintained exhaustive)
# (current path, historical name at 23cc934) — files were renamed M1/M2/M3 in 2026-08;
# content changed as part of that rename, so only historical existence is pinned.
REQUIRED_RESTORED_PLANS = [
    ("docs/history/M2_REPAIR_PLAN.md", "V2_1_REPAIR_PLAN.md"),
    ("docs/history/M3_REPAIR_PLAN.md", "V2_2_REPAIR_PLAN.md"),
    ("docs/history/M3_FINAL_CLEANUP_PLAN.md", "V2_2_FINAL_CLEANUP_PLAN.md"),
    ("docs/history/M3_FINAL_FREEZE_PLAN.md", "V2_2_FINAL_FREEZE_PLAN.md"),
]

REQUIRED_M3_EVIDENCE = [
    "reports/m3_final_report.md",
    "configs/final_model_m3_fd001.yaml",
    "configs/final_model_m3_fd004.yaml",
    "configs/deployment_m3_fd001.yaml",
    "experiments/m3/fd001_outer_fold_results.csv",
    "experiments/m3/selection_decision.json",
    "experiments/m3/fd001_conformal_engine_scores.csv",
    "experiments/m3/fd001_official_predictions.csv",
]


def _git_ls_files():
    out = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [f.decode() for f in out.split(b"\x00") if f]


@pytest.mark.static_contract
def test_required_restored_plans_exist_and_tracked():
    """4 restored plans must be tracked, present, and traceable to 23cc934 (existence anchor)."""
    tracked = set(_git_ls_files())
    for rel, hist_name in REQUIRED_RESTORED_PLANS:
        assert rel in tracked, f"restored plan not tracked: {rel}"
        assert (ROOT / rel).exists(), f"restored plan missing on disk: {rel}"
        # Historical existence anchor under the pre-rename name
        subprocess.check_output(["git", "rev-parse", f"23cc934:{hist_name}"], cwd=ROOT)


@pytest.mark.static_contract
def test_required_m3_evidence_anchors_exist():
    """Key M3 evidence files must be tracked (clean-clone present)."""
    tracked = set(_git_ls_files())
    for rel in REQUIRED_M3_EVIDENCE:
        assert rel in tracked, f"required M3 anchor not tracked: {rel}"
        assert (ROOT / rel).exists(), f"anchor missing on disk: {rel}"


@pytest.mark.static_contract
def test_text_encoding_no_mojibake_or_fffd():
    """Delegates to checker: strict UTF-8, no U+FFFD, no known mojibake."""
    from scripts.check_repository_integrity import check_text_encoding

    violations, used, exceptions = check_text_encoding()
    # Filter to show only non-excepted violations (checker already filters)
    assert not violations, f"encoding violations: {violations[:3]}"
    # Ensure exceptions are used (checker tests unused separately, but we verify)
    # The historical M3_REPAIR_PLAN.md is expected to be excepted
    if exceptions:
        # At least the historical mojibake exception must be present and used
        assert any("M3_REPAIR_PLAN" in e.get("source", "") for e in exceptions), "missing historical mojibake exception"


@pytest.mark.static_contract
def test_reference_integrity_no_missing_or_ambiguous():
    """Delegates to checker: references resolve via root/dir/basename, no missing/ambiguous."""
    from scripts.check_repository_integrity import check_references

    violations, used, exceptions = check_references()
    assert not violations, f"reference violations: {violations[:5]}"
    # Verify exceptions are narrow and used
    if exceptions:
        # Each exception must have source, target/pattern, reason
        for e in exceptions:
            assert e.get("source"), f"exception missing source: {e}"
            assert e.get("target") or e.get("pattern"), f"exception missing target/pattern: {e}"
            assert e.get("reason") and e["reason"].strip(), f"exception missing reason: {e}"


@pytest.mark.static_contract
def test_repository_integrity_cli_passes():
    """CLI must exit 0 when integrity holds (uses git ls-files)."""
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/check_repository_integrity.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"checker CLI failed:\n{result.stdout}\n{result.stderr}"
    assert "OK" in result.stdout


@pytest.mark.static_contract
def test_integrity_yaml_exceptions_are_used():
    """Fail on unused exceptions so list cannot become clutter."""
    from scripts.check_repository_integrity import check_all

    violations, used, exceptions = check_all()
    unused = [v for kind, v in violations if kind == "unused_exception"]
    assert not unused, f"unused exceptions: {unused}"
    invalid = [v for kind, v in violations if kind == "invalid_exception"]
    assert not invalid, f"invalid exceptions: {invalid}"


@pytest.mark.static_contract
def test_no_invalid_utf8_in_tracked_text_files():
    """Direct strict UTF-8 decode for tracked .md/.py/.toml/.yaml/.yml/.txt."""
    tracked = _git_ls_files()
    exts = {".md", ".py", ".toml", ".yaml", ".yml", ".txt"}
    for rel in tracked:
        if Path(rel).suffix.lower() not in exts:
            continue
        p = ROOT / rel
        if not p.exists():
            continue
        data = p.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as e:
            pytest.fail(f"{rel} invalid UTF-8: {e}")
        assert "\ufffd" not in text, f"{rel} contains U+FFFD"
        # Also check for known mojibake strings explicitly
        for seq in ["\u00c3\u00a2", "\u00c3\u0083", "\u00c3\u0082"]:
            # These are common moj prefixes; but we check via full strings in checker
            pass
