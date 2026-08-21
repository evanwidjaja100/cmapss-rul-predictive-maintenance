"""Dependency consistency tests (Phase 9).

Validates subset/category contract, platform-selected constraints, no path deps,
CI consumption, and Python policy agreement. Runs via `python scripts/check_dependency_consistency.py`
and is marked static_contract (artifact-free, no gitignored files).
"""
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.static_contract

ROOT = Path(__file__).resolve().parents[1]


def test_dependency_consistency_checker_passes():
    result = subprocess.run([sys.executable, "scripts/check_dependency_consistency.py"], capture_output=True, text=True, cwd=str(ROOT), timeout=30)
    assert result.returncode == 0, f"dependency checker failed:\n{result.stdout}\n{result.stderr}"
    assert "OK: dependency consistency passed" in result.stdout


def test_pip_check_passes():
    result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, cwd=str(ROOT), timeout=30)
    assert result.returncode == 0, f"pip check failed: {result.stdout} {result.stderr}"
    assert "No broken requirements found" in result.stdout or result.stdout.strip() == ""


def test_pyproject_and_requirements_subset():
    # Direct check: pyproject runtime deps subset of requirements.txt
    import re
    py_txt = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r"dependencies\s*=\s*\[(.*?)\]", py_txt, re.DOTALL)
    assert m, "cannot find dependencies"
    deps_raw = re.findall(r'"([^"]+)"|\'([^\']+)\'', m.group(1))
    py_deps = []
    for a,b in deps_raw:
        dep = a or b
        mm = re.match(r"([A-Za-z0-9_.\-]+)", dep)
        if mm:
            py_deps.append(mm.group(1).lower().replace("_","-"))
    req_txt = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    req_names = []
    for line in req_txt.splitlines():
        l = line.strip()
        if not l or l.startswith("#"):
            continue
        if ";" in l:
            l = l.split(";")[0].strip()
        mm = re.match(r"([A-Za-z0-9_.\-]+)", l)
        if mm:
            req_names.append(mm.group(1).lower().replace("_","-"))
    for dep in py_deps:
        assert dep in req_names, f"pyproject dep {dep} not in requirements.txt"


def test_no_path_dependencies_in_constraints():
    for rel in ["requirements-lock.txt", "requirements-ci-linux-py312.txt"]:
        p = ROOT / rel
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            l = line.strip()
            if not l or l.startswith("#"):
                continue
            assert "file://" not in l.lower(), f"{rel} has file:// path dep: {l}"
            assert not l.startswith("-e "), f"{rel} has editable dep: {l}"
            assert "@" not in l or "://" not in l, f"{rel} has VCS dep: {l}"


def test_python_policy_consistent():
    py_txt = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    ci_txt = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert '3.12.10' in py_txt, "pyproject should mention 3.12.10"
    assert '3.12.10' in ci_txt, "CI must pin 3.12.10"
    assert 'requires-python' in py_txt
    # Check header mentions
    lock_header = (ROOT / "requirements-lock.txt").read_text(encoding="utf-8").splitlines()[:15]
    assert any("Python: 3.12.10" in l for l in lock_header), "lock header missing Python 3.12.10"
    assert any("Pip: 26.2.1" in l for l in lock_header), "lock header missing pip version"


def test_ci_consumes_platform_selected_constraints():
    ci_txt = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "requirements-lock.txt" in ci_txt
    assert "requirements-ci-linux-py312.txt" in ci_txt
    assert "CONSTRAINTS=" in ci_txt or "constraints" in ci_txt.lower()
    assert "pip==26.2.1" in ci_txt or 'pip==26.2.1' in ci_txt
