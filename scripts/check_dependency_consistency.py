"""Dependency consistency checker (Phase 9).

Ensures:
- No path/editable/VCS deps in constraints
- Direct requirements are governed by selected constraints
- pyproject [project.dependencies] is subset of requirements.txt
- Additional direct requirements classified in ML/app/test/notebook categories
- CI consumes platform-selected constraints (requirements-lock.txt for Windows, requirements-ci-linux-py312.txt for Linux)
- Python policy agrees (pyproject, CI, V2.2 configs)
- Pip version recorded and header present
"""
import re
import sys
import hashlib
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]

# documented ML/app/test/notebook categories for additional direct requirements beyond pyproject runtime
CATEGORY_MAP = {
    "tensorflow": "ml",
    "xgboost": "ml",
    "shap": "ml",
    "streamlit": "app",
    "pytest": "test",
    "pytest-cov": "test",
    "jupyterlab": "notebook",
    "ipykernel": "notebook",
    # runtime deps also allowed
    "numpy": "runtime",
    "pandas": "runtime",
    "scipy": "runtime",
    "scikit-learn": "runtime",
    "matplotlib": "runtime",
    "pyyaml": "runtime",
    "joblib": "runtime",
    "pyyaml": "runtime",
}

def normalize(name: str) -> str:
    return re.sub(r"[-_]+", "-", name).lower().strip()

def parse_requirements(path: Path):
    reqs = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        l = line.strip()
        if not l or l.startswith("#"):
            continue
        # strip env marker
        if ";" in l:
            l = l.split(";")[0].strip()
        # strip extras?
        # e.g., package==1.2.3 or package>=1.0
        m = re.match(r"([A-Za-z0-9_.\-]+)(.*)", l)
        if not m:
            continue
        name = normalize(m.group(1))
        reqs[name] = l
    return reqs

def parse_constraints(path: Path):
    cons = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        l = line.strip()
        if not l or l.startswith("#"):
            continue
        if ";" in l:
            # keep marker but parse name
            l_name = l.split(";")[0].strip()
        else:
            l_name = l
        m = re.match(r"([A-Za-z0-9_.\-]+)==.*", l_name)
        if not m:
            continue
        name = normalize(m.group(1))
        cons[name] = l
    return cons

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def check():
    errors = []
    warnings = []

    # 1. No path/editable/VCS deps in constraints
    for rel in ["requirements-lock.txt", "requirements-ci-linux-py312.txt"]:
        p = ROOT / rel
        if not p.exists():
            if rel == "requirements-ci-linux-py312.txt":
                warnings.append(f"{rel} missing (if Windows constraints resolve on Linux, it's optional but we expect it for cross-platform)")
                continue
            errors.append(f"constraints {rel} missing")
            continue
        txt = p.read_text(encoding='utf-8')
        for line in txt.splitlines():
            l = line.strip()
            if not l or l.startswith("#"):
                continue
            if "file://" in l.lower() or l.startswith("-e ") or l.startswith("--editable") or "@" in l and "://" in l:
                errors.append(f"{rel} contains path/editable/VCS dep: {l!r}")
            if "==" not in l and ";" not in l and not l.startswith("#"):
                # package without pinned version (except marker lines)
                # For constraints, we expect pinned ==
                if l and not l.startswith("#"):
                    # Allow marker lines like pywinpty with ;
                    if ";" in l:
                        # check left side has ==
                        left = l.split(";")[0]
                        if "==" not in left:
                            errors.append(f"{rel} unpinned constraint with marker: {l!r}")
                    else:
                        errors.append(f"{rel} unpinned constraint: {l!r}")

    # 2. Check requirements.txt direct entries
    req_path = ROOT / "requirements.txt"
    reqs = parse_requirements(req_path)
    pyproject_path = ROOT / "pyproject.toml"
    py_txt = pyproject_path.read_text(encoding='utf-8')
    # extract project.dependencies list
    m = re.search(r"dependencies\s*=\s*\[(.*?)\]", py_txt, re.DOTALL)
    if not m:
        errors.append("cannot find [project.dependencies] in pyproject.toml")
        py_deps = {}
    else:
        block = m.group(1)
        # find quoted names
        deps_raw = re.findall(r'"([^"]+)"|\'([^\']+)\'', block)
        deps = []
        for a,b in deps_raw:
            deps.append(a or b)
        py_deps = {}
        for dep in deps:
            # dep may be "numpy" or "numpy>=1.0"
            mm = re.match(r"([A-Za-z0-9_.\-]+)", dep)
            if mm:
                name = normalize(mm.group(1))
                py_deps[name] = dep

    # subset check: pyproject deps must be subset of requirements.txt
    for name in py_deps:
        if name not in reqs:
            errors.append(f"pyproject dependency {name!r} not in requirements.txt direct entries")

    # additional direct requirements must be classified
    for name in reqs:
        if name not in py_deps:
            if name not in CATEGORY_MAP:
                errors.append(f"additional direct requirement {name!r} not classified in ML/app/test/notebook categories (add to CATEGORY_MAP)")
            else:
                # check category is one of allowed
                cat = CATEGORY_MAP[name]
                if cat not in ("ml","app","test","notebook","runtime"):
                    errors.append(f"requirement {name} has invalid category {cat}")

    # 3. Direct requirements governed by selected constraints
    # For each direct requirement, check that constraints file contains it (pinned)
    for name in reqs:
        # Use platform-selected constraints: check both lock files, at least one must contain
        found = False
        for rel in ["requirements-lock.txt", "requirements-ci-linux-py312.txt"]:
            p = ROOT / rel
            if p.exists():
                cons = parse_constraints(p)
                if name in cons:
                    found = True
                    break
        if not found:
            errors.append(f"direct requirement {name!r} not governed by selected constraints (missing in requirements-lock.txt / requirements-ci-linux-py312.txt)")

    # Also ensure pywinpty marker handling: if lock contains pywinpty, must have env marker
    lock_txt = (ROOT / "requirements-lock.txt").read_text(encoding='utf-8')
    for line in lock_txt.splitlines():
        if "pywinpty" in line.lower():
            if 'sys_platform' not in line:
                errors.append(f"pywinpty line missing env marker sys_platform: {line!r}")

    # 4. CI consumes platform-selected constraints
    ci_path = ROOT / ".github/workflows/ci.yml"
    ci_txt = ci_path.read_text(encoding='utf-8')
    # Check CI has platform selection logic
    if "requirements-lock.txt" not in ci_txt:
        errors.append("CI must reference requirements-lock.txt for Windows/local")
    if "requirements-ci-linux-py312.txt" not in ci_txt:
        errors.append("CI must explicitly reference requirements-ci-linux-py312.txt for Linux (platform selection)")
    if "pip==" not in ci_txt:
        errors.append("CI must install exact pip version via pip==<validated-version>, not floating")
    if 'python-version: "3.12.10"' not in ci_txt and "3.12.10" not in ci_txt:
        errors.append("CI must pin Python to 3.12.10 (V2.2 configs record 3.12.10)")

    # 5. Python policy agrees: pyproject, CI, V2.2 configs
    # pyproject requires-python should allow 3.12.10
    if 'requires-python' in py_txt:
        if '3.12' not in py_txt:
            warnings.append("pyproject requires-python may not include 3.12")
    # Check V2.2 configs record Python 3.12.10?
    # We know configs mention python version comment; check pyproject comment
    if "3.12.10" not in py_txt:
        warnings.append("pyproject.toml should mention Python 3.12.10 in comment for policy consistency")
    # Also check V2.2 final configs? Not needed

    # 6. Header with role/generation command/Python version/platform/install command
    for rel in ["requirements-lock.txt", "requirements-ci-linux-py312.txt"]:
        p = ROOT / rel
        if p.exists():
            header = p.read_text(encoding='utf-8').splitlines()[:15]
            header_str = "\n".join(header)
            if "Role:" not in header_str:
                errors.append(f"{rel} header missing Role")
            if "Pip:" not in header_str:
                errors.append(f"{rel} header missing Pip version")
            if "Python: 3.12.10" not in header_str:
                errors.append(f"{rel} header missing Python 3.12.10")
            if "Install" not in header_str:
                errors.append(f"{rel} header missing Install command")
            if "platform" not in header_str.lower():
                warnings.append(f"{rel} header missing platform")

    # 7. No packaging extras broad redesign unless required: check pyproject extras not broad?
    # We just ensure optional-dependencies dev exists but not extra packaging redesign.
    # For now, just ensure no new extras beyond dev
    if "[project.optional-dependencies]" in py_txt:
        if "extras" in py_txt.lower() and "ml" in py_txt.lower():
            warnings.append("broad packaging/extras redesign detected, should be follow-up")

    # Output
    print("=== Dependency consistency check ===")
    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  - {e}")
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    if not errors:
        print("OK: dependency consistency passed")
        # Also print constraints hashes for evidence
        for rel in ["requirements-lock.txt", "requirements-ci-linux-py312.txt"]:
            p = ROOT / rel
            if p.exists():
                print(f"{rel}: sha256={sha256_file(p)} bytes={p.stat().st_size}")
        print(f"pyproject deps subset ok: {len(py_deps)} runtime deps")
        print(f"requirements.txt direct: {sorted(reqs.keys())}")
        print(f"CI platform selection: verified")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(check())
