#!/usr/bin/env python3
"""Repository integrity checker — references and text encoding.

Enumerates via `git ls-files` (no filesystem walk). Provides:
  - check_references()
  - check_text_encoding()
  - CLI returning nonzero on violations

Reference behavior:
  - Scans tracked Markdown (.md) and Python (.py) documentation/comments for
    Markdown link targets and path-like tokens with extensions
    .md .py .yaml .yml .toml .json .csv .txt .ipynb .joblib .keras .npz
  - Resolves against repo root, referring file's directory, unique basename
  - Rejects missing or ambiguous with source file and line
  - Excludes URLs, anchors, templates/placeholders, gitignored artifacts
  - Exceptions in configs/repository_integrity.yaml (source, target/pattern, reason)
  - Fails on unused exceptions
  - Small required-anchor assertion for 4 restored plans + key V2.2 evidence

Encoding behavior:
  - Strict UTF-8 decode for tracked .md .py .toml .yaml .yml .txt
  - Rejects U+FFFD and known mojibake sequences (explicit, no blind transform)
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]

# Known mojibake byte sequences decoded as utf-8 strings (explicit, reviewed)
# Generated from cp1252 double-encoding: original utf-8 -> cp1252 decode -> utf-8 encode
# Each moj string is what appears when moj bytes are decoded as utf-8
_MOJ_BYTES_TO_CORRECT = [
    (b'\xc3\xa2\xe2\x82\xac\xe2\x80\x9d', b'\xe2\x80\x94'),  # em dash
    (b'\xc3\xa2\xe2\x82\xac\xe2\x80\x9c', b'\xe2\x80\x93'),  # en dash
    (b'\xc3\xa2\xe2\x80\xa0\xe2\x80\x99', b'\xe2\x86\x92'),  # arrow
    (b'\xc3\xa2\xe2\x82\xac\xc2\xa6', b'\xe2\x80\xa6'),  # ellipsis
    (b'\xc3\x83\xe2\x80\x94', b'\xc3\x97'),  # multiply
    (b'\xc3\x82\xc2\xb1', b'\xc2\xb1'),  # plusminus
    (b'\xc3\x82\xc2\xb2', b'\xc2\xb2'),  # squared
    (b'\xc3\x8e\xc2\xb1', b'\xce\xb1'),  # alpha
    (b'\xc3\xa2\xcb\x86\xe2\x80\x99', b'\xe2\x88\x92'),  # minus
    (b'\xc3\xa2\xe2\x80\xb0\xcb\x86', b'\xe2\x89\x88'),  # approx
    (b'\xc3\x82\xc2\xa7', b'\xc2\xa7'),  # section
]
# Moj strings for detection (decode moj bytes as utf-8)
MOJIBAKE_STRINGS = [moj.decode('utf-8') for moj, _ in _MOJ_BYTES_TO_CORRECT]

# Extensions for reference targets and for encoding check
REF_EXTS = {".md", ".py", ".yaml", ".yml", ".toml", ".json", ".csv", ".txt", ".ipynb", ".joblib", ".keras", ".npz"}
ENCODING_EXTS = {".md", ".py", ".toml", ".yaml", ".yml", ".txt"}

# Required anchors (small explicit list per spec)
REQUIRED_ANCHORS = [
    "V2_1_REPAIR_PLAN.md",
    "V2_2_REPAIR_PLAN.md",
    "V2_2_FINAL_CLEANUP_PLAN.md",
    "V2_2_FINAL_FREEZE_PLAN.md",
    "reports/v2_2_final_report.md",
    "configs/final_model_v2_2_fd001.yaml",
    "configs/final_model_v2_2_fd004.yaml",
    "configs/deployment_v2_2_fd001.yaml",
    "experiments/v2_2/fd001_outer_fold_results.csv",
    "experiments/v2_2/selection_decision.json",
    "experiments/v2_2/fd001_conformal_engine_scores.csv",
    "experiments/v2_2/fd001_conformal_quantiles.csv",
    "experiments/v2_2/fd001_official_predictions.csv",
    "experiments/v2_2/fd004_variant_results.csv",
]

# Gitignored prefixes (from .gitignore) to exclude from reference checking
IGNORED_PREFIXES = [
    "models/",
    "data/raw/",
    "data/processed/",
    "data/interim/",
    "reports/figures/",
    ".venv/",
    "__pycache__/",
    ".pytest_cache/",
    ".ipynb_checkpoints/",
    "build/",
    "dist/",
    ".opencode/",
]
# Allowed exceptions for ignored: experiments/* except splits/v2_1/v2_2
# We handle via function is_ignored_path

PLACEHOLDER_KEYWORDS = ["...", "example", "placeholder", "path/to", "your_", "my_", "<", ">", "*", "???"]

REPO_PREFIXES = [
    "V2_", "C_MAPSS", "configs/", "experiments/", "reports/", "scripts/", "src/", "tests/",
    "notebooks/", "requirements", "pyproject", ".github/", "LICENSE", "README", "PROJECT_SPEC",
    "CHANGELOG", "AUDIT", "THIRD_PARTY",
]

def _is_repo_like(token: str, basename_map) -> bool:
    # Check if token starts with known repo prefix
    for pref in REPO_PREFIXES:
        if token.startswith(pref):
            return True
    # Check if basename exists in tracked set (bare filename reference)
    base = PurePosixPath(token).name
    if base in basename_map:
        return True
    # Also check if token contains a repo prefix as part of path (e.g., "some/dir/reports/...")?
    for pref in REPO_PREFIXES:
        if pref.rstrip("/") in token:
            # If token contains a known prefix substring, consider repo-like
            # but be conservative: only if prefix appears at start of a path component
            if f"/{pref}" in token or token.startswith(pref):
                return True
    return False

def _git_ls_files(root: Path | None = None) -> list[str]:
    root = root or ROOT
    out = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    # -z gives NUL separated
    files = out.split(b"\x00")
    return [f.decode() for f in files if f]

def _load_exceptions(root: Path | None = None):
    root = root or ROOT
    cfg = root / "configs" / "repository_integrity.yaml"
    if not cfg.exists():
        return []
    import yaml  # pyyaml is in dependencies
    data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    # Support both `exceptions` key or top-level list
    if isinstance(data, dict):
        ex = data.get("exceptions", [])
        if ex is None:
            ex = []
    elif isinstance(data, list):
        ex = data
    else:
        ex = []
    # Normalize: ensure each has source, (target or pattern), reason
    normalized = []
    for e in ex:
        if not isinstance(e, dict):
            continue
        source = e.get("source")
        target = e.get("target")
        pattern = e.get("pattern")
        reason = e.get("reason")
        if not source or not reason or not reason.strip():
            # invalid, will be flagged as violation later
            normalized.append(e)
            continue
        if not target and not pattern:
            normalized.append(e)
            continue
        normalized.append({"source": source, "target": target, "pattern": pattern, "reason": reason, "_raw": e})
    return normalized

def _is_url(token: str) -> bool:
    return "://" in token or token.startswith("http:") or token.startswith("https:") or token.startswith("ftp:")

def _is_template(token: str) -> bool:
    low = token.lower()
    for kw in PLACEHOLDER_KEYWORDS:
        if kw in low:
            return True
    if "<" in token or ">" in token:
        return True
    return False

def _is_ignored_path(token: str) -> bool:
    # Check prefixes
    for pref in IGNORED_PREFIXES:
        if token.startswith(pref):
            return True
        # Also check if token is inside ignored dir via relative?
        if f"/{pref}" in token:
            return True
    # Handle experiments/* exception
    if token.startswith("experiments/"):
        if not (token.startswith("experiments/splits/") or token.startswith("experiments/v2_1/") or token.startswith("experiments/v2_2/")):
            # e.g., experiments/results.csv, experiments/FD001_final_test_results.json are ignored
            return True
    # Check for *.egg-info, *.pyc etc
    if token.endswith(".egg-info") or token.endswith(".pyc") or token.endswith(".pyo"):
        return True
    return False

def _strip_anchor(token: str) -> str:
    if "#" in token:
        base = token.split("#", 1)[0]
        return base
    return token

def check_text_encoding(root: Path | None = None, tracked_files: list[str] | None = None):
    """Strict UTF-8 decode for tracked .md .py .toml .yaml .yml .txt.

    Rejects:
      - invalid UTF-8 (UnicodeDecodeError)
      - U+FFFD replacement char
      - known mojibake sequences (explicit list)

    Returns list of violations: dict with file, line, message
    """
    root = root or ROOT
    if tracked_files is None:
        tracked_files = _git_ls_files(root)
    violations = []
    for rel in tracked_files:
        ext = Path(rel).suffix.lower()
        if ext not in ENCODING_EXTS:
            continue
        p = root / rel
        if not p.exists():
            continue
        data = p.read_bytes()
        # Strict UTF-8
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as e:
            violations.append({
                "file": rel,
                "line": 1,
                "message": f"invalid UTF-8: {e}",
                "snippet": repr(data[max(0, e.start-20):e.end+20]),
            })
            continue
        # Check BOM? Not flagged as error per spec, but we can note
        # Check for FFFD
        if "\ufffd" in text:
            # Find lines with FFFD
            for idx, line in enumerate(text.splitlines(), 1):
                if "\ufffd" in line:
                    col = line.index("\ufffd") + 1
                    violations.append({
                        "file": rel,
                        "line": idx,
                        "col": col,
                        "message": "contains U+FFFD replacement character",
                        "snippet": line.strip()[:300],
                    })
        # Check mojibake strings
        for moj_str in MOJIBAKE_STRINGS:
            if moj_str in text:
                for idx, line in enumerate(text.splitlines(), 1):
                    if moj_str in line:
                        col = line.index(moj_str) + 1
                        violations.append({
                            "file": rel,
                            "line": idx,
                            "col": col,
                            "message": f"contains mojibake sequence {moj_str!r}",
                            "snippet": line.strip()[:500],
                        })
        # Also check for decoded moj that is 2-char but original check may miss single char?
        # The moj strings above cover all 11, so sufficient
    # Check exceptions for encoding: if file is in exception list with pattern mojibake/FFFD, allow
    exceptions = _load_exceptions(root)
    filtered = []
    used_exceptions = set()
    for v in violations:
        matched = False
        for ei, e in enumerate(exceptions):
            src = e.get("source")
            pat = e.get("pattern")
            tgt = e.get("target")
            # For encoding, source should match file, pattern should match message or be generic
            if src and src not in v["file"] and not v["file"].endswith(src):
                continue
            # If exception has pattern, check if pattern in message or snippet
            if pat:
                try:
                    if re.search(pat, v["message"]) or re.search(pat, v["snippet"]):
                        matched = True
                        used_exceptions.add(ei)
                        break
                    # Also allow generic "mojibake" or "FFFD" keyword
                    if pat.lower() in v["message"].lower() or pat.lower() == "mojibake":
                        # For encoding, if pattern is mojibake and violation is mojibake, match
                        if "mojibake" in v["message"].lower() and "mojibake" in pat.lower():
                            matched = True
                            used_exceptions.add(ei)
                            break
                        if "fffd" in v["message"].lower() and "fffd" in pat.lower():
                            matched = True
                            used_exceptions.add(ei)
                            break
                except re.error:
                    if pat in v["message"] or pat in v["snippet"]:
                        matched = True
                        used_exceptions.add(ei)
                        break
            if tgt:
                # For encoding, target may be a substring to match snippet
                if tgt in v["snippet"] or tgt in v["message"]:
                    matched = True
                    used_exceptions.add(ei)
                    break
            # If exception is for this file generically (e.g., historical preservation)
            # and no specific pattern, consider it matched if source matches
            if src and (src == v["file"] or v["file"].endswith(src)) and not pat and not tgt:
                matched = True
                used_exceptions.add(ei)
                break
        if not matched:
            filtered.append(v)
    # Check for unused exceptions (only those that were intended for encoding)
    # We need to consider all exceptions, but we only mark used if matched above
    # For reference unused, will be checked in check_references; here we handle encoding unused separately
    # To avoid double reporting, we will let check_references handle unused, but we also check here
    # Instead, we will not yet flag unused here; the main CLI will check combined
    return filtered, used_exceptions, exceptions

def _extract_markdown_link_targets(line: str):
    # Matches [text](target)
    pattern = re.compile(r'\[([^\]]*)\]\(([^)]+)\)')
    return pattern.findall(line)

def _extract_path_tokens(text: str):
    # Path-like tokens with supported extensions
    # This matches strings like `reports/v2_2_final_report.md` or `configs/final_model.yaml`
    # Allow backticks, quotes around
    # Use regex for path with extension
    exts = "|".join(re.escape(e.lstrip(".")) for e in sorted(REF_EXTS, key=lambda x: -len(x)))
    # Pattern: word chars, dot, slash, hyphen, underscore, then dot + ext, word boundary
    pattern = re.compile(rf'(?:^|[^a-zA-Z0-9_\-./])([a-zA-Z0-9_\-./]+\.(?:{exts}))\b')
    # Also handle quoted tokens separately? This will capture inside quotes as well
    tokens = []
    for m in pattern.finditer(text):
        token = m.group(1)
        # Strip leading characters that are not part of path
        token = token.strip()
        tokens.append((token, m.start(1)))
    return tokens

def _extract_python_doc_text(path: Path) -> dict[int, str]:
    """Extract documentation text from Python file: comments and docstrings.

    Returns dict mapping line numbers to doc-relevant text for that line.
    Only the actual comment/docstring lines are included, not duplicated across ranges.
    """
    import ast
    import tokenize
    import io
    try:
        source = path.read_text(encoding="utf-8")
    except:
        return {}
    doc_lines: dict[int, str] = {}
    # Extract comments via tokenize
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok_type, tok_str, start, end, line in tokens:
            if tok_type == tokenize.COMMENT:
                row = start[0]
                doc_lines[row] = doc_lines.get(row, "") + " " + tok_str
    except Exception:
        for idx, line in enumerate(source.splitlines(), 1):
            if "#" in line:
                comment_part = line.split("#", 1)[1]
                doc_lines[idx] = doc_lines.get(idx, "") + " " + comment_part
    # Extract docstrings via ast - only add docstring text to its start line, not to every line in range
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)
                if doc and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                    doc_node = node.body[0]
                    start_line = doc_node.lineno
                    # Only add docstring to its start line to avoid duplicate violations per line
                    doc_lines[start_line] = doc_lines.get(start_line, "") + " " + doc
                elif doc:
                    lineno = getattr(node, 'lineno', 1)
                    doc_lines[lineno] = doc_lines.get(lineno, "") + " " + doc
    except Exception:
        pass
    return doc_lines

def check_references(root: Path | None = None, tracked_files: list[str] | None = None):
    """Check Markdown link targets and path-like tokens.

    Returns (violations, used_exception_indices, all_exceptions)
    violations: list of dict with file, line, col, target, message
    """
    root = root or ROOT
    if tracked_files is None:
        tracked_files = _git_ls_files(root)
    tracked_set = set(tracked_files)
    # basename map
    basename_map: dict[str, list[str]] = {}
    for f in tracked_files:
        base = PurePosixPath(f).name
        basename_map.setdefault(base, []).append(f)

    violations = []
    # Collect candidates to scan: .md and .py
    scan_files = [f for f in tracked_files if Path(f).suffix.lower() in {".md", ".py"}]

    for rel in scan_files:
        p = root / rel
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Will be flagged by encoding check; skip reference check for this file
            continue
        # For Python, only scan documentation/comments
        if Path(rel).suffix.lower() == ".py":
            doc_lines = _extract_python_doc_text(p)
            if not doc_lines:
                continue
            for idx, line in sorted(doc_lines.items()):
                candidates = []
                for _, target in _extract_markdown_link_targets(line):
                    candidates.append((target, line.find(target)))
                for token, col in _extract_path_tokens(line):
                    if token not in [c[0] for c in candidates]:
                        candidates.append((token, col))
                for token_raw, col in candidates:
                    token = token_raw.strip().strip("`'\"")
                    token = token.rstrip(".,;:)]}")
                    token = token.lstrip("([{'\"`")
                    if _is_url(token):
                        continue
                    base_token = _strip_anchor(token)
                    if not base_token:
                        continue
                    token_to_check = base_token
                    if _is_template(token_to_check):
                        continue
                    if _is_ignored_path(token_to_check):
                        continue
                    ext = PurePosixPath(token_to_check).suffix.lower()
                    if ext not in REF_EXTS:
                        continue
                    if not _is_repo_like(token_to_check, basename_map):
                        continue
                    resolved = False
                    ambiguous = None
                    if token_to_check in tracked_set:
                        resolved = True
                    else:
                        rel_dir = PurePosixPath(rel).parent
                        candidate_rel = (rel_dir / token_to_check).as_posix()
                        parts = []
                        for part in PurePosixPath(candidate_rel).parts:
                            if part == "..":
                                if parts and parts[-1] != "..":
                                    parts.pop()
                            elif part == ".":
                                continue
                            else:
                                parts.append(part)
                        normalized = PurePosixPath(*parts).as_posix() if parts else ""
                        if normalized.startswith("./"):
                            normalized = normalized[2:]
                        if normalized in tracked_set:
                            resolved = True
                        else:
                            base = PurePosixPath(token_to_check).name
                            if base in basename_map:
                                candidates_list = basename_map[base]
                                if len(candidates_list) == 1:
                                    resolved = True
                                elif len(candidates_list) > 1:
                                    ambiguous = f"ambiguous basename '{base}' matches {candidates_list}"
                    if not resolved:
                        msg = ambiguous if ambiguous else f"missing tracked reference '{token_to_check}'"
                        violations.append({
                            "file": rel,
                            "line": idx,
                            "col": col + 1,
                            "target": token_to_check,
                            "raw": token_raw,
                            "message": msg,
                            "snippet": line.strip()[:500],
                        })
            continue
        # For Markdown, scan all lines
        lines = text.splitlines()
        for idx, line in enumerate(lines, 1):
            candidates = []

            # Markdown links (only for .md, but also check .py docstrings that may contain them)
            for _, target in _extract_markdown_link_targets(line):
                candidates.append((target, line.find(target)))

            # Path-like tokens
            for token, col in _extract_path_tokens(line):
                # Avoid duplicate if already captured as link target
                if token not in [c[0] for c in candidates]:
                    candidates.append((token, col))

            for token_raw, col in candidates:
                token = token_raw.strip().strip("`'\"")
                # Strip trailing punctuation like ., ,, ;, :, ), ], }
                token = token.rstrip(".,;:)]}")
                token = token.lstrip("([{'\"`")
                # Exclude URLs
                if _is_url(token):
                    continue
                # Strip anchor for checking base
                base_token = _strip_anchor(token)
                if not base_token:
                    continue
                # If anchor was present, we already handle base
                token_to_check = base_token
                # Exclude templates/placeholders
                if _is_template(token_to_check):
                    continue
                # Exclude gitignored artifacts
                if _is_ignored_path(token_to_check):
                    continue
                # Check if token has supported extension, else skip (already filtered)
                ext = PurePosixPath(token_to_check).suffix.lower()
                if ext not in REF_EXTS:
                    continue
                # Only consider repo-like references to reduce false positives
                if not _is_repo_like(token_to_check, basename_map):
                    continue
                # Resolve
                resolved = False
                ambiguous = None
                # 1. Direct repo root
                if token_to_check in tracked_set:
                    resolved = True
                else:
                    # 2. Relative to referring file's directory
                    rel_dir = PurePosixPath(rel).parent
                    candidate_rel = (rel_dir / token_to_check).as_posix()
                    # Normalize: remove ./, handle ../
                    # Use PurePosixPath to normalize
                    # We need to handle .. segments: posix doesn't resolve .. without filesystem, so manual
                    # Simplistic: use Path to normalize via posix
                    # We can use PurePosixPath to get parts and resolve
                    # For simplicity, use Path(norm).as_posix after using PurePath?
                    # Let's normalize by splitting and handling ..
                    parts = []
                    for part in PurePosixPath(candidate_rel).parts:
                        if part == "..":
                            if parts and parts[-1] != "..":
                                parts.pop()
                        elif part == ".":
                            continue
                        else:
                            parts.append(part)
                    normalized = PurePosixPath(*parts).as_posix() if parts else ""
                    # Remove leading ./ if present
                    if normalized.startswith("./"):
                        normalized = normalized[2:]
                    if normalized in tracked_set:
                        resolved = True
                    else:
                        # 3. Unique basename match
                        base = PurePosixPath(token_to_check).name
                        if base in basename_map:
                            candidates_list = basename_map[base]
                            if len(candidates_list) == 1:
                                resolved = True
                            elif len(candidates_list) > 1:
                                ambiguous = f"ambiguous basename '{base}' matches {candidates_list}"
                            else:
                                ambiguous = None
                if not resolved:
                    msg = ambiguous if ambiguous else f"missing tracked reference '{token_to_check}'"
                    violations.append({
                        "file": rel,
                        "line": idx,
                        "col": col + 1,
                        "target": token_to_check,
                        "raw": token_raw,
                        "message": msg,
                        "snippet": line.strip()[:500],
                    })
    # Check required anchors
    for anchor in REQUIRED_ANCHORS:
        if anchor not in tracked_set:
            # Also check if anchor basename exists uniquely? But required anchors should be exact
            violations.append({
                "file": "REQUIRED_ANCHORS",
                "line": 0,
                "col": 0,
                "target": anchor,
                "raw": anchor,
                "message": f"required anchor missing: '{anchor}' not tracked",
                "snippet": "",
            })
        else:
            # Also check that anchor is actually referenced somewhere? The spec says retain assertion for 4 restored plans
            # We could check that at least one file references it, but for now just existence is required
            pass

    # Apply exceptions
    exceptions = _load_exceptions(root)
    filtered = []
    used = set()
    for v in violations:
        matched = False
        for ei, e in enumerate(exceptions):
            src = e.get("source")
            pat = e.get("pattern")
            tgt = e.get("target")
            # Source must match if specified
            if src:
                # src can be a file path or "REQUIRED_ANCHORS"
                if src != v["file"] and not v["file"].endswith(src) and src != "REQUIRED_ANCHORS":
                    # Also allow src to be a directory prefix
                    if not v["file"].startswith(src):
                        continue
            # Check target/pattern
            if pat:
                try:
                    # pattern is regex
                    if re.search(pat, v["target"]) or re.search(pat, v["message"]) or re.search(pat, v["snippet"]):
                        matched = True
                        used.add(ei)
                        break
                    # Also try plain substring
                    if pat in v["target"] or pat in v["message"]:
                        matched = True
                        used.add(ei)
                        break
                except re.error:
                    if pat in v["target"] or pat in v["message"]:
                        matched = True
                        used.add(ei)
                        break
            if tgt:
                if tgt == v["target"] or tgt in v["target"] or tgt in v["message"] or tgt == v["raw"]:
                    matched = True
                    used.add(ei)
                    break
                # Also check if tgt is a substring of snippet
                if tgt in v["snippet"]:
                    matched = True
                    used.add(ei)
                    break
            # If exception has only source and reason, and source matches, consider it matched
            # (e.g., for encoding historical files)
            if src and not pat and not tgt:
                # This is a generic exception for that source file; match any violation from that file
                matched = True
                used.add(ei)
                break
        if not matched:
            filtered.append(v)

    return filtered, used, exceptions

def check_all(root: Path | None = None):
    root = root or ROOT
    tracked = _git_ls_files(root)
    enc_violations, enc_used, enc_exceptions = check_text_encoding(root, tracked)
    ref_violations, ref_used, ref_exceptions = check_references(root, tracked)
    # Combine exceptions handling: they share same yaml, so we need to check overall unused
    # Load once and check which indices were used in either check
    all_exceptions = _load_exceptions(root)
    used_all = enc_used.union(ref_used)
    unused = []
    for ei, e in enumerate(all_exceptions):
        if ei not in used_all:
            # Check if exception is valid (has source and reason)
            # If it's unused, it's a violation per spec: fail on unused exceptions
            unused.append((ei, e))
    # Combine violations
    all_violations = []
    for v in enc_violations:
        all_violations.append(("encoding", v))
    for v in ref_violations:
        all_violations.append(("reference", v))
    for ei, e in unused:
        all_violations.append(("unused_exception", {"file": "configs/repository_integrity.yaml", "line": 0, "message": f"unused exception {ei}: {e}", "target": str(e)}))
    # Also check that exceptions have required fields
    for ei, e in enumerate(all_exceptions):
        src = e.get("source")
        tgt = e.get("target")
        pat = e.get("pattern")
        reason = e.get("reason")
        if not src or (not tgt and not pat) or not reason or not reason.strip():
            all_violations.append(("invalid_exception", {"file": "configs/repository_integrity.yaml", "line": 0, "message": f"invalid exception {ei}: must have source, target/pattern, nonempty reason: {e}", "target": str(e)}))
    return all_violations, used_all, all_exceptions

def _safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Repository integrity checker")
    parser.add_argument("--root", type=str, default=None, help="repository root override")
    parser.add_argument("--json", action="store_true", help="output json")
    args = parser.parse_args()
    root = Path(args.root) if args.root else ROOT
    violations, used, exceptions = check_all(root)
    if violations:
        _safe_print(f"Repository integrity: {len(violations)} violation(s) found")
        for kind, v in violations:
            file = v.get("file", "?")
            line = v.get("line", "?")
            msg = v.get("message", "")
            tgt = v.get("target", "")
            snippet = v.get("snippet", "")
            _safe_print(f"[{kind}] {file}:{line} -> {tgt} : {msg}")
            if snippet:
                _safe_print(f"  snippet: {snippet[:500]}")
        # Also print required anchors status
        _safe_print("\nRequired anchors:")
        tracked = _git_ls_files(root)
        for anchor in REQUIRED_ANCHORS:
            status = "OK" if anchor in tracked else "MISSING"
            _safe_print(f"  {anchor}: {status}")
        sys.exit(1)
    else:
        _safe_print("Repository integrity: OK")
        _safe_print(f"  checked {len(_git_ls_files(root))} tracked files")
        _safe_print(f"  required anchors: {len(REQUIRED_ANCHORS)} OK")
        if exceptions:
            _safe_print(f"  exceptions: {len(exceptions)} all used")
        sys.exit(0)

if __name__ == "__main__":
    main()
