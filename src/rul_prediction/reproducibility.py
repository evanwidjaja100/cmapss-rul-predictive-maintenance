"""Deterministic provenance and source hashing (Phase 3).

Execution-input scope is explicit and versioned; tracked files are enumerated via
``git ls-files -z`` and their CURRENT WORKTREE bytes are hashed, so staged or
unstaged tracked edits change the digest. Hash encoding is domain-separated,
length-delimited and fail-closed (``cmapss-tracked-source-v1``).

Dirty-state semantics distinguish whole-repository and execution-scope dirtiness
using NUL-delimited status and binary diffs. Future freezes capture provenance
before training and reject dirty execution inputs by default.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.metadata as _md
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

# ponytail: minimal stdlib implementation; no new dependency.
# Execution-input scope versioned explicitly per Section 10.1.
CANONICAL_ALGO = "cmapss-tracked-source-v1"
SCHEMA_VERSION = "cmapss-provenance-v1"
SNAPSHOT_LIMIT_BYTES = 5 * 1024 * 1024  # ponytail: 5 MiB per file, increase if training needs larger tracked snapshot

# Execution inputs that drive training/inference/output. Only these tracked files
# contribute to source_tree_hash. Unrelated docs/reports are recorded separately
# as whole-repo dirtiness but do not affect the execution hash.
EXECUTION_INPUT_PREFIXES = (
    "src/",
    "scripts/",
    "configs/",
    ".github/workflows/",
)
EXECUTION_INPUT_EXACT = {
    "app_v2.py",
    "pyproject.toml",
    "requirements.txt",
    "requirements-lock.txt",
}

# Sensitive filename patterns for snapshot scanning (report-only unless explicitly blocked)
SENSITIVE_FILENAME_RE = re.compile(
    r"(?:secret|credential|passwd|shadow|\.env|id_rsa|id_ed25519|\.pem|\.key|token)",
    re.IGNORECASE,
)
# Very naive secret content patterns for reporting (not exhaustive)
SECRET_CONTENT_RES = [
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key
    re.compile(r"-----BEGIN (?:RSA )?PRIVATE KEY-----"),
    re.compile(r"ghp_[A-Za-z0-9]{36}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
]


def _resolve_git_root(root: Path | str | None = None) -> Path:
    """Resolve repository root via git, fallback to parents lookup."""
    if root is not None:
        p = Path(root).resolve()
        # if given path is inside repo, find git root via rev-parse
        try:
            out = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="strict",
                check=False,
                cwd=str(p if p.is_dir() else p.parent),
            )
            if out.returncode == 0 and out.stdout.strip():
                return Path(out.stdout.strip()).resolve()
        except Exception:
            pass
        # fallback: walk up looking for .git
        cur = p if p.is_dir() else p.parent
        for _ in range(10):
            if (cur / ".git").exists():
                return cur.resolve()
            if cur.parent == cur:
                break
            cur = cur.parent
        return p if p.is_dir() else p.parent
    # no root given: use file's repo root (parents[3] from src/rul_prediction)
    here = Path(__file__).resolve()
    # rul_prediction/reproducibility.py -> src -> repo root
    repo = here.parents[2]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            check=False,
            cwd=str(repo),
        )
        if out.returncode == 0 and out.stdout.strip():
            return Path(out.stdout.strip()).resolve()
    except Exception:
        pass
    return repo.resolve()


def sha256_file(path: Path | str) -> str:
    """Streaming SHA-256 of file bytes. Fail-closed on read errors."""
    p = Path(path)
    h = hashlib.sha256()
    # fail closed: propagate OSError, do not silently skip
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_execution_input(posix_path: str) -> bool:
    """Return True if tracked POSIX path is an execution input per versioned scope.

    Explicitly excludes __pycache__, .pyc, .egg-info, models, raw data, pytest caches
    even if force-tracked (per 10.1, P2-1).
    """
    # Exclude non-execution artifacts even if force-tracked. Match path
    # components so similarly named real source directories remain included.
    parts = PurePosixPath(posix_path).parts
    if (
        "__pycache__" in parts
        or ".pytest_cache" in parts
        or any(part.endswith(".egg-info") for part in parts)
        or posix_path.lower().endswith(".pyc")
    ):
        return False
    if posix_path in EXECUTION_INPUT_EXACT:
        return True
    for pref in EXECUTION_INPUT_PREFIXES:
        if posix_path.startswith(pref):
            return True
    return False


def _git_ls_files_tracked(root: Path) -> list[str]:
    """Return sorted list of tracked POSIX paths via git ls-files -z."""
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        capture_output=True,
        check=False,
        cwd=str(root),
    )
    if out.returncode != 0:
        # fail closed: if git ls-files fails, propagate
        raise RuntimeError(f"git ls-files failed in {root}: {out.stderr.decode(errors='replace')}")
    raw = out.stdout  # bytes, NUL delimited
    if not raw:
        return []
    # split on NUL, filter empty tail
    parts = raw.split(b"\x00")
    paths = []
    for b in parts:
        if not b:
            continue
        # decode as utf-8; if invalid, fail closed (strict)
        try:
            s = b.decode("utf-8")
        except UnicodeDecodeError as e:
            raise RuntimeError(f"invalid UTF-8 in git ls-files entry: {b!r}") from e
        # normalize to POSIX (git already does, but ensure)
        # also reject absolute or traversal entries
        if os.path.isabs(s) or ".." in Path(s).parts:
            raise RuntimeError(f"invalid tracked path contains traversal or absolute: {s!r}")
        paths.append(s)
    # ensure sorted and normalized (POSIX)
    return sorted(paths)


def _git_ls_files_execution(root: Path) -> list[str]:
    """Filtered execution-input tracked files sorted POSIX."""
    all_tracked = _git_ls_files_tracked(root)
    exec_files = [p for p in all_tracked if _is_execution_input(p)]
    return sorted(exec_files)


def _hash_execution_tree(root: Path, exec_files: list[str]) -> tuple[str, int]:
    """Domain-separated length-delimited hash of execution inputs.

    Uses cmapss-tracked-source-v1 with type tag, path length, path bytes,
    content length, raw content per sorted POSIX path. No ambiguous
    concatenation. Content is ALWAYS the CURRENT WORKTREE bytes of each
    enumerated tracked path, so staged or unstaged edits change the digest
    (Python executes worktree bytes). Fail-closed on any read failure:
    raises, never falls back to HEAD/index or skips silently.
    """
    h = hashlib.sha256()
    # domain separation
    h.update(CANONICAL_ALGO.encode("utf-8") + b"\x00")
    # include versioned scope identifier as well for future changes
    # file count will be hashed as part of stream for determinism
    h.update(struct.pack(">I", len(exec_files)))
    for posix_path in exec_files:
        path_bytes = posix_path.encode("utf-8")
        file_path = root / posix_path
        # symlink escape check on filesystem if file exists
        try:
            if file_path.is_symlink():
                target = file_path.resolve()
                try:
                    target.relative_to(root.resolve())
                except ValueError:
                    raise RuntimeError(f"repository-escaping symlink: {posix_path} -> {target}")
        except OSError as e:
            raise RuntimeError(f"unreadable execution input {posix_path}: {e}") from e

        # header: type tag 'F', path_len, path_bytes, content_len
        h.update(b"F")
        h.update(struct.pack(">I", len(path_bytes)))
        h.update(path_bytes)
        # content: ALWAYS current worktree bytes, so staged/unstaged tracked
        # edits change the digest (Python executes worktree bytes, not HEAD).
        # Fail-closed: any read failure raises, never falls back to HEAD/index.
        try:
            content = file_path.read_bytes()
        except OSError as e:
            raise RuntimeError(f"unreadable execution input {posix_path}: {e}") from e
        h.update(struct.pack(">Q", len(content)))
        h.update(content)
    return h.hexdigest(), len(exec_files)


def tracked_source_tree_details(root: Path | str | None = None) -> dict:
    """Git-tracked execution-input details (cmapss-tracked-source-v1).

    Uses ``git ls-files -z`` only; ignores ignored/generated files.
    Hash encoding is domain-separated length-delimited (cmapss-tracked-source-v1)
    with type tag, path length, path bytes, content length, raw content per
    sorted POSIX path; content is the CURRENT WORKTREE bytes, so any staged or
    unstaged tracked edit changes the hash. Fail-closed on unreadable inputs.
    Returns dict with at least hash, algorithm, file count, normalized file list.
    Canonical digest field remains ``source_tree_hash`` per spec.
    """
    r = _resolve_git_root(root)
    exec_files = _git_ls_files_execution(r)
    digest, count = _hash_execution_tree(r, exec_files)
    return {
        "source_tree_hash": digest,
        "algorithm": CANONICAL_ALGO,
        "file_count": count,
        "files": exec_files,
        "file_list": exec_files,  # alias for convenience
        "root": str(r),
    }


def _git_status_porcelain_v2_z(root: Path) -> tuple[bytes, str]:
    """Run git status --porcelain=v2 -z and return (raw_bytes, decoded_text)."""
    out = subprocess.run(
        ["git", "status", "--porcelain=v2", "-z"],
        capture_output=True,
        check=False,
        cwd=str(root),
    )
    if out.returncode != 0:
        raise RuntimeError(f"git status failed: {out.stderr.decode(errors='replace')}")
    raw = out.stdout or b""
    # also get decoded for hashing
    text = raw.decode("utf-8", errors="strict") if raw else ""
    return raw, text


def _git_diff_binary(root: Path) -> bytes:
    """Binary diff including staged + unstaged (and untracked via status)."""
    # unstaged
    d1 = subprocess.run(["git", "diff", "--binary"], capture_output=True, check=False, cwd=str(root))
    # staged
    d2 = subprocess.run(["git", "diff", "--cached", "--binary"], capture_output=True, check=False, cwd=str(root))
    # untracked is not in diff, but we capture via status + we will hash relevant untracked file contents separately
    combined = (d1.stdout or b"") + b"\x00---STAGED---\x00" + (d2.stdout or b"")
    return combined


def _parse_status_for_dirty(root: Path) -> dict:
    """Parse NUL-delimited v2 status into whole-repo vs execution-scope dirtiness."""
    raw, text = _git_status_porcelain_v2_z(root)
    if not raw:
        return {
            "whole_dirty": False,
            "execution_dirty": False,
            "staged": [],
            "unstaged": [],
            "untracked": [],
            "relevant_untracked": [],
            "ignored_untracked": [],
            "raw_bytes": raw,
            "text": text,
        }
    entries = raw.split(b"\x00")
    staged = []
    unstaged = []
    untracked = []
    relevant_untracked = []
    for e in entries:
        if not e:
            continue
        try:
            s = e.decode("utf-8")
        except UnicodeDecodeError:
            raise RuntimeError(f"invalid UTF-8 in git status entry: {e!r}")
        # Robust extraction per git status v2 spec
        if s.startswith("?"):
            # "? <path>" untracked
            # Preserve tabs, spaces, and other valid pathname bytes after the
            # single status separator.
            path = s[2:] if s.startswith("? ") else s[1:]
            untracked.append(path)
            if _is_execution_input(path):
                relevant_untracked.append(path)
            continue
        if s.startswith("!"):
            # ignored: "! <path>"
            continue
        if s.startswith("1") or s.startswith("2") or s.startswith("u"):
            # Ordinary/rename/unmerged entries — robust to spaces in path (P2-2)
            # Porcelain v2: fields are SP-separated; path may contain spaces and is the remainder after fixed fields.
            # For "1"/"u": "1 <XY> <sub> <mH> <mI> <mW> <hH> <hI> <path>"
            # For "2": "2 <XY> <sub> <mH> <mI> <mW> <hH> <hI> <X><score> <path><tab><origPath>"
            if "\t" in s:
                # split on tab: header_and_new_path \t origPath (origPath may also contain spaces, but we preserve as is)
                header_and_new, orig = s.split("\t", 1)
                # header_and_new contains fixed fields + path. Extract path as remainder after 8th SP (for "2", after Xscore)
                # Use split with maxsplit to isolate path: for "2", there are 9 fields before path (1, XY, sub, mH, mI, mW, hH, hI, Xscore)
                # We split header_and_new with maxsplit 9 to get path as last part
                # Determine expected field count: 8 for "1"/"u", 9 for "2"
                maxsplit = 9 if s.startswith("2") else 10 if s.startswith("u") else 8
                # header_and_new starts with "2 " or "1 " etc., so split
                parts = header_and_new.split(" ", maxsplit)
                new_path = parts[-1] if len(parts) > maxsplit else header_and_new.split(" ", 1)[-1]
                # For safety, if new_path still contains leading hash-like token, split once more
                # The last field before path for "2" is Xscore (e.g., "R100"), for "1" is hI hash.
                # Our maxsplit already isolates path, but if path contains spaces, it remains intact as last part.
                paths = [new_path]
                if orig:
                    paths.append(orig.strip())
                canonical_path = new_path
            else:
                # no tab: path is remainder after fixed fields, may contain spaces
                maxsplit = 9 if s.startswith("2") else 10 if s.startswith("u") else 8
                # Split with maxsplit to preserve path with spaces
                parts = s.split(" ", maxsplit)
                if len(parts) > maxsplit:
                    canonical_path = parts[-1]
                else:
                    # fallback: take last token (for malformed)
                    canonical_path = s.split()[-1] if s.split() else ""
                paths = [canonical_path]
            # XY at s[2:4] for "1 " and "2 " lines
            xy = s[2:4] if len(s) >= 4 else "  "
            x = xy[0] if len(xy) > 0 else " "
            y = xy[1] if len(xy) > 1 else " "
            if x != "." and x != " ":
                staged.append(canonical_path)
            if y != "." and y != " ":
                unstaged.append(canonical_path)
            # For execution dirty, we already capture via staged/unstaged lists; also consider rename paths
            # No need to handle here, final exec_dirty loop will check staged/unstaged/relevant
            continue
        # unknown line, treat as dirty unstaged with last token as path
        tokens = s.split()
        path = tokens[-1].strip() if tokens else s.strip()
        unstaged.append(path)

    # For renames/deletions, those are tracked deletions/renames -> dirty

    whole_dirty = bool(raw.strip())
    # execution_dirty: any staged/unstaged/untracked relevant execution file dirty, or any tracked execution file changed
    exec_dirty = False
    # Check if any status entry path is execution input
    for p in staged + unstaged + relevant_untracked:
        if _is_execution_input(p):
            exec_dirty = True
            break
    # Also need to detect unstaged edits to tracked execution files that may not be captured as path parsing? raw != empty and any exec file appears in status text
    if not exec_dirty and raw:
        # fallback: if any exec file string appears in status text, mark dirty
        # This handles edge where parsing missed format
        for ef in _git_ls_files_execution(root):
            if ef.encode() in raw:
                exec_dirty = True
                break
        # Also if relevant untracked exists, already captured
        if relevant_untracked:
            exec_dirty = True

    return {
        "whole_dirty": whole_dirty,
        "execution_dirty": exec_dirty,
        "staged": staged,
        "unstaged": unstaged,
        "untracked": untracked,
        "relevant_untracked": relevant_untracked,
        "raw_bytes": raw,
        "text": text,
    }


def _hash_bytes(data: bytes) -> str | None:
    if not data:
        return None
    return hashlib.sha256(data).hexdigest()


def _list_untracked_under(root: Path, dir_posix: str) -> list[str]:
    """List files inside an untracked directory via git (excludes ignored), sorted."""
    out = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z", "--", dir_posix],
        capture_output=True,
        check=False,
        cwd=str(root),
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"git ls-files failed for untracked directory {dir_posix!r} in {root}: "
            f"{out.stderr.decode(errors='replace')}"
        )
    files = []
    for b in out.stdout.split(b"\x00"):
        if not b:
            continue
        files.append(b.decode("utf-8"))
    return sorted(files)


def _relevant_untracked_inventory(root: Path, relevant: list[str]) -> list[dict]:
    """Hash relevant untracked execution files, check traversal/symlink/size/secrets.

    Porcelain status collapses untracked directories (e.g. ``src/newpkg/``);
    such entries are expanded via ``git ls-files --others --exclude-standard``
    and every contained file is inventoried and hashed individually. A
    directory with nothing listable is recorded as a skipped note entry
    (``skipped: True``) instead of crashing; skipped entries carry no content.
    """
    inv = []

    def _inventory_file(posix: str) -> dict:
        p = root / posix
        # traversal check
        if ".." in Path(posix).parts or posix.startswith("/") or posix.startswith("\\"):
            raise RuntimeError(f"path traversal in untracked file: {posix!r}")
        # symlink escape check
        if p.is_symlink():
            target = p.resolve()
            try:
                target.relative_to(root.resolve())
            except ValueError:
                raise RuntimeError(f"repository-escaping symlink untracked: {posix} -> {target}")
        if not p.exists() or not p.is_file():
            raise RuntimeError(f"relevant untracked file missing or not a file: {posix}")
        # size check
        try:
            size = p.stat().st_size
        except OSError as e:
            raise RuntimeError(f"unreadable untracked file {posix}: {e}") from e
        if size > SNAPSHOT_LIMIT_BYTES:
            raise RuntimeError(
                f"relevant untracked file {posix} exceeds size limit {SNAPSHOT_LIMIT_BYTES} bytes ({size})"
            )
        # content hash
        try:
            content = p.read_bytes()
        except OSError as e:
            raise RuntimeError(f"failed to read untracked file {posix}: {e}") from e
        h = hashlib.sha256(content).hexdigest()
        # sensitive filename scan
        sensitive_reason = None
        if SENSITIVE_FILENAME_RE.search(posix):
            sensitive_reason = f"sensitive filename pattern matches {posix!r}"
        # secret content scan
        try:
            text = content.decode("utf-8", errors="ignore")
            for rx in SECRET_CONTENT_RES:
                if rx.search(text):
                    sensitive_reason = (sensitive_reason or "") + f"; secret pattern {rx.pattern!r} found in {posix!r}"
                    break
        except Exception:
            pass
        return {
            "path": posix,
            "posix_path": posix,
            "bytes": size,
            "sha256": h,
            "sensitive_flag": bool(sensitive_reason),
            "sensitive_reason": sensitive_reason,
        }

    for posix in relevant:
        if posix.endswith("/"):
            # untracked DIRECTORY entry: inventory each contained file individually
            files = _list_untracked_under(root, posix)
            if not files:
                # empty (or only-ignored) directory: skip with a note, never crash
                inv.append(
                    {
                        "path": posix,
                        "posix_path": posix,
                        "bytes": 0,
                        "sha256": None,
                        "sensitive_flag": False,
                        "sensitive_reason": None,
                        "skipped": True,
                        "note": "empty untracked directory (git lists no files); skipped",
                    }
                )
                continue
            for f in files:
                inv.append(_inventory_file(f))
        else:
            inv.append(_inventory_file(posix))
    return inv


def collect_git_provenance(
    root: Path | str | None = None,
    config_path: Path | str | None = None,
    config_file_sha256: str | None = None,
    config_canonical_sha256: str | None = None,
    split_paths: dict | None = None,
    constraints_path: Path | str | None = None,
    dirty_reason: str | None = None,
    snapshot_path: Path | str | None = None,
    snapshot_hash: str | None = None,
) -> dict:
    """Collect Git provenance at run start before fitting.

    Captures schema version, timestamp, commit, whole/execution dirty flags,
    status/diff hashes, source_tree_hash, relevant untracked list, config hashes,
    split/cutoff hashes, constraints hash, Python/package versions.
    """
    r = _resolve_git_root(root)
    # commit
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, encoding="utf-8", errors="strict", check=False, cwd=str(r))
        commit = out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        commit = None

    status_raw, status_text = _git_status_porcelain_v2_z(r)
    diff_bytes = _git_diff_binary(r)
    parsed = _parse_status_for_dirty(r)
    details = tracked_source_tree_details(r)

    # status/diff hashes
    status_hash = _hash_bytes(status_raw) if status_raw else None
    diff_hash = _hash_bytes(diff_bytes) if diff_bytes and diff_bytes.strip(b"\x00---STAGED---\x00") else None

    # relevant untracked inventory
    relevant = parsed["relevant_untracked"]
    try:
        rel_inv = _relevant_untracked_inventory(r, relevant) if relevant else []
    except Exception:
        # fail closed: propagate if inventory fails
        raise

    # execution source hash = same as source_tree_hash (tracked execution inputs)
    execution_source_hash = details["source_tree_hash"]

    # config hashes: hashing is evidence; unreadable input is a hard error
    cfg_file_sha = config_file_sha256
    cfg_canon_sha = config_canonical_sha256
    if config_path is not None and cfg_file_sha is None:
        cfg_file_sha = sha256_file(Path(config_path) if Path(config_path).is_absolute() else r / Path(config_path))

    # split hashes: fail closed on read errors (missing file is recorded as not exists)
    split_info = {}
    if split_paths:
        for k, v in split_paths.items():
            p = Path(v)
            if not p.is_absolute():
                p = r / p
            split_info[k] = {
                "path": str(v),
                "sha256": sha256_file(p) if p.exists() else None,
                "exists": p.exists(),
            }

    # constraints: fail closed on read errors
    constraints_info = {}
    if constraints_path is not None:
        p = Path(constraints_path)
        if not p.is_absolute():
            p = r / p
        constraints_info = {
            "path": str(constraints_path),
            "sha256": sha256_file(p) if p.exists() else None,
            "exists": p.exists(),
        }

    # package versions
    versions: dict[str, str] = {}
    for pkg in ("tensorflow", "numpy", "pandas", "scikit-learn", "xgboost", "joblib"):
        try:
            versions[pkg] = _md.version(pkg)
        except Exception:
            versions[pkg] = "unknown"

    ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "run_start_timestamp_utc": ts,
        "timestamp_utc": ts,  # alias for legacy callers (benchmark/v2_2)
        "git_commit": commit,
        "git_is_dirty": parsed["whole_dirty"],
        "git_is_dirty_whole": parsed["whole_dirty"],
        "git_is_dirty_execution": parsed["execution_dirty"],
        "git_status_hash": status_hash,
        "git_diff_hash": diff_hash,
        "source_tree_hash": details["source_tree_hash"],
        "source_tree_algorithm": details["algorithm"],
        "source_tree_file_count": details["file_count"],
        "file_count": details["file_count"],  # alias matching tracked_source_tree_details
        "source_tree_files": details["files"],
        "execution_source_hash": execution_source_hash,
        "execution_source_algorithm": CANONICAL_ALGO,
        "execution_source_file_count": details["file_count"],
        "relevant_untracked": relevant,
        "relevant_untracked_inventory": rel_inv,
        "dirty_reason": dirty_reason,
        "snapshot_path": str(snapshot_path) if snapshot_path else None,
        "snapshot_hash": snapshot_hash,
        "config_path": str(config_path) if config_path else None,
        "config_file_sha256": cfg_file_sha,
        "config_canonical_sha256": cfg_canon_sha,
        "split_paths": split_info if split_info else None,
        "constraints": constraints_info if constraints_info else None,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "python_full": sys.version,
        "package_versions": versions,
        "git_root": str(r),
    }
    return provenance


class DirtyExecutionError(RuntimeError):
    """Raised when dirty execution inputs are detected without explicit allowance."""


def assert_reproducible_run_state(
    root: Path | str | None = None,
    allow_dirty_execution: bool = False,
    dirty_reason: str | None = None,
    snapshot_dir: Path | str | None = None,
) -> dict:
    """Enforce reproducible run state before training/loading/output.

    - Rejects dirty execution inputs by default before training.
    - Allows unrelated non-execution files without bundle but records whole-repo status.
    - Permits dirty execution only with explicit flag, nonempty reason, durable snapshot destination.
    - Stores binary patch, copies of relevant untracked files, inventory, snapshot hash.
    - Rejects path traversal, repo-escaping symlinks, unreadable inputs, incomplete snapshot.
    - Rejects snapshot destinations under execution-scope dirs (src/, scripts/) outright;
      destinations under tracked evidence (experiments/, configs/) require explicit confirmation.
    - Never invents destination, never auto-stages/commits, rejects oversized, scans secrets.

    Returns provenance dict (as collect_git_provenance) if clean or allowed dirty.
    """
    r = _resolve_git_root(root)
    parsed = _parse_status_for_dirty(r)
    whole_dirty = parsed["whole_dirty"]
    exec_dirty = parsed["execution_dirty"]
    relevant = parsed["relevant_untracked"]

    if not exec_dirty:
        # execution clean: whole-repo dirtiness (e.g., docs, session-ses_feb9.md) is recorded but allowed
        return collect_git_provenance(root=r)

    # execution dirty
    if not allow_dirty_execution:
        raise DirtyExecutionError(
            "dirty execution inputs detected (staged/unstaged/untracked execution files). "
            "Refusing to train/load/write without explicit --allow-dirty. "
            f"Relevant untracked: {relevant} staged={parsed['staged'][:5]} unstaged={parsed['unstaged'][:5]} "
            "Provide allow_dirty_execution=True, nonempty dirty_reason, and snapshot_dir."
        )
    # dirty allowed: require reason and destination
    if not dirty_reason or not str(dirty_reason).strip():
        raise DirtyExecutionError("dirty execution allowed but dirty_reason is empty")
    if snapshot_dir is None or not str(snapshot_dir).strip():
        raise DirtyExecutionError("dirty execution allowed but snapshot_dir is missing (never invent destination)")

    snap = Path(snapshot_dir)
    # traversal check: snapshot_dir must not contain .. and must be inside or explicit absolute?
    # We allow any caller-supplied destination but reject path traversal that would escape via ..
    if ".." in snap.parts:
        raise DirtyExecutionError(f"snapshot_dir must not contain '..': {snap}")
    # Do not allow snapshot inside tracked repo evidence without explicit confirmation (P1-2 fix: explicit token, not substring)
    # Per spec 10.3, if snapshot is ever proposed for tracked evidence, stop and obtain explicit user confirmation after showing destination and sensitive-data risk summary.
    try:
        snap_resolved = snap.resolve()
        repo_resolved = r.resolve()
        try:
            snap_resolved.relative_to(repo_resolved)
            # inside repo
            # Hard-reject execution-scope dirs (src/, scripts/): a snapshot there
            # would seed future runs' untracked inventory (self-perpetuating dirt).
            if "src" in snap_resolved.parts or "scripts" in snap_resolved.parts:
                raise DirtyExecutionError(
                    f"snapshot destination {snap_resolved} rejected: inside execution-scope "
                    "directory (src/ or scripts/); snapshots must not be placed where they "
                    "can seed future runs' untracked execution inventory."
                )
            if "experiments" in snap_resolved.parts or "configs" in snap_resolved.parts:
                # Require explicit token, not weak substring, and include risk summary after inventory is available.
                # We check for explicit tokens "confirm_tracked_evidence" or "approved_tracked_evidence"
                confirmation_tokens = set(str(dirty_reason).split())
                has_explicit_confirm = bool(
                    confirmation_tokens
                    & {"confirm_tracked_evidence", "approved_tracked_evidence"}
                )
                if not has_explicit_confirm:
                    raise DirtyExecutionError(
                        f"snapshot destination {snap_resolved} is inside tracked evidence area (experiments/configs). "
                        "Explicit user confirmation required (include 'confirm_tracked_evidence' in dirty_reason) after reviewing destination and sensitive-data risk summary. "
                        f"Destination: {snap_resolved}, repo: {repo_resolved}, reason: {dirty_reason!r}. "
                        "Sensitive-data risk: snapshots of tracked evidence may contain sensitive experiment data; review inventory before confirming."
                    )
        except ValueError:
            pass  # outside repo, fine
    except Exception as e:
        if isinstance(e, DirtyExecutionError):
            raise
        raise DirtyExecutionError(f"invalid snapshot_dir {snap}: {e}") from e

    # Ensure snapshot dir exists or create it (durable destination)
    try:
        snap.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise DirtyExecutionError(f"cannot create snapshot_dir {snap}: {e}") from e

    # Verify snapshot dir is writable and not a file
    if not snap.is_dir():
        raise DirtyExecutionError(f"snapshot_dir is not a directory: {snap}")

    # Inventory relevant untracked files
    try:
        inv = _relevant_untracked_inventory(r, relevant)
    except Exception as e:
        raise DirtyExecutionError(f"dirty snapshot inventory failed: {e}") from e

    # Scan/report sensitive before copying (already in inventory)
    sensitive = [x for x in inv if x.get("sensitive_flag")]
    if sensitive:
        # Report but do not auto-block unless explicitly sensitive? We include in snapshot but warn
        # The spec says scan/report likely secrets or sensitive filenames before copying
        # We will write a report file and include in provenance
        pass

    # Store binary patch
    diff_bytes = _git_diff_binary(r)
    status_raw, _ = _git_status_porcelain_v2_z(r)
    patch_path = snap / "dirty_snapshot.patch"
    status_path = snap / "dirty_status.txt"
    inventory_path = snap / "inventory.json"
    # Also copy relevant untracked files exactly
    try:
        patch_path.write_bytes(diff_bytes)
        status_path.write_bytes(status_raw)
        # copy relevant untracked files
        for item in inv:
            if item.get("skipped"):
                continue  # note entry (e.g. empty untracked dir): no content to copy
            src = r / item["path"]
            dst = snap / item["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            # symlink check again before copy
            if src.is_symlink():
                target = src.resolve()
                try:
                    target.relative_to(r.resolve())
                except ValueError:
                    raise DirtyExecutionError(f"repository-escaping symlink: {item['path']}")
            dst.write_bytes(src.read_bytes())
        # inventory
        inventory_path.write_text(json.dumps(inv, indent=2), encoding="utf-8")
    except Exception as e:
        raise DirtyExecutionError(f"failed to write dirty snapshot to {snap}: {e}") from e

    # Snapshot hash: hash of patch + status + inventory + file contents
    h = hashlib.sha256()
    h.update(diff_bytes)
    h.update(status_raw)
    for item in sorted(inv, key=lambda x: x["path"]):
        if item.get("skipped"):
            continue  # note entry carries no content
        p = snap / item["path"]
        try:
            h.update(p.read_bytes())
        except Exception as e:
            raise DirtyExecutionError(f"incomplete snapshot, missing {item['path']}: {e}") from e
    snap_hash = h.hexdigest()
    # Write snapshot hash file
    (snap / "snapshot.sha256").write_text(snap_hash + "\n", encoding="utf-8")

    # Return provenance with snapshot info
    prov = collect_git_provenance(
        root=r,
        dirty_reason=dirty_reason,
        snapshot_path=str(snap),
        snapshot_hash=snap_hash,
    )
    # augment with snapshot specifics
    prov["dirty_snapshot"] = {
        "patch_path": str(patch_path),
        "status_path": str(status_path),
        "inventory_path": str(inventory_path),
        "snapshot_hash": snap_hash,
        "relevant_inventory": inv,
        "sensitive_findings": sensitive,
    }
    return prov


# ---------------------------------------------------------------------------
# Compatibility helpers for old imports (benchmark/v2_2)
# ---------------------------------------------------------------------------

def git_provenance(root: Path | str | None = None) -> dict:
    """Compatibility wrapper for old benchmark/v2_2.git_provenance."""
    return collect_git_provenance(root=root)


def source_tree_hash(root: Path | str | None = None) -> str:
    """Compatibility: return only the hash string (old API).

    Fail-closed: propagates the underlying error on any hashing failure
    (never returns None in place of evidence).
    """
    return tracked_source_tree_details(root)["source_tree_hash"]
