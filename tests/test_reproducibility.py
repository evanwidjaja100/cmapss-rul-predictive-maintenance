"""Provenance falsification tests (Section 10.5).

Exercises clean, staged, unstaged, untracked source, ignored cache, deletion,
rename, editable-install .egg-info, mtime change, enumeration order, Unicode,
Windows/POSIX normalization, missing/unreadable, dirty-snapshot reconstruction
in temporary repositories.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap

import pytest

from rul_prediction.reproducibility import (
    DirtyExecutionError,
    assert_reproducible_run_state,
    collect_git_provenance,
    sha256_file,
    tracked_source_tree_details,
)

pytestmark = pytest.mark.unit


def _run(cmd, cwd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd), **kw)


def _init_temp_repo(tmp: pathlib.Path) -> pathlib.Path:
    _run(["git", "init"], tmp)
    _run(["git", "config", "user.email", "test@test.com"], tmp)
    _run(["git", "config", "user.name", "Test"], tmp)
    # create execution inputs minimal set
    (tmp / "src").mkdir()
    (tmp / "scripts").mkdir()
    (tmp / "configs").mkdir()
    (tmp / ".github" / "workflows").mkdir(parents=True)
    (tmp / "src" / "a.py").write_text("x=1\n", encoding="utf-8")
    (tmp / "scripts" / "s.py").write_text("y=1\n", encoding="utf-8")
    (tmp / "configs" / "c.yaml").write_text("k: v\n", encoding="utf-8")
    (tmp / "app_m1.py").write_text("import streamlit as st\n", encoding="utf-8")
    (tmp / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (tmp / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    (tmp / "requirements-lock.txt").write_text("numpy==1.0\n", encoding="utf-8")
    (tmp / ".github" / "workflows" / "ci.yml").write_text("name: ci\n", encoding="utf-8")
    _run(["git", "add", "."], tmp)
    _run(["git", "commit", "-m", "init"], tmp)
    return tmp


def test_clean_tree_provenance():
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "repo"
        tmp.mkdir()
        _init_temp_repo(tmp)
        d = tracked_source_tree_details(tmp)
        assert d["source_tree_hash"] is not None
        assert d["file_count"] >= 7
        prov = collect_git_provenance(root=tmp)
        assert prov["git_is_dirty_execution"] is False
        assert prov["git_is_dirty"] is False
        # assert clean passes
        prov2 = assert_reproducible_run_state(root=tmp)
        assert prov2["git_is_dirty_execution"] is False


def test_staged_edit_detected():
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "repo"
        tmp.mkdir()
        _init_temp_repo(tmp)
        h0 = tracked_source_tree_details(tmp)["source_tree_hash"]
        # staged edit to execution file
        (tmp / "src" / "a.py").write_text("x=2\n", encoding="utf-8")
        _run(["git", "add", "src/a.py"], tmp)
        prov = collect_git_provenance(root=tmp)
        assert prov["git_is_dirty"] is True
        assert prov["git_is_dirty_execution"] is True
        # hash hashes CURRENT WORKTREE bytes: staged edit MUST change it
        h1 = tracked_source_tree_details(tmp)["source_tree_hash"]
        assert h1 != h0
        with pytest.raises(Exception, match="dirty execution"):
            assert_reproducible_run_state(root=tmp)


def test_unstaged_edit_detected():
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "repo"
        tmp.mkdir()
        _init_temp_repo(tmp)
        h0 = tracked_source_tree_details(tmp)["source_tree_hash"]
        (tmp / "configs" / "c.yaml").write_text("k: changed\n", encoding="utf-8")
        prov = collect_git_provenance(root=tmp)
        assert prov["git_is_dirty_execution"] is True
        # unstaged edit MUST change the hash
        h1 = tracked_source_tree_details(tmp)["source_tree_hash"]
        assert h1 != h0
        # restoring the original content restores the identical hash
        (tmp / "configs" / "c.yaml").write_text("k: v\n", encoding="utf-8")
        h2 = tracked_source_tree_details(tmp)["source_tree_hash"]
        assert h2 == h0
        # and the restored tree is execution-clean again
        prov2 = collect_git_provenance(root=tmp)
        assert prov2["git_is_dirty_execution"] is False


def test_untracked_source_relevant():
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "repo"
        tmp.mkdir()
        _init_temp_repo(tmp)
        # untracked execution file (new src file not yet added)
        (tmp / "src" / "new_module.py").write_text("z=99\n", encoding="utf-8")
        prov = collect_git_provenance(root=tmp)
        assert "src/new_module.py" in prov["relevant_untracked"]
        assert prov["git_is_dirty_execution"] is True
        with pytest.raises(Exception, match="dirty execution"):
            assert_reproducible_run_state(root=tmp)
        # allow dirty with snapshot
        with tempfile.TemporaryDirectory() as td2:
            snap = pathlib.Path(td2) / "snap"
            prov2 = assert_reproducible_run_state(
                root=tmp, allow_dirty_execution=True, dirty_reason="test untracked", snapshot_dir=snap
            )
            assert (snap / "inventory.json").exists()
            assert (snap / "src" / "new_module.py").exists()
            assert (snap / "src" / "new_module.py").read_text(encoding="utf-8") == "z=99\n"


def test_untracked_directory_expanded_in_inventory():
    import hashlib

    from rul_prediction.reproducibility import _relevant_untracked_inventory

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "repo"
        tmp.mkdir()
        _init_temp_repo(tmp)
        pkg = tmp / "src" / "newpkg"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "mod.py").write_text("q=7\n", encoding="utf-8")
        # porcelain collapses the untracked directory into a single dir entry
        assert "src/newpkg/" in collect_git_provenance(root=tmp)["relevant_untracked"]
        # inventory expands it into individual hashed files without crashing
        inv = _relevant_untracked_inventory(tmp, ["src/newpkg/"])
        by_path = {i["path"]: i for i in inv}
        assert set(by_path) == {"src/newpkg/__init__.py", "src/newpkg/mod.py"}
        # compare against actual on-disk bytes (Windows text mode writes CRLF)
        raw = (pkg / "mod.py").read_bytes()
        assert by_path["src/newpkg/mod.py"]["bytes"] == len(raw)
        assert by_path["src/newpkg/mod.py"]["sha256"] == hashlib.sha256(raw).hexdigest()
        assert by_path["src/newpkg/__init__.py"]["sha256"] == hashlib.sha256(b"").hexdigest()
        # full provenance collection does not crash either
        prov = collect_git_provenance(root=tmp)
        paths = {i["path"] for i in prov["relevant_untracked_inventory"]}
        assert {"src/newpkg/__init__.py", "src/newpkg/mod.py"} <= paths
        # snapshot flow copies the expanded files
        with tempfile.TemporaryDirectory() as td2:
            snap = pathlib.Path(td2) / "snap"
            assert_reproducible_run_state(
                root=tmp, allow_dirty_execution=True, dirty_reason="untracked dir", snapshot_dir=snap
            )
            assert (snap / "src" / "newpkg" / "mod.py").read_text(encoding="utf-8") == "q=7\n"


def test_empty_untracked_directory_skipped_with_note():
    from rul_prediction.reproducibility import _relevant_untracked_inventory

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "repo"
        tmp.mkdir()
        _init_temp_repo(tmp)
        # a directory git cannot list any file under: skipped with a note
        # entry, never a crash
        inv = _relevant_untracked_inventory(tmp, ["src/ghost_dir/"])
        assert len(inv) == 1
        assert inv[0]["skipped"] is True
        assert inv[0]["path"] == "src/ghost_dir/"
        assert inv[0]["sha256"] is None
        assert inv[0]["note"]


def test_untracked_source_path_with_tab_is_relevant(monkeypatch):
    from rul_prediction import reproducibility

    raw = b"? src/tab\tmodule.py" + bytes([0])
    monkeypatch.setattr(
        reproducibility,
        "_git_status_porcelain_m1_z",
        lambda _root: (raw, "? src/tab\tmodule.py"),
    )
    parsed = reproducibility._parse_status_for_dirty(pathlib.Path.cwd())
    assert parsed["relevant_untracked"] == ["src/tab\tmodule.py"]
    assert parsed["execution_dirty"] is True


def test_ignored_cache_does_not_affect_hash():
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "repo"
        tmp.mkdir()
        _init_temp_repo(tmp)
        h0 = tracked_source_tree_details(tmp)["source_tree_hash"]
        # ignored cache
        (tmp / "__pycache__").mkdir()
        (tmp / "__pycache__" / "a.pyc").write_bytes(b"\x00\x01")
        (tmp / ".pytest_cache").mkdir()
        (tmp / ".pytest_cache" / "CACHEDIR.TAG").write_text("cache\n")
        # also ensure .gitignore ignores them (but git ls-files already excludes ignored)
        (tmp / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n", encoding="utf-8")
        h1 = tracked_source_tree_details(tmp)["source_tree_hash"]
        assert h1 == h0
        prov = collect_git_provenance(root=tmp)
        # ignored should not make execution dirty (but .gitignore untracked is not execution input, so whole dirty true but exec false)
        assert prov["git_is_dirty_execution"] is False


def test_tracked_deletion_fails_closed():
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "repo"
        tmp.mkdir()
        _init_temp_repo(tmp)
        # tracked execution file deleted from worktree but still enumerated by
        # git ls-files: hashing is evidence, so this fails closed (no silent
        # fallback to HEAD content)
        (tmp / "src" / "a.py").unlink()
        with pytest.raises(RuntimeError, match="unreadable execution input"):
            tracked_source_tree_details(tmp)["source_tree_hash"]
        with pytest.raises(RuntimeError, match="unreadable execution input"):
            collect_git_provenance(root=tmp)


def test_rename_detected():
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "repo"
        tmp.mkdir()
        _init_temp_repo(tmp)
        # git mv
        _run(["git", "mv", "src/a.py", "src/b.py"], tmp)
        prov = collect_git_provenance(root=tmp)
        assert prov["git_is_dirty_execution"] is True


def test_egg_info_not_affects_hash():
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "repo"
        tmp.mkdir()
        _init_temp_repo(tmp)
        h0 = tracked_source_tree_details(tmp)["source_tree_hash"]
        (tmp / "src" / "rul_prediction.egg-info").mkdir()
        (tmp / "src" / "rul_prediction.egg-info" / "PKG-INFO").write_text("Metadata\n")
        h1 = tracked_source_tree_details(tmp)["source_tree_hash"]
        assert h1 == h0


def test_force_tracked_cache_filters_use_path_components():
    from rul_prediction.reproducibility import _is_execution_input

    assert _is_execution_input("src/__pycache__/module.py") is False
    assert _is_execution_input("src/.pytest_cache/state.py") is False
    assert _is_execution_input("src/package.egg-info/PKG-INFO") is False
    assert _is_execution_input("src/my__pycache__/module.py") is True
    assert _is_execution_input("src/package.egg-infoish/module.py") is True


def test_mtime_change_does_not_affect_hash():
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "repo"
        tmp.mkdir()
        _init_temp_repo(tmp)
        h0 = tracked_source_tree_details(tmp)["source_tree_hash"]
        p = tmp / "src" / "a.py"
        # touch mtime without content change
        os.utime(p, (0, 0))
        h1 = tracked_source_tree_details(tmp)["source_tree_hash"]
        assert h1 == h0


def test_enumeration_order_deterministic():
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "repo"
        tmp.mkdir()
        _init_temp_repo(tmp)
        # create many files in non-sorted filesystem order
        for name in ["z.py", "a.py", "m.py"]:
            (tmp / "src" / name).write_text(f"# {name}\n", encoding="utf-8")
        _run(["git", "add", "."], tmp)
        _run(["git", "commit", "-m", "add many"], tmp)
        h1 = tracked_source_tree_details(tmp)["source_tree_hash"]
        h2 = tracked_source_tree_details(tmp)["source_tree_hash"]
        assert h1 == h2
        # files list is sorted
        d = tracked_source_tree_details(tmp)
        assert d["files"] == sorted(d["files"])


def test_order_independence_across_repos():
    # same final file contents added in different orders with different mtimes
    # in two separate repos -> identical source_tree_hash (content-only)
    contents = {
        "src/z.py": "z=1\n",
        "src/a.py": "a=2\n",
        "src/m.py": "m=3\n",
        "scripts/s.py": "s=4\n",
    }
    orders = [
        ["src/z.py", "src/a.py", "src/m.py", "scripts/s.py"],
        ["scripts/s.py", "src/m.py", "src/a.py", "src/z.py"],
    ]
    hashes = []
    for order in orders:
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td) / "repo"
            tmp.mkdir()
            _init_temp_repo(tmp)
            for i, rel in enumerate(order):
                p = tmp / rel
                p.write_text(contents[rel], encoding="utf-8")
                os.utime(p, (1700000000 + i, 1700000000 + i))
                _run(["git", "add", rel], tmp)
            _run(["git", "commit", "-m", "files"], tmp)
            hashes.append(tracked_source_tree_details(tmp)["source_tree_hash"])
    assert hashes[0] is not None
    assert hashes[0] == hashes[1]


def test_unicode_content():
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "repo"
        tmp.mkdir()
        _init_temp_repo(tmp)
        (tmp / "src" / "unicode.py").write_text("# café — naïve → test\nx='α≈±'\n", encoding="utf-8")
        _run(["git", "add", "src/unicode.py"], tmp)
        _run(["git", "commit", "-m", "unicode"], tmp)
        h = tracked_source_tree_details(tmp)["source_tree_hash"]
        assert isinstance(h, str) and len(h) == 64


def test_windows_posix_normalization():
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "repo"
        tmp.mkdir()
        _init_temp_repo(tmp)
        # git always stores POSIX; we check our code normalizes
        d = tracked_source_tree_details(tmp)
        for f in d["files"]:
            assert "\\" not in f
            assert ".." not in f


def test_missing_unreadable_fails_closed():
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "repo"
        tmp.mkdir()
        _init_temp_repo(tmp)
        # missing file via direct sha256_file should raise (fail-closed, not silently skip)
        with pytest.raises((FileNotFoundError, OSError, RuntimeError)):
            sha256_file(tmp / "nonexistent.py")
        # deletion scenario: tracked execution file removed from working tree but
        # still enumerated by git ls-files. The canonical digest hashes CURRENT
        # WORKTREE bytes and is evidence, so this fails closed with a hard error
        # (no silent fallback to HEAD/index content).
        (tmp / "src" / "a.py").unlink()
        with pytest.raises(RuntimeError, match="unreadable execution input"):
            tracked_source_tree_details(tmp)
        with pytest.raises(RuntimeError, match="unreadable execution input"):
            collect_git_provenance(root=tmp)
        # direct filesystem read for missing file should also fail closed
        with pytest.raises((FileNotFoundError, OSError, RuntimeError)):
            sha256_file(tmp / "src" / "a.py")


def test_tracked_path_replaced_by_directory_raises():
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "repo"
        tmp.mkdir()
        _init_temp_repo(tmp)
        # replace an enumerated tracked file's path with a DIRECTORY of the same
        # name (deletion unstaged, so the path stays enumerated by ls-files).
        # Tree hashing must fail closed; no reliance on OS chmod (Windows-safe).
        (tmp / "src" / "a.py").unlink()
        (tmp / "src" / "a.py").mkdir()
        with pytest.raises(RuntimeError, match="unreadable execution input"):
            tracked_source_tree_details(tmp)["source_tree_hash"]
        with pytest.raises(RuntimeError, match="unreadable execution input"):
            collect_git_provenance(root=tmp)


def test_dirty_snapshot_reconstruction():
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "repo"
        tmp.mkdir()
        _init_temp_repo(tmp)
        (tmp / "src" / "a.py").write_text("x=dirty\n", encoding="utf-8")
        (tmp / "src" / "extra.py").write_text("y=extra\n", encoding="utf-8")
        with tempfile.TemporaryDirectory() as td2:
            snap = pathlib.Path(td2) / "snap"
            prov = assert_reproducible_run_state(
                root=tmp, allow_dirty_execution=True, dirty_reason="test snapshot reconstruct", snapshot_dir=snap
            )
            assert (snap / "dirty_snapshot.patch").exists()
            assert (snap / "dirty_status.txt").exists()
            assert (snap / "snapshot.sha256").exists()
            # snapshot hash should be reproducible
            h1 = (snap / "snapshot.sha256").read_text().strip()
            # reconstruct: re-hash patch+status+copies
            import hashlib

            patch = (snap / "dirty_snapshot.patch").read_bytes()
            status = (snap / "dirty_status.txt").read_bytes()
            extra = (snap / "src" / "extra.py").read_bytes() if (snap / "src" / "extra.py").exists() else b""
            h = hashlib.sha256()
            h.update(patch)
            h.update(status)
            # sorted inventory order
            inv = prov["dirty_snapshot"]["relevant_inventory"]
            for item in sorted(inv, key=lambda x: x["path"]):
                p = snap / item["path"]
                h.update(p.read_bytes())
            assert h.hexdigest() == h1


def test_path_traversal_rejected():
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "repo"
        tmp.mkdir()
        _init_temp_repo(tmp)
        (tmp / "src" / "a.py").write_text("x=dirty\n", encoding="utf-8")
        with tempfile.TemporaryDirectory() as td2:
            snap = pathlib.Path(td2) / "snap" / ".." / "evil"
            with pytest.raises(Exception, match="must not contain"):
                assert_reproducible_run_state(
                    root=tmp, allow_dirty_execution=True, dirty_reason="traversal", snapshot_dir=snap
                )


def test_snapshot_under_execution_scope_dirs_rejected():
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "repo"
        tmp.mkdir()
        _init_temp_repo(tmp)
        (tmp / "src" / "a.py").write_text("x=dirty\n", encoding="utf-8")
        # snapshots under src/ or scripts/ would seed future runs' untracked
        # inventory: rejected outright with a reason-naming error
        for dest in (tmp / "src" / "snapshot", tmp / "scripts" / "snapshot"):
            with pytest.raises(DirtyExecutionError, match="execution-scope"):
                assert_reproducible_run_state(
                    root=tmp,
                    allow_dirty_execution=True,
                    dirty_reason="bad destination",
                    snapshot_dir=dest,
                )


def test_tracked_evidence_confirmation_requires_exact_token():
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "repo"
        tmp.mkdir()
        _init_temp_repo(tmp)
        (tmp / "src" / "a.py").write_text("x=dirty\n", encoding="utf-8")
        with pytest.raises(DirtyExecutionError, match="Explicit user confirmation"):
            assert_reproducible_run_state(
                root=tmp,
                allow_dirty_execution=True,
                dirty_reason="not_confirm_tracked_evidence",
                snapshot_dir=tmp / "experiments" / "snapshot",
            )


def test_oversized_rejected():
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "repo"
        tmp.mkdir()
        _init_temp_repo(tmp)
        # create large relevant untracked file >5MiB
        big = tmp / "src" / "big.py"
        big.write_bytes(b"x" * (6 * 1024 * 1024))
        with tempfile.TemporaryDirectory() as td2:
            snap = pathlib.Path(td2) / "snap"
            with pytest.raises(Exception, match="exceeds size limit"):
                assert_reproducible_run_state(
                    root=tmp, allow_dirty_execution=True, dirty_reason="oversized test", snapshot_dir=snap
                )


def test_symlink_escape_rejected():
    # only test if symlink supported
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "repo"
        tmp.mkdir()
        _init_temp_repo(tmp)
        outside = pathlib.Path(td) / "outside.txt"
        outside.write_text("secret\n", encoding="utf-8")
        link = tmp / "src" / "link.py"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("symlink not supported")
        with tempfile.TemporaryDirectory() as td2:
            snap = pathlib.Path(td2) / "snap"
            with pytest.raises(Exception, match="symlink"):
                assert_reproducible_run_state(
                    root=tmp, allow_dirty_execution=True, dirty_reason="symlink test", snapshot_dir=snap
                )


def test_sensitive_scan_reported():
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "repo"
        tmp.mkdir()
        _init_temp_repo(tmp)
        (tmp / "src" / "secret.py").write_text("key = 'AKIA1234567890ABCDEF'\n", encoding="utf-8")
        with tempfile.TemporaryDirectory() as td2:
            snap = pathlib.Path(td2) / "snap"
            prov = assert_reproducible_run_state(
                root=tmp, allow_dirty_execution=True, dirty_reason="sensitive test", snapshot_dir=snap
            )
            # inventory should flag sensitive
            inv = prov["dirty_snapshot"]["relevant_inventory"]
            assert any(x["sensitive_flag"] for x in inv)


def test_ignored_vs_relevant_untracked_distinction():
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "repo"
        tmp.mkdir()
        _init_temp_repo(tmp)
        # relevant untracked execution file
        (tmp / "src" / "relevant.py").write_text("a=1\n", encoding="utf-8")
        # unrelated untracked (docs)
        (tmp / "docs").mkdir()
        (tmp / "docs" / "note.md").write_text("hello\n", encoding="utf-8")
        prov = collect_git_provenance(root=tmp)
        assert "src/relevant.py" in prov["relevant_untracked"]
        assert "docs/note.md" not in prov["relevant_untracked"]
        assert prov["git_is_dirty"] is True
        assert prov["git_is_dirty_execution"] is True  # because relevant
        # now remove relevant, keep unrelated only
        (tmp / "src" / "relevant.py").unlink()
        prov2 = collect_git_provenance(root=tmp)
        assert prov2["relevant_untracked"] == []
        # execution should be clean (unrelated docs not counted)
        assert prov2["git_is_dirty_execution"] is False
        assert prov2["git_is_dirty"] is True  # whole still dirty
        # clean execution should be allowed without snapshot
        prov3 = assert_reproducible_run_state(root=tmp)
        assert prov3["git_is_dirty_execution"] is False


def test_docstring_matches_behavior():
    """Ensure docstrings mention git ls-files and domain-separated encoding."""
    import inspect

    from rul_prediction import reproducibility

    src = inspect.getsource(reproducibility.tracked_source_tree_details)
    assert "git ls-files" in src or "git ls-files" in reproducibility.tracked_source_tree_details.__doc__
    assert "cmapss-tracked-source-v1" in src or "cmapss-tracked-source-v1" in reproducibility.tracked_source_tree_details.__doc__
    assert "domain-separated" in reproducibility.tracked_source_tree_details.__doc__ or "length-delimited" in reproducibility.tracked_source_tree_details.__doc__


def test_no_broad_except_continue():
    """Ensure no broad except OSError: continue remains."""
    content = pathlib.Path("src/rul_prediction/reproducibility.py").read_text(encoding="utf-8")
    # check for old pattern
    assert "except OSError:\n            continue" not in content
    assert "except OSError: continue" not in content
