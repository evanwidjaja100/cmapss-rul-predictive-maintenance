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
    (tmp / "app_v2.py").write_text("import streamlit as st\n", encoding="utf-8")
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
        # hash still reflects tracked HEAD, not staged? Our hash is based on committed tracked files (git ls-files shows old content until commit), but dirty flag indicates dirty
        # After staging, git ls-files still shows old blob until commit, so hash unchanged but dirty flag true
        h1 = tracked_source_tree_details(tmp)["source_tree_hash"]
        assert h1 == h0  # hash of committed content unchanged, but dirty detected via status/diff
        with pytest.raises(Exception, match="dirty execution"):
            assert_reproducible_run_state(root=tmp)


def test_unstaged_edit_detected():
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "repo"
        tmp.mkdir()
        _init_temp_repo(tmp)
        (tmp / "configs" / "c.yaml").write_text("k: changed\n", encoding="utf-8")
        prov = collect_git_provenance(root=tmp)
        assert prov["git_is_dirty_execution"] is True
        with pytest.raises(Exception, match="dirty execution"):
            assert_reproducible_run_state(root=tmp)


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


def test_untracked_source_path_with_tab_is_relevant(monkeypatch):
    from rul_prediction import reproducibility

    raw = b"? src/tab\tmodule.py" + bytes([0])
    monkeypatch.setattr(
        reproducibility,
        "_git_status_porcelain_v2_z",
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


def test_tracked_deletion_detected():
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "repo"
        tmp.mkdir()
        _init_temp_repo(tmp)
        (tmp / "src" / "a.py").unlink()
        prov = collect_git_provenance(root=tmp)
        assert prov["git_is_dirty_execution"] is True


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
        # deletion scenario: tracked file removed from working tree but still in HEAD
        # Our implementation hashes HEAD blob for stability across dirty states, so
        # tracked_source_tree_details remains stable and does NOT raise; instead
        # dirty is reported via provenance flags. This is documented behavior:
        # hash is of committed HEAD content (cmapss-tracked-source-v1), dirty
        # detection via NUL-delimited status + binary diff, not via hash failure.
        (tmp / "src" / "a.py").unlink()
        # should not raise; provenance should report execution dirty
        prov = collect_git_provenance(root=tmp)
        assert prov["git_is_dirty_execution"] is True
        # hash stays of HEAD content
        d = tracked_source_tree_details(tmp)
        assert d["source_tree_hash"] is not None
        # direct filesystem read for missing file should still fail closed if we try sha256_file
        with pytest.raises((FileNotFoundError, OSError, RuntimeError)):
            sha256_file(tmp / "src" / "a.py")


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
