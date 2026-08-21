"""Build V2.2 artifact manifests (FD001 + FD004).

Builder operational contract:
 - --check validates + deterministic regeneration comparison without rewriting
 - manifest creation accepts --generated-at <UTC> or preserves prior generated_at_utc when inputs unchanged
 - identical inputs + fixed timestamp => byte-for-byte deterministic JSON (stable ordering, formatting, newline, UTF-8)
 - library APIs and CLI accept explicit root override
 - all tamper/missing/wrong tests operate on temp copied bundles through that override and never modify real frozen artifacts
 - No manifest may hash itself. No broad globs. Reject absolute, .., duplicate, wrong identity, ambiguous mirrors
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ensure src on path for script invocation via python scripts/...
ROOT_FALLBACK = Path(__file__).resolve().parents[1]
if str(ROOT_FALLBACK / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_FALLBACK / "src"))

from rul_prediction.artifact_manifest import (
    MANIFEST_PATHS,
    ArtifactManifestError,
    ArtifactMissingError,
    build_manifest_dict,
    deterministic_json_bytes,
    load_manifest,
    validate_manifest_dict,
)

def _resolve_root(root: str | None) -> Path:
    from rul_prediction.artifact_manifest import _resolve_root as _r
    return _r(root)

def _maybe_preserve_timestamp(dataset: str, root: Path, new_dict: dict, generated_at_opt: str | None) -> str:
    """Return timestamp to use: --generated-at overrides, else preserve prior if inputs unchanged, else new."""
    if generated_at_opt is not None:
        return str(generated_at_opt)
    manifest_path = root / MANIFEST_PATHS[dataset]
    if not manifest_path.exists():
        return new_dict["generated_at_utc"]
    try:
        prior = json.loads(manifest_path.read_bytes().decode("utf-8"))
    except Exception:
        return new_dict["generated_at_utc"]
    # compare hashed inputs: artifacts sha/bytes + source/config/constraints hashes
    # if all equal, preserve prior timestamp
    def _inputs(d: dict) -> dict:
        # Use artifacts sorted + integrity hashes
        arts = sorted(d.get("artifacts", []), key=lambda x: x["role"])
        inp = {
            "artifacts": [(a["role"], a["sha256"], a["bytes"]) for a in arts],
            "source_tree_hash": d.get("source_integrity", {}).get("source_tree_hash"),
            "config_file_sha256": d.get("config_integrity", {}).get("config_file_sha256"),
            "config_canonical_sha256": d.get("config_integrity", {}).get("config_canonical_sha256"),
            "constraints": sorted([(c["path"], c["sha256"]) for c in d.get("constraints_integrity", [])]),
        }
        return inp
    if _inputs(prior) == _inputs(new_dict):
        # preserve prior generation time
        prior_ts = prior.get("generated_at_utc")
        if isinstance(prior_ts, str) and prior_ts:
            return prior_ts
    return new_dict["generated_at_utc"]

def build_one(dataset: str, root: Path, generated_at: str | None, check: bool = False) -> int:
    """Build single dataset manifest; honors --check and timestamp preservation; returns exit code."""
    manifest_path = root / MANIFEST_PATHS[dataset]
    # First build with requested or now timestamp
    new_dict = build_manifest_dict(dataset, root=root, generated_at_utc=generated_at)
    # Determine timestamp to use (preserve or override)
    ts = _maybe_preserve_timestamp(dataset, root, new_dict, generated_at)
    if ts != new_dict["generated_at_utc"]:
        # rebuild with preserved timestamp for determinism
        new_dict = build_manifest_dict(dataset, root=root, generated_at_utc=ts)
    # deterministic bytes
    new_bytes = deterministic_json_bytes(new_dict)
    if check:
        # validate and compare without writing
        validate_manifest_dict(new_dict, root=root)
        if not manifest_path.exists():
            print(f"[check] manifest missing for {dataset}: {manifest_path}")
            return 1
        prior_bytes = manifest_path.read_bytes()
        # Load prior and validate it too
        try:
            prior_dict = json.loads(prior_bytes.decode("utf-8"))
            validate_manifest_dict(prior_dict, root=root)
        except Exception as e:
            print(f"[check] prior manifest invalid for {dataset}: {e}")
            return 1
        # Compare regeneration: rebuild prior's inputs with same timestamp should be byte-identical if inputs unchanged?
        # For check we compare new_bytes == prior_bytes when inputs unchanged; else we expect mismatch?
        # Spec: --check performs validation and deterministic regeneration comparison without rewriting
        # Means if inputs unchanged and timestamp preserved, bytes must match; if inputs changed, check should indicate drift?
        # We implement: if new_bytes == prior_bytes => ok; else if inputs actually changed (hash diff) we still consider check failure because manifest not yet updated? But spec says check compares deterministic regeneration; if inputs changed, checksum will differ -> fail until manifest updated.
        if new_bytes != prior_bytes:
            # Provide diff hint: check if inputs changed vs formatting drift
            # Compare without timestamp: rebuild both with same ts and compare
            prior_inputs_ts = prior_dict.get("generated_at_utc")
            # If timestamps differ but inputs same, we already preserved, so should be equal. If still not equal, it's a real drift.
            print(f"[check] manifest for {dataset} would change (prior {len(prior_bytes)} bytes, new {len(new_bytes)} bytes)")
            # Show first diff line if possible
            import difflib
            prior_lines = prior_bytes.decode("utf-8").splitlines()
            new_lines = new_bytes.decode("utf-8").splitlines()
            diff = list(difflib.unified_diff(prior_lines, new_lines, fromfile="prior", tofile="new", lineterm=""))
            for line in diff[:40]:
                print(line)
            return 1
        print(f"[check] {dataset} manifest OK (no rewrite, deterministic, {len(new_bytes)} bytes)")
        return 0
    # normal write (atomic? ponytail: minimal write)
    # Ensure byte-for-byte deterministic: write deterministic_json_bytes exactly
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # atomic: write to temp then replace
    import tempfile, os
    tmp = Path(tempfile.mktemp(dir=str(manifest_path.parent)))
    try:
        tmp.write_bytes(new_bytes)
        os.replace(tmp, manifest_path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
    print(f"wrote {manifest_path} ({len(new_bytes)} bytes) dataset {dataset} generated_at {ts}")
    return 0

def main() -> None:
    parser = argparse.ArgumentParser(description="Build V2.2 artifact manifests")
    parser.add_argument("--root", default=None, help="repository/bundle root override")
    parser.add_argument("--generated-at", default=None, help="UTC timestamp override (ISO)")
    parser.add_argument("--check", action="store_true", help="validate + deterministic regeneration comparison without rewriting")
    parser.add_argument("--dataset", choices=["FD001", "FD004"], default=None, help="build single dataset manifest only")
    args = parser.parse_args()
    root = _resolve_root(args.root)
    datasets = [args.dataset] if args.dataset else ["FD001", "FD004"]
    # For FD004, ensure canonical official predictions derived exactly from mirror before manifest build
    # The build_manifest_dict will handle copying/validation, but we ensure mirror exists check
    exit_code = 0
    for ds in datasets:
        try:
            code = build_one(ds, root, args.generated_at, check=args.check)
            if code != 0:
                exit_code = code
        except (ArtifactManifestError, ArtifactMissingError) as e:
            print(f"error building {ds}: {e}", file=sys.stderr)
            exit_code = 1
        except Exception as e:
            print(f"unexpected error building {ds}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            exit_code = 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
