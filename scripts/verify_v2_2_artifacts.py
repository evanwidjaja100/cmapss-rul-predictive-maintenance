"""Verify V2.2 artifact manifests.

Modes:
 - tracked: every Git artifact required; absent local permitted; present wrong fails
 - full: every Git+local required and verified

Accepts explicit root override.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_FALLBACK = Path(__file__).resolve().parents[1]
if str(ROOT_FALLBACK / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_FALLBACK / "src"))

from rul_prediction.artifact_manifest import (
    MANIFEST_PATHS,
    ArtifactHashMismatchError,
    ArtifactManifestError,
    ArtifactMissingError,
    verify_manifest_file,
)

def _resolve_root(root: str | None) -> Path:
    from rul_prediction.artifact_manifest import _resolve_root as _r
    return _r(root)

def main() -> None:
    parser = argparse.ArgumentParser(description="Verify V2.2 artifact manifests")
    parser.add_argument("--mode", choices=["tracked", "full"], default="tracked", help="verification mode")
    parser.add_argument("--root", default=None, help="repository/bundle root override")
    parser.add_argument("--manifest", default=None, help="explicit manifest path (overrides dataset lookup)")
    parser.add_argument("--dataset", choices=["FD001", "FD004"], default=None, help="verify single dataset manifest only")
    args = parser.parse_args()
    root = _resolve_root(args.root)
    exit_code = 0
    # if explicit manifest path given, verify just that file
    if args.manifest:
        try:
            summary = verify_manifest_file(args.manifest, root=root, mode=args.mode)
            print(f"{summary['dataset']} {args.mode}: {summary['verified']}/{summary['total']} verified, skipped {summary['skipped_absent_local']} local absent")
        except (ArtifactMissingError, ArtifactHashMismatchError, ArtifactManifestError) as e:
            print(f"FAIL {args.manifest} mode {args.mode}: {e}", file=sys.stderr)
            sys.exit(1)
        return
    datasets = [args.dataset] if args.dataset else ["FD001", "FD004"]
    for ds in datasets:
        manifest_path = root / MANIFEST_PATHS[ds]
        if not manifest_path.exists():
            print(f"manifest missing for {ds}: {manifest_path}", file=sys.stderr)
            exit_code = 1
            continue
        try:
            summary = verify_manifest_file(manifest_path, root=root, mode=args.mode)
            print(f"{ds} {args.mode}: {summary['verified']}/{summary['total']} verified, skipped {summary['skipped_absent_local']} local absent (total {summary['total']})")
        except ArtifactMissingError as e:
            # friendly missing: still failure for required git, but message distinct
            print(f"FAIL {ds} missing artifact (mode {args.mode}): {e}", file=sys.stderr)
            exit_code = 1
        except ArtifactHashMismatchError as e:
            print(f"FAIL {ds} hash mismatch (mode {args.mode}, hard failure): {e}", file=sys.stderr)
            exit_code = 1
        except ArtifactManifestError as e:
            print(f"FAIL {ds} manifest invalid: {e}", file=sys.stderr)
            exit_code = 1
        except Exception as e:
            print(f"FAIL {ds} unexpected: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            exit_code = 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
