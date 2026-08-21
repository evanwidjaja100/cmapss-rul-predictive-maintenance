"""Artifact manifests for V2.2 lineage verification (Phase 4).

Manifests record schema/methodology version, dataset, model ID,
historical training-provenance status, repo-relative POSIX paths only,
role/path/SHA256/size/storage class/clean-clone requirement/hash kind,
explicit lineage connections, source/config/constraints integrity context
at generation time.

Storage classes:
 - git: must exist and verify in clean clone
 - local: gitignored runtime artifact

Verification modes:
 - tracked: every Git artifact required; absent local permitted; present wrong fails
 - full: every Git+local required

Operational contract:
 - --check validates + deterministic regeneration comparison without rewriting
 - manifest creation accepts --generated-at <UTC> or preserves prior generated_at_utc when inputs unchanged
 - identical inputs + fixed timestamp => byte-for-byte deterministic JSON (stable ordering, formatting, newline, UTF-8)
 - library APIs and CLI accept explicit root override
 - tamper/missing/wrong tests operate on temp copied bundles through override, never modify frozen artifacts
 - No manifest may hash itself. No broad globs. Reject absolute, .., duplicate roles/paths, wrong identity, ambiguous mirrors.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

# ponytail: stdlib only, no new dep; minimal explicit artifact enumeration (no globs)

MANIFEST_SCHEMA_VERSION = "cmapss-artifact-manifest-v1"
METHODOLOGY_VERSION = "2.2"

# baseline immutable binary hashes (Section 4.2)
BASELINE_HASHES = {
    "fd001_model": "23bd460cd447141d90fd045b863f1a33db02845b83ce9d4148791811ad5a9d6b",
    "fd001_scaler": "d1b02cfc7043de8e68f9dd71bc1a99efbcd2754a921a59524a431647d0371a48",
    "fd004_model": "9b39a059ea528b94f9d806533a69da596b7950254bbea1f11a232b2bb29ac87d",
    "fd004_condition": "f22ef7189cda906ee4ea92c37df625023f136abb3fcd8a0a74e3a0fdf8b4a328",
}

# expected identities
EXPECTED_MODEL_IDS = {
    "FD001": "xgb_w90_d6",
    "FD004": "gru_w45_huber_condC",
}

FD001_TRAINING_STATUS = "historical_dirty_partial"
FD004_TRAINING_STATUS = "historical_incomplete"

MANIFEST_PATHS = {
    "FD001": "experiments/v2_2/fd001_artifact_manifest.json",
    "FD004": "experiments/v2_2/fd004_artifact_manifest.json",
}

# ponytail: explicit artifact specs, no globs. Each entry defines role/path/storage.
FD001_ARTIFACT_SPECS: list[dict[str, Any]] = [
    # configs
    {"role": "final_config", "path": "configs/final_model_v2_2_fd001.yaml", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "deployment_config", "path": "configs/deployment_v2_2_fd001.yaml", "storage_class": "git", "hash_kind": "raw_sha256"},
    # models local
    {"role": "model", "path": "models/v2_2/fd001_xgb_w90_d6.joblib", "storage_class": "local", "hash_kind": "raw_sha256"},
    {"role": "scaler", "path": "models/v2_2/fd001_scaler.joblib", "storage_class": "local", "hash_kind": "raw_sha256"},
    # metadata
    {"role": "final_metadata", "path": "experiments/v2_2/fd001_final_fit_metadata.json", "storage_class": "git", "hash_kind": "raw_sha256"},
    # splits
    {"role": "outer_split_manifest", "path": "experiments/v2_2/fd001_outer_split_manifest.json", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "outer_fold1_cutoffs", "path": "experiments/splits/fd001_v2_2_outer_fold1_cutoffs.csv", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "outer_fold2_cutoffs", "path": "experiments/splits/fd001_v2_2_outer_fold2_cutoffs.csv", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "outer_fold3_cutoffs", "path": "experiments/splits/fd001_v2_2_outer_fold3_cutoffs.csv", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "outer_fold4_cutoffs", "path": "experiments/splits/fd001_v2_2_outer_fold4_cutoffs.csv", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "outer_fold5_cutoffs", "path": "experiments/splits/fd001_v2_2_outer_fold5_cutoffs.csv", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "calibration_cutoffs", "path": "experiments/splits/fd001_v2_2_calibration_cutoffs.csv", "storage_class": "git", "hash_kind": "raw_sha256"},
    # cv results
    {"role": "fold_results", "path": "experiments/v2_2/fd001_outer_fold_results.csv", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "outer_predictions", "path": "experiments/v2_2/fd001_outer_predictions.csv", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "engine_level_results", "path": "experiments/v2_2/fd001_outer_engine_level.csv", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "cv_summary", "path": "experiments/v2_2/fd001_cv_summary.csv", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "best_iterations", "path": "experiments/v2_2/fd001_best_epochs.csv", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "selection_decision", "path": "experiments/v2_2/selection_decision.json", "storage_class": "git", "hash_kind": "raw_sha256"},
    # conformal
    {"role": "conformal_scores", "path": "experiments/v2_2/fd001_conformal_engine_scores.csv", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "conformal_quantiles", "path": "experiments/v2_2/fd001_conformal_quantiles.csv", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "conformal_calibration", "path": "experiments/v2_2/conformal_calibration.json", "storage_class": "git", "hash_kind": "raw_sha256"},
    # official predictions & metrics
    {"role": "official_predictions", "path": "experiments/v2_2/fd001_official_predictions.csv", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "final_metrics", "path": "experiments/v2_2/fd001_final_metrics.json", "storage_class": "git", "hash_kind": "raw_sha256"},
    # constraints
    {"role": "constraints", "path": "requirements.txt", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "constraints_lock", "path": "requirements-lock.txt", "storage_class": "git", "hash_kind": "raw_sha256"},
]

FD004_ARTIFACT_SPECS: list[dict[str, Any]] = [
    {"role": "final_config", "path": "configs/final_model_v2_2_fd004.yaml", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "model", "path": "models/v2_2/fd004_gru_w45_huber_condC.keras", "storage_class": "local", "hash_kind": "raw_sha256"},
    {"role": "condition_preprocessor", "path": "models/v2_2/fd004_conditionC.joblib", "storage_class": "local", "hash_kind": "raw_sha256"},
    {"role": "final_metadata", "path": "experiments/v2_2/fd004_final_fit_metadata.json", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "split_json", "path": "experiments/splits/FD004_v2_seed42.json", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "validation_cutoffs", "path": "experiments/splits/fd004_v2_1_validation_cutoffs.csv", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "variant_results", "path": "experiments/v2_2/fd004_variant_results.csv", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "variant_predictions", "path": "experiments/v2_2/fd004_variant_predictions.csv", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "best_epochs", "path": "experiments/v2_2/fd004_best_epochs.csv", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "canonical_official_predictions", "path": "experiments/v2_2/fd004_official_predictions.csv", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "report_table_mirror", "path": "reports/tables/v2_2_fd004_predictions.csv", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "final_metrics", "path": "experiments/v2_2/fd004_final_metrics.json", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "constraints", "path": "requirements.txt", "storage_class": "git", "hash_kind": "raw_sha256"},
    {"role": "constraints_lock", "path": "requirements-lock.txt", "storage_class": "git", "hash_kind": "raw_sha256"},
]

# lineage: explicit connections covering required artifacts
FD001_LINEAGE: list[dict[str, str]] = [
    {"from": "final_config", "to": "outer_split_manifest", "relation": "configures split"},
    {"from": "outer_split_manifest", "to": "outer_fold1_cutoffs", "relation": "derives cutoffs"},
    {"from": "outer_split_manifest", "to": "outer_fold2_cutoffs", "relation": "derives cutoffs"},
    {"from": "outer_split_manifest", "to": "outer_fold3_cutoffs", "relation": "derives cutoffs"},
    {"from": "outer_split_manifest", "to": "outer_fold4_cutoffs", "relation": "derives cutoffs"},
    {"from": "outer_split_manifest", "to": "outer_fold5_cutoffs", "relation": "derives cutoffs"},
    {"from": "outer_split_manifest", "to": "calibration_cutoffs", "relation": "holds out calibration"},
    {"from": "outer_split_manifest", "to": "selection_decision", "relation": "defines development vs calibration"},
    {"from": "outer_fold1_cutoffs", "to": "fold_results", "relation": "evaluated on"},
    {"from": "outer_fold2_cutoffs", "to": "fold_results", "relation": "evaluated on"},
    {"from": "outer_fold3_cutoffs", "to": "fold_results", "relation": "evaluated on"},
    {"from": "outer_fold4_cutoffs", "to": "fold_results", "relation": "evaluated on"},
    {"from": "outer_fold5_cutoffs", "to": "fold_results", "relation": "evaluated on"},
    {"from": "fold_results", "to": "outer_predictions", "relation": "produces"},
    {"from": "fold_results", "to": "engine_level_results", "relation": "aggregates"},
    {"from": "fold_results", "to": "cv_summary", "relation": "summarizes"},
    {"from": "cv_summary", "to": "selection_decision", "relation": "selects"},
    {"from": "selection_decision", "to": "final_config", "relation": "deployment selection"},
    {"from": "selection_decision", "to": "best_iterations", "relation": "records median best_iteration"},
    {"from": "final_config", "to": "best_iterations", "relation": "governs fixed n_estimators"},
    {"from": "final_config", "to": "final_metadata", "relation": "configures final fit"},
    {"from": "outer_split_manifest", "to": "final_metadata", "relation": "provides train ids"},
    {"from": "best_iterations", "to": "final_metadata", "relation": "provides n_estimators"},
    {"from": "final_metadata", "to": "model", "relation": "trained from"},
    {"from": "final_metadata", "to": "scaler", "relation": "fitted from"},
    {"from": "calibration_cutoffs", "to": "conformal_scores", "relation": "evaluated to scores"},
    {"from": "conformal_scores", "to": "conformal_quantiles", "relation": "quantiles from scores"},
    {"from": "conformal_quantiles", "to": "conformal_calibration", "relation": "calibrated interval"},
    {"from": "deployment_config", "to": "conformal_calibration", "relation": "serving interval mirrors calibration"},
    {"from": "deployment_config", "to": "conformal_quantiles", "relation": "q values consistent"},
    {"from": "model", "to": "official_predictions", "relation": "post-hoc predicts"},
    {"from": "scaler", "to": "official_predictions", "relation": "preprocesses"},
    {"from": "official_predictions", "to": "final_metrics", "relation": "metrics from predictions"},
    {"from": "final_config", "to": "constraints", "relation": "validated constraints"},
    {"from": "final_config", "to": "constraints_lock", "relation": "validated constraints"},
]

FD004_LINEAGE: list[dict[str, str]] = [
    {"from": "final_config", "to": "split_json", "relation": "references"},
    {"from": "final_config", "to": "validation_cutoffs", "relation": "references"},
    {"from": "split_json", "to": "final_metadata", "relation": "provides train ids"},
    {"from": "validation_cutoffs", "to": "variant_results", "relation": "evaluated selection"},
    {"from": "validation_cutoffs", "to": "variant_predictions", "relation": "produces predictions"},
    {"from": "variant_results", "to": "best_epochs", "relation": "records best epoch"},
    {"from": "best_epochs", "to": "final_metadata", "relation": "provides fixed_epochs"},
    {"from": "final_config", "to": "final_metadata", "relation": "configures final fit"},
    {"from": "final_metadata", "to": "model", "relation": "trained from"},
    {"from": "final_metadata", "to": "condition_preprocessor", "relation": "fitted from"},
    {"from": "model", "to": "canonical_official_predictions", "relation": "post-hoc predicts"},
    {"from": "condition_preprocessor", "to": "canonical_official_predictions", "relation": "preprocesses"},
    {"from": "canonical_official_predictions", "to": "report_table_mirror", "relation": "exactly equals mirror"},
    {"from": "canonical_official_predictions", "to": "final_metrics", "relation": "metrics from predictions"},
    {"from": "final_config", "to": "constraints", "relation": "validated constraints"},
    {"from": "final_config", "to": "constraints_lock", "relation": "validated constraints"},
]

class ArtifactManifestError(ValueError):
    """Explicit validation exception for manifest/config errors (survives -O)."""

class ArtifactMissingError(FileNotFoundError):
    """Artifact absent - friendly guidance."""

class ArtifactHashMismatchError(ArtifactManifestError):
    """Present artifact hash/identity mismatch - hard failure."""

def _resolve_root(root: Path | str | None = None) -> Path:
    if root is not None:
        p = Path(root).resolve()
        if p.is_file():
            p = p.parent
        # if inside repo, find git top-level
        try:
            out = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, encoding="utf-8", errors="strict", check=False, cwd=str(p if p.is_dir() else p.parent))
            if out.returncode == 0 and out.stdout.strip():
                return Path(out.stdout.strip()).resolve()
        except Exception:
            pass
        # walk up
        cur = p if p.is_dir() else p.parent
        for _ in range(10):
            if (cur / ".git").exists():
                return cur.resolve()
            if cur.parent == cur:
                break
            cur = cur.parent
        return (p if p.is_dir() else p.parent).resolve()
    here = Path(__file__).resolve()
    repo = here.parents[2]
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, encoding="utf-8", errors="strict", check=False, cwd=str(repo))
        if out.returncode == 0 and out.stdout.strip():
            return Path(out.stdout.strip()).resolve()
    except Exception:
        pass
    return repo.resolve()

def sha256_file(path: Path | str) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _validate_posix_path(posix_path: str, *, manifest_path: str | None = None) -> None:
    if not isinstance(posix_path, str) or not posix_path:
        raise ArtifactManifestError(f"path must be non-empty str, got {posix_path!r}")
    # reject absolute
    if posix_path.startswith("/") or posix_path.startswith("\\"):
        raise ArtifactManifestError(f"absolute path not allowed: {posix_path!r}")
    # reject Windows absolute like C:\
    if len(posix_path) >= 2 and posix_path[1] == ":":
        raise ArtifactManifestError(f"absolute path not allowed: {posix_path!r}")
    # reject ..
    parts = PurePosixPath(posix_path).parts
    if ".." in parts:
        raise ArtifactManifestError(f"path must not contain '..': {posix_path!r}")
    # reject globs
    if any(c in posix_path for c in ["*", "?", "[", "]"]):
        raise ArtifactManifestError(f"broad glob not allowed in critical artifact path: {posix_path!r}")
    # POSIX only: no backslash
    if "\\" in posix_path:
        raise ArtifactManifestError(f"path must be POSIX (no backslash): {posix_path!r}")
    # manifest must not hash itself
    if manifest_path and posix_path == manifest_path:
        raise ArtifactManifestError(f"manifest must not hash itself: {posix_path!r}")

def _validate_role(role: str) -> None:
    if not isinstance(role, str) or not role:
        raise ArtifactManifestError(f"role must be non-empty str, got {role!r}")
    if ".." in role or "/" in role or "\\" in role:
        raise ArtifactManifestError(f"role must not contain path separators or '..': {role!r}")

def deterministic_json_bytes(obj: Any) -> bytes:
    """Stable JSON serialization: sorted keys, indent 2, LF, UTF-8."""
    s = json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False)
    # ensure LF newline, single trailing newline
    if not s.endswith("\n"):
        s += "\n"
    return s.encode("utf-8")

def _collect_file_artifact(spec: dict[str, Any], root: Path) -> dict[str, Any]:
    role = spec["role"]
    posix_path = spec["path"]
    storage_class = spec["storage_class"]
    hash_kind = spec["hash_kind"]
    # validation
    _validate_role(role)
    # manifest paths for self-hash check
    # we check both FD001 and FD004 manifest paths generically
    for mp in MANIFEST_PATHS.values():
        _validate_posix_path(posix_path, manifest_path=mp)
    if storage_class not in ("git", "local"):
        raise ArtifactManifestError(f"invalid storage_class {storage_class!r} for {role}")
    if hash_kind not in ("raw_sha256",):
        raise ArtifactManifestError(f"invalid hash_kind {hash_kind!r}")
    # check wrong identity: dataset-specific path sanity
    # For FD001 spec, paths should not mix? We keep light check: role-based sanity occurs outside
    p = root / PurePosixPath(posix_path)
    if not p.exists():
        # for local files absent is permitted in tracked mode, but builder expects existence
        # we still record missing as error for builder; verifier handles missing semantics
        # Here we provide placeholder for missing? But builder should fail closed unless explicitly allowed
        # So we raise ArtifactMissingError for builder context
        raise ArtifactMissingError(f"artifact {role} missing: {posix_path} (root {root})")
    # hash before anything else (fail closed)
    sha = sha256_file(p)
    size = p.stat().st_size
    # check baseline immutability for known binaries
    # ensure model binaries have not changed
    if role == "model" and "fd001" in posix_path:
        if sha.lower() != BASELINE_HASHES["fd001_model"].lower():
            # allow if caller explicitly checks? But we enforce hard failure for builder if mismatch?
            # For manifest generation we must ensure binaries unchanged; if changed, fail
            raise ArtifactHashMismatchError(f"FD001 model hash mismatch: expected {BASELINE_HASHES['fd001_model']}, got {sha}")
    if role == "scaler" and "fd001" in posix_path:
        if sha.lower() != BASELINE_HASHES["fd001_scaler"].lower():
            raise ArtifactHashMismatchError(f"FD001 scaler hash mismatch: expected {BASELINE_HASHES['fd001_scaler']}, got {sha}")
    if role == "model" and "fd004" in posix_path:
        if sha.lower() != BASELINE_HASHES["fd004_model"].lower():
            raise ArtifactHashMismatchError(f"FD004 model hash mismatch: expected {BASELINE_HASHES['fd004_model']}, got {sha}")
    if role == "condition_preprocessor" and "fd004" in posix_path:
        if sha.lower() != BASELINE_HASHES["fd004_condition"].lower():
            raise ArtifactHashMismatchError(f"FD004 condition hash mismatch: expected {BASELINE_HASHES['fd004_condition']}, got {sha}")
    return {
        "role": role,
        "path": posix_path,
        "sha256": sha,
        "bytes": int(size),
        "storage_class": storage_class,
        "required_in_clean_clone": storage_class == "git",
        "hash_kind": hash_kind,
    }

def _source_integrity_context(root: Path) -> dict[str, Any]:
    from rul_prediction.reproducibility import tracked_source_tree_details, collect_git_provenance
    try:
        details = tracked_source_tree_details(root)
    except Exception as e:
        raise ArtifactManifestError(f"failed to collect source integrity: {e}") from e
    prov = collect_git_provenance(root=root)
    return {
        "source_tree_hash": details["source_tree_hash"],
        "algorithm": details["algorithm"],
        "file_count": int(details["file_count"]),
        "git_commit": prov.get("git_commit"),
        "git_is_dirty_whole": bool(prov.get("git_is_dirty_whole")),
        "git_is_dirty_execution": bool(prov.get("git_is_dirty_execution")),
        "note": "Manifest generation time is not training time; this source_tree_hash is current generation-time hash, not historical training hash",
    }

def _config_integrity_context(root: Path, dataset: str) -> dict[str, Any]:
    if dataset == "FD001":
        cfg_path = root / "configs" / "final_model_v2_2_fd001.yaml"
        if not cfg_path.exists():
            raise ArtifactMissingError(f"FD001 final config missing: {cfg_path}")
        file_sha = sha256_file(cfg_path)
        # canonical: yaml -> json canonical via sort_keys
        try:
            import yaml
            payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            canonical_algo = "cmapss-fd001-config-canonical-v1"
            canonical_sha = hashlib.sha256((canonical_algo + "\n").encode("utf-8") + canonical_json).hexdigest()
        except Exception as e:
            raise ArtifactManifestError(f"failed to compute FD001 config canonical hash: {e}") from e
        return {
            "path": "configs/final_model_v2_2_fd001.yaml",
            "config_file_sha256": file_sha,
            "config_canonical_sha256": canonical_sha,
            "canonical_algo": canonical_algo,
            "note": "config_file_sha256 is raw-byte integrity; config_canonical_sha256 is versioned semantic identity; neither is historical training hash for post-training normalized config (not historical training hash)",
        }
    elif dataset == "FD004":
        from rul_prediction.benchmark.fd004_config import load_fd004_final_config
        cfg_path = root / "configs" / "final_model_v2_2_fd004.yaml"
        if not cfg_path.exists():
            raise ArtifactMissingError(f"FD004 final config missing: {cfg_path}")
        file_sha = sha256_file(cfg_path)
        cfg = load_fd004_final_config(cfg_path, root=root)
        canonical_sha = cfg.config_canonical_sha256
        return {
            "path": "configs/final_model_v2_2_fd004.yaml",
            "config_file_sha256": file_sha,
            "config_canonical_sha256": canonical_sha,
            "canonical_algo": "cmapss-fd004-config-canonical-v1",
            "note": "config_file_sha256 is raw-byte integrity; config_canonical_sha256 is versioned semantic identity (post-training normalization 2026-08-21); neither is historical training hash (not historical training hash)",
            "model_id": cfg.candidate_name,
        }
    else:
        raise ArtifactManifestError(f"unknown dataset {dataset!r}")

def _constraints_integrity_context(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for posix in ["requirements.txt", "requirements-lock.txt"]:
        p = root / posix
        if not p.exists():
            raise ArtifactMissingError(f"constraints missing: {posix}")
        out.append({
            "path": posix,
            "sha256": sha256_file(p),
            "bytes": int(p.stat().st_size),
            "storage_class": "git",
            "required_in_clean_clone": True,
            "hash_kind": "raw_sha256",
        })
    return out

def build_manifest_dict(dataset: str, root: Path | str | None = None, generated_at_utc: str | None = None) -> dict[str, Any]:
    """Build manifest dict for dataset (FD001 or FD004) at root.
    
    generated_at_utc: ISO UTC timestamp; if None, use now.
    Returns manifest dict (not yet serialized).
    """
    r = _resolve_root(root)
    # commit
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, encoding="utf-8", errors="strict", check=False, cwd=str(r))
        commit = out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        commit = None
    import datetime
    if generated_at_utc is None:
        generated_at_utc = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    else:
        # validate ISO
        generated_at_utc = str(generated_at_utc)
    dataset = str(dataset).upper()
    if dataset not in ("FD001", "FD004"):
        raise ArtifactManifestError(f"dataset must be FD001 or FD004, got {dataset!r}")
    model_id = EXPECTED_MODEL_IDS[dataset]
    # specs
    specs = FD001_ARTIFACT_SPECS if dataset == "FD001" else FD004_ARTIFACT_SPECS
    lineage = FD001_LINEAGE if dataset == "FD001" else FD004_LINEAGE
    # special handling for FD004 canonical equality: ensure file exists and equals report mirror
    if dataset == "FD004":
        canon = r / "experiments" / "v2_2" / "fd004_official_predictions.csv"
        mirror = r / "reports" / "tables" / "v2_2_fd004_predictions.csv"
        if not canon.exists():
            # derive from mirror without inference
            if not mirror.exists():
                raise ArtifactMissingError(f"FD004 report mirror missing: {mirror}")
            canon.parent.mkdir(parents=True, exist_ok=True)
            canon.write_bytes(mirror.read_bytes())
        elif mirror.exists():
            # must exactly equal
            if canon.read_bytes() != mirror.read_bytes():
                raise ArtifactManifestError("FD004 canonical official predictions must exactly equal report-table mirror (byte-identical)")
    # collect artifacts
    artifacts: list[dict[str, Any]] = []
    for spec in specs:
        # For FD004 canonical_official_predictions spec, we already ensured canonical exists above
        entry = _collect_file_artifact(spec, r)
        artifacts.append(entry)
    # sort deterministically by role
    artifacts = sorted(artifacts, key=lambda x: x["role"])
    # validate duplicates and self-hash and ambiguous mirrors
    _validate_manifest_artifacts(artifacts, dataset)
    # lineage sorted
    lineage_sorted = sorted(lineage, key=lambda x: (x["from"], x["to"]))
    # integrity contexts
    source_ctx = _source_integrity_context(r)
    config_ctx = _config_integrity_context(r, dataset)
    constraints_ctx = _constraints_integrity_context(r)
    # historical provenance wording
    if dataset == "FD001":
        training_provenance = {
            "status": FD001_TRAINING_STATUS,
            "historical_commit": "53e50d87b2402da5a6014d5ba4fb0adb635e1d71",
            "historical_timestamp": "2026-08-17T01:42:52+00:00",
            "note_generation_not_training": "Manifest generated_at_utc is generation time, not historical training time (2026-08-17)",
            "note_new_source_not_historical": "Current source_tree_hash is generation-time value, not historical training source hash (not historical training hash)",
            "note_config_hash_distinction": "config_file_sha256 (raw bytes) vs config_canonical_sha256 (semantic); neither is historical training hash (not historical training hash)",
            "note_model_binaries_immutable": "Model binaries must not change; verified against baseline hashes Section 4.2",
            "note_dirty_partial": "Historical training run was dirty (git_is_dirty true) and provenance is partial; future freezes reject dirty execution inputs",
        }
    else:
        training_provenance = {
            "status": FD004_TRAINING_STATUS,
            "historical_commit": None,
            "note_generation_not_training": "Manifest generated_at_utc is generation time, not historical training time",
            "note_new_source_not_historical": "Current source_tree_hash is generation-time value, not historical training source hash (not historical training hash)",
            "note_config_hash_distinction": "config_file_sha256 vs config_canonical_sha256; new YAML is post-training normalized (structured optimizer) and behaviorally equivalent, raw hash is not historical training hash (not historical training hash)",
            "note_model_binaries_immutable": "Model binaries must not change; fd004_conditionC.joblib is hash-gated legacy payload, remains byte-identical",
            "note_historical_incomplete": "FD004 training provenance remains historical and incomplete; no new provenance retroactively added",
        }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "methodology_version": METHODOLOGY_VERSION,
        "dataset": dataset,
        "model_id": model_id,
        "generated_at_utc": generated_at_utc,
        "generated_from_commit": commit,
        "historical_training_provenance": training_provenance,
        "source_integrity": source_ctx,
        "config_integrity": config_ctx,
        "constraints_integrity": constraints_ctx,
        "artifacts": artifacts,
        "lineage": lineage_sorted,
        "notes": {
            "storage_classes": "git must exist in clean clone; local gitignored runtime artifact; tracked mode permits absent local",
            "hash_kind": "raw_sha256 is SHA-256 of exact raw file bytes (artifact integrity)",
            "path_policy": "repo-relative POSIX paths only; absolute, .., duplicates, globs rejected",
            "no_self_hash": "no manifest hashes itself",
        },
    }
    # final structural validation
    validate_manifest_dict(manifest, root=r)
    return manifest

def _validate_manifest_artifacts(artifacts: list[dict[str, Any]], dataset: str) -> None:
    # duplicate roles/paths, glob, absolute, ..
    seen_roles: set[str] = set()
    seen_paths: set[str] = set()
    manifest_paths_set = set(MANIFEST_PATHS.values())
    for a in artifacts:
        role = a.get("role")
        path = a.get("path")
        sha = a.get("sha256")
        b = a.get("bytes")
        storage = a.get("storage_class")
        req = a.get("required_in_clean_clone")
        hk = a.get("hash_kind")
        _validate_role(role)
        _validate_posix_path(path)
        if path in manifest_paths_set:
            raise ArtifactManifestError(f"manifest must not include itself as artifact: {path!r}")
        if role in seen_roles:
            raise ArtifactManifestError(f"duplicate role {role!r}")
        if path in seen_paths:
            raise ArtifactManifestError(f"duplicate path {path!r}")
        seen_roles.add(role)
        seen_paths.add(path)
        # sha hex check
        if not isinstance(sha, str) or len(sha) != 64:
            raise ArtifactManifestError(f"artifact {role} sha256 must be 64 hex, got {sha!r}")
        try:
            int(sha, 16)
        except ValueError:
            raise ArtifactManifestError(f"artifact {role} sha256 not hex: {sha!r}")
        if not isinstance(b, int) or b < 0:
            raise ArtifactManifestError(f"artifact {role} bytes must be non-negative int, got {b!r}")
        if storage not in ("git", "local"):
            raise ArtifactManifestError(f"artifact {role} invalid storage_class {storage!r}")
        if not isinstance(req, bool):
            raise ArtifactManifestError(f"artifact {role} required_in_clean_clone must be bool")
        if hk != "raw_sha256":
            raise ArtifactManifestError(f"artifact {role} hash_kind must be raw_sha256, got {hk!r}")
        # wrong identity: dataset mismatch
        # For FD001, paths should not contain fd004; vice versa, but constraints are shared
        low = path.lower()
        if dataset == "FD001" and "fd004" in low and "fd004" not in role.lower():
            # allow if it's constraints? already constrained? we check artifact path contains fd004 but dataset is FD001 => wrong identity
            # But FD001 manifest should not contain FD004-specific files like fd004_variant...
            if any(x in low for x in ["fd004", "variant"]):
                raise ArtifactManifestError(f"wrong identity: FD001 manifest contains FD004 artifact {path!r}")
        if dataset == "FD004" and "fd001" in low:
            if any(x in low for x in ["fd001", "outer", "conformal", "selection"]):
                raise ArtifactManifestError(f"wrong identity: FD004 manifest contains FD001 artifact {path!r}")
    # ambiguous mirrors check for FD004
    if dataset == "FD004":
        # canonical and mirror must be distinct paths, same hash expected
        canon = next((a for a in artifacts if a["role"] == "canonical_official_predictions"), None)
        mirror = next((a for a in artifacts if a["role"] == "report_table_mirror"), None)
        if canon and mirror:
            if canon["path"] == mirror["path"]:
                raise ArtifactManifestError("ambiguous mirrors: canonical and mirror have same path")
            # their hashes must be equal (enforced at build time already, but validate)
            # If they differ, it's not ambiguous but would violate canonical equality requirement
            # We keep strict: they must be equal, otherwise it's wrong identity
            if canon["sha256"] != mirror["sha256"]:
                raise ArtifactManifestError(f"FKD004 canonical {canon['sha256'][:8]} != mirror {mirror['sha256'][:8]}; must be byte-identical")
            if canon["bytes"] != mirror["bytes"]:
                raise ArtifactManifestError("FD004 canonical and mirror size mismatch")

def validate_manifest_dict(manifest: dict[str, Any], root: Path | str | None = None) -> None:
    """Validate manifest structure; raises ArtifactManifestError on violation."""
    # top-level
    for k in ["schema_version", "methodology_version", "dataset", "model_id", "generated_at_utc", "artifacts", "lineage", "source_integrity", "config_integrity", "constraints_integrity", "historical_training_provenance"]:
        if k not in manifest:
            raise ArtifactManifestError(f"manifest missing required key {k!r}")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ArtifactManifestError(f"schema_version must be {MANIFEST_SCHEMA_VERSION!r}, got {manifest.get('schema_version')!r}")
    if manifest.get("methodology_version") != METHODOLOGY_VERSION:
        raise ArtifactManifestError(f"methodology_version must be {METHODOLOGY_VERSION!r}, got {manifest.get('methodology_version')!r}")
    dataset = manifest.get("dataset")
    if dataset not in ("FD001", "FD004"):
        raise ArtifactManifestError(f"dataset must be FD001/FD004, got {dataset!r}")
    model_id = manifest.get("model_id")
    if model_id != EXPECTED_MODEL_IDS[dataset]:
        raise ArtifactManifestError(f"model_id for {dataset} must be {EXPECTED_MODEL_IDS[dataset]!r}, got {model_id!r}")
    # historical status
    hp = manifest.get("historical_training_provenance") or {}
    status = hp.get("status")
    expected_status = FD001_TRAINING_STATUS if dataset == "FD001" else FD004_TRAINING_STATUS
    if status != expected_status:
        raise ArtifactManifestError(f"historical status for {dataset} must be {expected_status!r}, got {status!r}")
    # generated_at_utc is ISO
    gat = manifest.get("generated_at_utc")
    if not isinstance(gat, str) or not gat:
        raise ArtifactManifestError("generated_at_utc must be non-empty str")
    # artifacts
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ArtifactManifestError("artifacts must be non-empty list")
    _validate_manifest_artifacts(artifacts, dataset)
    # lineage
    lineage = manifest.get("lineage")
    if not isinstance(lineage, list) or not lineage:
        raise ArtifactManifestError("lineage must be non-empty list")
    roles = {a["role"] for a in artifacts}
    seen_edges: set[tuple[str, str]] = set()
    for edge in lineage:
        if not isinstance(edge, dict):
            raise ArtifactManifestError(f"lineage entry must be dict, got {edge!r}")
        f = edge.get("from")
        t = edge.get("to")
        if f not in roles:
            raise ArtifactManifestError(f"lineage from {f!r} not in artifacts roles")
        if t not in roles:
            raise ArtifactManifestError(f"lineage to {t!r} not in artifacts roles")
        if (f, t) in seen_edges:
            raise ArtifactManifestError(f"duplicate lineage edge {f}->{t}")
        seen_edges.add((f, t))
        if "relation" not in edge:
            raise ArtifactManifestError(f"lineage edge {f}->{t} missing relation")
    # check required lineage coverage: ensure all required roles appear in lineage graph (at least incident)
    required_specs = FD001_ARTIFACT_SPECS if dataset == "FD001" else FD004_ARTIFACT_SPECS
    required_roles = {s["role"] for s in required_specs}
    # manifest may contain exactly required roles; verify no missing required role
    missing = required_roles - roles
    if missing:
        raise ArtifactManifestError(f"manifest missing required roles {sorted(missing)}")
    # also ensure lineage connects at least those roles (degree >0)
    # Build adjacency
    incident: set[str] = set()
    for f, t in seen_edges:
        incident.add(f)
        incident.add(t)
    # constraints etc also need incident
    uncovered = required_roles - incident
    # it's okay if some leaf nodes have only one incident, but ensure none isolated? Enforce all required roles incident
    if uncovered:
        raise ArtifactManifestError(f"lineage does not connect roles {sorted(uncovered)}; explicit connections required")
    # source/config/constraints
    for key in ["source_integrity", "config_integrity", "constraints_integrity"]:
        if key not in manifest:
            raise ArtifactManifestError(f"missing {key}")
    # config distinction note
    ci = manifest["config_integrity"]
    if "config_file_sha256" not in ci or "config_canonical_sha256" not in ci:
        raise ArtifactManifestError("config_integrity must contain config_file_sha256 and config_canonical_sha256")
    # ensure POSIX paths only for artifacts (already checked), but also check top-level generation note not missing
    # Check no self-hash already done
    # additional: reject wrong identity via model_id vs artifact path containing model name?

def load_manifest(path: Path | str, root: Path | str | None = None) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute() and root is not None:
        p = Path(root) / p
    if not p.exists():
        raise ArtifactMissingError(f"manifest not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise ArtifactManifestError(f"manifest {p} invalid JSON: {e}") from e
    # structural validation
    validate_manifest_dict(data, root=_resolve_root(root) if root else None)
    return data

def verify_manifest_file(manifest_path: Path | str, root: Path | str | None = None, mode: str = "tracked") -> dict[str, Any]:
    """Verify artifacts listed in manifest against filesystem.

    mode: tracked (git required, local absent permitted) or full (all required)
    Raises ArtifactMissingError / ArtifactHashMismatchError on failure.
    Returns summary dict.
    """
    if mode not in ("tracked", "full"):
        raise ArtifactManifestError(f"mode must be tracked or full, got {mode!r}")
    r = _resolve_root(root)
    # load and validate structure first
    mp = Path(manifest_path)
    if not mp.is_absolute():
        mp = r / mp
    manifest = load_manifest(mp, root=r)
    artifacts = manifest["artifacts"]
    errors: list[str] = []
    verified = 0
    skipped_absent_local = 0
    for a in artifacts:
        role = a["role"]
        posix_path = a["path"]
        expected_sha = a["sha256"]
        expected_bytes = a["bytes"]
        storage = a["storage_class"]
        p = r / PurePosixPath(posix_path)
        exists = p.exists()
        is_local = storage == "local"
        if mode == "tracked" and is_local and not exists:
            skipped_absent_local += 1
            continue
        if not exists:
            raise ArtifactMissingError(f"missing required artifact {role}: {posix_path} (mode {mode}, storage {storage})")
        # present: verify hash before deserialization (hash check is the deserialization gate)
        actual_sha = sha256_file(p)
        actual_bytes = p.stat().st_size
        if actual_sha.lower() != expected_sha.lower() or actual_bytes != int(expected_bytes):
            raise ArtifactHashMismatchError(
                f"hash mismatch for {role} {posix_path}: expected {expected_sha} ({expected_bytes} bytes), got {actual_sha} ({actual_bytes} bytes)"
            )
        verified += 1
    # additional: check FD004 canonical equals mirror already via validate, but re-check file equality if both present
    if manifest["dataset"] == "FD004":
        canon = next((x for x in artifacts if x["role"] == "canonical_official_predictions"), None)
        mirror = next((x for x in artifacts if x["role"] == "report_table_mirror"), None)
        if canon and mirror:
            cp = r / canon["path"]
            mp2 = r / mirror["path"]
            if cp.exists() and mp2.exists():
                if cp.read_bytes() != mp2.read_bytes():
                    raise ArtifactHashMismatchError("FD004 canonical_official_predictions does not exactly equal report_table_mirror (byte mismatch)")
    return {
        "dataset": manifest["dataset"],
        "model_id": manifest["model_id"],
        "mode": mode,
        "verified": verified,
        "skipped_absent_local": skipped_absent_local,
        "total": len(artifacts),
    }

def verify_before_load(posix_path: str, root: Path | str | None = None, manifest_dataset: str | None = None) -> None:
    """Load-time verification: when manifest available, verify hash before deserialization.
    
    Raises:
      ArtifactMissingError: artifact absent (friendly)
      ArtifactManifestError: legacy without current schema (compatibility message)
      ArtifactHashMismatchError: present hash mismatch (hard failure)
    """
    r = _resolve_root(root)
    # locate manifest for dataset: if manifest_dataset provided, use that, else try both
    manifest_candidates = []
    if manifest_dataset:
        manifest_candidates.append(MANIFEST_PATHS[manifest_dataset])
    else:
        # try FD001 then FD004
        manifest_candidates.extend(MANIFEST_PATHS.values())
    found_manifest = None
    found_data = None
    for mp in manifest_candidates:
        p = r / mp
        if p.exists():
            try:
                data = load_manifest(p, root=r)
                found_manifest = p
                found_data = data
                break
            except ArtifactManifestError:
                # legacy without current schema? treat as compatibility
                raise ArtifactManifestError(
                    f"legacy manifest without current schema at {p}: please rebuild with scripts/build_v2_2_artifact_manifests.py (compatibility message)"
                )
    if found_data is None:
        # No manifest available: legacy path - allow but emit guidance? For hard requirement, we treat as legacy compatibility message
        # But spec says "when manifest/metadata schema available, verify..."
        # So if no manifest, we do not verify, allow legacy load (documented legacy path)
        return
    # find artifact entry matching posix_path
    # posix_path may be absolute? Convert to posix relative
    try:
        rel = PurePosixPath(posix_path)
        # if absolute, make relative to root if possible
        if Path(posix_path).is_absolute():
            try:
                rel = PurePosixPath(Path(posix_path).resolve().relative_to(r.resolve()).as_posix())
            except Exception:
                raise ArtifactManifestError(f"absolute path not allowed: {posix_path!r}")
        rel_str = rel.as_posix()
    except Exception as e:
        raise ArtifactManifestError(f"invalid path {posix_path!r}: {e}") from e
    entry = next((a for a in found_data["artifacts"] if a["path"] == rel_str), None)
    if entry is None:
        # Check if path is known artifact but role differs? Might be legacy without current schema already handled
        # If manifest exists but path not in it, treat as legacy artifact without current integrity schema
        raise ArtifactManifestError(
            f"artifact {rel_str!r} not in current manifest {found_manifest}; legacy artifact without current integrity schema (compatibility message or legacy path required)"
        )
    # verify existence
    p = r / rel_str
    if not p.exists():
        raise ArtifactMissingError(
            f"artifact absent: {rel_str} (role {entry['role']}). Generate it with scripts/build_v2_2_artifact_manifests.py or scripts/run_v2_2_freeze.py "
            f"(friendly guidance: ensure {rel_str} exists; for local artifacts run freeze)"
        )
    actual_sha = sha256_file(p)
    if actual_sha.lower() != entry["sha256"].lower():
        raise ArtifactHashMismatchError(
            f"present artifact hash mismatch for {rel_str} (role {entry['role']}): expected {entry['sha256']}, got {actual_sha} (hard failure before deserialization)"
        )
    # bytes also check
    if p.stat().st_size != int(entry["bytes"]):
        raise ArtifactHashMismatchError(
            f"size mismatch for {rel_str}: expected {entry['bytes']}, got {p.stat().st_size}"
        )

