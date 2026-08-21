"""FD004 authoritative typed configuration contract (Phase 2).

Distinction (Section 9.2):
  - ``config_file_sha256``: SHA-256 of exact raw file bytes (artifact integrity).
  - ``config_canonical_sha256``: SHA-256 of normalized typed mapping using
    versioned algorithm ``cmapss-fd004-config-canonical-v1`` (semantic identity).

Same distinction applies to split evidence: raw file hashes vs canonical
engine-ID/manifest hashes. Never compare a raw digest to a canonical digest.

Legacy optimizer string ``Adam(clipnorm=1.0)`` is accepted only if exact value,
normalized internally to ``{name: adam, clipnorm: 1.0}``. Always write structured.
Old-effective vs new-resolved mapping::

    legacy_effective = "Adam(clipnorm=1.0)"
    resolved_structured = {"name": "adam", "clipnorm": 1.0}

Numerics are unchanged; normalization is post-training for reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# ponytail: minimal dataclass contract; stdlib only, no new dependency.

REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_ALGO = "cmapss-fd004-config-canonical-v1"
LEGACY_OPTIMIZER_STRING = "Adam(clipnorm=1.0)"
# baseline hash of historical joblib (Section 4.2) – immutable gate
HISTORICAL_CONDITION_JOBLIB_SHA256 = "f22ef7189cda906ee4ea92c37df625023f136abb3fcd8a0a74e3a0fdf8b4a328"


class FD004ConfigError(ValueError):
    """Explicit validation exception for external FD004 config/manifest errors."""


def _fail(msg: str) -> None:
    raise FD004ConfigError(msg)


def _is_positive_int_no_bool(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def _is_finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _check_hex_sha256(s: Any, field: str) -> None:
    if not isinstance(s, str) or len(s) != 64:
        _fail(f"{field} must be 64-char hex sha256, got {s!r}")
    try:
        int(s, 16)
    except ValueError:
        _fail(f"{field} must be hex, got {s!r}")


def _normalize_optimizer(raw: Any) -> tuple[str, float]:
    """Return (name, clipnorm) from either legacy string or structured dict."""
    if isinstance(raw, str):
        if raw != LEGACY_OPTIMIZER_STRING:
            _fail(
                f"legacy optimizer string must be exactly {LEGACY_OPTIMIZER_STRING!r}, got {raw!r}"
            )
        return "adam", 1.0
    if isinstance(raw, dict):
        name = raw.get("name")
        clipnorm = raw.get("clipnorm")
        if not isinstance(name, str):
            _fail(f"optimizer.name must be str, got {name!r}")
        if not _is_finite_number(clipnorm):
            _fail(f"optimizer.clipnorm must be finite number, got {clipnorm!r}")
        name_lower = name.strip().lower()
        if name_lower != "adam":
            _fail(f"unsupported optimizer name {name!r}; only 'adam' allowed")
        cn = float(clipnorm)
        if not math.isfinite(cn) or cn <= 0:
            _fail(f"optimizer.clipnorm must be positive finite, got {clipnorm!r}")
        return name_lower, cn
    _fail(f"optimizer must be structured dict or legacy string {LEGACY_OPTIMIZER_STRING!r}, got {raw!r}")
    return "adam", 1.0  # unreachable


def _canonical_payload(config: FD004FinalConfig) -> dict:
    """Normalized behavior-driving payload for canonical hashing."""
    return {
        "methodology_version": config.methodology_version,
        "dataset": config.dataset,
        "model": {
            "candidate_name": config.candidate_name,
            "architecture": config.architecture,
            "window": config.window,
            "units": list(config.units),
            "dropout": float(config.dropout),
            "loss": config.loss,
            "optimizer": {"name": config.optimizer_name, "clipnorm": float(config.optimizer_clipnorm)},
            "learning_rate": float(config.learning_rate),
            "batch_size": int(config.batch_size),
            "seed": int(config.seed),
            "fixed_epochs": int(config.fixed_epochs),
        },
        "condition_preprocessing": {
            "variant": config.selected_variant,
            "clustering": {
                "method": config.clustering_method,
                "n_clusters": int(config.n_clusters),
                "random_state": int(config.clustering_random_state),
                "n_init": int(config.clustering_n_init),
            },
            "operating_setting_columns": list(config.operating_setting_columns),
            "sensor_scaling": {
                "mode": config.sensor_scaling_mode,
                "fit_scope": config.sensor_scaling_fit_scope,
            },
        },
        "splits": {
            "provenance": config.split_provenance,
            "validation_manifest": config.validation_manifest,
            "development_engine_count": int(config.development_engine_count),
            "validation_engine_count": int(config.validation_engine_count),
            "calibration_engine_count": int(config.calibration_engine_count),
            "final_engine_count": int(config.final_engine_count),
            "reserved_engine_count": int(config.reserved_engine_count),
            "development_engine_ids_sha256": config.development_engine_ids_sha256,
            "validation_engine_ids_sha256": config.validation_engine_ids_sha256,
            "calibration_engine_ids_sha256": config.calibration_engine_ids_sha256,
        },
        "training": {
            "validation_data_in_final_fit": bool(config.validation_data_in_final_fit),
        },
    }


def canonical_sha256_for_config(config: FD004FinalConfig) -> str:
    payload = _canonical_payload(config)
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    domain = CANONICAL_ALGO.encode("utf-8") + b"\n"
    return hashlib.sha256(domain + canonical_json).hexdigest()


def sha256_file(path: Path | str) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_repo_root(root: Path | str | None) -> Path:
    if root is None:
        return REPO_ROOT
    return Path(root).resolve()


@dataclass(frozen=True)
class FD004FinalConfig:
    """Immutable FD004 final config (GRU-only, huber, KMeans)."""

    methodology_version: str
    dataset: str
    candidate_name: str
    architecture: str
    window: int
    units: tuple[int, int]
    dropout: float
    loss: str
    optimizer_name: str
    optimizer_clipnorm: float
    learning_rate: float
    batch_size: int
    seed: int
    fixed_epochs: int
    selected_variant: str
    clustering_method: str
    n_clusters: int
    clustering_random_state: int
    clustering_n_init: int
    operating_setting_columns: tuple[str, ...]
    sensor_scaling_mode: str
    sensor_scaling_fit_scope: str
    split_provenance: str
    validation_manifest: str
    development_engine_count: int
    validation_engine_count: int
    calibration_engine_count: int
    final_engine_count: int
    reserved_engine_count: int
    development_engine_ids_sha256: str
    validation_engine_ids_sha256: str
    calibration_engine_ids_sha256: str
    validation_data_in_final_fit: bool
    # provenance for hash distinction
    config_file_sha256: str | None = None
    config_canonical_sha256: str | None = None
    source_path: Path | None = None

    def canonical_hash(self) -> str:
        return canonical_sha256_for_config(self)

    def model_artifact_path(self, root: Path | str | None = None) -> Path:
        r = _resolve_repo_root(root)
        # ponytail: derived name from identity; candidate validation ensures consistency
        name = f"fd004_gru_w{self.window}_{self.loss}_cond{self.selected_variant}.keras"
        # also verify candidate_name matches derived core (without prefix fd004_)
        # candidate_name is like gru_w45_huber_condC
        return r / "models" / "v2_2" / name

    def condition_artifact_path(self, root: Path | str | None = None) -> Path:
        r = _resolve_repo_root(root)
        return r / "models" / "v2_2" / f"fd004_condition{self.selected_variant}.joblib"

    def resolve_split_path(self, root: Path | str | None = None) -> Path:
        r = _resolve_repo_root(root)
        p = Path(self.split_provenance)
        # ponytail: allow absolute temp paths for testing (9.7 case 7); repo-relative is canonical
        if p.is_absolute():
            # still reject traversal that escapes via .. in absolute
            if ".." in p.parts:
                _fail(f"split provenance must not contain '..', got {p}")
            return p.resolve()
        if ".." in p.parts:
            _fail(f"split provenance must not contain '..', got {p}")
        return (r / p).resolve()

    def resolve_validation_manifest_path(self, root: Path | str | None = None) -> Path:
        r = _resolve_repo_root(root)
        p = Path(self.validation_manifest)
        if p.is_absolute():
            if ".." in p.parts:
                _fail(f"validation_manifest must not contain '..', got {p}")
            return p.resolve()
        if ".." in p.parts:
            _fail(f"validation_manifest must not contain '..', got {p}")
        return (r / p).resolve()

    def resolve_metadata_path(self, root: Path | str | None = None) -> Path:
        r = _resolve_repo_root(root)
        return r / "experiments" / "v2_2" / "fd004_final_fit_metadata.json"


def from_mapping(
    mapping: dict,
    *,
    source_path: Path | str | None = None,
    config_file_sha256: str | None = None,
) -> FD004FinalConfig:
    if not isinstance(mapping, dict):
        _fail("config mapping must be a dict")
    # top-level
    methodology_version = mapping.get("methodology_version")
    if str(methodology_version) != "2.2":
        _fail(f"methodology_version must be '2.2', got {methodology_version!r}")
    dataset = mapping.get("dataset")
    if dataset != "FD004":
        _fail(f"dataset must be 'FD004', got {dataset!r}")

    model = mapping.get("model")
    if not isinstance(model, dict):
        _fail("model must be a mapping")

    candidate_name = model.get("candidate_name")
    if not isinstance(candidate_name, str) or not candidate_name:
        _fail("model.candidate_name must be non-empty str")
    architecture = model.get("architecture")
    if not isinstance(architecture, str) or architecture.lower() != "gru":
        _fail(f"architecture must be 'gru', got {architecture!r}")

    window = model.get("window")
    if not _is_positive_int_no_bool(window):
        _fail(f"window must be positive int (non-bool), got {window!r}")

    units = model.get("units")
    if not isinstance(units, (list, tuple)) or len(units) != 2:
        _fail(f"units must be list/tuple of exactly 2 ints, got {units!r}")
    for u in units:
        if not _is_positive_int_no_bool(u):
            _fail(f"units entries must be positive ints (non-bool), got {u!r}")

    dropout = model.get("dropout")
    if not _is_finite_number(dropout):
        _fail(f"dropout must be finite number, got {dropout!r}")
    d = float(dropout)
    if not (0.0 <= d < 1.0):
        _fail(f"dropout must be in [0,1), got {dropout!r}")

    loss = model.get("loss")
    if not isinstance(loss, str) or loss.lower() not in {"huber", "mse"}:
        _fail(f"unsupported loss {loss!r}; allowed huber,mse (FD004 uses huber)")
    loss_norm = loss.lower()
    # FD004 historical uses huber; mse allowed only for synthetic test, but we keep strict:
    # Actually require huber for real config, but tests may use huber only.
    # Keep mse as allowed for future test flexibility; if plan says unsupported loss should fail, "foobar" will fail anyway.

    if "optimizer" not in model:
        _fail("model.optimizer missing")
    optimizer_name, optimizer_clipnorm = _normalize_optimizer(model.get("optimizer"))

    learning_rate = model.get("learning_rate")
    if not _is_finite_number(learning_rate):
        _fail(f"learning_rate must be finite, got {learning_rate!r}")
    lr = float(learning_rate)
    if not (lr > 0 and lr < 10):
        _fail(f"learning_rate must be positive finite <10, got {learning_rate!r}")

    batch_size = model.get("batch_size")
    if not _is_positive_int_no_bool(batch_size):
        _fail(f"batch_size must be positive int, got {batch_size!r}")

    seed = model.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        _fail(f"seed must be int, got {seed!r}")

    fixed_epochs = model.get("fixed_epochs")
    if not _is_positive_int_no_bool(fixed_epochs):
        _fail(f"fixed_epochs must be positive int, got {fixed_epochs!r}")

    cond = mapping.get("condition_preprocessing")
    if not isinstance(cond, dict):
        _fail("condition_preprocessing must be mapping")
    selected_variant = cond.get("variant")
    if selected_variant not in {"A", "B", "C", "D"}:
        _fail(f"condition_preprocessing.variant must be one of A,B,C,D, got {selected_variant!r}")

    clustering = cond.get("clustering")
    if not isinstance(clustering, dict):
        _fail("condition_preprocessing.clustering must be mapping")
    clustering_method = clustering.get("method")
    if not isinstance(clustering_method, str) or clustering_method.lower() != "kmeans":
        _fail(f"clustering.method must be 'kmeans' (case-insensitive), got {clustering_method!r}")
    clustering_method_norm = "kmeans"
    n_clusters = clustering.get("n_clusters")
    if not _is_positive_int_no_bool(n_clusters):
        _fail(f"clustering.n_clusters must be positive int, got {n_clusters!r}")
    clustering_random_state = clustering.get("random_state")
    if not isinstance(clustering_random_state, int) or isinstance(clustering_random_state, bool):
        _fail(f"clustering.random_state must be int, got {clustering_random_state!r}")
    clustering_n_init = clustering.get("n_init")
    if not _is_positive_int_no_bool(clustering_n_init):
        _fail(f"clustering.n_init must be positive int, got {clustering_n_init!r}")

    osc = cond.get("operating_setting_columns")
    if not isinstance(osc, (list, tuple)):
        _fail(f"operating_setting_columns must be list, got {osc!r}")
    expected_osc = ["setting_1", "setting_2", "setting_3"]
    if list(osc) != expected_osc:
        _fail(f"operating_setting_columns must be {expected_osc}, got {list(osc)!r}")

    sensor_scaling = cond.get("sensor_scaling")
    if not isinstance(sensor_scaling, dict):
        _fail("sensor_scaling must be mapping")
    mode = sensor_scaling.get("mode")
    if mode != "per_regime_standard_scaler":
        _fail(f"sensor_scaling.mode must be 'per_regime_standard_scaler', got {mode!r}")
    fit_scope = sensor_scaling.get("fit_scope")
    if fit_scope != "training_rows_only":
        _fail(f"sensor_scaling.fit_scope must be 'training_rows_only', got {fit_scope!r}")

    splits = mapping.get("splits")
    if not isinstance(splits, dict):
        _fail("splits must be mapping")
    split_provenance = splits.get("provenance")
    if not isinstance(split_provenance, str) or not split_provenance:
        _fail("splits.provenance must be non-empty str")
    validation_manifest = splits.get("validation_manifest")
    if not isinstance(validation_manifest, str) or not validation_manifest:
        _fail("splits.validation_manifest must be non-empty str")

    dev_count = splits.get("development_engine_count")
    val_count = splits.get("validation_engine_count")
    cal_count = splits.get("calibration_engine_count")
    for name, v in [("development_engine_count", dev_count), ("validation_engine_count", val_count), ("calibration_engine_count", cal_count)]:
        if not _is_positive_int_no_bool(v):
            _fail(f"splits.{name} must be positive int, got {v!r}")

    dev_sha = splits.get("development_engine_ids_sha256")
    val_sha = splits.get("validation_engine_ids_sha256")
    cal_sha = splits.get("calibration_engine_ids_sha256")
    for name, s in [("development_engine_ids_sha256", dev_sha), ("validation_engine_ids_sha256", val_sha), ("calibration_engine_ids_sha256", cal_sha)]:
        _check_hex_sha256(s, f"splits.{name}")

    training = mapping.get("training")
    if not isinstance(training, dict):
        _fail("training must be mapping")
    if "validation_data_in_final_fit" not in training:
        _fail("training.validation_data_in_final_fit missing")
    validation_data_in_final_fit = training.get("validation_data_in_final_fit")
    if not isinstance(validation_data_in_final_fit, bool):
        _fail(f"training.validation_data_in_final_fit must be bool, got {validation_data_in_final_fit!r}")
    if validation_data_in_final_fit is True:
        _fail("training.validation_data_in_final_fit must be false (validated config cannot allow validation data in final fit)")

    final_engine_count = training.get("final_engine_count")
    reserved_engine_count = training.get("reserved_engine_count")
    # these may be missing in very old mappings; derive defaults if needed? But require explicit for P2
    if final_engine_count is None:
        # derive as dev + val for validation, but require explicit in authoritative config
        _fail("training.final_engine_count missing")
    if not _is_positive_int_no_bool(final_engine_count):
        _fail(f"training.final_engine_count must be positive int, got {final_engine_count!r}")
    if reserved_engine_count is None:
        _fail("training.reserved_engine_count missing")
    if not _is_positive_int_no_bool(reserved_engine_count):
        _fail(f"training.reserved_engine_count must be positive int, got {reserved_engine_count!r}")

    # integrity: final = dev + val, reserved = cal
    if int(final_engine_count) != int(dev_count) + int(val_count):
        _fail(f"training.final_engine_count {final_engine_count} must equal dev {dev_count}+val {val_count}")
    if int(reserved_engine_count) != int(cal_count):
        _fail(f"training.reserved_engine_count {reserved_engine_count} must equal calibration {cal_count}")

    # candidate identity validation
    expected_candidate = f"gru_w{int(window)}_{loss_norm}_cond{selected_variant}"
    # candidate_name case-sensitive check but allow lower architecture part
    if candidate_name != expected_candidate:
        _fail(f"candidate_name {candidate_name!r} inconsistent with identity: expected {expected_candidate!r} from architecture/window/loss/variant")
    # architecture already validated gru
    # also ensure variant matches candidate suffix
    # fixed_epochs consistency vs variant_results if present
    variant_results = mapping.get("variant_results")
    if isinstance(variant_results, dict) and selected_variant in variant_results:
        vr = variant_results[selected_variant]
        if isinstance(vr, dict) and "best_epoch" in vr:
            be = vr["best_epoch"]
            if isinstance(be, int) and not isinstance(be, bool) and be != int(fixed_epochs):
                _fail(f"fixed_epochs {fixed_epochs} inconsistent with variant_results[{selected_variant}].best_epoch {be}")

    osc_tuple = tuple(str(x) for x in osc)

    cfg = FD004FinalConfig(
        methodology_version=str(methodology_version),
        dataset=str(dataset),
        candidate_name=str(candidate_name),
        architecture=str(architecture).lower(),
        window=int(window),
        units=(int(units[0]), int(units[1])),
        dropout=float(dropout),
        loss=loss_norm,
        optimizer_name=str(optimizer_name),
        optimizer_clipnorm=float(optimizer_clipnorm),
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        seed=int(seed),
        fixed_epochs=int(fixed_epochs),
        selected_variant=str(selected_variant),
        clustering_method=clustering_method_norm,
        n_clusters=int(n_clusters),
        clustering_random_state=int(clustering_random_state),
        clustering_n_init=int(clustering_n_init),
        operating_setting_columns=osc_tuple,
        sensor_scaling_mode=str(mode),
        sensor_scaling_fit_scope=str(fit_scope),
        split_provenance=str(split_provenance),
        validation_manifest=str(validation_manifest),
        development_engine_count=int(dev_count),
        validation_engine_count=int(val_count),
        calibration_engine_count=int(cal_count),
        final_engine_count=int(final_engine_count),
        reserved_engine_count=int(reserved_engine_count),
        development_engine_ids_sha256=str(dev_sha),
        validation_engine_ids_sha256=str(val_sha),
        calibration_engine_ids_sha256=str(cal_sha),
        validation_data_in_final_fit=bool(validation_data_in_final_fit),
        config_file_sha256=config_file_sha256,
        config_canonical_sha256=None,  # filled below
        source_path=Path(source_path) if source_path else None,
    )
    # compute canonical hash after construction (avoid frozen recursion)
    canonical = canonical_sha256_for_config(cfg)
    # need to return new instance with canonical filled (frozen so use object.__setattr__)
    object.__setattr__(cfg, "config_canonical_sha256", canonical)
    return cfg


def load_fd004_final_config(path: str | Path, *, root: Path | str | None = None) -> FD004FinalConfig:
    p = Path(path)
    if not p.is_absolute():
        # resolve relative to repo root for consistent behavior
        r = _resolve_repo_root(root)
        # if path is already repo-relative, try relative to root; fallback to cwd
        candidate = r / p
        if candidate.exists():
            p = candidate
        else:
            # also try cwd resolution
            p = p.resolve()
    if not p.exists():
        _fail(f"FD004 config file not found: {p}")
    file_sha = sha256_file(p)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg = from_mapping(raw, source_path=p, config_file_sha256=file_sha)
    return cfg


# helpers for file-integrity checks (used by freeze/posthoc sidecar)
def verify_condition_joblib_hash(path: Path | str, expected: str = HISTORICAL_CONDITION_JOBLIB_SHA256) -> None:
    p = Path(path)
    if not p.exists():
        _fail(f"condition joblib not found: {p}")
    actual = sha256_file(p)
    if actual.lower() != expected.lower():
        _fail(f"condition joblib hash mismatch: expected {expected}, got {actual} for {p}")

def artifact_paths_from_config(config: FD004FinalConfig, root: Path | str | None = None) -> dict:
    return {
        "model": config.model_artifact_path(root),
        "condition": config.condition_artifact_path(root),
        "metadata": config.resolve_metadata_path(root),
        "split": config.resolve_split_path(root),
        "validation_manifest": config.resolve_validation_manifest_path(root),
    }
