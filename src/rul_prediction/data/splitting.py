"""Deterministic, leakage-safe train/validation split by engine ID.

Split is performed on engine IDs (never on generated windows), with a
reproducible seed. The same engine can never appear in both partitions.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SEED = 42
VAL_FRACTION = 0.2

DEFAULT_SPLITS_DIR = Path("experiments/splits")


def split_engine_ids(
    engine_ids, seed: int = SEED, val_fraction: float = VAL_FRACTION
) -> tuple[set[int], set[int]]:
    """Return (train_engine_ids, validation_engine_ids) as disjoint sets."""
    ids = sorted(int(e) for e in engine_ids)
    rng = random.Random(seed)  # deterministic shuffle independent of global state
    rng.shuffle(ids)

    n_val = int(round(len(ids) * val_fraction))
    validation = set(ids[:n_val])
    train = set(ids[n_val:])
    assert train.isdisjoint(validation), "engine overlap between train and validation"
    return train, validation


def write_split_file(
    engine_ids,
    dataset: str,
    out_dir: str | Path = DEFAULT_SPLITS_DIR,
    seed: int = SEED,
    val_fraction: float = VAL_FRACTION,
) -> Path:
    """Persist the split as JSON and return the file path."""
    train, validation = split_engine_ids(engine_ids, seed, val_fraction)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dataset}_seed{seed}.json"
    payload = {
        "dataset": dataset,
        "seed": seed,
        "val_fraction": val_fraction,
        "n_train": len(train),
        "n_validation": len(validation),
        "train_engine_ids": sorted(train),
        "validation_engine_ids": sorted(validation),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_split_file(path: str | Path) -> tuple[set[int], set[int]]:
    """Read a previously written split JSON -> (train_engine_ids, validation_engine_ids)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    train = set(payload["train_engine_ids"])
    validation = set(payload["validation_engine_ids"])
    assert train.isdisjoint(validation)
    return train, validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Engine-level train/validation split")
    parser.add_argument("--dataset", default="FD001")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--out-dir", default=str(DEFAULT_SPLITS_DIR))
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()

    from .loader import load_train  # local import to avoid circular load at import time

    engine_ids = load_train(args.dataset, args.data_dir)["engine_id"].unique()
    path = write_split_file(engine_ids, args.dataset, args.out_dir, args.seed, args.val_fraction)
    payload = json.loads(path.read_text(encoding="utf-8"))
    train_ids = set(payload["train_engine_ids"])
    validation_ids = set(payload["validation_engine_ids"])
    assert train_ids.isdisjoint(validation_ids)

    print(f"Split file: {path}")
    print(f"Train engines: {len(train_ids)}")
    print(f"Validation engines: {len(validation_ids)}")
    print(f"Overlap: {len(train_ids & validation_ids)}")
    print(f"Random seed: {args.seed}")
    print("Train engine IDs:", sorted(train_ids))


if __name__ == "__main__":
    main()