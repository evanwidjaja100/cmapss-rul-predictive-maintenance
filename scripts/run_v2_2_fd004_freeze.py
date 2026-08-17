"""Methodology V2.2: freeze the FD004 condition-aware model (YAML-driven).

Reads configs/final_model_v2_2_fd004.yaml (source of truth). The 37 validation
engines have already served their variant-selection role; the final model is
retrained on 212 engines (175 training + 37 validation) with preprocessing fit
on those 212 rows only. The 37 reserved/calibration engines stay untouched.
Final epoch count comes from the development-only inner-fit/inner-stop control.
No official FD004 labels are read here (post-hoc evaluation is separate).

Artifacts:
    models/v2_2/fd004_gru_w45_huber_cond<V>.keras
    models/v2_2/fd004_condition<V>.joblib
    experiments/v2_2/fd004_final_fit_metadata.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import dump as dump_joblib

from rul_prediction.benchmark.v2 import ROOT
from rul_prediction.data.condition import SETTING_COLUMNS, condition_feature_matrix
from rul_prediction.data.loader import load_train
from rul_prediction.data.v2_preprocessing import add_raw_rul, build_v2_train_sequences
from rul_prediction.models.v2_models import v2_gru
from rul_prediction.training.trainer import set_seed

from run_v2_2_fd004 import build_matrix, fit_preprocessing

WINDOW = 45


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze V2.2 FD004 final model")
    parser.add_argument("--config", default="configs/final_model_v2_2_fd004.yaml")
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    assert cfg["methodology_version"] == "2.2" and cfg["dataset"] == "FD004"
    variant = cfg["preprocessing"]["variant"]
    best_epoch = int(cfg["training_control"]["best_epoch"])
    assert cfg["training_control"]["validation_data_in_final_fit"] is False

    frame = load_train("FD004", args.data_dir)
    split = json.loads((Path("experiments/splits") / "fd004_v2_seed42.json").read_text(encoding="utf-8"))
    train_ids = set(split["train_engine_ids"])
    val_ids = set(split["validation_engine_ids"])
    cal_ids = set(split["calibration_engine_ids"])
    final_ids = train_ids | val_ids          # 212 engines (validation already served selection)
    assert len(final_ids) == 212 and cal_ids.isdisjoint(final_ids)

    pre = fit_preprocessing(variant, frame, final_ids)
    rows = frame[frame["engine_id"].isin(final_ids)].sort_values(["engine_id", "cycle"])
    X = build_matrix(variant, rows, pre["kmeans"], pre["cluster_scalers"],
                     pre["settings_scaler"], pre["global_scaler"])
    rul = add_raw_rul(rows)["rul"].to_numpy(dtype=np.float32)
    X_seq, y_seq, _, _, masks = build_v2_train_sequences(
        X, rows["engine_id"].to_numpy(), rul, WINDOW)

    start = time.perf_counter()
    set_seed(cfg["model"]["seed"])
    model = v2_gru(WINDOW, X.shape[1], loss="huber", seed=cfg["model"]["seed"])
    model.fit([X_seq, masks], y_seq, batch_size=cfg["model"]["batch_size"],
              epochs=best_epoch, verbose=0)
    training_time = round(time.perf_counter() - start, 2)

    out_dir = ROOT / "models" / "v2_2"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / f"fd004_gru_w45_huber_cond{variant}.keras")
    dump_joblib({"kmeans": pre["kmeans"], "cluster_scalers": pre["cluster_scalers"],
                 "settings_scaler": pre["settings_scaler"]},
                out_dir / f"fd004_condition{variant}.joblib")
    meta = {
        "methodology_version": "2.2",
        "dataset": "FD004",
        "variant": variant,
        "best_epoch": best_epoch,
        "training_engine_count": len(final_ids),
        "train_ids": sorted(final_ids),
        "reserved_calibration_engine_count": len(cal_ids),
        "reserved_calibration_engine_ids": sorted(cal_ids),
        "official_labels_used_in_fitting": False,
        "training_time": training_time,
    }
    Path("experiments/v2_2/fd004_final_fit_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()