"""Methodology V2.1 bounded model-selection CV (FD001, 5 engine-group folds).

Runs each candidate in CV_CANDIDATES over the requested folds (default all 8
candidates x 5 folds) and appends results idempotently to:

    experiments/v2_1/fd001_cv_fold_results.csv
    experiments/v2_1/fd001_cv_engine_level.csv
    experiments/v2_1/fd001_cv_predictions.csv
    experiments/v2_1/fd001_cv_summary.csv     (mean +/- std over folds)

Usage:
    python scripts/run_v2_1_cv.py                          # all candidates, all folds
    python scripts/run_v2_1_cv.py --candidates gru_w45_huber lstm_w45_huber
    python scripts/run_v2_1_cv.py --folds 1 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from rul_prediction.benchmark.v2_1 import (
    CV_CANDIDATES,
    cv_summary,
    load_v2_1_cv_artifacts,
    run_cv_fold,
)

OUT_DIR = Path("experiments/v2_1")


def main() -> None:
    parser = argparse.ArgumentParser(description="V2.1 engine-group CV model selection")
    parser.add_argument("--candidates", nargs="*", default=None,
                        help="candidate ids (default: all)")
    parser.add_argument("--folds", nargs="*", type=int, default=None,
                        help="fold numbers (default: all 5)")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--splits-dir", default="experiments/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    candidates = [c for c in CV_CANDIDATES
                  if args.candidates is None or c["id"] in args.candidates]
    if not candidates:
        sys.exit("no candidates matched")

    artifacts = load_v2_1_cv_artifacts("FD001", args.data_dir, args.splits_dir)
    folds = [f for f in artifacts["folds"] if args.folds is None or f["fold"] in args.folds]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for candidate in candidates:
        for fold in folds:
            done_keys = set()
            path = OUT_DIR / "fd001_cv_fold_results.csv"
            if path.exists():
                done = pd.read_csv(path)
                done_keys = {(r.candidate_id, int(r.fold))
                             for r in done.itertuples(index=False)}
            if (candidate["id"], fold["fold"]) in done_keys:
                print(f"skip {candidate['id']} fold {fold['fold']} (already done)")
                continue
            print(f"running {candidate['id']} fold {fold['fold']} "
                  f"({len(fold['training'])} train / {len(fold['validation'])} val engines)")
            result = run_cv_fold(candidate, fold, artifacts["frame"],
                                 args.data_dir, args.splits_dir, seed=args.seed)
            pd.DataFrame([result["row"]]).to_csv(
                OUT_DIR / "fd001_cv_fold_results.csv", mode="a", header=not path.exists(),
                index=False)
            pd.DataFrame(result["engine_rows"]).to_csv(
                OUT_DIR / "fd001_cv_engine_level.csv", mode="a",
                header=not (OUT_DIR / "fd001_cv_engine_level.csv").exists(), index=False)
            pred_path = OUT_DIR / "fd001_cv_predictions.csv"
            pd.DataFrame(result["prediction_rows"]).to_csv(
                pred_path, mode="a", header=not pred_path.exists(), index=False)
            print(f"  done fold {fold['fold']}: RMSE={result['row']['RMSE']} "
                  f"NASA={result['row']['NASA_total']} ({result['row']['training_time']}s)")

    fold_rows = pd.read_csv(OUT_DIR / "fd001_cv_fold_results.csv").to_dict("records")
    summary = pd.DataFrame(cv_summary(fold_rows))
    summary.to_csv(OUT_DIR / "fd001_cv_summary.csv", index=False)
    print("\nCV summary (mean +/- std over folds):")
    print(summary.sort_values("RMSE_mean").to_string(index=False))


if __name__ == "__main__":
    main()