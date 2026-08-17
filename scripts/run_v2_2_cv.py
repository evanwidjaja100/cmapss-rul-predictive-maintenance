"""Methodology V2.2: nested engine-group CV (FD001) — 8 candidates x 5 folds.

Stage-1 (control) uses inner-fit / inner-stop splits (seeds 4201..4205);
Stage-2 retrains fixed-duration on all 68 outer-training engines and evaluates
the 17 untouched outer-evaluation engines. The summary is ONLY generated after
the hard completeness gate (every candidate has folds 1..5, 40 rows total).

Usage:
    python scripts/run_v2_2_cv.py                      # all 8 candidates x 5 folds
    python scripts/run_v2_2_cv.py --candidates gru_w45_huber --folds 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from rul_prediction.benchmark.v2_2 import (
    CV_CANDIDATES,
    assert_cv_complete,
    cv_summary,
    load_v2_2_cv_artifacts,
    run_cv_fold,
)

OUT_DIR = Path("experiments/v2_2")


def main() -> None:
    parser = argparse.ArgumentParser(description="V2.2 nested CV (FD001)")
    parser.add_argument("--candidates", nargs="*", default=None)
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--splits-dir", default="experiments/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    candidates = [c for c in CV_CANDIDATES
                  if args.candidates is None or c["id"] in args.candidates]
    if not candidates:
        sys.exit("no candidates matched")
    artifacts = load_v2_2_cv_artifacts("FD001", args.data_dir, args.splits_dir)
    folds = [f for f in artifacts["folds"] if args.folds is None or f["fold"] in args.folds]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for candidate in candidates:
        for fold in folds:
            done_keys = set()
            path = OUT_DIR / "fd001_outer_fold_results.csv"
            if path.exists():
                done = pd.read_csv(path)
                done_keys = {(r.candidate_id, int(r.fold)) for r in done.itertuples(index=False)}
            if (candidate["id"], fold["fold"]) in done_keys:
                print(f"skip {candidate['id']} fold {fold['fold']} (already done)")
                continue
            print(f"running {candidate['id']} fold {fold['fold']} "
                  f"(68 outer-train / 17 outer-eval; inner 58/10 seed {4200 + fold['fold']})")
            result = run_cv_fold(candidate, fold, artifacts["frame"], artifacts["cal_ids"],
                                 args.data_dir, args.splits_dir, seed=args.seed)
            pd.DataFrame([result["row"]]).to_csv(
                path, mode="a", header=not path.exists(), index=False)
            eng_path = OUT_DIR / "fd001_outer_engine_level.csv"
            pd.DataFrame(result["engine_rows"]).to_csv(
                eng_path, mode="a", header=not eng_path.exists(), index=False)
            pred_path = OUT_DIR / "fd001_outer_predictions.csv"
            pd.DataFrame(result["prediction_rows"]).to_csv(
                pred_path, mode="a", header=not pred_path.exists(), index=False)
            meta_path = OUT_DIR / "fd001_outer_metadata.jsonl"
            with meta_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(result["metadata"]) + "\n")
            print(f"  done fold {fold['fold']}: RMSE={result['row']['RMSE']} "
                  f"NASA={result['row']['NASA_total']} "
                  f"best_epoch={result['control'].get('best_epoch')} "
                  f"best_iteration={result['control'].get('best_iteration')} "
                  f"({result['row']['training_time']}s)")

    fold_rows = pd.read_csv(OUT_DIR / "fd001_outer_fold_results.csv").to_dict("records")
    n = assert_cv_complete(fold_rows, candidates)
    if args.candidates is None and args.folds is None:
        assert n == 40, "full V2.2 matrix must be exactly 8 candidates x 5 folds = 40"
    print(f"completeness gate passed: {n} candidate-fold rows")
    best_epochs = pd.DataFrame([
        {"candidate_id": r["candidate_id"], "fold": r["fold"],
         "inner_seed": r["inner_seed"], "best_epoch": r["best_epoch"],
         "best_iteration": r["best_iteration"]}
        for r in fold_rows])
    best_epochs.to_csv(OUT_DIR / "fd001_best_epochs.csv", index=False)
    summary = pd.DataFrame(cv_summary(fold_rows))
    summary.to_csv(OUT_DIR / "fd001_cv_summary.csv", index=False)
    print("\nV2.2 CV summary (mean +/- std over outer folds):")
    print(summary.sort_values("NASA_mean_per_engine_mean").to_string(index=False))


if __name__ == "__main__":
    main()