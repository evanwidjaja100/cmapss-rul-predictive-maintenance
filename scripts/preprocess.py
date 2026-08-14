"""CLI: validate C-MAPSS raw data and print integrity + summary reports.

Usage:
    .venv/Scripts/python.exe scripts/preprocess.py --dataset FD001 --validate-only
"""

from __future__ import annotations

import argparse

from rul_prediction.data.loader import (
    data_summary_lines,
    load_rul,
    load_test,
    load_train,
    summarize,
)
from rul_prediction.data.validation import validate_frame, validate_rul


def main() -> None:
    parser = argparse.ArgumentParser(description="C-MAPSS preprocessing/validation CLI")
    parser.add_argument("--dataset", default="FD001", help="e.g. FD001 (default: FD001)")
    parser.add_argument("--data-dir", default="data/raw", help="raw data directory")
    parser.add_argument("--validate-only", action="store_true", help="validate and summarize, write nothing")
    args = parser.parse_args()

    train = load_train(args.dataset, args.data_dir)
    test = load_test(args.dataset, args.data_dir)
    rul = load_rul(args.dataset, args.data_dir)

    reports = [validate_frame(train, args.dataset, "train"), validate_frame(test, args.dataset, "test")]
    reports.append(validate_rul(rul, args.dataset))

    for report in reports:
        print("\n".join(report.lines()))
        print()

    for summary in (summarize(train, args.dataset, "train"), summarize(test, args.dataset, "test")):
        print("Summary:")
        print("\n".join(f"  {line}" for line in data_summary_lines(summary)))
        print()

    if not all(r.passed for r in reports):
        raise SystemExit("Validation FAILED — see report above.")
    if args.validate_only:
        print("Validation PASSED (--validate-only: no writes performed).")


if __name__ == "__main__":
    main()