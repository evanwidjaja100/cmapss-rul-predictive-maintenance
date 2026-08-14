"""Data-integrity validation for C-MAPSS frames. Validation only — never mutates/removes columns."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .loader import DATA_COLUMNS, EXPECTED_ENGINE_COUNTS, sensor_columns


@dataclass
class ValidationReport:
    """Container of per-check pass/fail results plus diagnostic summaries."""

    dataset: str = ""
    kind: str = ""
    checks: dict[str, bool] = field(default_factory=dict)
    diagnostics: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def lines(self) -> list[str]:
        out = [f"Validation report - {self.kind} {self.dataset}"]
        for key, ok in self.checks.items():
            out.append(f"  [{'OK' if ok else 'FAIL'}] {key}")
        for key, val in self.diagnostics.items():
            out.append(f"  {key}: {val}")
        out.append(f"  OVERALL: {'PASS' if self.passed else 'FAIL'}")
        return out


def _check(name: str, report: ValidationReport, condition: bool, **diag) -> None:
    report.checks[name] = bool(condition)
    report.diagnostics.update({k: v for k, v in diag.items()})


def validate_rul(rul: np.ndarray, dataset: str, kind: str = "test") -> ValidationReport:
    report = ValidationReport(dataset=dataset, kind=kind)
    expected = EXPECTED_ENGINE_COUNTS[dataset]["test"]
    _check(
        "RUL file length matches number of test engines",
        report,
        len(rul) == expected,
        rul_length=len(rul),
        expected=expected,
    )
    _check("RUL values are positive integers", report, np.issubdtype(rul.dtype, np.integer) and (rul > 0).all())
    return report


def validate_frame(
    frame: pd.DataFrame, dataset: str, kind: str  # kind in {"train", "test"}
) -> ValidationReport:
    report = ValidationReport(dataset=dataset, kind=kind)

    _check("expected number of columns", report, frame.shape[1] == len(DATA_COLUMNS),
           shape=frame.shape)

    _check("all columns numeric", report,
           all(pd.api.types.is_numeric_dtype(frame[c]) for c in frame.columns))

    numeric = frame.select_dtypes("number")

    _check("no missing values", report, int(frame.isna().sum().sum()) == 0,
           missing_total=int(frame.isna().sum().sum()))

    _check("no infinite values", report,
           int(np.isinf(numeric).sum().sum()) == 0,
           inf_total=int(np.isinf(numeric).sum().sum()))

    # Constant columns are a known C-MAPSS property and are *reported* here
    # (removal is an experimental decision, Phase 8). All-NaN columns fail.
    numeric = frame.select_dtypes("number")
    constant_cols = [c for c in numeric.columns if numeric[c].nunique() <= 1]
    report.diagnostics["constant_columns"] = constant_cols
    _check("no fully empty columns", report,
           int(numeric.isna().all().sum()) == 0)

    dupes = int(frame.duplicated(subset=["engine_id", "cycle"]).sum())
    _check("no duplicate (engine_id, cycle) records", report, dupes == 0, duplicates=dupes)

    cycles_ordered = frame.groupby("engine_id")["cycle"].apply(
        lambda c: bool(np.all(np.diff(c) >= 1))  # strictly increasing by >=1
    ).all()
    _check("cycles strictly increasing within every engine", report, bool(cycles_ordered))

    expected = EXPECTED_ENGINE_COUNTS[dataset][kind]
    n_engines = int(frame["engine_id"].nunique())
    _check("engine count matches dataset cardinality", report, n_engines == expected,
           engines=n_engines, expected=expected)

    _check("engine_id starts at 1", report, int(frame["engine_id"].min()) == 1)
    _check("engine_id values are contiguous integers", report,
           int(frame["engine_id"].nunique()) == int(frame["engine_id"].max()))

    return report