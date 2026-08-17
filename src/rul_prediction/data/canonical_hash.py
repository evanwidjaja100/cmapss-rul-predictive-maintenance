"""Platform-independent canonical hashing (Methodology V2.2, issue V2.2-7).

Raw-file hashes of JSON/text manifests are not reproducible across platforms:
LF vs CRLF line endings, JSON whitespace and key ordering all change the byte
stream. V2.2 hashes SEMANTIC payloads instead:

- JSON payloads: json.dumps(sort_keys=True, separators=(",", ":"),
  ensure_ascii=False).encode("utf-8") -> sha256.
- CSV manifests: the dataframe is sorted into canonical order, NaN/null
  normalized, serialized to a stable CSV string with '\n' line endings
  (never platform newline), then hashed. Equivalent to hashing the semantic
  rows in deterministic JSON form.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import pandas as pd


def canonical_sha256_json(payload: object) -> str:
    """sha256 of a canonical JSON serialization (sorted keys, no whitespace)."""
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def canonical_sha256_csv(frame: pd.DataFrame) -> str:
    """sha256 of a dataframe's semantic rows in deterministic form.

    Rows are sorted by every column value (stable, string-aware), floats are
    normalized so 1.0 == 1, nulls become the empty string, and the serialized
    rows use '\n' line endings only.
    """
    df = frame.copy()
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].map(lambda v: "" if pd.isna(v) else f"{float(v):g}")
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].map(lambda v: "" if pd.isna(v) else int(v))
        else:
            df[col] = df[col].map(lambda v: "" if pd.isna(v) else str(v))
    df = df.sort_values(list(df.columns)).reset_index(drop=True)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(list(df.columns))
    writer.writerows(df.astype(str).itertuples(index=False, name=None))
    return hashlib.sha256(buffer.getvalue().encode("utf-8")).hexdigest()


def canonical_sha256_file(path: str | Path) -> str:
    """Canonical hash of a manifest file on disk by its semantic content.

    JSON -> canonical_sha256_json of the parsed payload.
    CSV  -> canonical_sha256_csv of the parsed dataframe.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        return canonical_sha256_json(json.loads(path.read_text(encoding="utf-8")))
    if path.suffix.lower() == ".csv":
        return canonical_sha256_csv(pd.read_csv(path))
    raise ValueError(f"unsupported manifest type: {path.suffix}")
