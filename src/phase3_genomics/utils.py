"""Genomics input utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, TextIO


def load_genomic_features_from_csv(
    source: str | Path | TextIO,
    patient_id: str | None = None,
) -> tuple[str, str | None, dict[str, Any]]:
    """Load one patient's model-ready genomic features from a CSV file or text stream."""

    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() == ".parquet":
            return _load_parquet(path, patient_id)
        handle = path.open("r", encoding="utf-8", newline="")
        should_close = True
    else:
        handle = source
        should_close = False

    try:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("genomic CSV must include a header row")
        if "patient_id" not in reader.fieldnames:
            raise ValueError("genomic CSV must include a patient_id column")

        first_row: dict[str, Any] | None = None
        for row in reader:
            if first_row is None:
                first_row = row
            if patient_id is None or row.get("patient_id") == patient_id:
                return _row_to_features(row)

        if patient_id is not None:
            raise ValueError(f"patient_id {patient_id!r} not found in genomic CSV")
        if first_row is None:
            raise ValueError("genomic CSV contains no rows")
        return _row_to_features(first_row)
    finally:
        if should_close:
            handle.close()


def _row_to_features(row: dict[str, Any]) -> tuple[str, str | None, dict[str, Any]]:
    patient_id = str(row.get("patient_id") or "").strip()
    if not patient_id:
        raise ValueError("genomic row is missing patient_id")
    sample_id = str(row.get("sample_id") or "").strip() or None
    features = {
        key: value
        for key, value in row.items()
        if key not in {"patient_id", "sample_id"} and value is not None
    }
    return patient_id, sample_id, features


def _load_parquet(path: Path, patient_id: str | None) -> tuple[str, str | None, dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Parquet input requires pandas/pyarrow; use CSV for the lightweight demo") from exc

    frame = pd.read_parquet(path)
    if "patient_id" not in frame.columns:
        raise ValueError("genomic parquet must include a patient_id column")
    row = frame.iloc[0] if patient_id is None else frame.loc[frame["patient_id"] == patient_id].iloc[0]
    return _row_to_features(row.to_dict())
