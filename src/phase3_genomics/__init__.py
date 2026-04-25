"""Compatibility aliases for genomics artifacts pickled under src.phase3_genomics."""

from __future__ import annotations

import importlib
import sys

_MODULES = (
    "artifact_model",
    "explain",
    "feature_selection",
    "inference",
    "model",
    "preprocess",
    "schema",
    "train",
    "utils",
)

for _name in _MODULES:
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"backend.ml.phase3_genomics.{_name}")

