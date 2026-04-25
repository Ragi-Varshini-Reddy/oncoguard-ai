"""Compatibility alias for the current patient-query module."""

from __future__ import annotations

import sys

from backend.ml.explainability import patient_query as _patient_query

sys.modules[__name__] = _patient_query

