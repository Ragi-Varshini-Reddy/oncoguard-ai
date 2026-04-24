"""Feature selection helpers for the genomics module."""

from __future__ import annotations

from typing import Any


def selected_feature_order(config: dict[str, Any]) -> list[str]:
    """Return the configured panel order used by preprocessing and inference."""

    return list(config["genomics"]["selected_gene_panel"])
