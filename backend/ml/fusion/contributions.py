"""Modality contribution scoring."""

from __future__ import annotations

from backend.ml.common.contracts import ALLOWED_MODALITIES, ModuleOutput
from backend.ml.common.utils import normalize_scores


def compute_modality_contributions(
    module_outputs: list[ModuleOutput],
    modality_mask: dict[str, bool],
) -> dict[str, float]:
    by_modality = {output.modality: output for output in module_outputs}
    raw_scores: dict[str, float] = {}
    for modality in ALLOWED_MODALITIES:
        output = by_modality.get(modality)
        if not modality_mask.get(modality, False) or output is None or output.status != "available":
            raw_scores[modality] = 0.0
            continue
        confidence = output.confidence if output.confidence is not None else 0.5
        embedding = output.embedding or []
        avg_abs = sum(abs(value) for value in embedding[:32]) / max(1, min(len(embedding), 32))
        raw_scores[modality] = confidence * (0.75 + min(avg_abs, 0.35))

    normalized = normalize_scores(raw_scores)
    return {key: round(normalized[key], 4) for key in ALLOWED_MODALITIES}
