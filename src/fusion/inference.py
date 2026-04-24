"""Fusion inference entry points."""

from __future__ import annotations

from typing import Any

from src.common.config import load_config
from src.common.contracts import ALLOWED_MODALITIES, FusionInput, FusionOutput, ModuleOutput
from src.common.utils import clamp
from src.fusion.contributions import compute_modality_contributions
from src.fusion.model import FusionScorer


def build_modality_mask(module_outputs: list[ModuleOutput]) -> dict[str, bool]:
    by_modality = {output.modality: output for output in module_outputs}
    return {
        modality: bool(by_modality.get(modality) and by_modality[modality].status == "available")
        for modality in ALLOWED_MODALITIES
    }


def run_fusion(
    patient_id: str,
    module_outputs: list[ModuleOutput | dict[str, Any]],
    modality_mask: dict[str, bool] | None = None,
    config: dict[str, Any] | None = None,
) -> FusionOutput:
    cfg = config or load_config()
    parsed_outputs = [
        output if isinstance(output, ModuleOutput) else ModuleOutput.model_validate(output)
        for output in module_outputs
    ]
    mask = modality_mask or build_modality_mask(parsed_outputs)
    fusion_input = FusionInput(patient_id=patient_id, module_outputs=parsed_outputs, modality_mask=mask)

    contributions = compute_modality_contributions(fusion_input.module_outputs, fusion_input.modality_mask)
    scorer = FusionScorer()
    prediction = scorer.predict(fusion_input.module_outputs, contributions)
    warnings = _collect_warnings(fusion_input.module_outputs, fusion_input.modality_mask)

    available_confidences = [
        output.confidence
        for output in fusion_input.module_outputs
        if output.status == "available" and output.confidence is not None
    ]
    coverage = sum(1 for value in fusion_input.modality_mask.values() if value) / len(ALLOWED_MODALITIES)
    mean_confidence = sum(available_confidences) / max(1, len(available_confidences))
    confidence = clamp(mean_confidence * (0.72 + coverage * 0.28))
    if confidence < float(cfg["fusion"].get("low_confidence_threshold", 0.6)):
        warnings.append("Fusion confidence is low because limited or uncertain modalities were available")

    return FusionOutput(
        patient_id=patient_id,
        diagnosis=prediction["diagnosis"],
        risk=prediction["risk"],
        confidence=round(confidence, 4),
        modality_contributions=contributions,
        warnings=warnings,
    )


def _collect_warnings(module_outputs: list[ModuleOutput], modality_mask: dict[str, bool]) -> list[str]:
    by_modality = {output.modality: output for output in module_outputs}
    warnings: list[str] = []
    for modality in ALLOWED_MODALITIES:
        output = by_modality.get(modality)
        if not modality_mask.get(modality, False):
            warnings.append(f"{modality.capitalize()} modality missing")
        if output:
            warnings.extend(output.warnings)
    deduped: list[str] = []
    for warning in warnings:
        if warning not in deduped:
            deduped.append(warning)
    return deduped
