"""Fusion inference entry points."""

from __future__ import annotations

from typing import Any

from backend.ml.common.config import load_config
from backend.ml.common.contracts import ALLOWED_MODALITIES, FusionInput, FusionOutput, ModuleOutput
from backend.ml.common.utils import clamp
from backend.ml.fusion.heads import PredictionHeads
from backend.ml.fusion.model import EvidenceWeightedFusion, FusionEvidence


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
    disabled_modalities: list[str] | None = None,
) -> FusionOutput:
    return _run_fusion(
        patient_id=patient_id,
        module_outputs=module_outputs,
        modality_mask=modality_mask,
        config=config,
        disabled_modalities=disabled_modalities,
        include_what_if=True,
    )


def explain_fusion(
    patient_id: str,
    module_outputs: list[ModuleOutput | dict[str, Any]],
    modality_mask: dict[str, bool] | None = None,
    config: dict[str, Any] | None = None,
    disabled_modalities: list[str] | None = None,
) -> FusionOutput:
    """Explicit rich-XAI fusion entry point used by API/UI."""

    return run_fusion(patient_id, module_outputs, modality_mask, config, disabled_modalities)


def _run_fusion(
    patient_id: str,
    module_outputs: list[ModuleOutput | dict[str, Any]],
    modality_mask: dict[str, bool] | None,
    config: dict[str, Any] | None,
    disabled_modalities: list[str] | None,
    include_what_if: bool,
) -> FusionOutput:
    cfg = config or load_config()
    parsed_outputs = [
        output if isinstance(output, ModuleOutput) else ModuleOutput.model_validate(output)
        for output in module_outputs
    ]
    mask = dict(modality_mask or build_modality_mask(parsed_outputs))
    disabled = set(disabled_modalities or [])
    unknown_disabled = disabled - set(ALLOWED_MODALITIES)
    if unknown_disabled:
        raise ValueError(f"Unknown disabled modalities: {sorted(unknown_disabled)}")
    for modality in disabled:
        mask[modality] = False

    fusion_input = FusionInput(patient_id=patient_id, module_outputs=parsed_outputs, modality_mask=mask)
    fusion = EvidenceWeightedFusion(cfg)
    evidence = fusion.build_evidence(fusion_input.module_outputs, fusion_input.modality_mask)
    raw_prediction = fusion.predict(evidence)
    fusion_details = raw_prediction.get("fusion_details", {})
    confidence, quality_summary = fusion.confidence(evidence)
    heads = PredictionHeads(cfg).run(
        diagnosis_probability=float(raw_prediction["diagnosis"]["probability"]),
        risk_score=float(raw_prediction["risk"]["score"]),
        raw_confidence=confidence,
        quality_summary=quality_summary,
    )
    prediction = {
        "diagnosis": {
            "class": heads["diagnosis_head"]["class"],
            "probability": heads["diagnosis_head"]["probability"],
        },
        "risk": {
            "class": heads["risk_head"]["class"],
            "score": heads["risk_head"]["score"],
        },
    }
    confidence = float(heads["confidence_calibration_head"]["confidence"])
    warnings = _collect_warnings(fusion_input.module_outputs, fusion_input.modality_mask, evidence, cfg)
    decision_trace = _decision_trace(evidence, prediction, confidence, disabled, fusion_details)
    what_if = (
        _compute_what_if(patient_id, parsed_outputs, modality_mask, cfg, disabled, prediction)
        if include_what_if
        else {}
    )

    return FusionOutput(
        patient_id=patient_id,
        diagnosis=prediction["diagnosis"],
        risk=prediction["risk"],
        confidence=round(confidence, 4),
        modality_contributions={modality: item.contribution for modality, item in evidence.items()},
        modality_evidence={modality: item.to_dict() for modality, item in evidence.items()},
        decision_trace=decision_trace,
        what_if=what_if,
        quality_summary=quality_summary,
        prediction_heads=heads,
        fusion_details=fusion_details,
        warnings=warnings,
    )


def _compute_what_if(
    patient_id: str,
    module_outputs: list[ModuleOutput],
    modality_mask: dict[str, bool] | None,
    config: dict[str, Any],
    disabled: set[str],
    selected_prediction: dict[str, Any],
) -> dict[str, Any]:
    base = _run_fusion(patient_id, module_outputs, modality_mask, config, [], include_what_if=False)
    selected_risk = float(selected_prediction["risk"]["score"])
    selected_probability = float(selected_prediction["diagnosis"]["probability"])
    available_modalities = [
        output.modality
        for output in module_outputs
        if output.status == "available" and output.modality not in disabled
    ]
    ablations: dict[str, Any] = {}
    for modality in available_modalities:
        candidate_disabled = sorted(disabled | {modality})
        candidate = _run_fusion(
            patient_id,
            module_outputs,
            modality_mask,
            config,
            candidate_disabled,
            include_what_if=False,
        )
        candidate_risk = candidate.risk.model_dump(by_alias=True)["score"]
        candidate_probability = candidate.diagnosis.model_dump(by_alias=True)["probability"]
        ablations[modality] = {
            "disabled_modalities": candidate_disabled,
            "risk_score": candidate_risk,
            "diagnosis_probability": candidate_probability,
            "risk_delta": round(candidate_risk - selected_risk, 4),
            "probability_delta": round(candidate_probability - selected_probability, 4),
            "confidence": candidate.confidence,
        }
    return {
        "selected_disabled_modalities": sorted(disabled),
        "all_modalities": {
            "risk_score": base.risk.model_dump(by_alias=True)["score"],
            "diagnosis_probability": base.diagnosis.model_dump(by_alias=True)["probability"],
            "confidence": base.confidence,
        },
        "selected_modalities": {
            "risk_score": round(selected_risk, 4),
            "diagnosis_probability": round(selected_probability, 4),
        },
        "leave_one_out": ablations,
    }


def _collect_warnings(
    module_outputs: list[ModuleOutput],
    modality_mask: dict[str, bool],
    evidence: dict[str, FusionEvidence],
    config: dict[str, Any],
) -> list[str]:
    by_modality = {output.modality: output for output in module_outputs}
    warnings: list[str] = []
    for modality in ALLOWED_MODALITIES:
        output = by_modality.get(modality)
        if not modality_mask.get(modality, False):
            warnings.append(f"{modality.capitalize()} modality missing or disabled")
        if output:
            warnings.extend(output.warnings)

    active = [item for item in evidence.values() if item.contribution > 0]
    risks = [item.risk_score for item in active]
    disagreement = max(risks) - min(risks) if len(risks) > 1 else 0.0
    threshold = float(config.get("fusion", {}).get("disagreement_warning_threshold", 0.35))
    if disagreement > threshold:
        warnings.append("Available modalities disagree substantially; review modality evidence before acting")

    confidence = sum(item.contribution * item.confidence for item in active)
    if active and confidence < float(config.get("fusion", {}).get("low_confidence_threshold", 0.6)):
        warnings.append("Fusion confidence is low because limited or uncertain modalities were available")

    deduped: list[str] = []
    for warning in warnings:
        if warning not in deduped:
            deduped.append(warning)
    return deduped


def _decision_trace(
    evidence: dict[str, FusionEvidence],
    prediction: dict[str, Any],
    confidence: float,
    disabled: set[str],
    fusion_details: dict[str, Any] | None = None,
) -> list[str]:
    active = sorted(
        [item for item in evidence.values() if item.contribution > 0],
        key=lambda item: item.contribution,
        reverse=True,
    )
    if not active:
        return ["No available enabled modalities were provided, so fusion returned zero-confidence output."]

    trace = [
        f"Fused {len(active)} enabled modality output(s) using confidence, quality, agreement, prediction strength, and configured modality priors.",
        "Top contributors: "
        + ", ".join(f"{item.modality} {item.contribution:.1%}" for item in active[:3])
        + ".",
        f"Final risk is {prediction['risk']['class']} ({prediction['risk']['score']:.1%}) with fusion confidence {confidence:.1%}.",
    ]
    guardrail = (fusion_details or {}).get("high_risk_guardrail")
    if guardrail:
        trace.append(
            f"High-risk guardrail lifted the fused score because {guardrail['modality']} produced high-confidence high-risk evidence."
        )
    if disabled:
        trace.append("User-disabled modalities: " + ", ".join(sorted(disabled)) + ".")
    return trace
