"""Evidence-weighted late fusion for multimodal decision support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.ml.common.contracts import ALLOWED_MODALITIES, ModuleOutput
from backend.ml.common.utils import clamp, diagnosis_from_probability, normalize_scores, risk_class_from_score


@dataclass(frozen=True)
class FusionEvidence:
    modality: str
    status: str
    enabled: bool
    confidence: float
    risk_score: float
    diagnosis_probability: float
    quality_factor: float
    raw_weight: float
    contribution: float
    mode: str
    explanation_summary: dict[str, Any]
    quality_flags: dict[str, Any]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "status": self.status,
            "enabled": self.enabled,
            "confidence": round(self.confidence, 4),
            "risk_score": round(self.risk_score, 4),
            "diagnosis_probability": round(self.diagnosis_probability, 4),
            "quality_factor": round(self.quality_factor, 4),
            "raw_weight": round(self.raw_weight, 4),
            "contribution": round(self.contribution, 4),
            "mode": self.mode,
            "explanation_summary": self.explanation_summary,
            "quality_flags": self.quality_flags,
            "warnings": self.warnings,
        }


class EvidenceWeightedFusion:
    """CPU-fast deterministic fusion with modality-level explainability."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.priors = {
            modality: float(config.get("fusion", {}).get("modality_priors", {}).get(modality, 1.0))
            for modality in ALLOWED_MODALITIES
        }

    def build_evidence(
        self,
        module_outputs: list[ModuleOutput],
        modality_mask: dict[str, bool],
    ) -> dict[str, FusionEvidence]:
        outputs_by_modality = {output.modality: output for output in module_outputs}
        raw_weights: dict[str, float] = {}
        partial: dict[str, FusionEvidence] = {}

        for modality in ALLOWED_MODALITIES:
            output = outputs_by_modality.get(modality)
            enabled = bool(modality_mask.get(modality, False))
            evidence = self._normalize_modality(modality, output, enabled)
            raw_weights[modality] = evidence.raw_weight
            partial[modality] = evidence

        contributions = normalize_scores(raw_weights)
        return {
            modality: FusionEvidence(
                **{
                    **evidence.__dict__,
                    "contribution": round(contributions.get(modality, 0.0), 4),
                }
            )
            for modality, evidence in partial.items()
        }

    def predict(self, evidence: dict[str, FusionEvidence]) -> dict[str, Any]:
        weighted_risk = sum(item.contribution * item.risk_score for item in evidence.values())
        weighted_probability = sum(item.contribution * item.diagnosis_probability for item in evidence.values())
        total_contribution = sum(item.contribution for item in evidence.values())
        if total_contribution <= 0:
            weighted_risk = 0.0
            weighted_probability = 0.0

        weighted_risk = clamp(weighted_risk)
        weighted_probability = clamp(weighted_probability)
        return {
            "diagnosis": {
                "class": diagnosis_from_probability(weighted_probability),
                "probability": round(weighted_probability, 4),
            },
            "risk": {
                "class": risk_class_from_score(weighted_risk),
                "score": round(weighted_risk, 4),
            },
        }

    def confidence(self, evidence: dict[str, FusionEvidence]) -> tuple[float, dict[str, Any]]:
        active = [item for item in evidence.values() if item.contribution > 0]
        if not active:
            return 0.0, {
                "available_modalities": 0,
                "enabled_modalities": 0,
                "coverage": 0.0,
                "weighted_confidence": 0.0,
                "risk_disagreement": 0.0,
                "disagreement_penalty": 0.0,
            }

        weighted_confidence = sum(item.contribution * item.confidence * item.quality_factor for item in active)
        coverage = len(active) / len(ALLOWED_MODALITIES)
        risks = [item.risk_score for item in active]
        disagreement = max(risks) - min(risks) if len(risks) > 1 else 0.0
        disagreement_penalty = min(0.35, disagreement * 0.35)
        confidence = clamp(weighted_confidence * (0.75 + coverage * 0.25) * (1.0 - disagreement_penalty))
        return round(confidence, 4), {
            "available_modalities": sum(1 for item in evidence.values() if item.status == "available"),
            "enabled_modalities": len(active),
            "coverage": round(coverage, 4),
            "weighted_confidence": round(weighted_confidence, 4),
            "risk_disagreement": round(disagreement, 4),
            "disagreement_penalty": round(disagreement_penalty, 4),
        }

    def _normalize_modality(
        self,
        modality: str,
        output: ModuleOutput | None,
        enabled: bool,
    ) -> FusionEvidence:
        if output is None:
            return self._empty_evidence(modality, "missing", enabled, ["No module output provided"])

        prediction = output.prediction or {}
        risk_score = _float(prediction.get("risk_score", prediction.get("diagnosis_probability", 0.5)), 0.5)
        diagnosis_probability = _float(prediction.get("diagnosis_probability", risk_score), risk_score)
        confidence = _float(output.confidence, 0.5)
        quality_factor = _quality_factor(output)

        available = output.status == "available" and enabled
        prediction_strength = 0.85 + abs(clamp(risk_score) - 0.5) * 0.3
        raw_weight = (
            self.priors.get(modality, 1.0) * confidence * quality_factor * prediction_strength
            if available
            else 0.0
        )
        return FusionEvidence(
            modality=modality,
            status=output.status,
            enabled=enabled,
            confidence=clamp(confidence),
            risk_score=clamp(risk_score),
            diagnosis_probability=clamp(diagnosis_probability),
            quality_factor=round(quality_factor, 4),
            raw_weight=round(raw_weight, 6),
            contribution=0.0,
            mode=output.mode,
            explanation_summary=_summarize_explanations(output),
            quality_flags=output.quality_flags,
            warnings=output.warnings,
        )

    @staticmethod
    def _empty_evidence(modality: str, status: str, enabled: bool, warnings: list[str]) -> FusionEvidence:
        return FusionEvidence(
            modality=modality,
            status=status,
            enabled=enabled,
            confidence=0.0,
            risk_score=0.0,
            diagnosis_probability=0.0,
            quality_factor=0.0,
            raw_weight=0.0,
            contribution=0.0,
            mode=status,
            explanation_summary={},
            quality_flags={"input_valid": False},
            warnings=warnings,
        )


def _quality_factor(output: ModuleOutput) -> float:
    if output.status != "available":
        return 0.0
    flags = output.quality_flags or {}
    factor = 1.0
    missing_rate = _float(flags.get("missing_feature_rate"), 0.0)
    factor *= 1.0 - min(0.45, missing_rate * 0.5)
    if flags.get("low_confidence"):
        factor *= 0.75
    if flags.get("out_of_distribution"):
        factor *= 0.7
    if flags.get("input_valid") is False:
        factor *= 0.5
    return clamp(factor, 0.05, 1.0)


def _summarize_explanations(output: ModuleOutput) -> dict[str, Any]:
    explanations = output.explanations or {}
    return {
        "method": explanations.get("method"),
        "note": explanations.get("note"),
        "top_features": explanations.get("top_features", [])[:5],
        "has_visual_explanation": bool(
            explanations.get("gradcam_placeholder") or explanations.get("heatmap") or explanations.get("attention_map")
        ),
    }


def _float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
