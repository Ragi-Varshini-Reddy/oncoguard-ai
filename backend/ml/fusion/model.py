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
    signal_strength: float
    agreement_factor: float
    raw_weight: float
    gated_weight: float
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
            "signal_strength": round(self.signal_strength, 4),
            "agreement_factor": round(self.agreement_factor, 4),
            "raw_weight": round(self.raw_weight, 4),
            "gated_weight": round(self.gated_weight, 4),
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

        consensus_risk = _weighted_average(
            {modality: evidence.risk_score for modality, evidence in partial.items()},
            raw_weights,
        )
        gated_weights = {
            modality: evidence.raw_weight * _agreement_factor(evidence, consensus_risk)
            for modality, evidence in partial.items()
        }
        contributions = normalize_scores(gated_weights)
        return {
            modality: FusionEvidence(
                **{
                    **evidence.__dict__,
                    "agreement_factor": round(_agreement_factor(evidence, consensus_risk), 4),
                    "gated_weight": round(gated_weights.get(modality, 0.0), 6),
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
        guardrail = self._high_risk_guardrail(evidence, weighted_risk, weighted_probability)
        if guardrail:
            weighted_risk = guardrail["risk_score"]
            weighted_probability = guardrail["diagnosis_probability"]
        return {
            "diagnosis": {
                "class": diagnosis_from_probability(weighted_probability),
                "probability": round(weighted_probability, 4),
            },
            "risk": {
                "class": risk_class_from_score(weighted_risk),
                "score": round(weighted_risk, 4),
            },
            "fusion_details": {
                "high_risk_guardrail": guardrail,
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
                "mean_agreement_factor": 0.0,
            }

        weighted_confidence = sum(item.contribution * item.confidence * item.quality_factor for item in active)
        coverage = len(active) / len(ALLOWED_MODALITIES)
        risks = [item.risk_score for item in active]
        disagreement = max(risks) - min(risks) if len(risks) > 1 else 0.0
        disagreement_penalty = min(0.35, disagreement * 0.35)
        mean_agreement = sum(item.contribution * item.agreement_factor for item in active)
        confidence = clamp(
            weighted_confidence
            * (0.75 + coverage * 0.25)
            * (1.0 - disagreement_penalty)
            * (0.9 + mean_agreement * 0.1)
        )
        return round(confidence, 4), {
            "available_modalities": sum(1 for item in evidence.values() if item.status == "available"),
            "enabled_modalities": len(active),
            "coverage": round(coverage, 4),
            "weighted_confidence": round(weighted_confidence, 4),
            "risk_disagreement": round(disagreement, 4),
            "disagreement_penalty": round(disagreement_penalty, 4),
            "mean_agreement_factor": round(mean_agreement, 4),
        }

    def _high_risk_guardrail(
        self,
        evidence: dict[str, FusionEvidence],
        weighted_risk: float,
        weighted_probability: float,
    ) -> dict[str, Any] | None:
        guardrail_cfg = self.config.get("fusion", {}).get("high_risk_guardrail", {})
        if guardrail_cfg.get("enabled", True) is False:
            return None

        min_risk = float(guardrail_cfg.get("min_modality_risk", 0.82))
        min_confidence = float(guardrail_cfg.get("min_confidence", 0.75))
        min_quality = float(guardrail_cfg.get("min_quality_factor", 0.7))
        blend = clamp(float(guardrail_cfg.get("escalation_blend", 0.35)), 0.0, 1.0)
        candidates = [
            item
            for item in evidence.values()
            if item.contribution > 0
            and item.risk_score >= min_risk
            and item.confidence >= min_confidence
            and item.quality_factor >= min_quality
        ]
        if not candidates:
            return None

        strongest = max(candidates, key=lambda item: (item.risk_score, item.confidence, item.contribution))
        if strongest.risk_score <= weighted_risk:
            return None

        risk_score = clamp(weighted_risk + (strongest.risk_score - weighted_risk) * blend)
        probability = clamp(weighted_probability + (strongest.diagnosis_probability - weighted_probability) * blend)
        return {
            "triggered": True,
            "modality": strongest.modality,
            "reason": "high-confidence high-risk modality should not be diluted by lower-risk modalities",
            "original_risk_score": round(weighted_risk, 4),
            "risk_score": round(risk_score, 4),
            "diagnosis_probability": round(probability, 4),
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
        signal_strength = _signal_strength(risk_score, diagnosis_probability)
        raw_weight = (
            self.priors.get(modality, 1.0) * confidence * quality_factor * signal_strength
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
            signal_strength=round(signal_strength, 4),
            agreement_factor=0.0,
            raw_weight=round(raw_weight, 6),
            gated_weight=0.0,
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
            signal_strength=0.0,
            agreement_factor=0.0,
            raw_weight=0.0,
            gated_weight=0.0,
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


def _signal_strength(risk_score: float, diagnosis_probability: float) -> float:
    risk_certainty = abs(clamp(risk_score) - 0.5)
    diagnosis_certainty = abs(clamp(diagnosis_probability) - 0.5)
    return clamp(0.72 + risk_certainty * 0.38 + diagnosis_certainty * 0.18, 0.72, 1.0)


def _agreement_factor(evidence: FusionEvidence, consensus_risk: float | None) -> float:
    if evidence.raw_weight <= 0 or consensus_risk is None:
        return 0.0
    alignment = 1.0 - abs(evidence.risk_score - consensus_risk)
    return clamp(0.65 + alignment * 0.35, 0.65, 1.0)


def _weighted_average(values: dict[str, float], weights: dict[str, float]) -> float | None:
    total = sum(max(0.0, weight) for weight in weights.values())
    if total <= 0:
        return None
    return sum(values[key] * max(0.0, weights.get(key, 0.0)) for key in values) / total


def _float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
