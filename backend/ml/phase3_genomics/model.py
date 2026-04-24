"""Deterministic demo genomics encoder and standalone risk head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.ml.common.utils import clamp, fixed_length_embedding, risk_class_from_score, sigmoid


DEMO_RISK_WEIGHTS: dict[str, float] = {
    "TP53_expr": 0.95,
    "CDKN2A_expr": -0.6,
    "EGFR_expr": 0.72,
    "PIK3CA_expr": 0.45,
    "NOTCH1_expr": -0.32,
    "CCND1_expr": 0.5,
    "FAT1_expr": 0.18,
    "CASP8_expr": -0.28,
    "HRAS_expr": 0.24,
    "MET_expr": 0.42,
    "MYC_expr": 0.68,
    "MDM2_expr": 0.22,
}


@dataclass
class GenomicsModelOutput:
    embedding: list[float]
    risk_score: float
    risk_class: str
    diagnosis_probability: float
    confidence: float


class DeterministicGenomicsModel:
    """A transparent model wrapper used when trained artifacts are unavailable."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.embedding_dim = int(config["genomics"].get("embedding_dim", 128))
        self.medium_threshold = float(config["genomics"].get("medium_risk_threshold", 0.4))
        self.high_threshold = float(config["genomics"].get("high_risk_threshold", 0.7))

    def predict(
        self,
        patient_id: str,
        feature_order: list[str],
        standardized_values: list[float],
        missing_rate: float,
    ) -> GenomicsModelOutput:
        weighted_sum = 0.0
        total_weight = 0.0
        for feature, value in zip(feature_order, standardized_values, strict=True):
            weight = DEMO_RISK_WEIGHTS.get(feature, 0.1)
            weighted_sum += value * weight
            total_weight += abs(weight)

        logit = weighted_sum / max(total_weight, 1.0)
        risk_score = clamp(sigmoid(logit + 0.15))
        confidence = clamp(0.58 + abs(risk_score - 0.5) * 0.65 - missing_rate * 0.3)
        diagnosis_probability = clamp(0.18 + risk_score * 0.72)
        embedding = fixed_length_embedding(
            standardized_values + [risk_score, confidence],
            self.embedding_dim,
            f"genomics:{patient_id}",
        )
        return GenomicsModelOutput(
            embedding=embedding,
            risk_score=round(risk_score, 4),
            risk_class=risk_class_from_score(risk_score, self.medium_threshold, self.high_threshold),
            diagnosis_probability=round(diagnosis_probability, 4),
            confidence=round(confidence, 4),
        )
