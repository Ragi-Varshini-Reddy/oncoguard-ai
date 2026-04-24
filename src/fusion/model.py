"""Simple fusion scorer for the hackathon prototype."""

from __future__ import annotations

from src.common.contracts import ModuleOutput
from src.common.utils import clamp, diagnosis_from_probability, risk_class_from_score


class FusionScorer:
    """Combines standalone module risk signals using contribution weights."""

    def predict(
        self,
        module_outputs: list[ModuleOutput],
        contributions: dict[str, float],
    ) -> dict[str, object]:
        by_modality = {output.modality: output for output in module_outputs}
        weighted_risk = 0.0
        weighted_probability = 0.0

        for modality, weight in contributions.items():
            output = by_modality.get(modality)
            prediction = output.prediction if output and output.prediction else {}
            risk_score = float(prediction.get("risk_score", prediction.get("diagnosis_probability", 0.45)))
            diagnosis_probability = float(prediction.get("diagnosis_probability", risk_score))
            weighted_risk += weight * risk_score
            weighted_probability += weight * diagnosis_probability

        if sum(contributions.values()) <= 0:
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
