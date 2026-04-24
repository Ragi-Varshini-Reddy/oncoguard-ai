"""Layer-6 prediction heads for diagnosis, risk, and confidence calibration."""

from __future__ import annotations

from typing import Any

from backend.ml.common.utils import clamp, diagnosis_from_probability, risk_class_from_score


class PredictionHeads:
    """Config-driven heads that make fusion outputs explicit and explainable."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        genomics = config.get("genomics", {})
        self.medium_threshold = float(genomics.get("medium_risk_threshold", 0.4))
        self.high_threshold = float(genomics.get("high_risk_threshold", 0.7))

    def run(
        self,
        diagnosis_probability: float,
        risk_score: float,
        raw_confidence: float,
        quality_summary: dict[str, Any],
    ) -> dict[str, Any]:
        probability = clamp(diagnosis_probability)
        score = clamp(risk_score)
        calibrated_confidence, penalties = self._calibrate_confidence(raw_confidence, quality_summary)
        return {
            "diagnosis_head": {
                "class": diagnosis_from_probability(probability),
                "probability": round(probability, 4),
                "classes": self.config.get("labels", {}).get("diagnosis_classes", ["benign", "precancer", "cancer"]),
                "input": "fused_diagnosis_probability",
            },
            "risk_head": {
                "class": risk_class_from_score(score, self.medium_threshold, self.high_threshold),
                "score": round(score, 4),
                "thresholds": {
                    "low_max": round(self.medium_threshold - 0.01, 4),
                    "medium_min": self.medium_threshold,
                    "high_min": self.high_threshold,
                },
                "input": "fused_risk_score",
            },
            "confidence_calibration_head": {
                "confidence": calibrated_confidence,
                "input": "weighted modality confidence, coverage, disagreement, quality",
                "coverage": quality_summary.get("coverage", 0.0),
                "risk_disagreement": quality_summary.get("risk_disagreement", 0.0),
                "warning": calibrated_confidence < float(self.config.get("fusion", {}).get("low_confidence_threshold", 0.6)),
                "penalties": penalties,
            },
        }

    @staticmethod
    def _calibrate_confidence(raw_confidence: float, quality_summary: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
        coverage = float(quality_summary.get("coverage", 0.0))
        disagreement = float(quality_summary.get("risk_disagreement", 0.0))
        penalties = []
        
        coverage_bonus = 0.08 if coverage >= 0.75 else 0.0
        disagreement_penalty = min(0.12, disagreement * 0.15)
        
        if coverage < 1.0:
            missing_pct = (1.0 - coverage)
            pen = min(0.20, missing_pct * 0.25)
            if pen > 0:
                penalties.append({"reason": "Missing modalities", "impact": round(-pen, 4)})
        
        if disagreement_penalty > 0:
            penalties.append({"reason": "Modality disagreement", "impact": round(-disagreement_penalty, 4)})
            
        if coverage_bonus > 0:
            penalties.append({"reason": "High coverage bonus", "impact": round(coverage_bonus, 4)})

        total_adjustment = sum(p["impact"] for p in penalties)
        calibrated = clamp(raw_confidence + total_adjustment)
        return round(calibrated, 4), penalties
