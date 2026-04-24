"""Feature-level explainability for genomics inference."""

from __future__ import annotations

from typing import Any

from backend.ml.phase3_genomics.model import DEMO_RISK_WEIGHTS


def explain_genomic_features(
    feature_order: list[str],
    raw_values: dict[str, float | None],
    standardized_values: list[float],
    config: dict[str, Any],
) -> dict[str, Any]:
    top_k = int(config["genomics"].get("top_k_features", 5))
    descriptions = dict(config["genomics"].get("feature_descriptions", {}))
    ranked: list[dict[str, Any]] = []

    for feature, standardized in zip(feature_order, standardized_values, strict=True):
        weight = DEMO_RISK_WEIGHTS.get(feature, 0.1)
        signed_effect = standardized * weight
        direction = "neutral"
        if signed_effect > 0.05:
            direction = "increases_risk"
        elif signed_effect < -0.05:
            direction = "decreases_risk"

        ranked.append(
            {
                "feature": feature,
                "value": raw_values.get(feature),
                "standardized_value": round(standardized, 4),
                "importance_score": round(abs(signed_effect), 4),
                "direction": direction,
                "description": descriptions.get(feature, "Model-ready molecular feature"),
            }
        )

    ranked.sort(key=lambda item: item["importance_score"], reverse=True)
    return {
        "method": "fallback_abs_standardized_value_x_demo_weight",
        "note": "Demo explanation only; features are model-important signals, not validated diagnostic biomarkers.",
        "top_features": ranked[:top_k],
    }
