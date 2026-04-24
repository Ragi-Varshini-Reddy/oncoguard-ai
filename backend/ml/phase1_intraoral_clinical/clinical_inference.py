"""Lightweight clinical structured-data inference wrapper."""

from __future__ import annotations

from typing import Any

from backend.ml.common.config import load_config
from backend.ml.common.contracts import ModuleOutput, missing_module_output
from backend.ml.common.utils import clamp, fixed_length_embedding, risk_class_from_score


def run_clinical_inference(
    patient_id: str,
    clinical_features: dict[str, Any] | None,
    config: dict[str, Any] | None = None,
) -> ModuleOutput:
    cfg = config or load_config()
    embedding_dim = int(cfg["modalities"]["clinical"]["embedding_dim"])
    if not clinical_features:
        return missing_module_output(patient_id, "clinical", embedding_dim)

    age = _float(clinical_features.get("age"), 45.0)
    lesion_size = _float(clinical_features.get("lesion_size_cm"), 1.0)
    ulcer_weeks = _float(clinical_features.get("persistent_ulcer_weeks"), 0.0)
    tobacco = 1.0 if clinical_features.get("tobacco_use") else 0.0
    alcohol = 1.0 if clinical_features.get("alcohol_use") else 0.0
    node = 1.0 if clinical_features.get("neck_node_present") else 0.0

    risk_score = clamp(
        0.12
        + (age - 35.0) / 100.0
        + lesion_size * 0.11
        + min(ulcer_weeks, 12.0) * 0.035
        + tobacco * 0.13
        + alcohol * 0.07
        + node * 0.18
    )
    confidence = clamp(0.62 + abs(risk_score - 0.5) * 0.35)
    embedding = fixed_length_embedding(
        [age / 100.0, lesion_size, ulcer_weeks / 12.0, tobacco, alcohol, node, risk_score],
        embedding_dim,
        f"clinical:{patient_id}",
    )
    top_features = [
        {"feature": "persistent_ulcer_weeks", "value": ulcer_weeks, "importance_score": round(ulcer_weeks * 0.035, 4)},
        {"feature": "lesion_size_cm", "value": lesion_size, "importance_score": round(lesion_size * 0.11, 4)},
        {"feature": "neck_node_present", "value": bool(node), "importance_score": round(node * 0.18, 4)},
        {"feature": "tobacco_use", "value": bool(tobacco), "importance_score": round(tobacco * 0.13, 4)},
    ]
    top_features.sort(key=lambda item: item["importance_score"], reverse=True)

    return ModuleOutput(
        patient_id=patient_id,
        modality="clinical",
        status="available",
        embedding=embedding,
        embedding_dim=embedding_dim,
        prediction={"risk_score": round(risk_score, 4), "risk_class": risk_class_from_score(risk_score)},
        confidence=round(confidence, 4),
        explanations={
            "method": "transparent_demo_clinical_score",
            "top_features": top_features,
            "note": "Demo clinical score only; not a validated clinical calculator.",
        },
        warnings=["Clinical branch uses transparent demo scoring, not a validated diagnostic model"],
        quality_flags={"input_valid": True, "artifact_loaded": False},
        mode="demo",
    )


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
