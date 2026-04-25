"""Lightweight clinical structured-data inference wrapper."""

from __future__ import annotations

from typing import Any

import os

try:
    import joblib
except ImportError:  # pragma: no cover - optional artifact dependency
    joblib = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional scientific dependency
    np = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional scientific dependency
    pd = None

try:
    import shap
except ImportError:  # pragma: no cover - optional explainability dependency
    shap = None

from backend.ml.common.config import load_config
from backend.ml.common.contracts import ModuleOutput, missing_module_output
from backend.ml.common.utils import clamp, fixed_length_embedding, risk_class_from_score

_CLINICAL_MODEL = None
_CLINICAL_ENCODERS = None

def _load_clinical_artifacts():
    global _CLINICAL_MODEL, _CLINICAL_ENCODERS
    if _CLINICAL_MODEL is not None:
        return
    if joblib is None:
        return
    
    artifact_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "artifacts", "models"
    )
    model_path = os.path.join(artifact_dir, "clinical_rf_model.pkl")
    encoder_path = os.path.join(artifact_dir, "clinical_encoders.pkl")
    
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        _CLINICAL_MODEL = joblib.load(model_path)
        _CLINICAL_ENCODERS = joblib.load(encoder_path)


def _clinical_input_values(clinical_features: dict[str, Any]) -> dict[str, Any]:
    return {
        "age": _float(clinical_features.get("age"), 45.0),
        "gender": clinical_features.get("gender") or clinical_features.get("sex") or "male",
        "tobacco_use": bool(clinical_features.get("tobacco_use")),
        "alcohol_use": bool(clinical_features.get("alcohol_use")),
        "lesion_site": clinical_features.get("lesion_site") or "unknown",
        "lesion_size_cm": _float(clinical_features.get("lesion_size_cm"), 1.0),
        "persistent_ulcer_weeks": _float(clinical_features.get("persistent_ulcer_weeks"), 0.0),
        "neck_node_present": bool(clinical_features.get("neck_node_present")),
        "poor_oral_hygiene": bool(clinical_features.get("poor_oral_hygiene")),
        "family_history": bool(clinical_features.get("family_history")),
    }


def _clinical_model_frame(values: dict[str, Any]) -> pd.DataFrame:
    if pd is None:
        raise RuntimeError("pandas is required for artifact-backed clinical inference")
    return pd.DataFrame(
        [
            {
                "Age": values["age"],
                "Gender": values["gender"],
                "Tobacco Use": "Yes" if values["tobacco_use"] else "No",
                "Alcohol Consumption": "Yes" if values["alcohol_use"] else "No",
                "Betel Quid Use": "No",
                "HPV Infection": "No",
                "Poor Oral Hygiene": "Yes" if values["poor_oral_hygiene"] else "No",
                "Family History of Cancer": "Yes" if values["family_history"] else "No",
                "Oral Lesions": "Yes" if values["lesion_size_cm"] > 0 else "No",
                "Unexplained Bleeding": "No",
                "White or Red Patches in Mouth": "Yes" if values["lesion_size_cm"] > 0 else "No",
            }
        ]
    )


def _encode_clinical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if pd is None:
        raise RuntimeError("pandas is required for artifact-backed clinical inference")
    encoded = frame.copy()
    if not _CLINICAL_ENCODERS:
        return encoded
    for column, encoder in _CLINICAL_ENCODERS.items():
        if column in encoded.columns:
            encoded[column] = encoded[column].astype(str).map(lambda value: value if value in encoder.classes_ else encoder.classes_[0])
            encoded[column] = encoder.transform(encoded[column])
    return encoded


def _positive_class_shap_values(shap_values: Any) -> np.ndarray:
    if np is None:
        raise RuntimeError("numpy is required for SHAP-backed clinical inference")
    if isinstance(shap_values, list):
        values = shap_values[-1] if len(shap_values) > 1 else shap_values[0]
    else:
        values = shap_values
    values_array = np.asarray(values)
    if values_array.ndim == 3:
        values_array = values_array[0, :, -1]
    elif values_array.ndim == 2:
        values_array = values_array[0]
    return np.asarray(values_array, dtype=float)


def _clinical_recommendations(values: dict[str, Any]) -> list[str]:
    recommendations = []
    if values["tobacco_use"]:
        recommendations.append("Tobacco use is a positive SHAP driver here; offer cessation counseling and document exposure history.")
    if values["alcohol_use"]:
        recommendations.append("Alcohol use is a positive SHAP driver here; counsel reduction or cessation and reinforce oral cancer surveillance.")
    return recommendations


def _heuristic_clinical_explanations(values: dict[str, Any]) -> dict[str, Any]:
    contribution_scores = {
        "persistent_ulcer_weeks": values["persistent_ulcer_weeks"] * 0.035,
        "lesion_size_cm": values["lesion_size_cm"] * 0.11,
        "neck_node_present": 0.18 if values["neck_node_present"] else 0.0,
        "tobacco_use": 0.13 if values["tobacco_use"] else 0.0,
        "alcohol_use": 0.07 if values["alcohol_use"] else 0.0,
        "age": max(0.0, (values["age"] - 35.0) / 100.0),
    }
    ranked = [
        {
            "feature": feature,
            "value": values.get(feature),
            "importance_score": round(abs(score), 4),
            "shap_value": round(score, 4),
            "direction": "increases_risk" if score > 0 else "decreases_risk" if score < 0 else "neutral",
        }
        for feature, score in contribution_scores.items()
    ]
    ranked.sort(key=lambda item: item["importance_score"], reverse=True)
    priority_features = [item for item in ranked if item["feature"] in {"tobacco_use", "alcohol_use"} and item["value"]]
    ordered = priority_features + [item for item in ranked if item not in priority_features]
    return {
        "method": "transparent_demo_clinical_score",
        "note": "Demo clinical score only; not a validated clinical calculator.",
        "top_features": ordered[: max(5, len(priority_features))],
        "feature_values": values,
        "recommendations": _clinical_recommendations(values),
    }


def _shap_clinical_explanations(model: Any, frame: pd.DataFrame, values: dict[str, Any]) -> dict[str, Any]:
    if shap is None or pd is None or np is None:
        return _heuristic_clinical_explanations(values)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(frame)
    positive_values = _positive_class_shap_values(shap_values)
    model_to_display = {
        "Age": "age",
        "Gender": "gender",
        "Tobacco Use": "tobacco_use",
        "Alcohol Consumption": "alcohol_use",
        "Betel Quid Use": "betel_quid_use",
        "HPV Infection": "hpv_infection",
        "Poor Oral Hygiene": "poor_oral_hygiene",
        "Family History of Cancer": "family_history",
        "Oral Lesions": "oral_lesions",
        "Unexplained Bleeding": "unexplained_bleeding",
        "White or Red Patches in Mouth": "white_or_red_patches_in_mouth",
    }
    ranked = []
    for index, column in enumerate(frame.columns):
        display_name = model_to_display.get(column, column.lower().replace(" ", "_"))
        ranked.append(
            {
                "feature": display_name,
                "model_feature": column,
                "value": values.get(display_name),
                "importance_score": round(abs(float(positive_values[index])), 4),
                "shap_value": round(float(positive_values[index]), 4),
                "direction": "increases_risk" if positive_values[index] > 0 else "decreases_risk" if positive_values[index] < 0 else "neutral",
            }
        )

    ranked.sort(key=lambda item: item["importance_score"], reverse=True)
    priority_features = [item for item in ranked if item["feature"] in {"tobacco_use", "alcohol_use"} and item["value"]]
    ordered = priority_features + [item for item in ranked if item not in priority_features]
    recommendations = _clinical_recommendations(values)

    return {
        "method": "shap_tree_explainer",
        "note": "SHAP local explanation for the trained clinical model; positive values indicate features that push risk upward for this patient.",
        "top_features": ordered[: max(5, len(priority_features))],
        "feature_values": values,
        "recommendations": recommendations,
    }

def run_clinical_inference(
    patient_id: str,
    clinical_features: dict[str, Any] | None,
    config: dict[str, Any] | None = None,
) -> ModuleOutput:
    cfg = config or load_config()
    embedding_dim = int(cfg["modalities"]["clinical"]["embedding_dim"])
    if not clinical_features:
        return missing_module_output(patient_id, "clinical", embedding_dim)

    _load_clinical_artifacts()
    values = _clinical_input_values(clinical_features)
    use_artifact_path = _CLINICAL_MODEL is not None and pd is not None and np is not None and shap is not None

    if not use_artifact_path:
        age = values["age"]
        lesion_size = values["lesion_size_cm"]
        ulcer_weeks = values["persistent_ulcer_weeks"]
        tobacco = 1.0 if values["tobacco_use"] else 0.0
        alcohol = 1.0 if values["alcohol_use"] else 0.0
        node = 1.0 if values["neck_node_present"] else 0.0

        # Fallback heuristic if model isn't available
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
        explanations = _heuristic_clinical_explanations(values)
    else:
        model_frame = _clinical_model_frame(values)
        encoded_frame = _encode_clinical_frame(model_frame)

        # Predict probability of class 1 ("Yes")
        probs = _CLINICAL_MODEL.predict_proba(encoded_frame)[0]
        risk_score = float(probs[1]) if len(probs) > 1 else float(probs[0])
        confidence = float(max(probs))
        explanations = _shap_clinical_explanations(_CLINICAL_MODEL, encoded_frame, values)

    age = values["age"]
    lesion_size = values["lesion_size_cm"]
    ulcer_weeks = values["persistent_ulcer_weeks"]
    tobacco = 1.0 if values["tobacco_use"] else 0.0
    alcohol = 1.0 if values["alcohol_use"] else 0.0
    node = 1.0 if values["neck_node_present"] else 0.0

    embedding = fixed_length_embedding(
        [age / 100.0, lesion_size, ulcer_weeks / 12.0, tobacco, alcohol, node, risk_score],
        embedding_dim,
        f"clinical:{patient_id}",
    )

    return ModuleOutput(
        patient_id=patient_id,
        modality="clinical",
        status="available",
        embedding=embedding,
        embedding_dim=embedding_dim,
        prediction={"risk_score": round(risk_score, 4), "risk_class": risk_class_from_score(risk_score)},
        confidence=round(confidence, 4),
        explanations=explanations,
        warnings=["Clinical branch uses transparent demo scoring, not a validated diagnostic model"],
        quality_flags={"input_valid": True, "artifact_loaded": False},
        mode="artifact" if _CLINICAL_MODEL is not None else "demo",
    )


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
