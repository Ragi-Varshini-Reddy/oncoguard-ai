"""Prototype histopathology patch/image inference wrapper."""

from __future__ import annotations

from src.common.config import load_config
from src.common.contracts import ModuleOutput, missing_module_output
from src.common.utils import clamp, fixed_length_embedding, risk_class_from_score, stable_hash_float


def run_histopathology_inference(
    patient_id: str,
    image_bytes: bytes | None,
    config: dict | None = None,
) -> ModuleOutput:
    cfg = config or load_config()
    embedding_dim = int(cfg["modalities"]["histopathology"]["embedding_dim"])
    if not image_bytes:
        return missing_module_output(patient_id, "histopathology", embedding_dim)

    byte_count = len(image_bytes)
    stain_proxy = stable_hash_float(image_bytes[-128:].hex() if image_bytes else patient_id, 0.0, 1.0)
    density_proxy = sum(image_bytes[: min(byte_count, 2048)]) / max(1, min(byte_count, 2048)) / 255.0
    risk_score = clamp(0.28 + stain_proxy * 0.34 + density_proxy * 0.22)
    confidence = clamp(0.6 + abs(risk_score - 0.5) * 0.3)
    embedding = fixed_length_embedding(
        [stain_proxy, density_proxy, byte_count / 1_000_000.0, risk_score],
        embedding_dim,
        f"histopathology:{patient_id}",
    )

    return ModuleOutput(
        patient_id=patient_id,
        modality="histopathology",
        status="available",
        embedding=embedding,
        embedding_dim=embedding_dim,
        prediction={"risk_score": round(risk_score, 4), "risk_class": risk_class_from_score(risk_score)},
        confidence=round(confidence, 4),
        explanations={
            "method": "demo_histology_byte_signature",
            "gradcam_placeholder": [[0.2, 0.3, 0.15], [0.45, 0.72, 0.48], [0.25, 0.38, 0.22]],
            "note": "Placeholder heatmap until a trained histopathology encoder is attached.",
        },
        warnings=["Histopathology branch uses a deterministic demo wrapper"],
        quality_flags={"input_valid": True, "artifact_loaded": False, "bytes": byte_count},
        mode="demo",
    )
