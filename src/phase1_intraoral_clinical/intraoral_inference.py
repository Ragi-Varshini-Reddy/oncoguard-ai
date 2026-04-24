"""Prototype intraoral image inference wrapper."""

from __future__ import annotations

from src.common.config import load_config
from src.common.contracts import ModuleOutput, missing_module_output
from src.common.utils import clamp, fixed_length_embedding, risk_class_from_score, stable_hash_float


def run_intraoral_inference(
    patient_id: str,
    image_bytes: bytes | None,
    config: dict | None = None,
) -> ModuleOutput:
    cfg = config or load_config()
    embedding_dim = int(cfg["modalities"]["intraoral"]["embedding_dim"])
    if not image_bytes:
        return missing_module_output(patient_id, "intraoral", embedding_dim)

    byte_count = len(image_bytes)
    mean_byte = sum(image_bytes[: min(byte_count, 4096)]) / max(1, min(byte_count, 4096)) / 255.0
    texture_proxy = stable_hash_float(image_bytes[:128].hex(), 0.0, 1.0)
    risk_score = clamp(0.22 + mean_byte * 0.32 + texture_proxy * 0.28)
    confidence = clamp(0.56 + byte_count / 2_000_000.0 + abs(risk_score - 0.5) * 0.25)
    embedding = fixed_length_embedding(
        [mean_byte, texture_proxy, byte_count / 1_000_000.0, risk_score],
        embedding_dim,
        f"intraoral:{patient_id}",
    )

    return ModuleOutput(
        patient_id=patient_id,
        modality="intraoral",
        status="available",
        embedding=embedding,
        embedding_dim=embedding_dim,
        prediction={"risk_score": round(risk_score, 4), "risk_class": risk_class_from_score(risk_score)},
        confidence=round(confidence, 4),
        explanations={
            "method": "demo_image_byte_signature",
            "gradcam_placeholder": [[0.1, 0.25, 0.4], [0.2, 0.55, 0.7], [0.15, 0.35, 0.5]],
            "note": "Placeholder visual explanation until a trained intraoral CNN is attached.",
        },
        warnings=["Intraoral image branch uses a deterministic demo wrapper"],
        quality_flags={"input_valid": True, "artifact_loaded": False, "bytes": byte_count},
        mode="demo",
    )
