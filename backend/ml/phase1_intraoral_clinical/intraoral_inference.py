"""Prototype intraoral image inference wrapper."""

from __future__ import annotations

from backend.ml.common.config import load_config
from backend.ml.common.contracts import ModuleOutput, missing_module_output
from backend.ml.common.utils import clamp, fixed_length_embedding, risk_class_from_score, stable_hash_float


import os
import io
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

_INTRAORAL_MODEL = None
_INTRAORAL_DEVICE = None

def _load_intraoral_artifacts():
    global _INTRAORAL_MODEL, _INTRAORAL_DEVICE
    if _INTRAORAL_MODEL is not None:
        return
        
    artifact_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "artifacts", "models"
    )
    model_path = os.path.join(artifact_dir, "intraoral_efficientnet.pt")
    
    if os.path.exists(model_path):
        if torch.backends.mps.is_available():
            _INTRAORAL_DEVICE = torch.device("mps")
        elif torch.cuda.is_available():
            _INTRAORAL_DEVICE = torch.device("cuda")
        else:
            _INTRAORAL_DEVICE = torch.device("cpu")
            
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 1)
        
        model.load_state_dict(torch.load(model_path, map_location=_INTRAORAL_DEVICE))
        model.to(_INTRAORAL_DEVICE)
        model.eval()
        _INTRAORAL_MODEL = model

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
    _load_intraoral_artifacts()
    
    # Preprocessing
    mean_byte = sum(image_bytes[: min(byte_count, 4096)]) / max(1, min(byte_count, 4096)) / 255.0
    texture_proxy = stable_hash_float(image_bytes[:128].hex(), 0.0, 1.0)

    if _INTRAORAL_MODEL is None:
        # Fallback heuristic
        risk_score = clamp(0.22 + mean_byte * 0.32 + texture_proxy * 0.28)
        confidence = clamp(0.56 + byte_count / 2_000_000.0 + abs(risk_score - 0.5) * 0.25)
        embedding = fixed_length_embedding(
            [mean_byte, texture_proxy, byte_count / 1_000_000.0, risk_score],
            embedding_dim,
            f"intraoral:{patient_id}",
        )
        mode = "demo"
        warnings = ["Intraoral image branch uses a deterministic demo wrapper"]
    else:
        # Actual Inference
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(_INTRAORAL_DEVICE)
        
        with torch.no_grad():
            # Get logits
            logits = _INTRAORAL_MODEL(input_tensor)
            prob = torch.sigmoid(logits).item()
            
            # To get embeddings, we need features before the classifier. 
            # In EfficientNet, features are available via _INTRAORAL_MODEL.features and _INTRAORAL_MODEL.avgpool
            features = _INTRAORAL_MODEL.features(input_tensor)
            pooled = _INTRAORAL_MODEL.avgpool(features)
            flattened = torch.flatten(pooled, 1)
            raw_embedding = flattened[0].cpu().numpy().tolist()
            
        risk_score = float(prob)
        confidence = 0.85 # Assume high confidence for actual model run
        
        # We project the 1280-dim feature vector down to the required embedding_dim securely
        embedding = fixed_length_embedding(
            raw_embedding,
            embedding_dim,
            f"intraoral_model:{patient_id}",
        )
        mode = "active"
        warnings = []

    return ModuleOutput(
        patient_id=patient_id,
        modality="intraoral",
        status="available",
        embedding=embedding,
        embedding_dim=embedding_dim,
        prediction={"risk_score": round(risk_score, 4), "risk_class": risk_class_from_score(risk_score)},
        confidence=round(confidence, 4),
        explanations={
            "method": "efficientnet_b0" if mode == "active" else "demo_image_byte_signature",
            "gradcam_placeholder": [[0.1, 0.25, 0.4], [0.2, 0.55, 0.7], [0.15, 0.35, 0.5]],
            "note": "Intraoral inference completed." if mode == "active" else "Placeholder visual explanation.",
        },
        warnings=warnings,
        quality_flags={"input_valid": True, "artifact_loaded": (_INTRAORAL_MODEL is not None), "bytes": byte_count},
        mode=mode,
    )
