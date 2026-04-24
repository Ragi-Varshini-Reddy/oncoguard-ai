"""Standalone genomics inference entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.config import load_config
from src.common.contracts import ModuleOutput, missing_module_output
from src.phase3_genomics.explain import explain_genomic_features
from src.phase3_genomics.artifact_model import infer_with_genomics_artifacts
from src.phase3_genomics.model import DeterministicGenomicsModel
from src.phase3_genomics.preprocess import GenomicsPreprocessor
from src.phase3_genomics.schema import GenomicsInferenceRequest
from src.phase3_genomics.utils import load_genomic_features_from_csv


def run_genomics_inference(
    patient_id: str,
    genomic_features: dict[str, Any] | None,
    sample_id: str | None = None,
    config: dict[str, Any] | None = None,
    source: str = "demo",
) -> ModuleOutput:
    cfg = config or load_config()
    embedding_dim = int(cfg["genomics"].get("embedding_dim", 128))
    request = GenomicsInferenceRequest(
        patient_id=patient_id,
        genomic_features=genomic_features,
        sample_id=sample_id,
        source=source,
    )

    if request.genomic_features is None:
        return missing_module_output(patient_id, "genomics", embedding_dim)

    if cfg["genomics"].get("mode") == "artifact":
        try:
            result = infer_with_genomics_artifacts(
                request.patient_id,
                request.genomic_features,
                cfg,
            )
            warnings = ["Genomics inference used trained local artifacts; clinical validation is still required"]
            if result["missing_features"]:
                warnings.append(f"Imputed missing genomic features: {', '.join(result['missing_features'])}")
            return ModuleOutput(
                patient_id=request.patient_id,
                modality="genomics",
                status="available",
                embedding=result["embedding"],
                embedding_dim=result["embedding_dim"],
                prediction={
                    "diagnosis_probability": result["diagnosis_probability"],
                    "risk_score": result["risk_score"],
                    "risk_class": result["risk_class"],
                },
                confidence=result["confidence"],
                explanations={
                    "method": "linear_coefficient_x_standardized_value",
                    "top_features": result["top_features"],
                    "model_card": result["model_card"],
                    "training_metrics": result["metrics"],
                    "note": "Feature effects are model explanations, not validated biomarkers.",
                },
                warnings=warnings,
                quality_flags={
                    "input_valid": True,
                    "missing_feature_count": len(result["missing_features"]),
                    "missing_feature_rate": round(len(result["missing_features"]) / max(1, len(cfg["genomics"]["selected_gene_panel"])), 4),
                    "used_imputation": bool(result["missing_features"]),
                    "artifact_loaded": True,
                },
                mode="artifact",
            )
        except Exception as exc:
            return ModuleOutput(
                patient_id=patient_id,
                modality="genomics",
                status="error",
                embedding=None,
                embedding_dim=embedding_dim,
                prediction=None,
                confidence=None,
                explanations={},
                warnings=[f"Artifact-backed genomics inference unavailable: {exc}"],
                quality_flags={"input_valid": False, "artifact_loaded": False},
                mode="artifact_error",
            )

    try:
        preprocessor = GenomicsPreprocessor(cfg)
        prepared = preprocessor.transform(request.genomic_features)
        missing_rate = float(prepared.quality_flags["missing_feature_rate"])
        model = DeterministicGenomicsModel(cfg)
        model_output = model.predict(
            patient_id=request.patient_id,
            feature_order=prepared.feature_order,
            standardized_values=prepared.standardized_values,
            missing_rate=missing_rate,
        )
        explanations = explain_genomic_features(
            prepared.feature_order,
            prepared.raw_values,
            prepared.standardized_values,
            cfg,
        )
        quality_flags = dict(prepared.quality_flags)
        quality_flags["low_confidence"] = model_output.confidence < float(
            cfg["genomics"].get("low_confidence_threshold", 0.6)
        )

        warnings = list(prepared.warnings)
        if quality_flags["low_confidence"]:
            warnings.append("Genomics module confidence is low")
        warnings.append("Genomics model is a deterministic demo wrapper, not clinically validated")

        return ModuleOutput(
            patient_id=request.patient_id,
            modality="genomics",
            status="available",
            embedding=model_output.embedding,
            embedding_dim=embedding_dim,
            prediction={
                "diagnosis_probability": model_output.diagnosis_probability,
                "risk_score": model_output.risk_score,
                "risk_class": model_output.risk_class,
            },
            confidence=model_output.confidence,
            explanations=explanations,
            warnings=warnings,
            quality_flags=quality_flags,
            mode=cfg["genomics"].get("mode", "demo"),
        )
    except Exception as exc:  # pragma: no cover - defensive contract boundary
        return ModuleOutput(
            patient_id=patient_id,
            modality="genomics",
            status="error",
            embedding=None,
            embedding_dim=embedding_dim,
            prediction=None,
            confidence=None,
            explanations={},
            warnings=[f"Genomics inference failed: {exc}"],
            quality_flags={"input_valid": False},
            mode="error_fallback",
        )


def run_genomics_from_table(
    table_path: str | Path,
    patient_id: str | None = None,
    config: dict[str, Any] | None = None,
) -> ModuleOutput:
    resolved_patient_id, sample_id, features = load_genomic_features_from_csv(table_path, patient_id)
    return run_genomics_inference(
        resolved_patient_id,
        features,
        sample_id=sample_id,
        config=config,
        source=str(table_path),
    )
