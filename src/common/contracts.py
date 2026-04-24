"""Shared Pydantic contracts for OralCare-AI module and fusion payloads."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "1.0"
ALLOWED_MODALITIES = ("intraoral", "clinical", "histopathology", "genomics")
ALLOWED_STATUSES = ("available", "missing", "error")

ModalityName = Literal["intraoral", "clinical", "histopathology", "genomics"]
StatusName = Literal["available", "missing", "error"]


class ModuleOutput(BaseModel):
    """Contract returned by every standalone modality module."""

    model_config = ConfigDict(extra="allow")

    patient_id: str = Field(min_length=1)
    modality: ModalityName
    status: StatusName
    embedding: list[float] | None
    embedding_dim: int = Field(gt=0)
    prediction: dict[str, Any] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    explanations: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    warnings: list[str] = Field(default_factory=list)
    quality_flags: dict[str, Any] = Field(default_factory=dict)
    mode: str = "demo"

    @field_validator("embedding")
    @classmethod
    def coerce_embedding(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        return [float(item) for item in value]

    @model_validator(mode="after")
    def validate_embedding_contract(self) -> "ModuleOutput":
        if self.status == "available":
            if self.embedding is None:
                raise ValueError("available module outputs must include an embedding")
            if len(self.embedding) != self.embedding_dim:
                raise ValueError("embedding length must match embedding_dim")
        return self


class FusionInput(BaseModel):
    """Input contract consumed by the fusion layer."""

    patient_id: str = Field(min_length=1)
    module_outputs: list[ModuleOutput]
    modality_mask: dict[ModalityName, bool]
    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="after")
    def validate_mask(self) -> "FusionInput":
        missing = set(ALLOWED_MODALITIES) - set(self.modality_mask)
        if missing:
            raise ValueError(f"modality_mask missing keys: {sorted(missing)}")
        return self


class DiagnosisOutput(BaseModel):
    class_: str = Field(alias="class")
    probability: float = Field(ge=0.0, le=1.0)

    model_config = ConfigDict(populate_by_name=True)


class RiskOutput(BaseModel):
    class_: str = Field(alias="class")
    score: float = Field(ge=0.0, le=1.0)

    model_config = ConfigDict(populate_by_name=True)


class FusionOutput(BaseModel):
    """Output contract returned by multimodal fusion."""

    patient_id: str
    diagnosis: DiagnosisOutput
    risk: RiskOutput
    confidence: float = Field(ge=0.0, le=1.0)
    modality_contributions: dict[ModalityName, float]
    warnings: list[str] = Field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="after")
    def validate_contributions(self) -> "FusionOutput":
        missing = set(ALLOWED_MODALITIES) - set(self.modality_contributions)
        if missing:
            raise ValueError(f"modality_contributions missing keys: {sorted(missing)}")
        total = sum(self.modality_contributions.values())
        if total > 0 and abs(total - 1.0) > 0.02:
            raise ValueError("nonzero modality contributions must sum to approximately 1.0")
        return self


def missing_module_output(patient_id: str, modality: ModalityName, embedding_dim: int) -> ModuleOutput:
    """Create a contract-compatible missing-modality output."""

    return ModuleOutput(
        patient_id=patient_id,
        modality=modality,
        status="missing",
        embedding=None,
        embedding_dim=embedding_dim,
        prediction=None,
        confidence=None,
        explanations={},
        warnings=[f"{modality.capitalize()} modality missing"],
        quality_flags={"input_valid": False},
        mode="missing",
    )
