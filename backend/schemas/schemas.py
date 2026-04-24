"""API request and response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GenomicsJsonRequest(BaseModel):
    patient_id: str = Field(min_length=1)
    sample_id: str | None = None
    genomic_features: dict[str, Any]


class FusionRequest(BaseModel):
    patient_id: str = Field(min_length=1)
    module_outputs: list[dict[str, Any]]
    modality_mask: dict[str, bool] | None = None
    disabled_modalities: list[str] = Field(default_factory=list)


class ReportRequest(BaseModel):
    patient_id: str = Field(min_length=1)
    module_outputs: list[dict[str, Any]]
    fusion_output: dict[str, Any]


class PatientQueryRequest(BaseModel):
    patient_id: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=1000)
    module_outputs: list[dict[str, Any]]
    fusion_output: dict[str, Any] | None = None
    use_llm: bool = True


class PatientChatRequest(BaseModel):
    patient_id: str = Field(min_length=1)
    session_id: str | None = None
    message: str = Field(min_length=1, max_length=1000)
    module_outputs: list[dict[str, Any]]
    fusion_output: dict[str, Any] | None = None
    use_llm: bool = True


class DemoLoginRequest(BaseModel):
    user_id: str = Field(min_length=1)


class AppointmentRequest(BaseModel):
    requested_date: str = Field(min_length=1, max_length=80)
    issue: str = Field(min_length=1, max_length=160)
    reason: str = Field(min_length=1, max_length=1000)


class AppointmentUpdate(BaseModel):
    status: str = Field(pattern="^(requested|scheduled|completed|cancelled)$")
    requested_date: str | None = Field(default=None, max_length=80)
    doctor_notes: str | None = Field(default=None, max_length=1000)
