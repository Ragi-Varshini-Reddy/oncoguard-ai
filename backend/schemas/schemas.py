"""API request and response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GenomicsJsonRequest(BaseModel):
    patient_id: str = Field(min_length=1)
    sample_id: str | None = None
    genomic_features: dict[str, Any]


class ClinicalDataRequest(BaseModel):
    age: float | None = None
    sex: str | None = None
    gender: str | None = None
    tobacco_use: bool = False
    alcohol_use: bool = False
    lesion_site: str | None = None
    lesion_size_cm: float | None = None
    persistent_ulcer_weeks: float | None = None
    neck_node_present: bool = False
    poor_oral_hygiene: bool = False
    family_history: bool = False


class FusionRequest(BaseModel):
    patient_id: str = Field(min_length=1)
    module_outputs: list[dict[str, Any]]
    modality_mask: dict[str, bool] | None = None
    disabled_modalities: list[str] = Field(default_factory=list)


class ReportRequest(BaseModel):
    patient_id: str = Field(min_length=1)
    module_outputs: list[dict[str, Any]]
    fusion_output: dict[str, Any]
    report_text: str | None = Field(default=None, max_length=8000)
    doctor_notes: str | None = Field(default=None, max_length=2000)


class ReportApprovalRequest(BaseModel):
    report_text: str = Field(min_length=1, max_length=8000)
    approval_status: str = Field(default="approved", pattern="^(draft|approved|rejected)$")
    doctor_notes: str | None = Field(default=None, max_length=2000)


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


class NewPatientRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    age: int | None = None
    sex: str | None = Field(default=None, max_length=20)
    summary: str | None = None
    email: str | None = None
    phone: str | None = None

