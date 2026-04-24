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


class ReportRequest(BaseModel):
    patient_id: str = Field(min_length=1)
    module_outputs: list[dict[str, Any]]
    fusion_output: dict[str, Any]
