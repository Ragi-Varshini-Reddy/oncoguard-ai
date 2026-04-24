"""Typed request schema for genomics inference."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class GenomicsInferenceRequest(BaseModel):
    patient_id: str = Field(min_length=1)
    genomic_features: dict[str, float | int | str | None] | None = None
    sample_id: str | None = None
    source: str = "demo"

    @field_validator("genomic_features")
    @classmethod
    def validate_feature_mapping(
        cls, value: dict[str, float | int | str | None] | None
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("genomic_features must be a key-value mapping")
        return value
