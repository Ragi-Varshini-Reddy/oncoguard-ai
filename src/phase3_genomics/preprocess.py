"""Validation, feature mapping, imputation, and scaling for genomics features."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenomicsPreprocessResult:
    feature_order: list[str]
    raw_values: dict[str, float | None]
    imputed_values: dict[str, float]
    standardized_values: list[float]
    missing_features: list[str]
    invalid_features: list[str]
    warnings: list[str] = field(default_factory=list)
    quality_flags: dict[str, Any] = field(default_factory=dict)


class GenomicsPreprocessor:
    """Prepares model-ready gene-panel features without fitting on inference data."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.gene_panel = list(config["genomics"]["selected_gene_panel"])
        self.aliases = dict(config["genomics"].get("aliases", {}))
        self.medians = dict(config["genomics"].get("demo_medians", {}))
        self.means = dict(config["genomics"].get("demo_means", {}))
        self.stds = dict(config["genomics"].get("demo_stds", {}))
        self.missing_rate_warning_threshold = float(
            config["genomics"].get("missing_rate_warning_threshold", 0.3)
        )

    def transform(self, features: dict[str, Any]) -> GenomicsPreprocessResult:
        canonical = self._canonicalize(features)
        raw_values: dict[str, float | None] = {}
        imputed_values: dict[str, float] = {}
        standardized: list[float] = []
        missing_features: list[str] = []
        invalid_features: list[str] = []
        warnings: list[str] = []

        for feature in self.gene_panel:
            raw = canonical.get(feature)
            numeric = self._to_float(raw)
            if raw is None or raw == "":
                missing_features.append(feature)
                numeric = None
            elif numeric is None:
                invalid_features.append(feature)
                missing_features.append(feature)

            raw_values[feature] = numeric
            value = numeric if numeric is not None else float(self.medians.get(feature, 0.0))
            imputed_values[feature] = value

            mean = float(self.means.get(feature, 0.0))
            std = float(self.stds.get(feature, 1.0)) or 1.0
            standardized.append((value - mean) / std)

        missing_rate = len(missing_features) / max(1, len(self.gene_panel))
        used_imputation = bool(missing_features)
        out_of_distribution = any(abs(value) > 5.0 for value in standardized)

        if used_imputation:
            warnings.append(f"Imputed {len(missing_features)} missing genomic feature(s)")
        if invalid_features:
            warnings.append(f"Invalid genomic values treated as missing: {', '.join(invalid_features)}")
        if missing_rate > self.missing_rate_warning_threshold:
            warnings.append("High genomic missingness; confidence reduced")
        if out_of_distribution:
            warnings.append("One or more genomic values are outside the demo training range")

        return GenomicsPreprocessResult(
            feature_order=self.gene_panel,
            raw_values=raw_values,
            imputed_values=imputed_values,
            standardized_values=standardized,
            missing_features=missing_features,
            invalid_features=invalid_features,
            warnings=warnings,
            quality_flags={
                "input_valid": not invalid_features,
                "missing_feature_count": len(missing_features),
                "missing_feature_rate": round(missing_rate, 4),
                "used_imputation": used_imputation,
                "out_of_distribution": out_of_distribution,
                "artifact_loaded": False,
            },
        )

    def _canonicalize(self, features: dict[str, Any]) -> dict[str, Any]:
        canonical: dict[str, Any] = {}
        for key, value in features.items():
            canonical_key = self.aliases.get(key, key)
            canonical[canonical_key] = value
        return canonical

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
