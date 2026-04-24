"""Config loading with a dependency-light fallback for hackathon environments."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "name": "OralCare-AI",
        "schema_version": "1.0",
        "random_seed": 468,
        "demo_mode": True,
        "disclaimer": "Decision-support prototype only. Not a final medical diagnosis or clinically validated risk tool.",
    },
    "modalities": {
        "intraoral": {"embedding_dim": 256},
        "clinical": {"embedding_dim": 128},
        "histopathology": {"embedding_dim": 256},
        "genomics": {"embedding_dim": 128},
    },
    "labels": {
        "diagnosis_classes": ["benign", "precancer", "cancer"],
        "risk_classes": {
            "low": {"min": 0.0, "max": 0.39},
            "medium": {"min": 0.4, "max": 0.69},
            "high": {"min": 0.7, "max": 1.0},
        },
    },
    "genomics": {
        "mode": "demo",
        "embedding_dim": 128,
        "top_k_features": 5,
        "missing_rate_warning_threshold": 0.3,
        "low_confidence_threshold": 0.6,
        "high_risk_threshold": 0.7,
        "medium_risk_threshold": 0.4,
        "selected_gene_panel": [
            "TP53_expr",
            "CDKN2A_expr",
            "EGFR_expr",
            "PIK3CA_expr",
            "NOTCH1_expr",
            "CCND1_expr",
            "FAT1_expr",
            "CASP8_expr",
            "HRAS_expr",
            "MET_expr",
            "MYC_expr",
            "MDM2_expr",
        ],
        "aliases": {
            "tp53": "TP53_expr",
            "TP53": "TP53_expr",
            "cdkn2a": "CDKN2A_expr",
            "EGFR Expression": "EGFR_expr",
            "egfr": "EGFR_expr",
        },
        "demo_medians": {
            "TP53_expr": 0.52,
            "CDKN2A_expr": 0.42,
            "EGFR_expr": 0.58,
            "PIK3CA_expr": 0.46,
            "NOTCH1_expr": 0.38,
            "CCND1_expr": 0.5,
            "FAT1_expr": 0.44,
            "CASP8_expr": 0.35,
            "HRAS_expr": 0.3,
            "MET_expr": 0.48,
            "MYC_expr": 0.55,
            "MDM2_expr": 0.41,
        },
        "demo_means": {
            "TP53_expr": 0.52,
            "CDKN2A_expr": 0.42,
            "EGFR_expr": 0.58,
            "PIK3CA_expr": 0.46,
            "NOTCH1_expr": 0.38,
            "CCND1_expr": 0.5,
            "FAT1_expr": 0.44,
            "CASP8_expr": 0.35,
            "HRAS_expr": 0.3,
            "MET_expr": 0.48,
            "MYC_expr": 0.55,
            "MDM2_expr": 0.41,
        },
        "demo_stds": {
            "TP53_expr": 0.18,
            "CDKN2A_expr": 0.16,
            "EGFR_expr": 0.2,
            "PIK3CA_expr": 0.17,
            "NOTCH1_expr": 0.14,
            "CCND1_expr": 0.19,
            "FAT1_expr": 0.16,
            "CASP8_expr": 0.13,
            "HRAS_expr": 0.12,
            "MET_expr": 0.16,
            "MYC_expr": 0.21,
            "MDM2_expr": 0.15,
        },
        "feature_descriptions": {
            "TP53_expr": "Model-ready TP53 expression signal",
            "CDKN2A_expr": "Model-ready CDKN2A expression signal",
            "EGFR_expr": "Model-ready EGFR expression signal",
            "PIK3CA_expr": "Model-ready PIK3CA expression signal",
            "NOTCH1_expr": "Model-ready NOTCH1 expression signal",
            "CCND1_expr": "Model-ready CCND1 expression signal",
            "FAT1_expr": "Model-ready FAT1 expression signal",
            "CASP8_expr": "Model-ready CASP8 expression signal",
            "HRAS_expr": "Model-ready HRAS expression signal",
            "MET_expr": "Model-ready MET expression signal",
            "MYC_expr": "Model-ready MYC expression signal",
            "MDM2_expr": "Model-ready MDM2 expression signal",
        },
    },
    "fusion": {"hidden_dim": 256, "confidence_floor": 0.35, "low_confidence_threshold": 0.6},
    "reporting": {"report_title": "OralCare-AI Multimodal Decision-Support Report"},
}


def load_config(config_path: str | Path = "configs/prototype_config.yaml") -> dict[str, Any]:
    """Load YAML config, falling back to the built-in prototype config if PyYAML is absent."""

    path = Path(config_path)
    if not path.exists():
        return copy.deepcopy(DEFAULT_CONFIG)
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return copy.deepcopy(DEFAULT_CONFIG)

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    merged = copy.deepcopy(DEFAULT_CONFIG)
    _deep_update(merged, loaded)
    return merged


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
