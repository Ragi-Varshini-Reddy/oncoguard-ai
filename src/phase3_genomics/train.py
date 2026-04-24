"""Genomics training entry points."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any

from src.common.config import load_config
from src.phase3_genomics.artifact_model import train_genomics_artifacts
from src.phase3_genomics.feature_selection import selected_feature_order


def save_feature_order(artifact_dir: str | Path = "artifacts", config: dict[str, Any] | None = None) -> Path:
    cfg = config or load_config()
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "genomics_feature_order.json"
    output_path.write_text(
        json.dumps({"schema_version": "1.0", "feature_order": selected_feature_order(cfg)}, indent=2),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train real artifact-backed genomics model")
    parser.add_argument("--input", help="CSV/parquet training table with patient_id, label, and gene-panel columns")
    parser.add_argument("--artifact", default=None, help="Output joblib artifact path")
    parser.add_argument("--feature-order-only", action="store_true", help="Only write the configured feature-order artifact")
    args = parser.parse_args()
    config = load_config()
    if args.feature_order_only or not args.input:
        print(save_feature_order(config=config))
        return
    artifact_path, summary = train_genomics_artifacts(args.input, config, args.artifact)
    print(json.dumps({"artifact_path": str(artifact_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
