from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.common.config import load_config
from src.phase3_genomics.artifact_model import train_genomics_artifacts
from src.phase3_genomics.inference import run_genomics_from_table, run_genomics_inference


class GenomicsInferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config("configs/prototype_config.yaml")
        self.demo_config = load_config("configs/prototype_config.yaml")
        self.demo_config["genomics"]["mode"] = "demo"
        sample_path = Path("data_samples/sample_patient_001.json")
        self.sample = json.loads(sample_path.read_text(encoding="utf-8"))

    def test_available_genomics_payload_fields_and_embedding_shape(self) -> None:
        genomics = self.sample["genomics"]
        output = run_genomics_inference(
            self.sample["patient_id"],
            genomics["features"],
            sample_id=genomics["sample_id"],
            config=self.demo_config,
        )
        self.assertEqual(output.status, "available")
        self.assertEqual(output.modality, "genomics")
        self.assertEqual(output.embedding_dim, 128)
        self.assertEqual(len(output.embedding or []), 128)
        self.assertIn("risk_score", output.prediction or {})
        self.assertIn("top_features", output.explanations)

    def test_missing_genomics_is_safe(self) -> None:
        output = run_genomics_inference("P-MISSING", None, config=self.config)
        self.assertEqual(output.status, "missing")
        self.assertIsNone(output.embedding)
        self.assertEqual(output.embedding_dim, 128)

    def test_csv_ingestion(self) -> None:
        output = run_genomics_from_table("data_samples/sample_genomics.csv", patient_id="P001", config=self.demo_config)
        self.assertEqual(output.patient_id, "P001")
        self.assertEqual(output.status, "available")
        self.assertEqual(len(output.embedding or []), 128)

    def test_artifact_backed_training_and_inference(self) -> None:
        features = self.config["genomics"]["selected_gene_panel"]
        with tempfile.TemporaryDirectory() as temp_dir:
            train_path = Path(temp_dir) / "training.csv"
            artifact_path = Path(temp_dir) / "genomics_model.joblib"
            rows = ["patient_id,risk_label," + ",".join(features)]
            for index in range(30):
                label = "high" if index >= 15 else "low"
                base = 0.75 if label == "high" else 0.25
                values = [str(round(base + (feature_index * 0.01), 4)) for feature_index, _ in enumerate(features)]
                rows.append(f"P{index:03d},{label}," + ",".join(values))
            train_path.write_text("\n".join(rows), encoding="utf-8")
            train_genomics_artifacts(train_path, self.config, artifact_path)
            artifact_config = load_config("configs/prototype_config.yaml")
            artifact_config["genomics"]["artifact_path"] = str(artifact_path)
            output = run_genomics_inference(
                "P999",
                {feature: 0.82 for feature in features},
                config=artifact_config,
            )
            self.assertEqual(output.status, "available")
            self.assertEqual(output.mode, "artifact")
            self.assertEqual(len(output.embedding or []), 128)
            self.assertIn("top_features", output.explanations)


if __name__ == "__main__":
    unittest.main()
