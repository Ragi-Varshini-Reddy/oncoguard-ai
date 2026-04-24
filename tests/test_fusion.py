from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.common.config import load_config
from src.fusion.inference import build_modality_mask, run_fusion
from src.phase1_intraoral_clinical.clinical_inference import run_clinical_inference
from src.phase1_intraoral_clinical.intraoral_inference import run_intraoral_inference
from src.phase2_histopathology.inference import run_histopathology_inference
from src.phase3_genomics.inference import run_genomics_inference


class FusionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config("configs/prototype_config.yaml")
        self.config["genomics"]["mode"] = "demo"
        self.sample = json.loads(Path("data_samples/sample_patient_001.json").read_text(encoding="utf-8"))

    def test_fusion_mask_handling(self) -> None:
        patient_id = self.sample["patient_id"]
        outputs = [
            run_intraoral_inference(patient_id, None, self.config),
            run_clinical_inference(patient_id, self.sample["clinical"], self.config),
            run_histopathology_inference(patient_id, None, self.config),
            run_genomics_inference(patient_id, self.sample["genomics"]["features"], config=self.config),
        ]
        mask = build_modality_mask(outputs)
        self.assertFalse(mask["intraoral"])
        self.assertTrue(mask["clinical"])
        self.assertFalse(mask["histopathology"])
        self.assertTrue(mask["genomics"])

        fusion = run_fusion(patient_id, outputs, modality_mask=mask, config=self.config)
        self.assertEqual(fusion.modality_contributions["intraoral"], 0.0)
        self.assertEqual(fusion.modality_contributions["histopathology"], 0.0)
        self.assertGreater(fusion.modality_contributions["genomics"], 0.0)
        self.assertIn(fusion.risk.class_, {"low", "medium", "high"})


if __name__ == "__main__":
    unittest.main()
