from __future__ import annotations

import unittest

from backend.ml.phase1_intraoral_clinical.clinical_inference import run_clinical_inference
from backend.ml.common.config import load_config


class ClinicalInferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config("configs/prototype_config.yaml")

    def test_clinical_inference_handles_missing_age_and_sex(self) -> None:
        output = run_clinical_inference(
            "P-CLINICAL",
            {
                "tobacco_use": True,
                "alcohol_use": True,
                "lesion_site": "lateral tongue",
                "lesion_size_cm": 2.4,
                "persistent_ulcer_weeks": 7,
                "neck_node_present": True,
                "poor_oral_hygiene": False,
                "family_history": False,
            },
            self.config,
        )

        self.assertEqual(output.status, "available")
        feature_names = [item["feature"] for item in output.explanations.get("top_features", [])]
        self.assertIn("tobacco_use", feature_names)
        self.assertIn("alcohol_use", feature_names)
        recommendations = output.explanations.get("recommendations", [])
        self.assertTrue(any("Tobacco use" in item for item in recommendations))
        self.assertTrue(any("Alcohol use" in item for item in recommendations))


if __name__ == "__main__":
    unittest.main()
