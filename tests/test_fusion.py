from __future__ import annotations

import json
import unittest
from pathlib import Path

from backend.ml.common.config import load_config
from backend.ml.common.contracts import ModuleOutput
from backend.ml.fusion.inference import build_modality_mask, run_fusion
from backend.ml.phase1_intraoral_clinical.clinical_inference import run_clinical_inference
from backend.ml.phase1_intraoral_clinical.intraoral_inference import run_intraoral_inference
from backend.ml.phase2_histopathology.inference import run_histopathology_inference
from backend.ml.phase3_genomics.inference import run_genomics_inference


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

    def test_genomics_only_fusion_has_interactive_xai(self) -> None:
        patient_id = self.sample["patient_id"]
        genomics = run_genomics_inference(patient_id, self.sample["genomics"]["features"], config=self.config)
        fusion = run_fusion(patient_id, [genomics], config=self.config)
        self.assertEqual(fusion.modality_contributions["genomics"], 1.0)
        self.assertEqual(fusion.modality_contributions["clinical"], 0.0)
        self.assertIn("genomics", fusion.modality_evidence)
        self.assertTrue(fusion.decision_trace)
        self.assertIn("leave_one_out", fusion.what_if)
        self.assertIn("quality_summary", fusion.model_dump())

    def test_disabling_modality_changes_what_if_without_mutating_outputs(self) -> None:
        patient_id = self.sample["patient_id"]
        clinical = run_clinical_inference(patient_id, self.sample["clinical"], self.config)
        genomics = run_genomics_inference(patient_id, self.sample["genomics"]["features"], config=self.config)
        original_genomics_embedding = list(genomics.embedding or [])
        fusion = run_fusion(patient_id, [clinical, genomics], config=self.config, disabled_modalities=["clinical"])
        self.assertEqual(fusion.modality_contributions["clinical"], 0.0)
        self.assertGreater(fusion.modality_contributions["genomics"], 0.0)
        self.assertEqual(genomics.embedding, original_genomics_embedding)
        self.assertIn("clinical", fusion.what_if["selected_disabled_modalities"])

    def test_clinical_inference_regression_without_age_sex(self) -> None:
        patient_id = self.sample["patient_id"]
        clinical_data = {key: value for key, value in self.sample["clinical"].items() if key not in ["age", "sex"]}
        clinical_without_age_sex = run_clinical_inference(patient_id, clinical_data, self.config)

        self.assertEqual(clinical_without_age_sex.status, "available")
        feature_names = [item["feature"] for item in clinical_without_age_sex.explanations.get("top_features", [])]
        self.assertIn("tobacco_use", feature_names)
        self.assertIn("alcohol_use", feature_names)
        recommendations = clinical_without_age_sex.explanations.get("recommendations", [])
        self.assertTrue(any("Tobacco use" in item for item in recommendations))
        self.assertTrue(any("Alcohol use" in item for item in recommendations))

    def test_error_and_low_confidence_modalities_are_penalized_or_ignored(self) -> None:
        patient_id = "P-XAI"
        low_confidence = ModuleOutput(
            patient_id=patient_id,
            modality="clinical",
            status="available",
            embedding=[0.01] * 128,
            embedding_dim=128,
            prediction={"risk_score": 0.8, "diagnosis_probability": 0.8},
            confidence=0.4,
            explanations={},
            quality_flags={"low_confidence": True},
        )
        error_output = ModuleOutput(
            patient_id=patient_id,
            modality="genomics",
            status="error",
            embedding=None,
            embedding_dim=128,
            prediction=None,
            confidence=None,
            explanations={},
        )
        fusion = run_fusion(patient_id, [low_confidence, error_output], config=self.config)
        self.assertGreater(fusion.modality_contributions["clinical"], 0.0)
        self.assertEqual(fusion.modality_contributions["genomics"], 0.0)
        self.assertLess(fusion.modality_evidence["clinical"]["quality_factor"], 1.0)

    def test_all_four_modality_contract_outputs_fuse(self) -> None:
        patient_id = "P-ALL"
        outputs = [
            ModuleOutput(
                patient_id=patient_id,
                modality="intraoral",
                status="available",
                embedding=[0.1] * 256,
                embedding_dim=256,
                prediction={"risk_score": 0.2, "diagnosis_probability": 0.2},
                confidence=0.8,
                explanations={},
            ),
            ModuleOutput(
                patient_id=patient_id,
                modality="clinical",
                status="available",
                embedding=[0.1] * 128,
                embedding_dim=128,
                prediction={"risk_score": 0.9, "diagnosis_probability": 0.9},
                confidence=0.85,
                explanations={},
            ),
            ModuleOutput(
                patient_id=patient_id,
                modality="histopathology",
                status="available",
                embedding=[0.1] * 256,
                embedding_dim=256,
                prediction={"risk_score": 0.55, "diagnosis_probability": 0.55},
                confidence=0.75,
                explanations={},
            ),
            ModuleOutput(
                patient_id=patient_id,
                modality="genomics",
                status="available",
                embedding=[0.1] * 128,
                embedding_dim=128,
                prediction={"risk_score": 0.65, "diagnosis_probability": 0.65},
                confidence=0.7,
                explanations={},
            ),
        ]
        fusion = run_fusion(patient_id, outputs, config=self.config)
        self.assertAlmostEqual(sum(fusion.modality_contributions.values()), 1.0, places=2)
        self.assertIn("disagree", " ".join(fusion.warnings).lower())

    def test_fusion_exposes_gated_weighting_factors(self) -> None:
        patient_id = "P-GATED"
        outputs = [
            ModuleOutput(
                patient_id=patient_id,
                modality="clinical",
                status="available",
                embedding=[0.1] * 128,
                embedding_dim=128,
                prediction={"risk_score": 0.62, "diagnosis_probability": 0.62},
                confidence=0.82,
                explanations={},
            ),
            ModuleOutput(
                patient_id=patient_id,
                modality="genomics",
                status="available",
                embedding=[0.1] * 128,
                embedding_dim=128,
                prediction={"risk_score": 0.68, "diagnosis_probability": 0.68},
                confidence=0.78,
                explanations={},
            ),
        ]
        fusion = run_fusion(patient_id, outputs, config=self.config)
        clinical_evidence = fusion.modality_evidence["clinical"]
        self.assertGreater(clinical_evidence["signal_strength"], 0.0)
        self.assertGreater(clinical_evidence["agreement_factor"], 0.0)
        self.assertGreater(clinical_evidence["gated_weight"], 0.0)
        self.assertIn("agreement", fusion.decision_trace[0])

    def test_high_risk_guardrail_limits_dilution(self) -> None:
        patient_id = "P-GUARD"
        outputs = [
            ModuleOutput(
                patient_id=patient_id,
                modality="histopathology",
                status="available",
                embedding=[0.1] * 256,
                embedding_dim=256,
                prediction={"risk_score": 0.94, "diagnosis_probability": 0.94},
                confidence=0.95,
                explanations={},
            ),
            ModuleOutput(
                patient_id=patient_id,
                modality="clinical",
                status="available",
                embedding=[0.1] * 128,
                embedding_dim=128,
                prediction={"risk_score": 0.18, "diagnosis_probability": 0.18},
                confidence=0.8,
                explanations={},
            ),
            ModuleOutput(
                patient_id=patient_id,
                modality="genomics",
                status="available",
                embedding=[0.1] * 128,
                embedding_dim=128,
                prediction={"risk_score": 0.25, "diagnosis_probability": 0.25},
                confidence=0.76,
                explanations={},
            ),
        ]
        fusion = run_fusion(patient_id, outputs, config=self.config)
        details = fusion.model_dump().get("fusion_details", {})
        self.assertEqual(details["high_risk_guardrail"]["modality"], "histopathology")
        self.assertGreater(
            fusion.risk.model_dump(by_alias=True)["score"],
            details["high_risk_guardrail"]["original_risk_score"],
        )
        self.assertIn("guardrail", " ".join(fusion.decision_trace).lower())


if __name__ == "__main__":
    unittest.main()
