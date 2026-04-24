from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from backend.main import app
from backend.ml.common.contracts import ModuleOutput


class FusionApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.output = ModuleOutput(
            patient_id="P-API",
            modality="genomics",
            status="available",
            embedding=[0.1] * 128,
            embedding_dim=128,
            prediction={"risk_score": 0.72, "diagnosis_probability": 0.72},
            confidence=0.8,
            explanations={"top_features": [{"feature": "TP53_expr", "importance_score": 0.4}]},
        ).model_dump()

    def test_fusion_infer_backward_compatible(self) -> None:
        response = self.client.post(
            "/api/fusion/infer",
            json={"patient_id": "P-API", "module_outputs": [self.output]},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("diagnosis", body)
        self.assertIn("risk", body)
        self.assertIn("modality_contributions", body)

    def test_fusion_explain_returns_xai_fields(self) -> None:
        response = self.client.post(
            "/api/fusion/explain",
            json={"patient_id": "P-API", "module_outputs": [self.output]},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("modality_evidence", body)
        self.assertIn("decision_trace", body)
        self.assertIn("what_if", body)
        self.assertIn("quality_summary", body)

    def test_invalid_modality_mask_returns_400(self) -> None:
        response = self.client.post(
            "/api/fusion/explain",
            json={
                "patient_id": "P-API",
                "module_outputs": [self.output],
                "modality_mask": {"genomics": True},
            },
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
