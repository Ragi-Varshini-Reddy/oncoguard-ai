from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.common.config import load_config
from src.common.contracts import ModuleOutput


class ConfigAndContractsTest(unittest.TestCase):
    def test_config_loading(self) -> None:
        config = load_config("configs/prototype_config.yaml")
        self.assertEqual(config["project"]["name"], "OralCare-AI")
        self.assertEqual(config["genomics"]["embedding_dim"], 128)
        self.assertIn("TP53_expr", config["genomics"]["selected_gene_panel"])

    def test_module_output_contract_validation(self) -> None:
        output = ModuleOutput(
            patient_id="P001",
            modality="genomics",
            status="available",
            embedding=[0.1] * 128,
            embedding_dim=128,
            prediction={"risk_score": 0.7},
            confidence=0.82,
            explanations={},
        )
        self.assertEqual(output.modality, "genomics")
        self.assertEqual(len(output.embedding or []), 128)

    def test_contract_rejects_bad_embedding_shape(self) -> None:
        with self.assertRaises(ValidationError):
            ModuleOutput(
                patient_id="P001",
                modality="genomics",
                status="available",
                embedding=[0.1],
                embedding_dim=128,
                prediction={},
                confidence=0.8,
                explanations={},
            )


if __name__ == "__main__":
    unittest.main()
