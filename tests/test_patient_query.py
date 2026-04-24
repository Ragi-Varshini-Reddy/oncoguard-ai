from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.ml.common.contracts import ModuleOutput
from backend.ml.explainability.llm_provider import LLMResult
from backend.ml.explainability.patient_query import answer_patient_query
from backend.ml.fusion.inference import run_fusion


class PatientQueryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.module_output = ModuleOutput(
            patient_id="P-Q",
            modality="genomics",
            status="available",
            embedding=[0.1] * 128,
            embedding_dim=128,
            prediction={"risk_score": 0.7, "diagnosis_probability": 0.68},
            confidence=0.78,
            explanations={
                "top_features": [
                    {"feature": "TP53_expr", "direction": "increases_risk", "importance_score": 0.4}
                ]
            },
        )
        self.fusion = run_fusion("P-Q", [self.module_output])

    def test_patient_query_answers_from_available_context(self) -> None:
        from backend.app import main

        original_enabled = main.CONFIG.get("llm", {}).get("enabled")
        main.CONFIG.setdefault("llm", {})["enabled"] = False
        self.addCleanup(lambda: main.CONFIG["llm"].update({"enabled": original_enabled}))
        response = self.client.post(
            "/api/patient/query",
            json={
                "patient_id": "P-Q",
                "query": "What is my health status and why?",
                "module_outputs": [self.module_output.model_dump()],
                "fusion_output": self.fusion.model_dump(by_alias=True),
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("answer", body)
        self.assertIn("disclaimer", body)
        self.assertIn("genomics", body["sources_used"])
        self.assertIn("fusion", body["sources_used"])
        self.assertEqual(body["answer_mode"], "rule_based")

    def test_llm_patient_query_path_uses_provider(self) -> None:
        config = {"llm": {"enabled": True, "allow_rule_based_fallback": True, "provider": "ollama"}}
        with patch(
            "src.explainability.patient_query.generate_patient_answer",
            return_value=LLMResult(text="LLM grounded answer", provider="ollama", model="llama3.1:8b"),
        ):
            body = answer_patient_query(
                "P-Q",
                "What should I ask my doctor?",
                [self.module_output],
                self.fusion,
                config=config,
                use_llm=True,
            )
        self.assertEqual(body["answer"], "LLM grounded answer")
        self.assertEqual(body["answer_mode"], "llm")
        self.assertEqual(body["llm_model"], "llama3.1:8b")

    def test_patient_chat_stores_compact_history(self) -> None:
        from backend.app import main

        original_enabled = main.CONFIG.get("llm", {}).get("enabled")
        main.CONFIG.setdefault("llm", {})["enabled"] = False
        self.addCleanup(lambda: main.CONFIG["llm"].update({"enabled": original_enabled}))
        first = self.client.post(
            "/api/patient/chat",
            json={
                "patient_id": "P-Q",
                "message": "Hi",
                "use_llm": False,
                "module_outputs": [self.module_output.model_dump()],
                "fusion_output": self.fusion.model_dump(by_alias=True),
            },
        )
        self.assertEqual(first.status_code, 200)
        first_body = first.json()
        self.assertIn("session_id", first_body)
        self.assertEqual(len(first_body["messages"]), 2)

        second = self.client.post(
            "/api/patient/chat",
            json={
                "patient_id": "P-Q",
                "session_id": first_body["session_id"],
                "message": "Why?",
                "use_llm": False,
                "module_outputs": [self.module_output.model_dump()],
                "fusion_output": self.fusion.model_dump(by_alias=True),
            },
        )
        self.assertEqual(second.status_code, 200)
        second_body = second.json()
        self.assertEqual(second_body["session_id"], first_body["session_id"])
        self.assertEqual(len(second_body["messages"]), 4)

    def test_greeting_does_not_dump_medical_context(self) -> None:
        response = self.client.post(
            "/api/patient/chat",
            json={
                "patient_id": "P-GREET",
                "message": "hey",
                "use_llm": True,
                "module_outputs": [self.module_output.model_dump()],
                "fusion_output": self.fusion.model_dump(by_alias=True),
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["intent"], "greeting")
        self.assertNotIn("medium risk", body["answer"].lower())
        self.assertIn("ask", body["answer"].lower())


if __name__ == "__main__":
    unittest.main()
