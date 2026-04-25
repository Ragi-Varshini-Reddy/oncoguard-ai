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
        self.clinical_module_output = ModuleOutput(
            patient_id="P-Q",
            modality="clinical",
            status="available",
            embedding=[0.2] * 128,
            embedding_dim=128,
            prediction={"risk_score": 0.74, "diagnosis_probability": 0.71},
            confidence=0.81,
            explanations={
                "feature_values": {
                    "age": 54,
                    "gender": "male",
                    "tobacco_use": True,
                    "alcohol_use": False,
                    "poor_oral_hygiene": True,
                },
                "top_features": [
                    {
                        "feature": "tobacco_use",
                        "direction": "increases_risk",
                        "importance_score": 0.28,
                        "shap_value": 0.28,
                    },
                    {
                        "feature": "poor_oral_hygiene",
                        "direction": "increases_risk",
                        "importance_score": 0.12,
                        "shap_value": 0.12,
                    },
                ],
                "recommendations": ["Tobacco use is a positive SHAP driver here; offer cessation counseling."],
            },
        )
        self.clinical_fusion = run_fusion("P-Q", [self.module_output, self.clinical_module_output])

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

    def test_patient_query_includes_shap_risk_change_suggestion(self) -> None:
        body = answer_patient_query(
            "P-Q",
            "What should I do now?",
            [self.module_output, self.clinical_module_output],
            self.clinical_fusion,
            config={"llm": {"enabled": False}},
            use_llm=False,
        )
        self.assertIn("Current risk is about", body["answer"])
        self.assertIn("Quit smoking", body["answer"])
        self.assertIn("illustrative SHAP-adjusted risk", body["answer"])
        self.assertIn("Build a daily oral self-check", body["answer"])

    def test_patient_query_skips_habit_change_without_bad_habit(self) -> None:
        clean_clinical_output = ModuleOutput(
            patient_id="P-Q",
            modality="clinical",
            status="available",
            embedding=[0.15] * 128,
            embedding_dim=128,
            prediction={"risk_score": 0.42, "diagnosis_probability": 0.38},
            confidence=0.64,
            explanations={
                "feature_values": {
                    "age": 41,
                    "gender": "female",
                    "tobacco_use": False,
                    "alcohol_use": False,
                    "poor_oral_hygiene": False,
                },
                "top_features": [
                    {
                        "feature": "poor_oral_hygiene",
                        "direction": "increases_risk",
                        "importance_score": 0.16,
                        "shap_value": 0.16,
                    }
                ],
                "recommendations": [],
            },
        )
        clean_fusion = run_fusion("P-Q", [self.module_output, clean_clinical_output])
        body = answer_patient_query(
            "P-Q",
            "What should I do now?",
            [self.module_output, clean_clinical_output],
            clean_fusion,
            config={"llm": {"enabled": False}},
            use_llm=False,
        )
        self.assertNotIn("Quit smoking", body["answer"])
        self.assertNotIn("Reduce or stop alcohol", body["answer"])
        self.assertIn("Build a daily oral self-check", body["answer"])

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
