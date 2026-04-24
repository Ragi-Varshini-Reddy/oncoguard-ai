from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from backend.main import app


class RoleWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_demo_login_returns_linked_profile(self) -> None:
        response = self.client.post("/api/auth/demo-login", json={"user_id": "DOC-001"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["user"]["role"], "doctor")
        self.assertEqual(body["profile"]["doctor_id"], "DR-RAO")

    def test_doctor_can_list_assigned_patients(self) -> None:
        response = self.client.get("/api/doctors/DR-RAO/patients", headers={"X-User-Id": "DOC-001"})
        self.assertEqual(response.status_code, 200)
        patients = response.json()["patients"]
        self.assertGreaterEqual(len(patients), 1)
        self.assertTrue(any(patient["patient_id"] == "TCGA-BB-4227" for patient in patients))

    def test_patient_cannot_access_another_patient(self) -> None:
        response = self.client.get("/api/patients/TCGA-CV-5441", headers={"X-User-Id": "PAT-001"})
        self.assertEqual(response.status_code, 403)

    def test_lab_can_upload_document_by_patient_id_but_not_read_dashboard(self) -> None:
        denied = self.client.get("/api/patients/TCGA-BB-4227", headers={"X-User-Id": "LAB-001"})
        self.assertEqual(denied.status_code, 403)

        uploaded = self.client.post(
            "/api/patients/TCGA-BB-4227/documents",
            headers={"X-User-Id": "LAB-001"},
            data={"document_type": "genomics", "notes": "unit test upload"},
            files={"document_file": ("test.csv", b"patient_id,TP53_expr\nTCGA-BB-4227,1.2\n", "text/csv")},
        )
        self.assertEqual(uploaded.status_code, 200)
        self.assertEqual(uploaded.json()["patient_id"], "TCGA-BB-4227")

    def test_patient_can_request_and_doctor_can_schedule_appointment(self) -> None:
        requested = self.client.post(
            "/api/patients/TCGA-BB-4227/appointments/request",
            headers={"X-User-Id": "PAT-001"},
            json={"requested_date": "2026-04-30 10:00", "issue": "Mouth ulcer", "reason": "Follow-up discussion"},
        )
        self.assertEqual(requested.status_code, 200)
        appointment_id = requested.json()["appointment_id"]

        scheduled = self.client.patch(
            f"/api/appointments/{appointment_id}",
            headers={"X-User-Id": "DOC-001"},
            json={"status": "scheduled", "requested_date": "2026-04-30 10:30", "doctor_notes": "Confirmed"},
        )
        self.assertEqual(scheduled.status_code, 200)
        self.assertEqual(scheduled.json()["status"], "scheduled")


if __name__ == "__main__":
    unittest.main()
