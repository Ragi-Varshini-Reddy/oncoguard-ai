from backend.db.db import connect, utc_now
import uuid
from datetime import datetime, timedelta

def seed_varshini():
    doctor_id = "DR-RAO"
    lab_tech_id = "LABTECH-001"
    patient_name = "Varshini"
    patient_id = "VAR-2024-001"
    user_id = "USR-VARSHINI"

    now_dt = datetime.now()
    start_date = datetime(2026, 4, 1)
    end_date = now_dt
    days_count = (end_date - start_date).days + 1

    with connect() as cursor:
        # Create User
        cursor.execute(
            "INSERT IGNORE INTO users (user_id, role, name, email, phone, created_at) VALUES (%s, %s, %s, %s, %s, %s)",
            (user_id, 'patient', patient_name, 'varshini@example.local', '+91-9876543210', utc_now())
        )
        
        # Create Patient
        cursor.execute(
            "INSERT IGNORE INTO patients (patient_id, user_id, doctor_id, lab_technician_id, age, sex, summary, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (patient_id, user_id, doctor_id, lab_tech_id, 24, 'female', 'Monitoring rising lesion risk on lateral tongue border.', start_date.isoformat())
        )

        # Clear existing history for this patient if any to start fresh
        cursor.execute("DELETE FROM risk_history WHERE patient_id = %s", (patient_id,))
        cursor.execute("DELETE FROM model_runs WHERE patient_id = %s", (patient_id,))
        cursor.execute("DELETE FROM patient_documents WHERE patient_id = %s", (patient_id,))

        start_risk = 0.30
        end_risk = 0.45

        for i in range(days_count):
            current_dt = start_date + timedelta(days=i)
            # Linearly interpolate risk
            if days_count > 1:
                risk_score = start_risk + (end_risk - start_risk) * (i / (days_count - 1))
            else:
                risk_score = start_risk
            
            # Add some slight jitter
            import random
            risk_score += random.uniform(-0.01, 0.01)
            risk_score = max(0.0, min(1.0, risk_score))
            
            risk_level = "high" if risk_score >= 0.7 else "medium" if risk_score >= 0.4 else "low"
            ts = current_dt.isoformat()
            
            # Create Document
            doc_id = f"DOC-V-{i}"
            cursor.execute(
                "INSERT INTO patient_documents (document_id, patient_id, uploader_user_id, document_type, filename, storage_path, notes, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (doc_id, patient_id, 'LAB-001', 'intraoral', f"intraoral_scan_{i}.jpg", f"/mock/path/varshini/scan_{i}.jpg", "Routine intraoral check", ts)
            )

            # Create Model Run
            run_id = f"RUN-V-{i}"
            module_outputs = [
                {
                    "patient_id": patient_id,
                    "modality": "intraoral",
                    "status": "available",
                    "embedding": [0.1] * 128,  # Mock embedding
                    "embedding_dim": 128,
                    "prediction": {"risk_score": risk_score, "risk_class": risk_level},
                    "confidence": 0.85 + random.uniform(0, 0.1),
                    "explanations": {"top_features": [{"feature": "Surface Color", "importance_score": 0.4, "direction": "Increased vascularity"}]}
                }
            ]
            fusion_output = {
                "patient_id": patient_id,
                "risk": {"score": risk_score, "class": risk_level},
                "confidence": 0.88,
                "diagnosis": {"class": "OSCC" if risk_score > 0.6 else "Leukoplakia", "probability": risk_score},
                "modality_contributions": {"intraoral": 1.0},
                "decision_trace": ["Image features indicate evolving lesion", "Risk score interpolated for demo"]
            }
            
            import json
            cursor.execute(
                "INSERT INTO model_runs (run_id, patient_id, created_by_user_id, module_outputs_json, fusion_output_json, created_at) VALUES (%s, %s, %s, %s, %s, %s)",
                (run_id, patient_id, 'LAB-001', json.dumps(module_outputs), json.dumps(fusion_output), ts)
            )

            # Create Risk History
            history_id = f"HIST-V-{i}"
            cursor.execute(
                "INSERT INTO risk_history (history_id, patient_id, run_id, risk_score, risk_level, confidence, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (history_id, patient_id, run_id, risk_score, risk_level, 0.88, ts)
            )

    print(f"Successfully seeded data for {patient_name} from {start_date.date()} to {end_date.date()}")

if __name__ == "__main__":
    seed_varshini()
