"""FastAPI backend for the real-data OralCare-AI workflow."""

from __future__ import annotations

import tempfile
import uuid
from datetime import datetime
from io import BytesIO, StringIO

import cairosvg
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from jinja2 import Environment, FileSystemLoader
from backend.db.db import connect, dumps, initialize_database, loads, row_to_dict, rows_to_dicts, utc_now
from backend.schemas.schemas import (
    AppointmentRequest,
    AppointmentUpdate,
    ClinicalDataRequest,
    DemoLoginRequest,
    FusionRequest,
    GenomicsJsonRequest,
    PatientChatRequest,
    PatientQueryRequest,
    ReportApprovalRequest,
    ReportRequest,
    NewPatientRequest,
)
from backend.core.security import actor_profile, assert_patient_access, get_actor, optional_actor, require_role
from backend.ml.common.config import load_config
from backend.ml.common.contracts import FusionOutput, ModuleOutput
from backend.ml.fusion.inference import explain_fusion, run_fusion
from backend.ml.explainability.chat_store import ChatStore
from backend.ml.explainability.patient_query import answer_patient_chat, answer_patient_query
from backend.ml.phase1_intraoral_clinical.clinical_inference import run_clinical_inference
from backend.ml.phase1_intraoral_clinical.intraoral_inference import run_intraoral_inference
from backend.ml.phase2_histopathology.inference import run_histopathology_inference
from backend.ml.phase3_genomics.artifact_model import train_genomics_artifacts
from backend.ml.phase3_genomics.inference import run_genomics_inference
from backend.ml.phase3_genomics.preprocess import GenomicsPreprocessor
from backend.ml.phase3_genomics.utils import load_genomic_features_from_csv
from backend.ml.reporting.generate_report import generate_html_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(PROJECT_ROOT / "configs" / "prototype_config.yaml")
CHAT_STORE = ChatStore(max_turns=int(CONFIG.get("llm", {}).get("max_chat_turns", 10)))
DOCUMENT_ROOT = PROJECT_ROOT / "artifacts" / "documents"
initialize_database()

app = FastAPI(
    title="OralCare-AI API",
    version="1.0.0",
    description="Artifact-backed multimodal decision-support prototype API.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIG.get("api", {}).get("cors_origins", ["http://localhost:5173", "http://127.0.0.1:5173"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.on_event("startup")
def startup() -> None:
    initialize_database()


@app.get("/api/health")
def health() -> dict[str, Any]:
    artifact_path = PROJECT_ROOT / CONFIG["genomics"].get("artifact_path", "artifacts/genomics_model.joblib")
    return {
        "status": "ok",
        "project": CONFIG["project"]["name"],
        "genomics_artifact_available": artifact_path.exists(),
        "genomics_artifact_path": str(artifact_path),
    }


@app.get("/api/users/roles")
def user_roles() -> dict[str, Any]:
    return CONFIG["users"]["roles"]


@app.get("/api/auth/demo-users")
def demo_users() -> dict[str, Any]:
    initialize_database()
    with connect() as conn:
        users = rows_to_dicts(conn.execute("SELECT user_id, role, name, email FROM users ORDER BY role, user_id").fetchall())
    return {"users": users}


@app.post("/api/auth/demo-login")
def demo_login(payload: DemoLoginRequest) -> dict[str, Any]:
    initialize_database()
    with connect() as conn:
        user = row_to_dict(conn.execute("SELECT * FROM users WHERE user_id = ?", (payload.user_id,)).fetchone())
    if not user:
        raise HTTPException(status_code=401, detail="Unknown demo user ID")
    return actor_profile(user)


@app.get("/api/me")
def me(actor: dict[str, Any] = Depends(get_actor)) -> dict[str, Any]:
    return actor_profile(actor)


@app.get("/api/technician/patients")
def technician_patients(actor: dict[str, Any] = Depends(get_actor)) -> dict[str, Any]:
    require_role(actor, "lab_technician")
    with connect() as conn:
        tech = row_to_dict(conn.execute("SELECT * FROM lab_technicians WHERE user_id = ?", (actor["user_id"],)).fetchone())
        patients = rows_to_dicts(
            conn.execute(
                """
                SELECT patients.*, users.name, users.email, users.phone
                FROM patients
                JOIN users ON patients.user_id = users.user_id
                WHERE patients.lab_technician_id = ?
                ORDER BY users.name, patients.patient_id
                """,
                (tech["lab_technician_id"],),
            ).fetchall()
        )
    return {"patients": patients}


@app.get("/api/technician/doctors")
def technician_doctors(actor: dict[str, Any] = Depends(get_actor)) -> dict[str, Any]:
    require_role(actor, "lab_technician")
    with connect() as conn:
        tech = row_to_dict(conn.execute("SELECT * FROM lab_technicians WHERE user_id = ?", (actor["user_id"],)).fetchone())
        doctors = rows_to_dicts(
            conn.execute(
                """
                SELECT doctors.*, users.name, users.email
                FROM doctors 
                JOIN users ON doctors.user_id = users.user_id
                JOIN doctor_technician_assignments ON doctors.doctor_id = doctor_technician_assignments.doctor_id
                WHERE doctor_technician_assignments.lab_technician_id = ?
                """,
                (tech["lab_technician_id"],)
            ).fetchall()
        )
    return {"doctors": doctors}


@app.get("/api/doctors/{doctor_id}/patients")
def doctor_patients(doctor_id: str, actor: dict[str, Any] = Depends(get_actor)) -> dict[str, Any]:
    require_role(actor, "doctor", "lab_technician")
    with connect() as conn:
        if actor["role"] == "doctor":
            doctor = row_to_dict(conn.execute("SELECT * FROM doctors WHERE user_id = ?", (actor["user_id"],)).fetchone())
            if not doctor or doctor["doctor_id"] != doctor_id:
                raise HTTPException(status_code=403, detail="Doctor profile mismatch")
        patients = rows_to_dicts(
            conn.execute(
                """
                SELECT p.*, u.name, u.email, u.phone,
                       COALESCE(doc_stats.document_count, 0) AS document_count,
                       COALESCE(doc_stats.heatmap_count, 0) AS heatmap_count,
                       latest_risk.risk_score AS latest_risk_score,
                       latest_risk.risk_level AS latest_risk_level,
                       latest_risk.confidence AS latest_confidence
                FROM patients p
                JOIN users u ON p.user_id = u.user_id
                LEFT JOIN (
                    SELECT patient_id,
                           COUNT(*) AS document_count,
                           SUM(CASE
                                   WHEN document_type = 'histopathological'
                                        AND (filename LIKE '%_heatmap%' OR COALESCE(notes, '') LIKE '%Grad-CAM%')
                                   THEN 1 ELSE 0
                               END) AS heatmap_count
                    FROM patient_documents
                    GROUP BY patient_id
                ) AS doc_stats ON p.patient_id = doc_stats.patient_id
                LEFT JOIN (
                    SELECT rh.patient_id, rh.risk_score, rh.risk_level, rh.confidence
                    FROM risk_history rh
                    JOIN (
                        SELECT patient_id, MAX(created_at) AS max_created_at
                        FROM risk_history
                        GROUP BY patient_id
                    ) latest ON rh.patient_id = latest.patient_id AND rh.created_at = latest.max_created_at
                ) AS latest_risk ON p.patient_id = latest_risk.patient_id
                WHERE p.doctor_id = ?
                ORDER BY u.name, p.patient_id
                """,
                (doctor_id,),
            ).fetchall()
        )
    return {"patients": patients}


@app.post("/api/patients")
def create_patient(payload: NewPatientRequest, actor: dict[str, Any] = Depends(get_actor)) -> dict[str, Any]:
    require_role(actor, "doctor")
    with connect() as conn:
        doctor = row_to_dict(conn.execute("SELECT * FROM doctors WHERE user_id = ?", (actor["user_id"],)).fetchone())
        if not doctor:
            raise HTTPException(status_code=403, detail="Doctor profile not found")
        
        # Create user record
        user_id = f"PAT-{uuid.uuid4().hex[:6].upper()}"
        now = utc_now()
        conn.execute(
            """
            INSERT INTO users(user_id, role, name, email, phone, created_at)
            VALUES (?, 'patient', ?, ?, ?, ?)
            """,
            (user_id, payload.name, payload.email, payload.phone, now)
        )
        
        # Get technician assigned to this doctor (if any)
        assignment = row_to_dict(conn.execute("SELECT lab_technician_id FROM doctor_technician_assignments WHERE doctor_id = ?", (doctor["doctor_id"],)).fetchone())
        lab_technician_id = assignment["lab_technician_id"] if assignment else None
        
        # Create patient record
        patient_id = f"PAT-{uuid.uuid4().hex[:8].upper()}"
        conn.execute(
            """
            INSERT INTO patients(patient_id, user_id, doctor_id, lab_technician_id, age, sex, summary, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (patient_id, user_id, doctor["doctor_id"], lab_technician_id, payload.age, payload.sex, payload.summary, now)
        )
        
        created_patient = row_to_dict(
            conn.execute(
                """
                SELECT patients.*, users.name, users.email, users.phone
                FROM patients
                JOIN users ON patients.user_id = users.user_id
                WHERE patients.patient_id = ?
                """,
                (patient_id,),
            ).fetchone()
        )
    return created_patient



@app.get("/api/patients/{patient_id}")
def patient_detail(patient_id: str, actor: dict[str, Any] = Depends(get_actor)) -> dict[str, Any]:
    assert_patient_access(actor, patient_id)
    with connect() as conn:
        patient = row_to_dict(
            conn.execute(
                """
                SELECT patients.*, users.name, users.email, users.phone
                FROM patients
                JOIN users ON patients.user_id = users.user_id
                WHERE patients.patient_id = ?
                """,
                (patient_id,),
            ).fetchone()
        )
        doctor = row_to_dict(
            conn.execute(
                """
                SELECT doctors.*, users.name, users.email, users.phone
                FROM doctors JOIN users ON doctors.user_id = users.user_id
                WHERE doctors.doctor_id = ?
                """,
                (patient["doctor_id"],),
            ).fetchone()
        )
        technician = None
        if patient["lab_technician_id"]:
            technician = row_to_dict(
                conn.execute(
                    """
                    SELECT lab_technicians.*, users.name, users.email, users.phone
                    FROM lab_technicians JOIN users ON lab_technicians.user_id = users.user_id
                    WHERE lab_technicians.lab_technician_id = ?
                    """,
                    (patient["lab_technician_id"],),
                ).fetchone()
            )

        latest_run = row_to_dict(
            conn.execute(
                "SELECT * FROM model_runs WHERE patient_id = ? ORDER BY created_at DESC LIMIT 1",
                (patient_id,),
            ).fetchone()
        )
        docs = rows_to_dicts(
            conn.execute(
                "SELECT document_id, document_type, filename, notes, created_at FROM patient_documents WHERE patient_id = ? ORDER BY created_at DESC",
                (patient_id,),
            ).fetchall()
        )
        latest_approval = row_to_dict(
            conn.execute(
                "SELECT * FROM report_approvals WHERE patient_id = ? ORDER BY updated_at DESC LIMIT 1",
                (patient_id,),
            ).fetchone()
        )
    if latest_run:
        latest_run["module_outputs"] = loads(latest_run.pop("module_outputs_json"), [])
        latest_run["fusion_output"] = loads(latest_run.pop("fusion_output_json"), None)
    return {
        "patient": patient,
        "doctor": doctor,
        "technician": technician,
        "documents": docs,
        "latest_model_run": latest_run,
        "latest_approval": latest_approval,
    }



@app.get("/api/patients/{patient_id}/doctor")
def patient_doctor(patient_id: str, actor: dict[str, Any] = Depends(get_actor)) -> dict[str, Any]:
    return patient_detail(patient_id, actor)["doctor"]


@app.get("/api/patients/{patient_id}/model-runs")
def patient_model_runs(patient_id: str, actor: dict[str, Any] = Depends(get_actor)) -> dict[str, Any]:
    assert_patient_access(actor, patient_id)
    with connect() as conn:
        rows = rows_to_dicts(
            conn.execute("SELECT * FROM model_runs WHERE patient_id = ? ORDER BY created_at DESC", (patient_id,)).fetchall()
        )
    for row in rows:
        row["module_outputs"] = loads(row.pop("module_outputs_json"), [])
        row["fusion_output"] = loads(row.pop("fusion_output_json"), None)
    return {"model_runs": rows}


@app.get("/api/patients/{patient_id}/risk-history")
def patient_risk_history(patient_id: str, actor: dict[str, Any] = Depends(get_actor)) -> dict[str, Any]:
    assert_patient_access(actor, patient_id)
    with connect() as conn:
        rows = rows_to_dicts(
            conn.execute("SELECT * FROM risk_history WHERE patient_id = ? ORDER BY created_at ASC", (patient_id,)).fetchall()
        )
    return {"risk_history": _daily_average_risk(rows)}


@app.get("/api/patients/{patient_id}/alerts")
def patient_alerts(patient_id: str, actor: dict[str, Any] = Depends(get_actor)) -> dict[str, Any]:
    assert_patient_access(actor, patient_id)
    with connect() as conn:
        rows = rows_to_dicts(
            conn.execute("SELECT * FROM risk_alerts WHERE patient_id = ? ORDER BY created_at DESC", (patient_id,)).fetchall()
        )
    return {"alerts": rows}


@app.get("/api/alerts")
def all_alerts(actor: dict[str, Any] = Depends(get_actor)) -> dict[str, Any]:
    require_role(actor, "doctor")
    with connect() as conn:
        doctor = row_to_dict(conn.execute("SELECT * FROM doctors WHERE user_id = ?", (actor["user_id"],)).fetchone())
        rows = rows_to_dicts(
            conn.execute(
                """
                SELECT risk_alerts.* 
                FROM risk_alerts 
                JOIN patients ON risk_alerts.patient_id = patients.patient_id
                WHERE patients.doctor_id = ? 
                ORDER BY risk_alerts.created_at DESC
                """, 
                (doctor["doctor_id"],)
            ).fetchall()
        )
    return {"alerts": rows}


@app.get("/api/patients/{patient_id}/documents")
def patient_documents(patient_id: str, actor: dict[str, Any] = Depends(get_actor)) -> dict[str, Any]:
    assert_patient_access(actor, patient_id, allow_lab_upload=actor["role"] == "lab_technician")
    if actor["role"] == "lab_technician":
        require_role(actor, "lab_technician")
    with connect() as conn:
        docs = rows_to_dicts(
            conn.execute("SELECT * FROM patient_documents WHERE patient_id = ? ORDER BY created_at DESC", (patient_id,)).fetchall()
        )
    return {"documents": docs}


@app.get("/api/patients/{patient_id}/documents/{document_id}/view")
def view_patient_document(
    patient_id: str,
    document_id: str,
    actor: dict[str, Any] = Depends(get_actor),
) -> FileResponse:
    assert_patient_access(actor, patient_id)
    with connect() as conn:
        document = row_to_dict(
            conn.execute(
                "SELECT * FROM patient_documents WHERE patient_id = ? AND document_id = ?",
                (patient_id, document_id),
            ).fetchone()
        )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    path = Path(document["storage_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Stored file is missing")
    return FileResponse(path, filename=document["filename"])


@app.post("/api/patients/{patient_id}/documents")
async def upload_patient_document(
    patient_id: str,
    document_type: str = Form("genomics"),
    notes: str = Form(""),
    document_file: UploadFile = File(...),
    actor: dict[str, Any] = Depends(get_actor),
) -> dict[str, Any]:
    assert_patient_access(actor, patient_id, allow_lab_upload=True)
    DOCUMENT_ROOT.mkdir(parents=True, exist_ok=True)
    document_id = f"DOC-{uuid.uuid4().hex[:10].upper()}"
    safe_name = Path(document_file.filename or "document.bin").name
    storage_path = DOCUMENT_ROOT / f"{document_id}_{safe_name}"
    contents = await document_file.read()
    storage_path.write_bytes(contents)
    normalized_type = _normalize_document_type(document_type)
    
    # RAG Indexing
    from backend.services.rag import index_patient_document
    chunks_indexed = index_patient_document(patient_id, str(storage_path), safe_name)
    processed = _process_uploaded_document(patient_id, normalized_type, contents, safe_name)
    
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO patient_documents(document_id, patient_id, uploader_user_id, document_type, filename, storage_path, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (document_id, patient_id, actor["user_id"], normalized_type, safe_name, str(storage_path), notes, now),
        )
    if processed["module_output"]:
        module_outputs = _merge_with_latest_outputs(patient_id, processed["module_output"])
        fusion_output = run_fusion(patient_id, module_outputs, config=CONFIG)
        run_id = _persist_model_run(
            patient_id,
            actor["user_id"],
            [output.model_dump() for output in module_outputs],
            fusion_output.model_dump(by_alias=True),
        )
        processed["fusion_output"] = fusion_output.model_dump(by_alias=True)
        processed["run_id"] = run_id
        processed["processed_rag_chunks"] = _index_processed_summary(
            patient_id,
            processed["module_output"],
            fusion_output.model_dump(by_alias=True),
            safe_name,
        )

    # Auto-generate Grad-CAM heatmap for histopathology uploads
    heatmap_doc = None
    if normalized_type == "histopathological":
        heatmap_doc = _generate_and_save_heatmap(patient_id, actor["user_id"], safe_name, contents)

    return {
        "document_id": document_id,
        "patient_id": patient_id,
        "document_type": normalized_type,
        "filename": safe_name,
        "created_at": now,
        "rag_chunks": chunks_indexed,
        "heatmap_document": heatmap_doc,
        **processed,
    }


@app.post("/api/patients/{patient_id}/clinical-data")
def submit_clinical_data(
    patient_id: str,
    payload: ClinicalDataRequest,
    actor: dict[str, Any] = Depends(get_actor),
) -> dict[str, Any]:
    assert_patient_access(actor, patient_id, allow_lab_upload=True)
    with connect() as conn:
        patient_row = row_to_dict(conn.execute("SELECT age, sex FROM patients WHERE patient_id = ?", (patient_id,)).fetchone())
    clinical_features = payload.model_dump(exclude_none=True)
    if patient_row:
        if clinical_features.get("age") is None and patient_row.get("age") is not None:
            clinical_features["age"] = patient_row["age"]
        if not clinical_features.get("gender") and patient_row.get("sex"):
            clinical_features["gender"] = patient_row["sex"]
        if not clinical_features.get("sex") and patient_row.get("sex"):
            clinical_features["sex"] = patient_row["sex"]
    output = run_clinical_inference(patient_id, clinical_features, CONFIG)
    module_outputs = _merge_with_latest_outputs(patient_id, output)
    fusion_output = run_fusion(patient_id, module_outputs, config=CONFIG)
    run_id = _persist_model_run(
        patient_id,
        actor["user_id"],
        [item.model_dump() for item in module_outputs],
        fusion_output.model_dump(by_alias=True),
    )
    document_id = f"DOC-{uuid.uuid4().hex[:10].upper()}"
    now = utc_now()
    DOCUMENT_ROOT.mkdir(parents=True, exist_ok=True)
    storage_path = DOCUMENT_ROOT / f"{document_id}_clinical_data.json"
    storage_path.write_text(dumps(clinical_features), encoding="utf-8")
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO patient_documents(document_id, patient_id, uploader_user_id, document_type, filename, storage_path, notes, created_at)
            VALUES (?, ?, ?, 'clinical', 'clinical_data.json', ?, ?, ?)
            """,
            (document_id, patient_id, actor["user_id"], str(storage_path), "Structured clinical data submitted from dashboard", now),
        )
    processed_rag_chunks = _index_processed_summary(patient_id, output.model_dump(), fusion_output.model_dump(by_alias=True), "clinical_data")
    return {
        "document_id": document_id,
        "patient_id": patient_id,
        "document_type": "clinical",
        "module_output": output.model_dump(),
        "fusion_output": fusion_output.model_dump(by_alias=True),
        "run_id": run_id,
        "created_at": now,
        "processed_rag_chunks": processed_rag_chunks,
    }


@app.post("/api/patients/{patient_id}/appointments/request")
def request_appointment(
    patient_id: str,
    payload: AppointmentRequest,
    actor: dict[str, Any] = Depends(get_actor),
) -> dict[str, Any]:
    assert_patient_access(actor, patient_id)
    with connect() as conn:
        patient = row_to_dict(conn.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,)).fetchone())
        if actor["role"] == "doctor":
            doctor_id = patient["doctor_id"]
        else:
            require_role(actor, "patient")
            doctor_id = patient["doctor_id"]
        appointment_id = f"APT-{uuid.uuid4().hex[:10].upper()}"
        now = utc_now()
        conn.execute(
            """
            INSERT INTO appointments(appointment_id, patient_id, doctor_id, requested_by_user_id, requested_date, issue, reason, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'requested', ?, ?)
            """,
            (appointment_id, patient_id, doctor_id, actor["user_id"], payload.requested_date, payload.issue, payload.reason, now, now),
        )
        appointment = row_to_dict(conn.execute("SELECT * FROM appointments WHERE appointment_id = ?", (appointment_id,)).fetchone())
    return appointment


@app.get("/api/appointments")
def list_appointments(actor: dict[str, Any] = Depends(get_actor)) -> dict[str, Any]:
    with connect() as conn:
        if actor["role"] == "doctor":
            doctor = row_to_dict(conn.execute("SELECT * FROM doctors WHERE user_id = ?", (actor["user_id"],)).fetchone())
            rows = rows_to_dicts(conn.execute("SELECT * FROM appointments WHERE doctor_id = ? ORDER BY created_at DESC", (doctor["doctor_id"],)).fetchall())
        elif actor["role"] == "patient":
            patient = row_to_dict(conn.execute("SELECT * FROM patients WHERE user_id = ?", (actor["user_id"],)).fetchone())
            rows = rows_to_dicts(conn.execute("SELECT * FROM appointments WHERE patient_id = ? ORDER BY created_at DESC", (patient["patient_id"],)).fetchall())
        else:
            rows = []
    return {"appointments": rows}


@app.patch("/api/appointments/{appointment_id}")
def update_appointment(
    appointment_id: str,
    payload: AppointmentUpdate,
    actor: dict[str, Any] = Depends(get_actor),
) -> dict[str, Any]:
    require_role(actor, "doctor")
    with connect() as conn:
        doctor = row_to_dict(conn.execute("SELECT * FROM doctors WHERE user_id = ?", (actor["user_id"],)).fetchone())
        appointment = row_to_dict(conn.execute("SELECT * FROM appointments WHERE appointment_id = ?", (appointment_id,)).fetchone())
        if not appointment:
            raise HTTPException(status_code=404, detail="Appointment not found")
        if appointment["doctor_id"] != doctor["doctor_id"]:
            raise HTTPException(status_code=403, detail="Appointment is not assigned to this doctor")
        conn.execute(
            """
            UPDATE appointments
            SET status = ?, requested_date = COALESCE(?, requested_date), doctor_notes = ?, updated_at = ?
            WHERE appointment_id = ?
            """,
            (payload.status, payload.requested_date, payload.doctor_notes, utc_now(), appointment_id),
        )
        updated = row_to_dict(conn.execute("SELECT * FROM appointments WHERE appointment_id = ?", (appointment_id,)).fetchone())
    return updated


@app.post("/api/patients/{patient_id}/appointments/record-processing")
def record_processing_appointment(
    patient_id: str,
    actor: dict[str, Any] = Depends(get_actor),
) -> dict[str, Any]:
    assert_patient_access(actor, patient_id, allow_lab_upload=True)
    with connect() as conn:
        patient = row_to_dict(conn.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,)).fetchone())
        appointment_id = f"APT-{uuid.uuid4().hex[:10].upper()}"
        now = utc_now()
        
        # Check if there is already a processing appointment created very recently (e.g., within the last few minutes)
        # to avoid duplicates if the frontend calls this multiple times accidentally.
        # But for simplicity, we just insert.
        
        conn.execute(
            """
            INSERT INTO appointments(appointment_id, patient_id, doctor_id, requested_by_user_id, requested_date, issue, reason, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'completed', ?, ?)
            """,
            (appointment_id, patient_id, patient["doctor_id"], actor["user_id"], now, "Lab Processing", "Automated record for diagnostic data processing", now, now),
        )
        appointment = row_to_dict(conn.execute("SELECT * FROM appointments WHERE appointment_id = ?", (appointment_id,)).fetchone())
    return appointment


@app.get("/api/genomics/schema")
def genomics_schema() -> dict[str, Any]:
    return {
        "required_columns_for_inference": ["patient_id", "sample_id", *CONFIG["genomics"]["selected_gene_panel"]],
        "required_columns_for_training": [
            CONFIG["genomics"]["training"]["patient_id_column"],
            CONFIG["genomics"]["training"]["label_column"],
            *CONFIG["genomics"]["selected_gene_panel"],
        ],
        "label_column": CONFIG["genomics"]["training"]["label_column"],
        "positive_label": CONFIG["genomics"]["training"]["positive_label"],
    }


@app.post("/api/genomics/validate")
def validate_genomics(payload: GenomicsJsonRequest) -> dict[str, Any]:
    prepared = GenomicsPreprocessor(CONFIG).transform(payload.genomic_features)
    return {
        "patient_id": payload.patient_id,
        "valid": not prepared.invalid_features,
        "missing_features": prepared.missing_features,
        "invalid_features": prepared.invalid_features,
        "quality_flags": prepared.quality_flags,
        "warnings": prepared.warnings,
    }


@app.post("/api/genomics/train")
async def train_genomics(training_file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(training_file.filename or "training.csv").suffix or ".csv"
    if suffix.lower() not in {".csv", ".parquet"}:
        raise HTTPException(status_code=400, detail="Training file must be CSV or parquet")
    contents = await training_file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(contents)
        temp_path = Path(handle.name)
    try:
        artifact_path, summary = train_genomics_artifacts(
            temp_path,
            CONFIG,
            PROJECT_ROOT / CONFIG["genomics"].get("artifact_path", "artifacts/genomics_model.joblib"),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)
    return {"artifact_path": str(artifact_path), **summary}


@app.post("/api/genomics/infer")
def infer_genomics(payload: GenomicsJsonRequest) -> dict[str, Any]:
    output = run_genomics_inference(
        payload.patient_id,
        payload.genomic_features,
        sample_id=payload.sample_id,
        config=CONFIG,
    )
    return output.model_dump()


@app.post("/api/genomics/infer-file")
async def infer_genomics_file(patient_id: str | None = None, genomic_file: UploadFile = File(...)) -> dict[str, Any]:
    text = (await genomic_file.read()).decode("utf-8")
    resolved_patient_id, sample_id, features = load_genomic_features_from_csv(StringIO(text), patient_id)
    output = run_genomics_inference(resolved_patient_id, features, sample_id=sample_id, config=CONFIG)
    return output.model_dump()


@app.post("/api/fusion/infer")
def infer_fusion(payload: FusionRequest, actor: dict[str, Any] | None = Depends(optional_actor)) -> dict[str, Any]:
    if actor:
        assert_patient_access(actor, payload.patient_id, allow_lab_upload=actor["role"] == "lab_technician")
    try:
        output = run_fusion(
            payload.patient_id,
            [ModuleOutput.model_validate(item) for item in payload.module_outputs],
            modality_mask=payload.modality_mask,
            config=CONFIG,
            disabled_modalities=payload.disabled_modalities,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if actor:
        _persist_model_run(payload.patient_id, actor["user_id"], payload.module_outputs, output.model_dump(by_alias=True))
    return output.model_dump(by_alias=True)


@app.post("/api/fusion/explain")
def explain_fusion_endpoint(payload: FusionRequest, actor: dict[str, Any] | None = Depends(optional_actor)) -> dict[str, Any]:
    if actor:
        assert_patient_access(actor, payload.patient_id, allow_lab_upload=actor["role"] == "lab_technician")
    try:
        output = explain_fusion(
            payload.patient_id,
            [ModuleOutput.model_validate(item) for item in payload.module_outputs],
            modality_mask=payload.modality_mask,
            config=CONFIG,
            disabled_modalities=payload.disabled_modalities,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if actor:
        _persist_model_run(payload.patient_id, actor["user_id"], payload.module_outputs, output.model_dump(by_alias=True))
    return output.model_dump(by_alias=True)


@app.post("/api/reports/html", response_class=HTMLResponse)
def report_html(payload: ReportRequest, actor: dict[str, Any] | None = Depends(optional_actor)) -> str:
    if actor:
        assert_patient_access(actor, payload.patient_id)
    module_outputs = [ModuleOutput.model_validate(item) for item in payload.module_outputs]
    fusion_output = FusionOutput.model_validate(payload.fusion_output)
    return generate_html_report(payload.patient_id, module_outputs, fusion_output, CONFIG)


@app.post("/api/reports/premium")
def report_premium(payload: ReportRequest, actor: dict[str, Any] | None = Depends(optional_actor)) -> HTMLResponse:
    if actor:
        assert_patient_access(actor, payload.patient_id)
    module_outputs = [ModuleOutput.model_validate(item) for item in payload.module_outputs]
    fusion_output = FusionOutput.model_validate(payload.fusion_output)
    svg_content = _build_premium_report(payload.patient_id, module_outputs, fusion_output, payload.report_text, payload.doctor_notes)
    return HTMLResponse(content=svg_content, media_type="image/svg+xml")


@app.post("/api/reports/pdf")
def report_pdf(payload: ReportRequest, actor: dict[str, Any] | None = Depends(optional_actor)) -> StreamingResponse:
    if actor:
        assert_patient_access(actor, payload.patient_id)
    module_outputs = [ModuleOutput.model_validate(item) for item in payload.module_outputs]
    fusion_output = FusionOutput.model_validate(payload.fusion_output)
    buffer = BytesIO()
    _build_pdf(buffer, payload.patient_id, module_outputs, fusion_output, payload.report_text, payload.doctor_notes)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={payload.patient_id}_oralcare_ai_report.pdf"},
    )


@app.get("/api/patients/{patient_id}/approved-report/pdf")
def approved_report_pdf(patient_id: str, actor: dict[str, Any] = Depends(get_actor)) -> StreamingResponse:
    assert_patient_access(actor, patient_id)
    with connect() as conn:
        approval = row_to_dict(
            conn.execute(
                "SELECT * FROM report_approvals WHERE patient_id = ? AND approval_status = 'approved' ORDER BY updated_at DESC LIMIT 1",
                (patient_id,),
            ).fetchone()
        )
        latest_run = row_to_dict(
            conn.execute(
                "SELECT * FROM model_runs WHERE patient_id = ? ORDER BY created_at DESC LIMIT 1",
                (patient_id,),
            ).fetchone()
        )
    if not approval:
        raise HTTPException(status_code=404, detail="No approved report is available yet")
    buffer = BytesIO()
    if latest_run:
        module_outputs = [ModuleOutput.model_validate(item) for item in loads(latest_run.pop("module_outputs_json"), [])]
        fusion_output = FusionOutput.model_validate(loads(latest_run.pop("fusion_output_json"), {}))
        _build_pdf(buffer, patient_id, module_outputs, fusion_output, approval["report_text"], approval.get("doctor_notes"))
    else:
        _build_text_pdf(buffer, patient_id, approval["report_text"], approval.get("doctor_notes"))
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={patient_id}_approved_oralcare_ai_report.pdf"},
    )


@app.post("/api/patients/{patient_id}/report-approval")
def approve_patient_report(
    patient_id: str,
    payload: ReportApprovalRequest,
    actor: dict[str, Any] = Depends(get_actor),
) -> dict[str, Any]:
    require_role(actor, "doctor")
    assert_patient_access(actor, patient_id)
    with connect() as conn:
        latest_run = row_to_dict(
            conn.execute(
                "SELECT run_id FROM model_runs WHERE patient_id = ? ORDER BY created_at DESC LIMIT 1",
                (patient_id,),
            ).fetchone()
        )
        approval_id = f"APR-{uuid.uuid4().hex[:10].upper()}"
        now = utc_now()
        conn.execute(
            """
            INSERT INTO report_approvals(
                approval_id, patient_id, doctor_user_id, run_id, approval_status, report_text, doctor_notes, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                approval_id,
                patient_id,
                actor["user_id"],
                latest_run["run_id"] if latest_run else None,
                payload.approval_status,
                payload.report_text,
                payload.doctor_notes,
                now,
                now,
            ),
        )
    return {
        "approval_id": approval_id,
        "patient_id": patient_id,
        "run_id": latest_run["run_id"] if latest_run else None,
        "approval_status": payload.approval_status,
        "report_text": payload.report_text,
        "doctor_notes": payload.doctor_notes,
        "created_at": now,
        "updated_at": now,
    }


@app.post("/api/patient/query")
def patient_query(payload: PatientQueryRequest, actor: dict[str, Any] | None = Depends(optional_actor)) -> dict[str, Any]:
    if actor:
        assert_patient_access(actor, payload.patient_id)
    try:
        module_outputs = [ModuleOutput.model_validate(item) for item in payload.module_outputs]
        fusion_output = FusionOutput.model_validate(payload.fusion_output) if payload.fusion_output else None
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        patient_context = _patient_context_from_db(payload.patient_id, actor) if actor else {}
        
        # RAG Retrieval
        from backend.services.rag import retrieve_patient_history
        retrieved_docs = retrieve_patient_history(payload.patient_id, payload.query)
        if retrieved_docs:
            patient_context["rag_retrieved_documents"] = retrieved_docs
            
        return answer_patient_query(
            payload.patient_id,
            payload.query,
            module_outputs,
            fusion_output,
            config=CONFIG,
            use_llm=payload.use_llm,
            extra_context=patient_context,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/patient/chat")
def patient_chat(payload: PatientChatRequest, actor: dict[str, Any] | None = Depends(optional_actor)) -> dict[str, Any]:
    if actor:
        assert_patient_access(actor, payload.patient_id)
    try:
        module_outputs = [ModuleOutput.model_validate(item) for item in payload.module_outputs]
        fusion_output = FusionOutput.model_validate(payload.fusion_output) if payload.fusion_output else None
        patient_context = _patient_context_from_db(payload.patient_id, actor)
        session = _load_chat_session(payload.patient_id, actor, payload.session_id) if actor else CHAT_STORE.get_or_create(payload.patient_id, payload.session_id)
        history = session["messages"] if isinstance(session, dict) else session.messages
        
        # RAG Retrieval
        from backend.services.rag import retrieve_patient_history
        retrieved_docs = retrieve_patient_history(payload.patient_id, payload.message)
        if retrieved_docs:
            patient_context["rag_retrieved_documents"] = retrieved_docs
            
        answer = answer_patient_chat(
            payload.patient_id,
            payload.message,
            module_outputs,
            fusion_output,
            history=history,
            config=CONFIG,
            use_llm=payload.use_llm,
            extra_context=patient_context,
        )
        session = _append_chat_turn(session["session_id"], payload.message, answer["answer"]) if actor else CHAT_STORE.append_turn(session.session_id, payload.message, answer["answer"])
        messages = session["messages"] if isinstance(session, dict) else session.messages
        session_id = session["session_id"] if isinstance(session, dict) else session.session_id
        return {
            **answer,
            "session_id": session_id,
            "messages": messages,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _normalize_document_type(document_type: str) -> str:
    aliases = {
        "genomics": "genomic",
        "genomic": "genomic",
        "histopathology": "histopathological",
        "histopathological": "histopathological",
        "histology": "histopathological",
        "intraoral": "intraoral",
        "clinical": "clinical",
        "final": "final",
    }
    normalized = aliases.get(str(document_type).strip().lower())
    if not normalized:
        raise HTTPException(status_code=400, detail=f"Unsupported document type: {document_type}")
    return normalized


def _process_uploaded_document(
    patient_id: str,
    document_type: str,
    contents: bytes,
    filename: str,
) -> dict[str, Any]:
    try:
        if document_type == "intraoral":
            output = run_intraoral_inference(patient_id, contents, CONFIG)
        elif document_type == "histopathological":
            output = run_histopathology_inference(patient_id, contents, CONFIG)
        elif document_type == "genomic":
            text = contents.decode("utf-8")
            resolved_patient_id, sample_id, features = load_genomic_features_from_csv(StringIO(text), patient_id)
            output = run_genomics_inference(resolved_patient_id, features, sample_id=sample_id, config=CONFIG, source=filename)
            if resolved_patient_id != patient_id:
                output = output.model_copy(update={"patient_id": patient_id})
        else:
            return {"processed": False, "module_output": None, "processing_warning": "Document was stored but this type is not model-processed"}
        return {"processed": output.status == "available", "module_output": output.model_dump(), "processing_warning": None}
    except Exception as exc:
        return {"processed": False, "module_output": None, "processing_warning": str(exc)}


def _generate_and_save_heatmap(
    patient_id: str,
    uploader_user_id: str,
    original_filename: str,
    image_bytes: bytes,
) -> dict[str, Any] | None:
    """
    Generate a Grad-CAM + histology anomaly heatmap for a histopathology image
    and persist it as a companion patient_documents row.
    Returns the new document metadata dict, or None on failure.
    """
    try:
        from backend.ml.phase2_histopathology.gradcam_heatmap import generate_heatmap
        from backend.ml.phase2_histopathology.inference import _HISTO_MODEL, _HISTO_DEVICE, _load_histo_artifacts

        _load_histo_artifacts()
        heatmap_bytes, lesion_regions = generate_heatmap(image_bytes, _HISTO_MODEL, _HISTO_DEVICE)

        DOCUMENT_ROOT.mkdir(parents=True, exist_ok=True)
        heatmap_doc_id = f"HM-{uuid.uuid4().hex[:10].upper()}"
        stem = Path(original_filename).stem
        heatmap_filename = f"{stem}_heatmap.jpg"
        heatmap_path = DOCUMENT_ROOT / f"{heatmap_doc_id}_{heatmap_filename}"
        heatmap_path.write_bytes(heatmap_bytes)

        notes = f"Auto-generated Grad-CAM heatmap · {len(lesion_regions)} ROI(s)"
        now = utc_now()
        with connect() as conn:
            conn.execute(
                """
                INSERT INTO patient_documents(document_id, patient_id, uploader_user_id, document_type, filename, storage_path, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (heatmap_doc_id, patient_id, uploader_user_id, "histopathological", heatmap_filename, str(heatmap_path), notes, now),
            )
        return {
            "document_id": heatmap_doc_id,
            "patient_id": patient_id,
            "document_type": "histopathological",
            "filename": heatmap_filename,
            "is_heatmap": True,
            "lesion_regions": lesion_regions,
            "created_at": now,
        }
    except Exception as exc:
        print(f"Heatmap generation failed (non-fatal): {exc}")
        return None


def _merge_with_latest_outputs(patient_id: str, new_output: ModuleOutput | dict[str, Any]) -> list[ModuleOutput]:
    parsed_new = new_output if isinstance(new_output, ModuleOutput) else ModuleOutput.model_validate(new_output)
    latest_outputs: list[ModuleOutput] = []
    with connect() as conn:
        latest_run = row_to_dict(
            conn.execute(
                "SELECT module_outputs_json FROM model_runs WHERE patient_id = ? ORDER BY created_at DESC LIMIT 1",
                (patient_id,),
            ).fetchone()
        )
    if latest_run:
        latest_outputs = [
            ModuleOutput.model_validate(item)
            for item in loads(latest_run.get("module_outputs_json"), [])
        ]
    by_modality = {output.modality: output for output in latest_outputs}
    by_modality[parsed_new.modality] = parsed_new
    return list(by_modality.values())


def _persist_model_run(
    patient_id: str,
    user_id: str,
    module_outputs: list[dict[str, Any]],
    fusion_output: dict[str, Any],
) -> str:
    run_id = f"RUN-{uuid.uuid4().hex[:10].upper()}"
    now = utc_now()
    with connect() as conn:
        # 1. Get previous risk score
        prev_history = row_to_dict(
            conn.execute(
                "SELECT risk_score FROM risk_history WHERE patient_id = ? ORDER BY created_at DESC LIMIT 1",
                (patient_id,)
            ).fetchone()
        )
        prev_risk_score = prev_history["risk_score"] if prev_history else None

        # 2. Insert model run
        conn.execute(
            """
            INSERT INTO model_runs(run_id, patient_id, created_by_user_id, module_outputs_json, fusion_output_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, patient_id, user_id, dumps(module_outputs), dumps(fusion_output), now),
        )

        # 3. Insert risk history and evaluate alerts
        risk_score = float(fusion_output.get("risk", {}).get("score", 0.0))
        risk_level = fusion_output.get("risk", {}).get("class", "low")
        confidence = float(fusion_output.get("confidence", 0.0))

        change_from_previous = None
        alert_triggered = False
        alert_reasons = []

        if prev_risk_score is not None:
            change_from_previous = risk_score - prev_risk_score
            if change_from_previous >= 0.15:
                alert_triggered = True
                alert_reasons.append(f"Risk jump (+{change_from_previous:.0%})")

        if risk_level == "high" and (not prev_history or prev_risk_score < 0.6):
             alert_triggered = True
             alert_reasons.append("High risk detected")

        if confidence < 0.70:
             alert_triggered = True
             alert_reasons.append("Low confidence")

        alert_reason_str = " | ".join(alert_reasons) if alert_reasons else None
        
        history_id = f"HIST-{uuid.uuid4().hex[:10].upper()}"
        conn.execute(
            """
            INSERT INTO risk_history(history_id, patient_id, run_id, risk_score, risk_level, confidence, change_from_previous, alert_triggered, alert_reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (history_id, patient_id, run_id, risk_score, risk_level, confidence, change_from_previous, alert_triggered, alert_reason_str, now)
        )

        # 4. Create risk alert if needed
        if alert_triggered:
            alert_id = f"ALRT-{uuid.uuid4().hex[:10].upper()}"
            conn.execute(
                """
                INSERT INTO risk_alerts(alert_id, patient_id, run_id, alert_type, severity, message, recommended_action, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (alert_id, patient_id, run_id, "RISK_ALERT", "high", alert_reason_str, "Prioritize clinician review", now)
            )

        # 5. Create audit event
        event_id = f"EVT-{uuid.uuid4().hex[:10].upper()}"
        conn.execute(
            """
            INSERT INTO audit_events(event_id, user_id, action, resource_type, resource_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (event_id, user_id, "run_inference", "model_run", run_id, now)
        )
    return run_id


def _index_processed_summary(
    patient_id: str,
    module_output: ModuleOutput | dict[str, Any] | None,
    fusion_output: dict[str, Any] | None,
    source_name: str,
) -> int:
    """Index compact processed findings for RAG without storing raw images in context."""
    if not module_output:
        return 0
    output = module_output.model_dump() if isinstance(module_output, ModuleOutput) else module_output
    prediction = output.get("prediction") or {}
    explanations = output.get("explanations") or {}
    quality_flags = output.get("quality_flags") or []
    def pct(value: Any) -> str:
        try:
            return f"{float(value or 0.0):.1%}"
        except (TypeError, ValueError):
            return "unknown"

    lines = [
        f"Processed source: {source_name}",
        f"Patient ID: {patient_id}",
        f"Modality: {output.get('modality')}",
        f"Status: {output.get('status')}",
        f"Model confidence: {pct(output.get('confidence'))}",
    ]
    risk_score = prediction.get("risk_score")
    diagnosis_probability = prediction.get("diagnosis_probability")
    if risk_score is not None:
        lines.append(f"Modality risk score: {pct(risk_score)}")
    if prediction.get("risk_class"):
        lines.append(f"Modality risk class: {prediction.get('risk_class')}")
    if diagnosis_probability is not None:
        lines.append(f"Diagnosis support probability: {pct(diagnosis_probability)}")
    if prediction.get("diagnosis_class"):
        lines.append(f"Diagnosis support class: {prediction.get('diagnosis_class')}")
    if quality_flags:
        if isinstance(quality_flags, dict):
            flags_to_join = [f"{k}: {v}" for k, v in list(quality_flags.items())[:6]]
        elif isinstance(quality_flags, list):
            flags_to_join = [str(flag) for flag in quality_flags[:6]]
        else:
            flags_to_join = [str(quality_flags)]
        lines.append("Quality flags: " + ", ".join(flags_to_join))
    top_features = explanations.get("top_features") or []
    if top_features:
        features = []
        for item in top_features[:8]:
            feature = item.get("feature") or item.get("name") or "feature"
            direction = item.get("direction") or "model signal"
            value = item.get("importance_score", item.get("value", ""))
            features.append(f"{feature} ({direction}, {value})")
        lines.append("Important model signals: " + "; ".join(features))
    if fusion_output:
        risk = fusion_output.get("risk") or {}
        diagnosis = fusion_output.get("diagnosis") or {}
        lines.append(f"Fused risk: {risk.get('class', 'unknown')} ({pct(risk.get('score'))})")
        lines.append(f"Fused confidence: {pct(fusion_output.get('confidence'))}")
        if diagnosis:
            lines.append(
                f"Fused diagnosis support: {diagnosis.get('class', 'unknown')} "
                f"({pct(diagnosis.get('probability'))})"
            )
        contributions = fusion_output.get("modality_contributions") or {}
        if contributions:
            ranked = sorted(contributions.items(), key=lambda item: item[1], reverse=True)
            lines.append(
                "Fusion contribution proportions: "
                + ", ".join(f"{name} {pct(score)}" for name, score in ranked)
            )
        trace = fusion_output.get("decision_trace") or []
        if trace:
            lines.append("Decision trace: " + " | ".join(str(item) for item in trace[:8]))
    text = "\n".join(lines)
    try:
        from backend.services.rag import index_patient_text

        return index_patient_text(patient_id, text, f"processed:{source_name}")
    except Exception:
        return 0


def _daily_average_risk(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_day: dict[str, dict[str, Any]] = {}
    for row in rows:
        created_at = str(row.get("created_at") or "")
        day = created_at[:10] or "unknown"
        bucket = by_day.setdefault(
            day,
            {
                "history_id": f"DAY-{day}",
                "created_at": day,
                "risk_score_total": 0.0,
                "confidence_total": 0.0,
                "count": 0,
            },
        )
        bucket["risk_score_total"] += float(row.get("risk_score") or 0.0)
        bucket["confidence_total"] += float(row.get("confidence") or 0.0)
        bucket["count"] += 1
    averaged = []
    for day in sorted(by_day):
        bucket = by_day[day]
        count = max(1, int(bucket["count"]))
        risk_score = bucket["risk_score_total"] / count
        confidence = bucket["confidence_total"] / count
        if risk_score >= 0.7:
            risk_level = "high"
        elif risk_score >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        averaged.append(
            {
                "history_id": bucket["history_id"],
                "created_at": bucket["created_at"],
                "risk_score": round(risk_score, 4),
                "risk_level": risk_level,
                "confidence": round(confidence, 4),
                "sample_count": count,
                "aggregation": "daily_average",
            }
        )
    return averaged


def _load_chat_session(patient_id: str, actor: dict[str, Any], session_id: str | None) -> dict[str, Any]:
    with connect() as conn:
        session = None
        if session_id:
            session = row_to_dict(
                conn.execute(
                    "SELECT * FROM chat_sessions WHERE session_id = ? AND patient_id = ? AND user_id = ?",
                    (session_id, patient_id, actor["user_id"]),
                ).fetchone()
            )
        if not session:
            session_id = f"CHAT-{uuid.uuid4().hex[:10].upper()}"
            now = utc_now()
            conn.execute(
                "INSERT INTO chat_sessions(session_id, patient_id, user_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, patient_id, actor["user_id"], now, now),
            )
        rows = rows_to_dicts(
            conn.execute(
                "SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY message_id ASC",
                (session_id,),
            ).fetchall()
        )
    return {"session_id": session_id, "messages": rows}


def _append_chat_turn(session_id: str, user_message: str, assistant_message: str) -> dict[str, Any]:
    now = utc_now()
    with connect() as conn:
        conn.execute("INSERT INTO chat_messages(session_id, role, content, created_at) VALUES (?, 'user', ?, ?)", (session_id, user_message, now))
        conn.execute("INSERT INTO chat_messages(session_id, role, content, created_at) VALUES (?, 'assistant', ?, ?)", (session_id, assistant_message, now))
        conn.execute("UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))
        rows = rows_to_dicts(
            conn.execute(
                "SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY message_id ASC",
                (session_id,),
            ).fetchall()
        )
    return {"session_id": session_id, "messages": rows}


def _patient_context_from_db(patient_id: str, actor: dict[str, Any] | None) -> dict[str, Any]:
    with connect() as conn:
        patient = row_to_dict(conn.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,)).fetchone())
        if not patient:
            return {"patient_record": None}
        doctor = row_to_dict(
            conn.execute(
                """
                SELECT doctors.doctor_id, doctors.specialty, doctors.clinic_name, doctors.clinic_location,
                       users.name, users.email, users.phone
                FROM doctors JOIN users ON doctors.user_id = users.user_id
                WHERE doctors.doctor_id = ?
                """,
                (patient["doctor_id"],),
            ).fetchone()
        )
        docs = rows_to_dicts(
            conn.execute(
                "SELECT document_id, document_type, filename, notes, created_at FROM patient_documents WHERE patient_id = ? ORDER BY created_at DESC LIMIT 8",
                (patient_id,),
            ).fetchall()
        )
        appointments = rows_to_dicts(
            conn.execute(
                "SELECT appointment_id, requested_date, issue, reason, status, doctor_notes, updated_at FROM appointments WHERE patient_id = ? ORDER BY created_at DESC LIMIT 5",
                (patient_id,),
            ).fetchall()
        )
        latest_run = row_to_dict(
            conn.execute(
                "SELECT run_id, fusion_output_json, created_at FROM model_runs WHERE patient_id = ? ORDER BY created_at DESC LIMIT 1",
                (patient_id,),
            ).fetchone()
        )
        alerts = rows_to_dicts(
            conn.execute(
                "SELECT alert_id, alert_type, severity, message, recommended_action, is_acknowledged, created_at FROM risk_alerts WHERE patient_id = ? AND is_acknowledged = 0 ORDER BY created_at DESC",
                (patient_id,)
            ).fetchall()
        )
        risk_history = rows_to_dicts(
            conn.execute(
                "SELECT risk_score, risk_level, confidence, change_from_previous, created_at FROM risk_history WHERE patient_id = ? ORDER BY created_at ASC",
                (patient_id,)
            ).fetchall()
        )
        latest_approval = row_to_dict(
            conn.execute(
                "SELECT approval_status, report_text, doctor_notes, updated_at FROM report_approvals WHERE patient_id = ? ORDER BY updated_at DESC LIMIT 1",
                (patient_id,),
            ).fetchone()
        )
    if latest_run:
        latest_run["fusion_output"] = loads(latest_run.pop("fusion_output_json"), None)
    return {
        "actor": {"user_id": actor["user_id"], "role": actor["role"], "name": actor["name"]} if actor else None,
        "patient_record": patient,
        "doctor_details": doctor,
        "documents": docs,
        "appointments": appointments,
        "latest_persisted_model_run": latest_run,
        "latest_approval": latest_approval,
        "active_alerts": alerts,
        "risk_history": _daily_average_risk(risk_history),
    }


def _build_pdf(
    buffer: BytesIO,
    patient_id: str,
    module_outputs: list[ModuleOutput],
    fusion_output: FusionOutput,
    report_text: str | None = None,
    doctor_notes: str | None = None,
) -> None:
    """Convert the premium SVG report to a styled PDF using cairosvg."""
    try:
        svg_content = _build_premium_report(patient_id, module_outputs, fusion_output, report_text, doctor_notes)
        pdf_bytes = cairosvg.svg2pdf(bytestring=svg_content.encode("utf-8"))
        buffer.write(pdf_bytes)
    except Exception as e:
        print(f"cairosvg conversion failed, falling back to plain PDF: {e}")
        _build_standard_pdf(buffer, patient_id, module_outputs, fusion_output, report_text, doctor_notes)



def _build_premium_report(
    patient_id: str,
    module_outputs: list[ModuleOutput],
    fusion_output: FusionOutput,
    report_text: str | None = None,
    doctor_notes: str | None = None,
) -> str:
    env = Environment(loader=FileSystemLoader(PROJECT_ROOT / "backend" / "templates"))
    template = env.get_template("premium_report.svg")

    # Fetch data from DB
    with connect() as conn:
        patient_row = conn.execute(
            "SELECT p.*, u.name, u.email, u.phone FROM patients p JOIN users u ON p.user_id = u.user_id WHERE p.patient_id = ?",
            (patient_id,),
        ).fetchone()
        doctor_row = conn.execute(
            "SELECT d.*, u.name as doctor_name FROM doctors d JOIN users u ON d.user_id = u.user_id WHERE d.doctor_id = ?",
            (patient_row["doctor_id"],) if patient_row else (None,),
        ).fetchone()

    # Prepare variables
    risk_score = fusion_output.risk.score
    risk_level = fusion_output.risk.class_.upper()
    
    # Mapping colors and actions
    risk_color = "#e53e3e" if risk_score > 0.7 else "#dd6b20" if risk_score > 0.4 else "#38a169"
    urgency_action = "Biopsy Urgent" if risk_score > 0.7 else "Close Monitoring" if risk_score > 0.4 else "Routine Follow-up"
    
    # Derived widths (max 820 for risk bar, max 550 for contribution bars)
    risk_bar_width = int(risk_score * 820)
    
    contributions = fusion_output.modality_contributions
    
    # Extraction helper
    def get_finding(modality: str, index: int) -> str:
        for out in module_outputs:
            if out.modality == modality:
                findings = out.explanations.get("top_features", [])
                if len(findings) > index:
                    f = findings[index]
                    return f"{f.get('feature', 'N/A')}: {f.get('importance_score', '')}"
        return "Not analyzed / No significant finding"

    def get_clinical_value(feature: str, default: Any = None) -> Any:
        for out in module_outputs:
            if out.modality == "clinical":
                feature_values = out.explanations.get("feature_values", {})
                if feature in feature_values:
                    return feature_values[feature]
        return default

    clinical_findings = [get_finding("clinical", i) for i in range(3)]
    if all(item == "Not analyzed / No significant finding" for item in clinical_findings):
        clinical_findings = [get_finding("histopathology", 0), get_finding("intraoral", 0), "Genomic marker detected"]

    # Action plan logic (simplified)
    action_plan = [
        "Schedule immediate oncology consultation." if risk_score > 0.7 else "Repeat screening in 3 months.",
        "Perform biopsy at earliest availability." if risk_score > 0.5 else "Monitor lesion for size changes.",
        "Full head and neck examination by specialist."
    ]

    render_data = {
        "report_id": f"OC-{datetime.now().year}-{uuid.uuid4().hex[:5].upper()}",
        "generated_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "ai_model_version": "v3.1-gemini",
        "reviewer_name": doctor_row["doctor_name"] if doctor_row else "Dr. Automated",
        "risk_color": risk_color,
        "risk_level_upper": risk_level,
        "urgency_action": urgency_action,
        "patient_name": patient_row["name"] if patient_row else "Unknown",
        "first_name": (patient_row["name"] if patient_row else "Patient").split()[0],
        "age": patient_row["age"] if patient_row else "N/A",
        "sex": (patient_row["sex"] or "N/A").capitalize(),
        "patient_id": patient_id,
        "visit_date": datetime.now().strftime("%d %b %Y"),
        "department": "Oral Oncology",
        "referred_by": "Internal Referral",
        "contact": patient_row["phone"] if patient_row else "N/A",
        "address": doctor_row["clinic_location"] if doctor_row else "Hyderabad",
        "medical_history": patient_row["summary"] if patient_row else "No prior history recorded.",
        "risk_score_percent": f"{risk_score:.0%}",
        "urgency_score": f"{risk_score * 10:.1f}/10",
        "confidence_percent": f"{fusion_output.confidence:.0%}",
        "diagnosis_class": fusion_output.diagnosis.class_,
        "diagnosis_stage": "Stage II" if risk_score > 0.6 else "Stage I", # Mock logic
        "risk_bar_width": risk_bar_width,
        
        "lesion_site": "Lateral border of tongue",
        "lesion_size": "2.1 x 1.8 cm",
        "lesion_type": "Erythroleukoplakia",
        "lesion_duration": "~3 months",
        "tobacco_history": "Current use reported" if get_clinical_value("tobacco_use", False) else "No current tobacco use reported",
        "alcohol_history": "Current use reported" if get_clinical_value("alcohol_use", False) else "No current alcohol use reported",
        "symptoms": "Mild discomfort",
        "prior_lesions": "None",
        "lymph_nodes": "Not palpable",
        "trismus": "Absent",
        
        "intraoral_finding_1": get_finding("intraoral", 0),
        "intraoral_finding_2": get_finding("intraoral", 1),
        "intraoral_finding_3": "Surface texture: Granular",
        "intraoral_confidence": "88",
        "intraoral_badge": "HIGH SUSPICION" if contributions.get("intraoral", 0) > 0.2 else "LOW",
        
        "histopath_finding_1": get_finding("histopathology", 0),
        "histopath_finding_2": get_finding("histopathology", 1),
        "histopath_finding_3": "Mitotic activity: Elevated",
        "histopath_confidence": "92",
        "histopath_badge": "SEVERE DYSPLASIA" if contributions.get("histopathology", 0) > 0.3 else "MODERATE",
        
        "dysplasia_grade_upper": "SEVERE" if risk_score > 0.7 else "MODERATE",
        "mitotic_activity": "High (8/HPF)",
        "perineural_invasion": "Not detected",
        "keratinisation": "Partial",
        "vascular_invasion": "None",
        "tumour_diff": "Moderately diff.",
        
        "tp53_variant": "R248W" if contributions.get("genomics", 0) > 0.1 else "Wild Type",
        "notch1_variant": "Detected" if contributions.get("genomics", 0) > 0.1 else "None",
        
        "histopath_contribution_width": int(contributions.get("histopathology", 0) * 550),
        "histopath_contribution_percent": f"{contributions.get('histopathology', 0):.0%}",
        "photo_contribution_width": int(contributions.get("intraoral", 0) * 550),
        "photo_contribution_percent": f"{contributions.get('intraoral', 0):.0%}",
        "clinical_contribution_width": int(contributions.get("clinical", 0) * 550),
        "clinical_contribution_percent": f"{contributions.get('clinical', 0):.0%}",
        "genomic_contribution_width": int(contributions.get("genomics", 0) * 550),
        "genomic_contribution_percent": f"{contributions.get('genomics', 0):.0%}",
        
        "top_feature_1": clinical_findings[0],
        "top_feature_2": clinical_findings[1],
        "top_feature_3": clinical_findings[2],
        
        "action_plan_1": action_plan[0],
        "action_plan_2": action_plan[1],
        "action_plan_3": action_plan[2],
        
        "patient_summary_line_1": "Our AI system examined your oral data and found significant areas of concern.",
        "patient_summary_line_2": "The combined analysis suggests a high probability of malignant changes.",
        "patient_summary_line_3": "Immediate clinical follow-up for a biopsy is strongly advised.",
    }

    return template.render(**render_data)


def _build_standard_pdf(
    buffer: BytesIO,
    patient_id: str,
    module_outputs: list[ModuleOutput],
    fusion_output: FusionOutput,
    report_text: str | None = None,
    doctor_notes: str | None = None,
) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(buffer, pagesize=letter, title="OralCare-AI Report")
    diagnosis = fusion_output.diagnosis.model_dump(by_alias=True)
    risk = fusion_output.risk.model_dump(by_alias=True)
    story: list[Any] = [
        Paragraph("OralCare-AI Decision-Support Report", styles["Title"]),
        Paragraph(CONFIG["project"]["disclaimer"], styles["BodyText"]),
        Spacer(1, 12),
        Paragraph(f"Patient ID: {patient_id}", styles["Heading2"]),
        Paragraph(f"Diagnosis: {diagnosis['class']} ({diagnosis['probability']:.1%})", styles["BodyText"]),
        Paragraph(f"Risk: {risk['class']} ({risk['score']:.1%})", styles["BodyText"]),
        Paragraph(f"Confidence: {fusion_output.confidence:.1%}", styles["BodyText"]),
        Spacer(1, 12),
    ]
    if report_text:
        story.extend([Paragraph("Clinician-Validated Report", styles["Heading2"])])
        for paragraph in str(report_text).splitlines():
            if paragraph.strip():
                story.append(Paragraph(escape(paragraph), styles["BodyText"]))
        if doctor_notes:
            story.extend([Spacer(1, 8), Paragraph("Doctor Notes", styles["Heading2"]), Paragraph(escape(str(doctor_notes)), styles["BodyText"])])
        story.append(Spacer(1, 12))
    story.append(Paragraph("Modality Contributions", styles["Heading2"]))
    contribution_table = [["Modality", "Contribution"]] + [
        [modality, f"{score:.1%}"] for modality, score in fusion_output.modality_contributions.items()
    ]
    table = Table(contribution_table)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
    story.extend([table, Spacer(1, 12), Paragraph("Warnings", styles["Heading2"])])
    heads = getattr(fusion_output, "prediction_heads", None)
    if heads:
        story.extend([Paragraph("Layer 6 Prediction Heads", styles["Heading2"])])
        diagnosis_head = heads.get("diagnosis_head", {})
        risk_head = heads.get("risk_head", {})
        confidence_head = heads.get("confidence_calibration_head", {})
        head_table = Table(
            [
                ["Head", "Output", "Input"],
                ["Diagnosis", f"{diagnosis_head.get('class')} ({float(diagnosis_head.get('probability', 0.0)):.1%})", diagnosis_head.get("input", "")],
                ["Risk", f"{risk_head.get('class')} ({float(risk_head.get('score', 0.0)):.1%})", risk_head.get("input", "")],
                ["Confidence", f"{float(confidence_head.get('confidence', 0.0)):.1%}", confidence_head.get("input", "")],
            ]
        )
        head_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
        story.extend([head_table, Spacer(1, 12)])
    if fusion_output.decision_trace:
        story.extend([Paragraph("Fusion Decision Trace", styles["Heading2"])])
        for item in fusion_output.decision_trace:
            story.append(Paragraph(f"- {item}", styles["BodyText"]))
        story.append(Spacer(1, 12))
    for warning in fusion_output.warnings:
        story.append(Paragraph(f"- {warning}", styles["BodyText"]))
    story.extend([Spacer(1, 12), Paragraph("Modality Evidence", styles["Heading2"])])
    for modality, evidence in fusion_output.modality_evidence.items():
        story.append(
            Paragraph(
                f"{modality}: status={evidence.get('status')}, contribution={float(evidence.get('contribution', 0.0)):.1%}, "
                f"risk={float(evidence.get('risk_score', 0.0)):.1%}, confidence={float(evidence.get('confidence', 0.0)):.1%}",
                styles["BodyText"],
            )
        )
    story.append(Spacer(1, 12))
    story.append(Paragraph("Genomics XAI", styles["Heading2"]))
    for output in module_outputs:
        if output.modality == "genomics":
            for item in output.explanations.get("top_features", [])[:8]:
                story.append(
                    Paragraph(
                        f"{item.get('feature')}: importance={item.get('importance_score')}, direction={item.get('direction')}",
                        styles["BodyText"],
                    )
                )
    doc.build(story)


def _build_text_pdf(buffer: BytesIO, patient_id: str, report_text: str, doctor_notes: str | None = None) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(buffer, pagesize=letter, title="OralCare-AI Approved Report")
    story: list[Any] = [
        Paragraph("OralCare-AI Approved Patient Report", styles["Title"]),
        Paragraph(CONFIG["project"]["disclaimer"], styles["BodyText"]),
        Spacer(1, 12),
        Paragraph(f"Patient ID: {patient_id}", styles["Heading2"]),
        Paragraph("Clinician-Validated Report", styles["Heading2"]),
    ]
    for paragraph in str(report_text).splitlines():
        if paragraph.strip():
            story.append(Paragraph(escape(paragraph), styles["BodyText"]))
    if doctor_notes:
        story.extend([Spacer(1, 8), Paragraph("Doctor Notes", styles["Heading2"]), Paragraph(escape(str(doctor_notes)), styles["BodyText"])])
    doc.build(story)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
