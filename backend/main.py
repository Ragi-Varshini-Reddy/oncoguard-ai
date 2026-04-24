"""FastAPI backend for the real-data OralCare-AI workflow."""

from __future__ import annotations

import tempfile
import uuid
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.db.db import connect, dumps, initialize_database, loads, row_to_dict, rows_to_dicts, utc_now
from backend.schemas.schemas import (
    AppointmentRequest,
    AppointmentUpdate,
    DemoLoginRequest,
    FusionRequest,
    GenomicsJsonRequest,
    PatientChatRequest,
    PatientQueryRequest,
    ReportRequest,
)
from backend.core.security import actor_profile, assert_patient_access, get_actor, optional_actor, require_role
from backend.ml.common.config import load_config
from backend.ml.common.contracts import FusionOutput, ModuleOutput
from backend.ml.fusion.inference import explain_fusion, run_fusion
from backend.ml.explainability.chat_store import ChatStore
from backend.ml.explainability.patient_query import answer_patient_chat, answer_patient_query
from backend.ml.phase3_genomics.artifact_model import train_genomics_artifacts
from backend.ml.phase3_genomics.inference import run_genomics_inference
from backend.ml.phase3_genomics.preprocess import GenomicsPreprocessor
from backend.ml.phase3_genomics.utils import load_genomic_features_from_csv
from backend.ml.reporting.generate_report import generate_html_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    allow_origins=CONFIG.get("api", {}).get("cors_origins", ["http://localhost:5173"]),
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


@app.get("/api/doctors/{doctor_id}/patients")
def doctor_patients(doctor_id: str, actor: dict[str, Any] = Depends(get_actor)) -> dict[str, Any]:
    require_role(actor, "doctor")
    with connect() as conn:
        doctor = row_to_dict(conn.execute("SELECT * FROM doctors WHERE user_id = ?", (actor["user_id"],)).fetchone())
        if not doctor or doctor["doctor_id"] != doctor_id:
            raise HTTPException(status_code=403, detail="Doctor profile mismatch")
        patients = rows_to_dicts(conn.execute("SELECT * FROM patients WHERE doctor_id = ? ORDER BY patient_id", (doctor_id,)).fetchall())
    return {"patients": patients}


@app.get("/api/patients/{patient_id}")
def patient_detail(patient_id: str, actor: dict[str, Any] = Depends(get_actor)) -> dict[str, Any]:
    assert_patient_access(actor, patient_id)
    with connect() as conn:
        patient = row_to_dict(conn.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,)).fetchone())
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
        latest_run = row_to_dict(
            conn.execute(
                "SELECT * FROM model_runs WHERE patient_id = ? ORDER BY created_at DESC LIMIT 1",
                (patient_id,),
            ).fetchone()
        )
    if latest_run:
        latest_run["module_outputs"] = loads(latest_run.pop("module_outputs_json"), [])
        latest_run["fusion_output"] = loads(latest_run.pop("fusion_output_json"), None)
    return {"patient": patient, "doctor": doctor, "latest_model_run": latest_run}


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
    return {"risk_history": rows}


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
    storage_path.write_bytes(await document_file.read())
    
    # RAG Indexing
    from backend.services.rag import index_patient_document
    chunks_indexed = index_patient_document(patient_id, str(storage_path), safe_name)
    
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO patient_documents(document_id, patient_id, uploader_user_id, document_type, filename, storage_path, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (document_id, patient_id, actor["user_id"], document_type, safe_name, str(storage_path), notes, now),
        )
    return {
        "document_id": document_id,
        "patient_id": patient_id,
        "document_type": document_type,
        "filename": safe_name,
        "created_at": now,
        "rag_chunks": chunks_indexed,
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


@app.post("/api/reports/pdf")
def report_pdf(payload: ReportRequest, actor: dict[str, Any] | None = Depends(optional_actor)) -> StreamingResponse:
    if actor:
        assert_patient_access(actor, payload.patient_id)
    module_outputs = [ModuleOutput.model_validate(item) for item in payload.module_outputs]
    fusion_output = FusionOutput.model_validate(payload.fusion_output)
    buffer = BytesIO()
    _build_pdf(buffer, payload.patient_id, module_outputs, fusion_output)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={payload.patient_id}_oralcare_ai_report.pdf"},
    )


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


def _persist_model_run(
    patient_id: str,
    user_id: str,
    module_outputs: list[dict[str, Any]],
    fusion_output: dict[str, Any],
) -> None:
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
                "SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY message_id DESC LIMIT ?",
                (session_id, int(CONFIG.get("llm", {}).get("max_chat_turns", 10)) * 2),
            ).fetchall()
        )
    return {"session_id": session_id, "messages": list(reversed(rows))}


def _append_chat_turn(session_id: str, user_message: str, assistant_message: str) -> dict[str, Any]:
    now = utc_now()
    with connect() as conn:
        conn.execute("INSERT INTO chat_messages(session_id, role, content, created_at) VALUES (?, 'user', ?, ?)", (session_id, user_message, now))
        conn.execute("INSERT INTO chat_messages(session_id, role, content, created_at) VALUES (?, 'assistant', ?, ?)", (session_id, assistant_message, now))
        conn.execute("UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))
        rows = rows_to_dicts(
            conn.execute(
                "SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY message_id DESC LIMIT ?",
                (session_id, int(CONFIG.get("llm", {}).get("max_chat_turns", 10)) * 2),
            ).fetchall()
        )
    return {"session_id": session_id, "messages": list(reversed(rows))}


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
    if latest_run:
        latest_run["fusion_output"] = loads(latest_run.pop("fusion_output_json"), None)
    return {
        "actor": {"user_id": actor["user_id"], "role": actor["role"], "name": actor["name"]} if actor else None,
        "patient_record": patient,
        "doctor_details": doctor,
        "documents": docs,
        "appointments": appointments,
        "latest_persisted_model_run": latest_run,
        "active_alerts": alerts,
        "risk_history": risk_history,
    }


def _build_pdf(buffer: BytesIO, patient_id: str, module_outputs: list[ModuleOutput], fusion_output: FusionOutput) -> None:
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
        Paragraph("Modality Contributions", styles["Heading2"]),
    ]
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
