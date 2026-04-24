"""SQLite persistence for demo users, patients, appointments, documents, and model runs."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "artifacts" / "oralcare_ai.sqlite3"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def initialize_database() -> None:
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                role TEXT NOT NULL CHECK (role IN ('patient', 'doctor', 'lab_technician')),
                name TEXT NOT NULL,
                email TEXT,
                phone TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS doctors (
                doctor_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL UNIQUE REFERENCES users(user_id),
                specialty TEXT NOT NULL,
                clinic_name TEXT NOT NULL,
                clinic_location TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL UNIQUE REFERENCES users(user_id),
                doctor_id TEXT NOT NULL REFERENCES doctors(doctor_id),
                age INTEGER,
                sex TEXT,
                summary TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS lab_technicians (
                lab_technician_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL UNIQUE REFERENCES users(user_id),
                lab_name TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS patient_documents (
                document_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL REFERENCES patients(patient_id),
                uploader_user_id TEXT NOT NULL REFERENCES users(user_id),
                document_type TEXT NOT NULL,
                filename TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS model_runs (
                run_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL REFERENCES patients(patient_id),
                created_by_user_id TEXT REFERENCES users(user_id),
                module_outputs_json TEXT NOT NULL,
                fusion_output_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS appointments (
                appointment_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL REFERENCES patients(patient_id),
                doctor_id TEXT NOT NULL REFERENCES doctors(doctor_id),
                requested_by_user_id TEXT NOT NULL REFERENCES users(user_id),
                requested_date TEXT NOT NULL,
                issue TEXT NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('requested', 'scheduled', 'completed', 'cancelled')),
                doctor_notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS risk_alerts (
                alert_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL REFERENCES patients(patient_id),
                run_id TEXT REFERENCES model_runs(run_id),
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                recommended_action TEXT,
                is_acknowledged BOOLEAN DEFAULT 0,
                created_at TEXT NOT NULL,
                acknowledged_by TEXT REFERENCES users(user_id),
                acknowledged_at TEXT
            );

            CREATE TABLE IF NOT EXISTS risk_history (
                history_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL REFERENCES patients(patient_id),
                run_id TEXT REFERENCES model_runs(run_id),
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                confidence REAL NOT NULL,
                change_from_previous REAL,
                alert_triggered BOOLEAN DEFAULT 0,
                alert_reason TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                user_id TEXT REFERENCES users(user_id),
                role TEXT,
                action TEXT NOT NULL,
                resource_type TEXT,
                resource_id TEXT,
                created_at TEXT NOT NULL,
                details_json TEXT
            );

            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL REFERENCES patients(patient_id),
                user_id TEXT NOT NULL REFERENCES users(user_id),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES chat_sessions(session_id),
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        _seed(conn)


def _seed(conn: sqlite3.Connection) -> None:
    now = utc_now()
    users = [
        ("DOC-001", "doctor", "Dr. Anika Rao", "doctor@oralcare.local", "+1-555-0101"),
        ("PAT-001", "patient", "Goutham Demo Patient", "patient@oralcare.local", "+1-555-0201"),
        ("PAT-002", "patient", "Maya Demo Patient", "maya@oralcare.local", "+1-555-0202"),
        ("LAB-001", "lab_technician", "Ravi Lab Technician", "lab@oralcare.local", "+1-555-0301"),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO users(user_id, role, name, email, phone, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        [(user_id, role, name, email, phone, now) for user_id, role, name, email, phone in users],
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO doctors(doctor_id, user_id, specialty, clinic_name, clinic_location)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("DR-RAO", "DOC-001", "Oral oncology", "OralCare Demo Clinic", "San Francisco, CA"),
    )
    patients = [
        ("TCGA-BB-4227", "PAT-001", "DR-RAO", 57, "male", "Demo patient with available genomics workflow."),
        ("TCGA-CV-5441", "PAT-002", "DR-RAO", 63, "female", "Demo patient ready for multimodal uploads."),
    ]
    conn.executemany(
        """
        INSERT OR IGNORE INTO patients(patient_id, user_id, doctor_id, age, sex, summary, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [(patient_id, user_id, doctor_id, age, sex, summary, now) for patient_id, user_id, doctor_id, age, sex, summary in patients],
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO lab_technicians(lab_technician_id, user_id, lab_name)
        VALUES (?, ?, ?)
        """,
        ("LABTECH-001", "LAB-001", "OralCare Molecular Lab"),
    )


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def dumps(data: Any) -> str:
    return json.dumps(data, sort_keys=True, default=str)


def loads(text: str | None, default: Any = None) -> Any:
    if not text:
        return default
    return json.loads(text)
