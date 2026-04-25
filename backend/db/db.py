"""MySQL persistence for demo users, patients, appointments, documents, and model runs."""

from __future__ import annotations

import json
import os
import mysql.connector
from mysql.connector import pooling
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "oncoguard_ai")


class DatabaseCursor:
    """Compatibility wrapper that gives MySQL cursors sqlite-like chaining."""

    def __init__(self, cursor: mysql.connector.cursor.MySQLCursor) -> None:
        self._cursor = cursor

    def execute(self, operation: str, params: tuple[Any, ...] | list[Any] | None = None) -> "DatabaseCursor":
        self._cursor.execute(_mysql_placeholders(operation), params)
        return self

    def executemany(self, operation: str, seq_params: list[tuple[Any, ...]]) -> "DatabaseCursor":
        self._cursor.executemany(_mysql_placeholders(operation), seq_params)
        return self

    def fetchone(self) -> dict[str, Any] | None:
        return self._cursor.fetchone()

    def fetchall(self) -> list[dict[str, Any]]:
        return self._cursor.fetchall()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


def _mysql_placeholders(operation: str) -> str:
    return operation.replace("?", "%s")


# Initialize connection pool
try:
    db_pool = mysql.connector.pooling.MySQLConnectionPool(
        pool_name="oncoguard_pool",
        pool_size=5,
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )
except mysql.connector.Error as err:
    print(f"Error connecting to MySQL: {err}")
    # Fallback for initialization if DB doesn't exist yet
    db_pool = None

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

@contextmanager
def connect() -> Iterator[DatabaseCursor]:
    if not db_pool:
        # Emergency fallback for first-time setup or if pool failed
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE
        )
    else:
        conn = db_pool.get_connection()
    
    raw_cursor = conn.cursor(dictionary=True)
    cursor = DatabaseCursor(raw_cursor)
    try:
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        raw_cursor.close()
        conn.close()

def initialize_database() -> None:
    # First, ensure the database exists
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DATABASE}")
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        print(f"Warning during DB creation: {err}")

    with connect() as cursor:
        # MySQL doesn't have executescript, so we split by semicolon
        statements = [
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id VARCHAR(255) PRIMARY KEY,
                role ENUM('patient', 'doctor', 'lab_technician') NOT NULL,
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255),
                phone VARCHAR(50),
                created_at VARCHAR(100) NOT NULL
            ) ENGINE=InnoDB;
            """,
            """
            CREATE TABLE IF NOT EXISTS doctors (
                doctor_id VARCHAR(255) PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL UNIQUE,
                specialty VARCHAR(255) NOT NULL,
                clinic_name VARCHAR(255) NOT NULL,
                clinic_location VARCHAR(255) NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            ) ENGINE=InnoDB;
            """,
            """
            CREATE TABLE IF NOT EXISTS lab_technicians (
                lab_technician_id VARCHAR(255) PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL UNIQUE,
                lab_name VARCHAR(255) NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            ) ENGINE=InnoDB;
            """,
            """
            CREATE TABLE IF NOT EXISTS doctor_technician_assignments (
                assignment_id INT AUTO_INCREMENT PRIMARY KEY,
                doctor_id VARCHAR(255) NOT NULL,
                lab_technician_id VARCHAR(255) NOT NULL,
                created_at VARCHAR(100) NOT NULL,
                UNIQUE(doctor_id, lab_technician_id),
                FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id) ON DELETE CASCADE,
                FOREIGN KEY (lab_technician_id) REFERENCES lab_technicians(lab_technician_id) ON DELETE CASCADE
            ) ENGINE=InnoDB;
            """,
            """
            CREATE TABLE IF NOT EXISTS patients (
                patient_id VARCHAR(255) PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL UNIQUE,
                doctor_id VARCHAR(255) NOT NULL,
                lab_technician_id VARCHAR(255),
                age INT,
                sex VARCHAR(20),
                summary TEXT,
                created_at VARCHAR(100) NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id) ON DELETE CASCADE,
                FOREIGN KEY (lab_technician_id) REFERENCES lab_technicians(lab_technician_id) ON DELETE SET NULL
            ) ENGINE=InnoDB;
            """,
            """
            CREATE TABLE IF NOT EXISTS patient_documents (
                document_id VARCHAR(255) PRIMARY KEY,
                patient_id VARCHAR(255) NOT NULL,
                uploader_user_id VARCHAR(255) NOT NULL,
                document_type ENUM('intraoral', 'histopathological', 'genomic', 'clinical', 'final') NOT NULL,
                filename VARCHAR(255) NOT NULL,
                storage_path TEXT NOT NULL,
                notes TEXT,
                created_at VARCHAR(100) NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
                FOREIGN KEY (uploader_user_id) REFERENCES users(user_id) ON DELETE CASCADE
            ) ENGINE=InnoDB;
            """,
            """
            CREATE TABLE IF NOT EXISTS model_runs (
                run_id VARCHAR(255) PRIMARY KEY,
                patient_id VARCHAR(255) NOT NULL,
                created_by_user_id VARCHAR(255),
                module_outputs_json LONGTEXT NOT NULL,
                fusion_output_json LONGTEXT NOT NULL,
                created_at VARCHAR(100) NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
                FOREIGN KEY (created_by_user_id) REFERENCES users(user_id) ON DELETE SET NULL
            ) ENGINE=InnoDB;
            """,
            """
            CREATE TABLE IF NOT EXISTS appointments (
                appointment_id VARCHAR(255) PRIMARY KEY,
                patient_id VARCHAR(255) NOT NULL,
                doctor_id VARCHAR(255) NOT NULL,
                requested_by_user_id VARCHAR(255) NOT NULL,
                requested_date VARCHAR(100) NOT NULL,
                issue TEXT NOT NULL,
                reason TEXT NOT NULL,
                status ENUM('requested', 'scheduled', 'completed', 'cancelled') NOT NULL,
                doctor_notes TEXT,
                created_at VARCHAR(100) NOT NULL,
                updated_at VARCHAR(100) NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
                FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id) ON DELETE CASCADE,
                FOREIGN KEY (requested_by_user_id) REFERENCES users(user_id) ON DELETE CASCADE
            ) ENGINE=InnoDB;
            """,
            """
            CREATE TABLE IF NOT EXISTS risk_alerts (
                alert_id VARCHAR(255) PRIMARY KEY,
                patient_id VARCHAR(255) NOT NULL,
                run_id VARCHAR(255),
                alert_type VARCHAR(255) NOT NULL,
                severity VARCHAR(100) NOT NULL,
                message TEXT NOT NULL,
                recommended_action TEXT,
                is_acknowledged BOOLEAN DEFAULT FALSE,
                created_at VARCHAR(100) NOT NULL,
                acknowledged_by VARCHAR(255),
                acknowledged_at VARCHAR(100),
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
                FOREIGN KEY (run_id) REFERENCES model_runs(run_id) ON DELETE SET NULL,
                FOREIGN KEY (acknowledged_by) REFERENCES users(user_id) ON DELETE SET NULL
            ) ENGINE=InnoDB;
            """,
            """
            CREATE TABLE IF NOT EXISTS risk_history (
                history_id VARCHAR(255) PRIMARY KEY,
                patient_id VARCHAR(255) NOT NULL,
                run_id VARCHAR(255),
                risk_score DOUBLE NOT NULL,
                risk_level VARCHAR(100) NOT NULL,
                confidence DOUBLE NOT NULL,
                change_from_previous DOUBLE,
                alert_triggered BOOLEAN DEFAULT FALSE,
                alert_reason TEXT,
                created_at VARCHAR(100) NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
                FOREIGN KEY (run_id) REFERENCES model_runs(run_id) ON DELETE SET NULL
            ) ENGINE=InnoDB;
            """,
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id VARCHAR(255) PRIMARY KEY,
                user_id VARCHAR(255),
                role VARCHAR(100),
                action VARCHAR(255) NOT NULL,
                resource_type VARCHAR(100),
                resource_id VARCHAR(255),
                created_at VARCHAR(100) NOT NULL,
                details_json LONGTEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL
            ) ENGINE=InnoDB;
            """,
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id VARCHAR(255) PRIMARY KEY,
                patient_id VARCHAR(255) NOT NULL,
                user_id VARCHAR(255) NOT NULL,
                created_at VARCHAR(100) NOT NULL,
                updated_at VARCHAR(100) NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            ) ENGINE=InnoDB;
            """,
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                role ENUM('user', 'assistant') NOT NULL,
                content TEXT NOT NULL,
                created_at VARCHAR(100) NOT NULL,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
            ) ENGINE=InnoDB;
            """,
            """
            CREATE TABLE IF NOT EXISTS report_approvals (
                approval_id VARCHAR(255) PRIMARY KEY,
                patient_id VARCHAR(255) NOT NULL,
                doctor_user_id VARCHAR(255) NOT NULL,
                run_id VARCHAR(255),
                approval_status ENUM('draft', 'approved', 'rejected') NOT NULL,
                report_text LONGTEXT NOT NULL,
                doctor_notes TEXT,
                created_at VARCHAR(100) NOT NULL,
                updated_at VARCHAR(100) NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
                FOREIGN KEY (doctor_user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                FOREIGN KEY (run_id) REFERENCES model_runs(run_id) ON DELETE SET NULL
            ) ENGINE=InnoDB;
            """
        ]
        for statement in statements:
            if statement.strip():
                cursor.execute(statement)
        
        _seed(cursor)

def _seed(cursor: mysql.connector.cursor.MySQLCursor) -> None:
    now = utc_now()
    users = [
        ("DOC-001", "doctor", "Dr. Anika Rao", "doctor@oralcare.local", "+1-555-0101"),
        ("PAT-001", "patient", "Goutham Demo Patient", "patient@oralcare.local", "+1-555-0201"),
        ("PAT-002", "patient", "Maya Demo Patient", "maya@oralcare.local", "+1-555-0202"),
        ("LAB-001", "lab_technician", "Ravi Lab Technician", "lab@oralcare.local", "+1-555-0301"),
    ]
    # Use INSERT IGNORE for MySQL
    cursor.executemany(
        "INSERT IGNORE INTO users(user_id, role, name, email, phone, created_at) VALUES (%s, %s, %s, %s, %s, %s)",
        [(user_id, role, name, email, phone, now) for user_id, role, name, email, phone in users],
    )
    cursor.execute(
        """
        INSERT IGNORE INTO doctors(doctor_id, user_id, specialty, clinic_name, clinic_location)
        VALUES (%s, %s, %s, %s, %s)
        """,
        ("DR-RAO", "DOC-001", "Oral oncology", "OralCare Demo Clinic", "Hyderabad"),
    )
    cursor.execute(
        """
        INSERT IGNORE INTO lab_technicians(lab_technician_id, user_id, lab_name)
        VALUES (%s, %s, %s)
        """,
        ("LABTECH-001", "LAB-001", "OralCare Molecular Lab"),
    )
    cursor.execute(
        """
        INSERT IGNORE INTO doctor_technician_assignments(doctor_id, lab_technician_id, created_at)
        VALUES (%s, %s, %s)
        """,
        ("DR-RAO", "LABTECH-001", now),
    )
    patients = [
        ("TCGA-BB-4227", "PAT-001", "DR-RAO", "LABTECH-001", 57, "male", "Demo patient with available genomics workflow."),
        ("TCGA-CV-5441", "PAT-002", "DR-RAO", "LABTECH-001", 63, "female", "Demo patient ready for multimodal uploads."),
    ]
    cursor.executemany(
        """
        INSERT IGNORE INTO patients(patient_id, user_id, doctor_id, lab_technician_id, age, sex, summary, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        [(p[0], p[1], p[2], p[3], p[4], p[5], p[6], now) for p in patients],
    )

def row_to_dict(row: dict | None) -> dict[str, Any] | None:
    return row

def rows_to_dicts(rows: list[dict]) -> list[dict[str, Any]]:
    return rows

def dumps(data: Any) -> str:
    return json.dumps(data, sort_keys=True, default=str)

def loads(text: str | None, default: Any = None) -> Any:
    if not text:
        return default
    return json.loads(text)
