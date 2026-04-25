"""Role-aware demo access control."""

from __future__ import annotations

from typing import Any

from fastapi import Header, HTTPException

from backend.db.db import connect, row_to_dict


def get_actor(x_user_id: str | None = Header(default=None, alias="X-User-Id")) -> dict[str, Any]:
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Demo login required")
    with connect() as conn:
        user = row_to_dict(conn.execute("SELECT * FROM users WHERE user_id = ?", (x_user_id,)).fetchone())
    if not user:
        raise HTTPException(status_code=401, detail="Unknown demo user")
    return user


def optional_actor(x_user_id: str | None = Header(default=None, alias="X-User-Id")) -> dict[str, Any] | None:
    if not x_user_id:
        return None
    with connect() as conn:
        return row_to_dict(conn.execute("SELECT * FROM users WHERE user_id = ?", (x_user_id,)).fetchone())


def require_role(actor: dict[str, Any], *roles: str) -> None:
    if actor["role"] not in roles:
        raise HTTPException(status_code=403, detail="This user role cannot perform that action")


def assert_patient_access(actor: dict[str, Any], patient_id: str, *, allow_lab_upload: bool = False) -> None:
    role = actor["role"]
    with connect() as conn:
        patient = row_to_dict(conn.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,)).fetchone())
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        if role == "patient" and patient["user_id"] == actor["user_id"]:
            return
        if role == "doctor":
            doctor = row_to_dict(conn.execute("SELECT * FROM doctors WHERE user_id = ?", (actor["user_id"],)).fetchone())
            if doctor and patient["doctor_id"] == doctor["doctor_id"]:
                return
        if role == "lab_technician":
            lab_tech = row_to_dict(conn.execute("SELECT * FROM lab_technicians WHERE user_id = ?", (actor["user_id"],)).fetchone())
            if lab_tech and patient["lab_technician_id"] == lab_tech["lab_technician_id"]:
                return
    raise HTTPException(status_code=403, detail="Patient is not accessible to this user")



def actor_profile(actor: dict[str, Any]) -> dict[str, Any]:
    with connect() as conn:
        profile: dict[str, Any] = {}
        if actor["role"] == "doctor":
            profile = row_to_dict(conn.execute("SELECT * FROM doctors WHERE user_id = ?", (actor["user_id"],)).fetchone()) or {}
        elif actor["role"] == "patient":
            profile = row_to_dict(conn.execute("SELECT * FROM patients WHERE user_id = ?", (actor["user_id"],)).fetchone()) or {}
        elif actor["role"] == "lab_technician":
            profile = row_to_dict(conn.execute("SELECT * FROM lab_technicians WHERE user_id = ?", (actor["user_id"],)).fetchone()) or {}
    return {"user": actor, "profile": profile}
