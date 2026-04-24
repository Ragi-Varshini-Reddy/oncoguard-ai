"""FastAPI backend for the real-data OralCare-AI workflow."""

from __future__ import annotations

import tempfile
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.app.schemas import FusionRequest, GenomicsJsonRequest, ReportRequest
from src.common.config import load_config
from src.common.contracts import FusionOutput, ModuleOutput
from src.fusion.inference import run_fusion
from src.phase3_genomics.artifact_model import train_genomics_artifacts
from src.phase3_genomics.inference import run_genomics_inference
from src.phase3_genomics.preprocess import GenomicsPreprocessor
from src.phase3_genomics.utils import load_genomic_features_from_csv
from src.reporting.generate_report import generate_html_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_config(PROJECT_ROOT / "configs" / "prototype_config.yaml")

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
def infer_fusion(payload: FusionRequest) -> dict[str, Any]:
    try:
        output = run_fusion(
            payload.patient_id,
            [ModuleOutput.model_validate(item) for item in payload.module_outputs],
            modality_mask=payload.modality_mask,
            config=CONFIG,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return output.model_dump(by_alias=True)


@app.post("/api/reports/html", response_class=HTMLResponse)
def report_html(payload: ReportRequest) -> str:
    module_outputs = [ModuleOutput.model_validate(item) for item in payload.module_outputs]
    fusion_output = FusionOutput.model_validate(payload.fusion_output)
    return generate_html_report(payload.patient_id, module_outputs, fusion_output, CONFIG)


@app.post("/api/reports/pdf")
def report_pdf(payload: ReportRequest) -> StreamingResponse:
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
    for warning in fusion_output.warnings:
        story.append(Paragraph(f"- {warning}", styles["BodyText"]))
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
