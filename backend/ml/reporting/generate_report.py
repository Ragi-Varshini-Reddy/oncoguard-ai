"""Doctor-readable HTML report generation."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any

from backend.ml.common.config import load_config
from backend.ml.common.contracts import FusionOutput, ModuleOutput


def generate_html_report(
    patient_id: str,
    module_outputs: list[ModuleOutput],
    fusion_output: FusionOutput,
    config: dict[str, Any] | None = None,
) -> str:
    cfg = config or load_config()
    disclaimer = cfg["project"]["disclaimer"]
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows = "\n".join(
        f"<tr><td>{html.escape(output.modality)}</td><td>{html.escape(output.status)}</td>"
        f"<td>{'' if output.confidence is None else round(output.confidence, 3)}</td></tr>"
        for output in module_outputs
    )
    contribution_rows = "\n".join(
        f"<tr><td>{html.escape(modality)}</td><td>{score:.2%}</td></tr>"
        for modality, score in fusion_output.modality_contributions.items()
    )
    evidence_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(modality)}</td>"
        f"<td>{html.escape(str(evidence.get('status', '')))}</td>"
        f"<td>{float(evidence.get('contribution', 0.0)):.1%}</td>"
        f"<td>{float(evidence.get('risk_score', 0.0)):.1%}</td>"
        f"<td>{float(evidence.get('confidence', 0.0)):.1%}</td>"
        "</tr>"
        for modality, evidence in fusion_output.modality_evidence.items()
    )
    decision_trace = "".join(f"<li>{html.escape(item)}</li>" for item in fusion_output.decision_trace)
    explanation_sections = "\n".join(_explanation_block(output) for output in module_outputs)
    warnings = "".join(f"<li>{html.escape(warning)}</li>" for warning in fusion_output.warnings)
    heads = getattr(fusion_output, "prediction_heads", {}) or {}
    head_rows = _head_rows(heads)

    diagnosis = fusion_output.diagnosis.model_dump(by_alias=True)
    risk = fusion_output.risk.model_dump(by_alias=True)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(cfg["reporting"]["report_title"])}</title>
  <style>
    body {{ font-family: Arial, sans-serif; color: #1d2433; margin: 32px; line-height: 1.45; }}
    h1, h2 {{ color: #12343b; }}
    .banner {{ padding: 12px 14px; background: #fff4d6; border-left: 5px solid #d99a00; margin-bottom: 18px; }}
    .metric {{ display: inline-block; min-width: 170px; margin: 8px 12px 8px 0; padding: 12px; border: 1px solid #d8dee9; border-radius: 6px; }}
    .metric strong {{ display: block; font-size: 18px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 8px 0 20px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px; text-align: left; }}
    th {{ background: #eef3f7; }}
    .small {{ color: #5b677a; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>{html.escape(cfg["reporting"]["report_title"])}</h1>
  <p class="small">Generated {generated_at}</p>
  <div class="banner"><strong>Clinical disclaimer:</strong> {html.escape(disclaimer)}</div>
  <h2>Patient Summary</h2>
  <p><strong>Patient ID:</strong> {html.escape(patient_id)}</p>
  <div class="metric">Diagnosis<strong>{html.escape(diagnosis["class"])}</strong>{diagnosis["probability"]:.1%}</div>
  <div class="metric">Risk<strong>{html.escape(risk["class"])}</strong>{risk["score"]:.1%}</div>
  <div class="metric">Confidence<strong>{fusion_output.confidence:.1%}</strong></div>
  <h2>Available Modalities</h2>
  <table><tr><th>Modality</th><th>Status</th><th>Confidence</th></tr>{rows}</table>
  <h2>Modality Contribution</h2>
  <table><tr><th>Modality</th><th>Contribution</th></tr>{contribution_rows}</table>
  <h2>Layer 6 Prediction Heads</h2>
  <table><tr><th>Head</th><th>Output</th><th>Input</th></tr>{head_rows}</table>
  <h2>Fusion Decision Trace</h2>
  <ul>{decision_trace}</ul>
  <h2>Modality Evidence</h2>
  <table><tr><th>Modality</th><th>Status</th><th>Contribution</th><th>Risk</th><th>Confidence</th></tr>{evidence_rows}</table>
  <h2>Explanations</h2>
  {explanation_sections}
  <h2>Warnings</h2>
  <ul>{warnings}</ul>
</body>
</html>"""


def _head_rows(heads: dict[str, Any]) -> str:
    diagnosis = heads.get("diagnosis_head", {})
    risk = heads.get("risk_head", {})
    confidence = heads.get("confidence_calibration_head", {})
    rows = [
        ("Diagnosis", f"{diagnosis.get('class', '')} ({float(diagnosis.get('probability', 0.0)):.1%})", diagnosis.get("input", "")),
        ("Risk", f"{risk.get('class', '')} ({float(risk.get('score', 0.0)):.1%})", risk.get("input", "")),
        ("Confidence", f"{float(confidence.get('confidence', 0.0)):.1%}", confidence.get("input", "")),
    ]
    return "".join(
        f"<tr><td>{html.escape(name)}</td><td>{html.escape(output)}</td><td>{html.escape(str(source))}</td></tr>"
        for name, output, source in rows
    )


def _explanation_block(output: ModuleOutput) -> str:
    if output.status != "available":
        return f"<h3>{html.escape(output.modality.title())}</h3><p>Modality not provided.</p>"
    top_features = output.explanations.get("top_features", [])
    if top_features:
        rows = "".join(
            "<tr>"
            f"<td>{html.escape(str(item.get('feature', '')))}</td>"
            f"<td>{html.escape(str(item.get('value', '')))}</td>"
            f"<td>{html.escape(str(item.get('importance_score', '')))}</td>"
            f"<td>{html.escape(str(item.get('direction', '')))}</td>"
            "</tr>"
            for item in top_features
        )
        detail = f"<table><tr><th>Feature</th><th>Value</th><th>Importance</th><th>Direction</th></tr>{rows}</table>"
    elif output.explanations.get("gradcam_placeholder"):
        detail = "<p>Visual explanation placeholder available in the dashboard.</p>"
    else:
        detail = "<p>No detailed explanation returned.</p>"
    note = output.explanations.get("note", "")
    return f"<h3>{html.escape(output.modality.title())}</h3>{detail}<p class=\"small\">{html.escape(note)}</p>"
