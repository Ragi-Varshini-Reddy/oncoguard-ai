"""Streamlit dashboard for the OralCare-AI hackathon prototype."""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.common.config import load_config
from backend.ml.fusion.inference import run_fusion
from backend.ml.phase1_intraoral_clinical.clinical_inference import run_clinical_inference
from backend.ml.phase1_intraoral_clinical.intraoral_inference import run_intraoral_inference
from backend.ml.phase2_histopathology.inference import run_histopathology_inference
from backend.ml.phase3_genomics.inference import run_genomics_inference
from backend.ml.phase3_genomics.utils import load_genomic_features_from_csv
from backend.ml.reporting.generate_report import generate_html_report

try:
    import streamlit as st
except ModuleNotFoundError as exc:  # pragma: no cover - only relevant outside app runtime
    raise RuntimeError("Install streamlit to run the dashboard: pip install -r requirements.txt") from exc


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "prototype_config.yaml")
    st.set_page_config(page_title="OralCare-AI", layout="wide")
    st.title("OralCare-AI")
    st.warning(cfg["project"]["disclaimer"])

    sample = _load_sample_patient(PROJECT_ROOT / "data_samples" / "sample_patient_001.json")
    with st.sidebar:
        st.header("Patient")
        use_sample = st.toggle("Use sample patient", value=True)
        patient_id = st.text_input("Patient ID", sample["patient_id"] if use_sample else "P-DEMO")
        st.caption("Demo data is synthetic and not patient-matched to external datasets.")

    left, right = st.columns([1, 1])
    with left:
        intraoral_file = st.file_uploader("Intraoral image", type=["jpg", "jpeg", "png"])
        histology_file = st.file_uploader("Histopathology image or patch", type=["jpg", "jpeg", "png", "tif", "tiff"])
        genomic_file = st.file_uploader("Genomic CSV", type=["csv"])
    with right:
        clinical = _clinical_form(sample["clinical"] if use_sample else {})
        use_sample_genomics = st.checkbox("Use sample genomics", value=use_sample)

    if st.button("Run inference", type="primary"):
        genomic_features, sample_id = _resolve_genomics(
            patient_id=patient_id,
            use_sample_genomics=use_sample_genomics,
            sample=sample,
            uploaded_file=genomic_file,
        )
        module_outputs = [
            run_intraoral_inference(patient_id, intraoral_file.read() if intraoral_file else None, cfg),
            run_clinical_inference(patient_id, clinical, cfg),
            run_histopathology_inference(patient_id, histology_file.read() if histology_file else None, cfg),
            run_genomics_inference(patient_id, genomic_features, sample_id=sample_id, config=cfg),
        ]
        fusion_output = run_fusion(patient_id, module_outputs, config=cfg)
        _render_results(module_outputs, fusion_output)
        report_html = generate_html_report(patient_id, module_outputs, fusion_output, cfg)
        st.download_button(
            "Download report",
            data=report_html.encode("utf-8"),
            file_name=f"{patient_id}_oralcare_ai_report.html",
            mime="text/html",
        )


def _clinical_form(defaults: dict[str, Any]) -> dict[str, Any]:
    st.subheader("Clinical form")
    age = st.number_input("Age", min_value=18, max_value=100, value=int(defaults.get("age", 50)))
    sex = st.selectbox("Sex", ["female", "male", "other"], index=["female", "male", "other"].index(defaults.get("sex", "female")))
    lesion_site = st.selectbox(
        "Lesion site",
        ["lateral tongue", "floor of mouth", "buccal mucosa", "lip", "gingiva", "other"],
        index=0,
    )
    lesion_size = st.number_input(
        "Lesion size (cm)",
        min_value=0.0,
        max_value=10.0,
        value=float(defaults.get("lesion_size_cm", 1.2)),
        step=0.1,
    )
    ulcer_weeks = st.number_input(
        "Persistent ulcer duration (weeks)",
        min_value=0,
        max_value=52,
        value=int(defaults.get("persistent_ulcer_weeks", 0)),
    )
    tobacco = st.checkbox("Tobacco use", value=bool(defaults.get("tobacco_use", False)))
    alcohol = st.checkbox("Alcohol use", value=bool(defaults.get("alcohol_use", False)))
    node = st.checkbox("Neck node present", value=bool(defaults.get("neck_node_present", False)))
    return {
        "age": age,
        "sex": sex,
        "lesion_site": lesion_site,
        "lesion_size_cm": lesion_size,
        "persistent_ulcer_weeks": ulcer_weeks,
        "tobacco_use": tobacco,
        "alcohol_use": alcohol,
        "neck_node_present": node,
    }


def _resolve_genomics(
    patient_id: str,
    use_sample_genomics: bool,
    sample: dict[str, Any],
    uploaded_file: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    if uploaded_file:
        text = uploaded_file.getvalue().decode("utf-8")
        resolved_patient_id, sample_id, features = load_genomic_features_from_csv(StringIO(text), patient_id)
        if resolved_patient_id != patient_id:
            st.info(f"Using genomic patient ID {resolved_patient_id} from uploaded CSV.")
        return features, sample_id
    if use_sample_genomics:
        genomics = sample.get("genomics", {})
        return genomics.get("features"), genomics.get("sample_id")
    return None, None


def _render_results(module_outputs: list[Any], fusion_output: Any) -> None:
    diagnosis = fusion_output.diagnosis.model_dump(by_alias=True)
    risk = fusion_output.risk.model_dump(by_alias=True)
    st.subheader("Final multimodal output")
    c1, c2, c3 = st.columns(3)
    c1.metric("Diagnosis", diagnosis["class"], f"{diagnosis['probability']:.1%}")
    c2.metric("Risk", risk["class"], f"{risk['score']:.1%}")
    c3.metric("Confidence", f"{fusion_output.confidence:.1%}")

    st.subheader("Modality availability")
    st.dataframe(
        [
            {"modality": output.modality, "status": output.status, "confidence": output.confidence}
            for output in module_outputs
        ],
        use_container_width=True,
    )

    st.subheader("Modality contribution")
    st.bar_chart(fusion_output.modality_contributions)

    st.subheader("Explainability")
    for output in module_outputs:
        with st.expander(output.modality.title(), expanded=output.modality == "genomics"):
            if output.explanations.get("top_features"):
                st.dataframe(output.explanations["top_features"], use_container_width=True)
            elif output.explanations.get("gradcam_placeholder"):
                st.caption(output.explanations.get("note", "Visual explanation placeholder."))
                st.dataframe(output.explanations["gradcam_placeholder"], use_container_width=True)
            else:
                st.write("No explanation available for this modality.")

    if fusion_output.warnings:
        st.subheader("Warnings")
        for warning in fusion_output.warnings:
            st.warning(warning)


def _load_sample_patient(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
