# OralCare-AI — Improvement Plan and USP Enhancement Document

## 1. Current Implemented Direction

The system has been upgraded from a basic multimodal AI prototype into a clinical workflow-oriented platform with:

- Demo user login
- Role-based access for patient, doctor, and lab technician
- SQLite-backed local persistence
- Patient records
- Model run history
- Appointments
- Uploaded documents
- LLM-based patient/doctor assistant
- Layer-6 prediction heads
- Report generation
- Genomics/fusion/XAI panels

This improves the project from a simple “AI classifier” into a usable clinical decision-support workflow.

## 2. Core Product Positioning

**Product Name:** OralCare AI Companion

**One-Line Pitch:**
OralCare AI Companion is an explainable multimodal clinical decision-support prototype that combines oral images, histopathology, clinical parameters, and genomics to diagnose, explain, track, and prioritize oral cancer risk across patient visits.

**Strong Hackathon Positioning:**
Most competitors will build:
`Image Upload → AI Prediction → Cancer / Non-Cancer`

Our system provides:
`Patient Login → Multimodal Data Upload → Genomics + Image + Clinical + Histopathology Evidence → Missing-Modality-Aware Fusion → Layer-6 Diagnosis/Risk/Confidence Heads → Explainable Report Card → Longitudinal Risk Tracking → Doctor Alerting → Context-Aware AI Companion`

## 3. Gap-Filling Improvements

### Gap 1: Existing systems are mostly one-shot classifiers
**Problem:** Most AI systems give a single prediction for a single uploaded image. This is not enough for oral cancer because risk can evolve over time through precancerous stages such as leukoplakia, erythroplakia, and OSMF.
**Improvement:** Add a Longitudinal Risk Tracker.
**How it works:**
For every patient, store each model run:
`patient_id, run_id, date, diagnosis, risk_score, risk_level, confidence, available_modalities, modality_contribution, warnings`
Then plot risk over time:
- Visit 1: 24% risk — Low
- Visit 2: 41% risk — Medium
- Visit 3: 67% risk — High
**Alert rule:** If current risk score - previous risk score >= 15 percentage points: trigger priority alert
**Why it is unique:** This shifts the product from single prediction to patient disease progression monitoring.

### Gap 2: Existing systems do not support real clinical roles
**Problem:** Most prototypes have one generic dashboard. But real clinical workflows involve different users: patient, doctor, lab technician.
**Improvement:** Use role-specific dashboards.
- **Patient dashboard:** Should show latest health status, latest diagnosis/risk/confidence, assigned doctor, reports, documents, appointment request form, patient-friendly AI companion.
- **Doctor dashboard:** Should show assigned patient list, patient detail page, run/review inference, XAI and fusion panel, report download, appointment approval/reschedule, doctor-facing AI assistant.
- **Lab technician dashboard:** Should show patient ID lookup, genomics upload, genomics validation, genomics quality flags, upload status.
**Why it is unique:** This makes the system look like a hospital workflow product, not just an ML demo.

### Gap 3: Existing multimodal systems use weak fusion
**Problem:** Many multimodal systems simply concatenate features: `image_features + clinical_features + genomic_features → classifier`. This is weak because all modalities are treated equally, missing modalities break the system, modality contribution is not explainable, interaction between modalities is not patient-specific.
**Improvement:** Use Adaptive Missing-Modality-Aware Gated Fusion.
**Fusion logic:**
1. Each modality produces an embedding.
2. Each embedding is projected to a common dimension.
3. A modality mask indicates which data is available.
4. A gated attention layer computes modality weights.
5. Only available modalities contribute to the final fused representation.
6. The model outputs diagnosis, risk, confidence, and modality contribution.
**Why it is unique:** The model can still run even if biopsy or genomic data is unavailable.

### Gap 4: Existing tools lack fusion-level explainability
**Problem:** Some systems show only a Grad-CAM heatmap. That explains an image model, not the full multimodal decision.
**Improvement:** Add multi-layer explainability.
**XAI stack:**
- Intraoral image: Grad-CAM heatmap
- Histopathology: patch attention or Grad-CAM
- Clinical: SHAP feature importance
- Genomics: top gene-level contributors
- Fusion: modality contribution chart
- Layer-6 heads: diagnosis threshold explanation, risk threshold explanation, confidence penalty explanation
**Why it is unique:** This explains not only what the model predicted, but also which modality influenced the prediction, why risk was high, which missing data reduced confidence, and which factors changed between visits.

### Gap 5: Existing systems are not clinically deployable
**Problem:** Many AI papers and prototypes stop at model accuracy. They do not include user access, patient history, reports, appointments, model run history, role-specific workflow, audit trail.
**Improvement:** Use a clinical workflow layer.
**Why it is unique:** This fills the “deployability” gap by creating an end-to-end platform.

## 4. USP Stack

**USP 1: OralCare AI Companion**
A contextual AI assistant that answers questions using patient-specific context.

**USP 2: Longitudinal Risk Tracker**
A timeline that shows risk progression across multiple visits.

**USP 3: Explainable Risk Report Card**
A one-click clinician-ready report.

**USP 4: Role-Based Clinical Workflow**
The system supports three real users (Patient, Doctor, Lab Technician).

**USP 5: Layer-6 Prediction Heads**
Layer-6 heads make final outputs more structured and explainable (Diagnosis Head, Risk Head, Confidence Calibration Head).

## 5. Recommended Improvements to Add Now

- **Improvement 1: Add a Longitudinal Risk Table** (`risk_history` table or derive from `model_runs`)
- **Improvement 2: Add Alert Engine** (e.g. `RISK_JUMP_ALERT`, `HIGH_RISK_ALERT`, `LOW_CONFIDENCE_ALERT`)
- **Improvement 3: Improve LLM Companion Grounding** (retrieve patient profile, latest model run, missing modalities, etc.)
- **Improvement 4: Add Modality Coverage Score** (e.g., Coverage Score: 75%)
- **Improvement 5: Add Confidence Penalty Explanation** (e.g., Penalty: Histopathology missing -8%)
- **Improvement 6: Add Genomics Quality Panel** (Features received, missing rate, top genomic contributors)
- **Improvement 7: Add Report Consistency** (UI result and PDF report must use the same `run_id`)

## 6. Fusion Efficiency Improvements

**Current safe hackathon approach:** Use evidence-weighted late fusion until true patient-matched multimodal training data exists.

## 7. Suggested Additional Database Tables

- `risk_alerts`
- `risk_history`
- `audit_events`

## 8. Recommended UI Improvements

- **Patient Dashboard**: Latest Risk Status, Risk Timeline, Assigned Doctor, Reports, Appointments, Ask OralCare Companion
- **Doctor Dashboard**: Patient List, Priority Alerts, Risk Timeline, XAI Summary, Modality Contributions, Report Generator, Appointment Requests, Doctor Assistant
- **Lab Dashboard**: Upload Genomics, Validate Genomics, Quality Flags, Urgent Patient Queue
