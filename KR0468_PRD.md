# Product Requirements Document — KR0468 OralCare-AI

## 1. Product name
**OralCare-AI: Explainable Multimodal Oral Cancer Diagnosis and Risk Assessment Platform**

## 2. Hackathon context
Time remaining is limited, so the product must be built as a **working clinical-decision-support prototype**, not a regulatory-grade medical product. The goal is to impress judges through a clear end-to-end demo, modular engineering, visible explainability, and a realistic path to clinical deployment.

## 3. Problem statement
Current oral cancer AI systems are often single-modality models that work only on one data type such as clinical images, histopathology, clinical parameters, or genomic data. They often lack explainability, cannot handle missing patient modalities, and are difficult to use in a clinical workflow. OralCare-AI addresses this by providing a web-based system that combines available patient data, produces diagnosis/risk predictions, and explains the evidence by modality.

## 4. Product vision
Build a multimodal AI prototype that allows a clinician or evaluator to upload oral lesion images, histopathology images, clinical parameters, and optional genomic features, then receive:

- diagnosis support
- risk stratification
- confidence and uncertainty flags
- image heatmaps
- clinical/genomic feature explanations
- modality contribution summary
- downloadable report

## 5. Target users
1. **Dentist / oral clinician** — wants quick screening support.
2. **Pathologist** — wants visual evidence from histopathology patches.
3. **Oncology researcher** — wants multimodal risk explanation.
4. **Hackathon judges** — want a complete, impressive, explainable prototype.

## 6. Core value proposition
OralCare-AI is not just a classifier. It is a **modular multimodal clinical AI workflow** that can operate even when some modalities are missing and can explain which modality influenced the final decision.

## 7. Hackathon USP factors
These features should be prioritized because they create strong demo impact:

1. **Missing-modality-aware inference** — system works with only image + clinical data, then improves when histopathology/genomics are added.
2. **Modality contribution score** — shows how much intraoral image, clinical, histopathology, and genomics influenced the final output.
3. **Doctor-readable report** — produces a PDF/HTML report with prediction, evidence, confidence, and disclaimer.
4. **Genomic biomarker explanation** — shows top genes/features affecting the risk score.
5. **Patient simulator** — preloaded sample patients for reliable demo even if live uploads fail.
6. **Ablation comparison** — show image-only vs image+clinical vs multimodal prediction.
7. **Clinical warning banner** — “Decision support only, not final diagnosis.” This makes the prototype look clinically responsible.

## 8. Scope

### In scope for hackathon
- Intraoral image upload and classification.
- Clinical form ingestion.
- Histopathology patch/image upload and classification.
- Genomic CSV upload or predefined gene panel input.
- Separate modality encoders.
- Simple or transformer-based multimodal fusion.
- Explainability outputs.
- Web demo using Streamlit or FastAPI backend with simple frontend.
- Downloadable report.

### Out of scope for hackathon
- Full clinical validation.
- Regulatory approval.
- Raw FASTQ/BAM/VCF processing.
- Real hospital EHR integration.
- Whole-slide image processing at production scale.
- Claims of definitive diagnosis.

## 9. User stories

### US1 — Clinician uploads patient data
As a clinician, I want to upload available patient data so that the system can generate diagnosis support and risk assessment.

### US2 — Clinician sees explainability
As a clinician, I want to see why the model made a decision so that I can judge whether to trust the output.

### US3 — Researcher compares modalities
As a researcher, I want to see how each modality changes prediction so that I can understand the added value of multimodal AI.

### US4 — Hackathon judge sees reliable demo
As a judge, I want a smooth demo with sample patients, visual outputs, and a clear workflow.

## 10. Functional requirements

### FR1 — Data ingestion
The system shall accept:
- intraoral image file: JPG/PNG
- histopathology image file or patch image: JPG/PNG/TIFF where feasible
- clinical form values
- genomic CSV file or preloaded sample genomic features

### FR2 — Data validation
The system shall validate:
- file type
- missing fields
- feature names
- value ranges
- patient ID format
- modality availability

### FR3 — Cohort builder
The system shall maintain a patient-level record containing:
- patient_id
- label, if known
- available modalities
- modality masks
- split assignment
- paths or feature payloads

### FR4 — Intraoral module
The system shall preprocess intraoral images, generate a fixed embedding, produce a standalone prediction, and optionally create a Grad-CAM heatmap.

### FR5 — Clinical module
The system shall preprocess structured clinical features, generate an embedding, and produce feature importance explanations.

### FR6 — Histopathology module
The system shall preprocess histopathology images or patches, generate a fixed embedding, and produce visual evidence using attention/Grad-CAM.

### FR7 — Genomics module
The system shall ingest tabular genomic features, validate schema, impute missing values, normalize features, perform feature selection or gene panel selection, generate a fixed-size embedding, produce standalone risk output, and show top genomic features.

### FR8 — Fusion module
The system shall accept modality embeddings and a modality mask, then generate a joint embedding using either:
- concatenation baseline, or
- gated attention, or
- cross-modal transformer.

### FR9 — Prediction heads
The system shall output:
- diagnosis class
- risk score or risk class
- calibrated confidence
- low-confidence warning when necessary

### FR10 — Explainability layer
The system shall display:
- image heatmap for intraoral/histopathology branches
- SHAP/top-feature explanation for clinical and genomic branches
- modality contribution score for fusion

### FR11 — Report generation
The system shall generate a report containing:
- patient ID
- available modalities
- predictions
- confidence
- modality contributions
- explanation summary
- clinical disclaimer

## 11. Non-functional requirements

### NFR1 — Speed
For hackathon demo, inference should complete in under 10 seconds on CPU for sample patients. GPU acceleration is optional.

### NFR2 — Reliability
The app must not crash when a modality is missing. It should display “modality not provided” and continue inference.

### NFR3 — Reproducibility
All model runs must use fixed seeds and saved train/validation/test splits.

### NFR4 — Maintainability
The codebase must be modular, with separate directories for each modality and common contracts.

### NFR5 — Safety
The app must clearly say that the system is for decision support only and not a final medical diagnosis.

## 12. Data requirements

### Main data sources for prototype
- Kaggle oral cancer/lips/tongue image datasets for intraoral images.
- Mendeley oral cancer dataset for image/histopathology experiments, depending on contents.
- TCGA-HNSC through GDC for clinical and genomic features.
- GEO as optional external gene-expression validation.
- SEER as optional population-level background, not direct patient-level multimodal fusion.

### Required tables
- `patients.csv`
- `clinical.csv`
- `genomics.csv`
- `image_manifest.csv`
- `histopath_manifest.csv`
- `splits.csv`
- `labels.csv`

## 13. Model requirements

### Modality encoder output dimensions
- intraoral image: 256-d embedding
- clinical: 128-d embedding
- histopathology: 256-d embedding
- genomics: 128-d embedding

### Fusion hidden dimension
- 256 by default

### Output tasks
For hackathon, use the most feasible label setup:
1. Binary: cancer vs non-cancer, or
2. Three-class: benign / precancer / cancer, if labels are available.

Risk can be derived as:
- low, medium, high from predicted probability thresholds, or
- standalone genomic/clinical risk score if labels allow.

## 14. Evaluation requirements

Minimum metrics:
- accuracy
- precision
- recall
- F1 score
- ROC-AUC if binary
- confusion matrix

Demo metrics:
- modality ablation table
- inference time
- missing-modality behavior test

## 15. Integration requirements

Every module must return this payload:

```json
{
  "patient_id": "P001",
  "modality": "genomics",
  "status": "available",
  "embedding": [0.01, 0.02],
  "embedding_dim": 128,
  "prediction": {},
  "confidence": 0.82,
  "explanations": {},
  "schema_version": "1.0"
}
```

The fusion module must accept:

```json
{
  "patient_id": "P001",
  "embeddings": {
    "intraoral": [],
    "clinical": [],
    "histopathology": [],
    "genomics": []
  },
  "modality_mask": {
    "intraoral": true,
    "clinical": true,
    "histopathology": false,
    "genomics": true
  }
}
```

## 16. Acceptance criteria

The project is acceptable if:
- each phase works independently
- end-to-end demo works with sample patient
- missing modality does not break inference
- final report is generated
- XAI output is visible
- genomics module produces top-feature explanation
- architecture and contracts are documented

## 17. Demo flow

1. Open app dashboard.
2. Select sample patient or upload patient data.
3. Show available modalities.
4. Run separate modality predictions.
5. Run multimodal fusion.
6. Display final diagnosis/risk.
7. Display heatmap and feature explanations.
8. Display modality contribution chart.
9. Download report.

## 18. Recommended 17-hour execution priority

### First 3 hours
- finalize contracts
- create repo structure
- create sample patient payloads
- make dummy embeddings if model training is incomplete

### Hours 4–8
- complete modality inference wrappers
- complete genomics preprocessing + encoder baseline
- connect Streamlit UI

### Hours 9–12
- implement fusion baseline plus contribution scores
- generate explanations
- create report template

### Hours 13–15
- polish UI, add sample patients, add demo charts

### Hours 16–17
- rehearse demo, fix bugs, prepare pitch

## 19. Critical hackathon recommendation
If full model training is not stable, still deliver the system using pretrained encoders or saved dummy/sample embeddings, but be transparent in the demo. Judges will value an integrated, explainable, clinically realistic workflow more than a fragile model that only shows accuracy.
