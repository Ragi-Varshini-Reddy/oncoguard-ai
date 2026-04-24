# OralCare-AI

Explainable multimodal oral cancer diagnosis-support prototype for the KR0468 hackathon track.

This repository is a working decision-support demo, not a clinically validated diagnostic product. It integrates standalone modality wrappers for:

- intraoral image upload
- clinical structured data
- histopathology image upload
- genomic structured data
- missing-modality-aware fusion
- explainability summaries
- downloadable HTML report

## Quick Start: FastAPI + React/Vite

Backend:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## Genomics Real-Data Flow

Train artifact:

```bash
.venv/bin/python -m src.phase3_genomics.train --input /path/to/labeled_tcga_hnsc_gene_panel.csv
```

Required training columns are listed by:

```bash
curl http://127.0.0.1:8000/api/genomics/schema
```

Inference requires a trained artifact at `artifacts/genomics_model.joblib`. The FastAPI path reports an error instead of silently using mock predictions when the artifact is missing.

See [docs/REAL_MODEL_PLAN.md](docs/REAL_MODEL_PLAN.md).

## Team Phase Folders

The work is split so three people can develop in parallel while sharing one backend, one frontend, and one contract system.

- [Phase 1](<Phase 1/README.md>) - intraoral imaging + clinical data
- [Phase 2](<Phase 2/README.md>) - histopathology
- [Phase 3 - Genomics](<Phase 3 - Genomics/README.md>) - current genomics training/inference/XAI/fusion/report flow

Frontend routes:

- `#/phase-1`
- `#/phase-2`
- `#/phase-3-genomics`

The frontend remains single-app React/Vite. Add phase-specific screens inside `frontend/src/main.jsx` or split into components under `frontend/src/` later.

## Legacy Streamlit Prototype

```bash
python3 -m pip install -r requirements.txt
streamlit run src/app/streamlit_app.py
```

The code is designed to keep running when heavyweight ML dependencies are unavailable. The tested core path uses dependency-light deterministic demo wrappers and clearly marks them as prototype/demo outputs.

## Demo Data

Sample payloads live in `data_samples/`.

- `sample_patient_001.json` is synthetic demo data.
- `sample_genomics.csv` contains model-ready gene-panel features.

External Kaggle, Mendeley, GEO, TCGA, and SEER sources should not be treated as patient-matched unless IDs are truly aligned. For hackathon demos, use explicit sample patient payloads.

## Validation

```bash
python3 -m unittest discover -s tests
```

If `pytest` is installed:

```bash
pytest
```
