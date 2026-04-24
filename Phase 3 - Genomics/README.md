# Phase 3 - Genomics

## Owner Scope

Phase 3 owns the genomics branch and the current highest-priority real-data flow:

- TCGA-HNSC/GDC gene-panel dataset builder
- artifact-backed genomics training
- artifact-backed genomics inference
- genomics XAI
- fusion integration
- report integration
- backend endpoints for genomics train/infer/schema
- frontend route for genomics workflow

## Current Implementation

Core code:

```text
src/phase3_genomics/
├── artifact_model.py
├── inference.py
├── preprocess.py
├── train.py
├── utils.py
└── explain.py
```

Backend:

```text
backend/app/main.py
```

Frontend:

```text
frontend/src/main.jsx
```

Data builder:

```text
scripts/build_tcga_hnsc_genomics_dataset.py
```

Current route:

```text
http://127.0.0.1:5173/#/phase-3-genomics
```

## Real Data Flow

Build TCGA-HNSC training table:

```bash
.venv/bin/python scripts/build_tcga_hnsc_genomics_dataset.py --output data/processed/tcga_hnsc_genomics_training.csv --max-cases 500
```

Train artifact:

```bash
.venv/bin/python -m src.phase3_genomics.train --input data/processed/tcga_hnsc_genomics_training.csv
```

Artifact path:

```text
artifacts/genomics_model.joblib
```

## Input Contract

Training CSV:

```text
patient_id,risk_label,TP53_expr,CDKN2A_expr,EGFR_expr,PIK3CA_expr,NOTCH1_expr,CCND1_expr,FAT1_expr,CASP8_expr,HRAS_expr,MET_expr,MYC_expr,MDM2_expr
```

Inference CSV:

```text
patient_id,sample_id,TP53_expr,CDKN2A_expr,EGFR_expr,PIK3CA_expr,NOTCH1_expr,CCND1_expr,FAT1_expr,CASP8_expr,HRAS_expr,MET_expr,MYC_expr,MDM2_expr
```

## Notes

The current label is a stage-derived risk proxy from TCGA-HNSC metadata. It is real GDC data, but not a clinically validated outcome label.
