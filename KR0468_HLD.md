# High-Level Design — KR0468 OralCare-AI

## 1. System overview
OralCare-AI is a modular multimodal AI platform for oral cancer diagnosis support and risk assessment. The system combines four modalities:

1. intraoral images
2. clinical structured data
3. histopathology images
4. genomic structured data

Each modality has its own preprocessing pipeline and encoder. The encoder outputs are projected to a common representation and fused using a missing-modality-aware fusion module. The system outputs diagnosis, risk, confidence, explainability, and a report.

## 2. High-level architecture

```text
Data Sources
  ↓
Cohort Builder + Contracts
  ↓
Modality Preprocessors
  ↓
Modality Encoders
  ↓
Projection Layer
  ↓
Missing-Modality-Aware Fusion
  ↓
Prediction Heads
  ↓
Explainability Layer
  ↓
Web UI + Report Generator
```

## 3. Major components

### 3.1 Data ingestion layer
Responsible for accepting raw inputs:
- intraoral image files
- histopathology image files
- clinical forms
- genomic CSV/features
- sample patient presets

### 3.2 Data governance and cohort builder
This is the most important integration component.

Responsibilities:
- normalize patient IDs
- normalize labels
- manage common splits
- create modality availability masks
- prevent accidental patient leakage
- ensure every module follows the same contracts

### 3.3 Preprocessing layer
Each modality preprocesses data separately:

| Modality | Preprocessing |
|---|---|
| Intraoral | resize, normalize, brightness/blur checks, augmentation |
| Clinical | impute, categorical encoding, numeric scaling |
| Histopathology | patch extraction, stain normalization, tissue filtering |
| Genomics | missingness filtering, imputation, log/z-score scaling, feature selection |

### 3.4 Encoder layer
Each encoder converts modality data to a fixed embedding.

| Encoder | Output |
|---|---|
| Image encoder | 256-d vector |
| Clinical encoder | 128-d vector |
| Histopathology encoder | 256-d vector |
| Genomics encoder | 128-d vector |

### 3.5 Projection layer
All embeddings are projected into a common hidden dimension, recommended `hidden_dim=256`.

### 3.6 Fusion layer
The fusion layer accepts:
- projected modality embeddings
- modality availability mask
- optional modality confidence scores

Recommended hackathon implementation:
1. Concatenation baseline.
2. Gated attention fusion.
3. If time permits, transformer encoder over modality tokens.

### 3.7 Prediction heads
Outputs:
- diagnosis class
- risk class or score
- confidence score
- warning if uncertainty is high

### 3.8 Explainability layer
Responsibilities:
- Grad-CAM for image branches
- SHAP/top-feature explanation for clinical and genomic modules
- modality contribution chart from fusion weights

### 3.9 Deployment layer
Recommended stack:
- Streamlit for fastest demo
- FastAPI optional if separating backend
- PyTorch for model loading
- ReportLab/Jinja2/WeasyPrint for report generation

## 4. Data flow

### 4.1 Single patient inference
1. User uploads patient data.
2. System validates modality availability.
3. Each available modality is preprocessed.
4. Each available encoder returns an embedding and standalone prediction.
5. Fusion module combines available embeddings using the mask.
6. Prediction heads output diagnosis/risk.
7. XAI module generates explanations.
8. UI displays results and report.

### 4.2 Missing modality behavior
If a modality is missing:
- status is set to `missing`
- embedding is omitted or zeroed only after mask is set
- fusion mask prevents fake contribution
- report says the modality was not provided

## 5. API-level contract

### Modality module output
```json
{
  "patient_id": "P001",
  "modality": "genomics",
  "status": "available",
  "embedding": [0.0],
  "embedding_dim": 128,
  "prediction": {
    "risk_score": 0.73,
    "risk_class": "high"
  },
  "confidence": 0.81,
  "explanations": {
    "top_features": []
  },
  "schema_version": "1.0"
}
```

### Fusion input
```json
{
  "patient_id": "P001",
  "module_outputs": [],
  "modality_mask": {
    "intraoral": true,
    "clinical": true,
    "histopathology": false,
    "genomics": true
  }
}
```

### Fusion output
```json
{
  "patient_id": "P001",
  "diagnosis": {
    "class": "cancer",
    "probability": 0.87
  },
  "risk": {
    "class": "high",
    "score": 0.82
  },
  "confidence": 0.79,
  "modality_contributions": {
    "intraoral": 0.34,
    "clinical": 0.18,
    "histopathology": 0.0,
    "genomics": 0.48
  },
  "warnings": ["Histopathology modality missing"]
}
```

## 6. Repository structure

```text
project/
├── configs/
├── data_contracts/
├── data_samples/
├── src/
│   ├── common/
│   ├── phase1_intraoral_clinical/
│   ├── phase2_histopathology/
│   ├── phase3_genomics/
│   ├── fusion/
│   ├── explainability/
│   ├── reporting/
│   └── app/
├── artifacts/
├── tests/
└── docs/
```

## 7. Risk and mitigation

| Risk | Mitigation |
|---|---|
| Dataset mismatch | Separate matched dataset from auxiliary pretraining datasets |
| Integration failure | Use strict contracts and tests |
| Model underperforms | Show ablation, XAI, and sample patients; do not rely only on accuracy |
| Missing modalities | Use modality masks |
| Time shortage | Use pretrained encoders and lightweight MLPs |

## 8. Hackathon implementation strategy
1. Build API contracts first.
2. Use lightweight models.
3. Ensure sample patient demo works.
4. Add one strong XAI view per modality.
5. Polish the UI and report.
