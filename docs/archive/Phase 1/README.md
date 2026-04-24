# Phase 1 - Intraoral Imaging + Clinical Data

## Owner Scope

Phase 1 owns the first clinical screening layer of OralCare-AI:

- intraoral image upload and preprocessing
- intraoral image model or pretrained/fine-tuned vision encoder
- Grad-CAM or visual heatmap explanation
- clinical structured-data form
- clinical preprocessing, model, and feature-level explanation
- standalone modality outputs for `intraoral` and `clinical`
- integration with the shared FastAPI backend and the single React/Vite frontend

## Contract To Follow

Every Phase 1 module must return the shared module output contract:

```json
{
  "patient_id": "P001",
  "modality": "intraoral",
  "status": "available",
  "embedding": [],
  "embedding_dim": 256,
  "prediction": {},
  "confidence": 0.82,
  "explanations": {},
  "schema_version": "1.0"
}
```

For clinical outputs, use:

```json
{
  "modality": "clinical",
  "embedding_dim": 128
}
```

Allowed statuses:

- `available`
- `missing`
- `error`

## Current Code Locations

The current placeholder/demo wrappers live in:

- `src/phase1_intraoral_clinical/intraoral_inference.py`
- `src/phase1_intraoral_clinical/clinical_inference.py`

Frontend route placeholder:

- `#/phase-1`

## Files To Build Next

Suggested ownership layout:

```text
src/phase1_intraoral_clinical/
├── image_schema.py
├── image_preprocess.py
├── image_model.py
├── image_inference.py
├── image_explain.py
├── clinical_schema.py
├── clinical_preprocess.py
├── clinical_model.py
├── clinical_train.py
├── clinical_inference.py
└── clinical_explain.py
```

## Data Needed

### Intraoral Images

Need image files with labels and patient/image IDs:

- JPG/PNG intraoral lesion images
- label: benign / precancer / cancer, or binary cancer / non-cancer
- train/validation/test split
- source metadata

Possible sources:

- Kaggle oral cancer / lips / tongue image datasets
- Mendeley oral cancer image datasets, depending on dataset contents

Do not pretend these patients are matched with TCGA/GDC genomics unless IDs actually match.

### Clinical Data

Need structured rows:

- `patient_id`
- age
- sex
- tobacco use
- alcohol use
- lesion site
- lesion size
- ulcer duration
- neck node status
- label or risk endpoint

## Modeling Recommendation

### Intraoral

Start with a pretrained vision backbone:

- EfficientNetV2
- ConvNeXt
- ViT

Fine-tune classification head first. Add Grad-CAM for explainability.

### Clinical

Start with transparent tabular models:

- logistic regression
- random forest
- gradient boosting

Use coefficients, permutation importance, or SHAP for explanations.

## API Placeholders

Suggested FastAPI endpoints:

- `POST /api/phase1/intraoral/infer`
- `POST /api/phase1/clinical/infer`
- `POST /api/phase1/intraoral/train`
- `POST /api/phase1/clinical/train`

Do not change the shared fusion contract without coordinating with Phase 3.
