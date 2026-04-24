# Phase 2 - Histopathology

## Owner Scope

Phase 2 owns the histopathology branch of OralCare-AI:

- histopathology image or patch upload
- tissue/patch preprocessing
- pathology encoder or foundation-model embedding extraction
- standalone histopathology prediction
- histopathology visual explanation
- output contract for the `histopathology` modality
- integration with shared fusion and reporting

## Contract To Follow

Every Phase 2 module must return:

```json
{
  "patient_id": "P001",
  "modality": "histopathology",
  "status": "available",
  "embedding": [],
  "embedding_dim": 256,
  "prediction": {},
  "confidence": 0.82,
  "explanations": {},
  "schema_version": "1.0"
}
```

Allowed statuses:

- `available`
- `missing`
- `error`

## Current Code Location

The current placeholder/demo wrapper lives in:

- `src/phase2_histopathology/inference.py`

Frontend route placeholder:

- `#/phase-2`

## Files To Build Next

Suggested ownership layout:

```text
src/phase2_histopathology/
├── schema.py
├── preprocess.py
├── patching.py
├── model.py
├── train.py
├── inference.py
├── explain.py
└── utils.py
```

## Data Needed

Need histopathology patches or whole-slide image-derived patches:

- image/patch files
- slide ID
- patient ID if available
- diagnosis/risk label
- patch-level or slide-level labels
- train/validation/test split

Possible source:

- Mendeley oral cancer histopathology datasets, depending on dataset contents

Important: do not treat histopathology images as matched with TCGA/GDC genomics unless IDs truly match.

## Modeling Recommendation

Best practical path:

1. Start with patch-level classification if patches are available.
2. Use pretrained pathology foundation embeddings where license/access permits.
3. Train a lightweight classifier head.
4. If whole-slide data exists, move toward multiple-instance learning.

Possible pretrained/foundation options:

- pathology foundation encoders such as CONCH or UNI, subject to access/license
- ImageNet-pretrained ConvNeXt/EfficientNet as fallback

## Explainability

Start with:

- Grad-CAM for CNN-style models
- patch attention scores for MIL models
- top contributing patch thumbnails for report/UI

## API Placeholders

Suggested FastAPI endpoints:

- `POST /api/phase2/histopathology/infer`
- `POST /api/phase2/histopathology/train`

Do not change the shared fusion contract without coordinating with Phase 3.
