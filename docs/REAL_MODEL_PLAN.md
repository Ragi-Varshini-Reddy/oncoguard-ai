# OralCare-AI Real Model Plan

## Immediate Priority

Build the genomics flow first:

1. Collect a labeled gene-expression or mutation-summary table.
2. Train and save preprocessing artifacts on training data only.
3. Run patient inference from uploaded molecular table.
4. Produce feature-level XAI.
5. Send genomics module output to missing-modality-aware fusion.
6. Generate clinician-facing report.

The current FastAPI path is artifact-backed. It does not silently fall back to a mock model when the trained genomics artifact is missing.

## What We Need

### Genomics

Required training table:

- `patient_id`
- `risk_label` or configured target label
- configured gene panel columns:
  - `TP53_expr`
  - `CDKN2A_expr`
  - `EGFR_expr`
  - `PIK3CA_expr`
  - `NOTCH1_expr`
  - `CCND1_expr`
  - `FAT1_expr`
  - `CASP8_expr`
  - `HRAS_expr`
  - `MET_expr`
  - `MYC_expr`
  - `MDM2_expr`

Recommended real source:

- TCGA-HNSC from GDC for transcriptomic, mutation, clinical, and survival-related fields.

Labels we can realistically build:

- tumor vs normal, if normal samples are available
- high-risk vs low-risk using survival/event proxy
- stage-derived risk, if clinical stage is available
- recurrence/progression proxy, if present

Avoid:

- FASTQ/BAM/VCF processing in this prototype
- pretending TCGA/GEO/Kaggle/Mendeley patients are matched unless IDs actually align

### Intraoral Images

Need:

- image files
- patient/image IDs
- lesion labels
- train/validation/test splits
- source metadata

Model approach:

- start with ImageNet-pretrained EfficientNetV2, ConvNeXt, or ViT
- fine-tune classification head first
- unfreeze later layers only if dataset size allows
- Grad-CAM or attention rollout for visual explanation

### Histopathology

Need:

- WSI/patch images
- slide or patch labels
- patient IDs
- patch extraction strategy
- train/validation/test splits

Model approach:

- use pathology foundation embeddings where license/access permits, such as CONCH or UNI
- train a lightweight classifier or multiple-instance-learning head on extracted embeddings
- fine-tune only if enough labeled pathology data and compute are available

### Clinical

Need:

- structured clinical table
- clean label mapping
- missing-value policy
- categorical vocabulary

Model approach:

- start with calibrated logistic regression, random forest, or gradient boosting
- use SHAP/permutation/coefficients for feature explanations

## Do We Train a New Model for Each Modality?

Yes, each modality needs its own encoder or model wrapper because the input distributions are different.

Recommended sequence:

1. Genomics: train tabular model now.
2. Clinical: train lightweight tabular model.
3. Intraoral: fine-tune pretrained vision model.
4. Histopathology: extract pretrained pathology embeddings, then train a classifier/MIL head.
5. Fusion: train only when we have patient-matched module outputs.

## Pretrained Models and Fine-Tuning

Use pretrained models where they match the modality:

- Intraoral: general pretrained vision backbones are useful; fine-tuning is required.
- Histopathology: pathology foundation models are preferred; often train a downstream head first.
- Genomics: single-cell foundation models like scGPT/Geneformer are not a clean first choice for bulk TCGA gene panels. Start with strong tabular baselines. Consider foundation embeddings only if the input data type matches.
- Clinical: no pretrained model needed for the prototype; use transparent tabular models.

## Fusion Strategy

Best practical approach:

1. Each modality produces:
   - embedding
   - prediction
   - confidence
   - explanation
2. Fusion consumes only patient-matched module outputs.
3. Missing modalities are masked.
4. In early real-data phase, use calibrated late fusion:
   - weighted ensemble from modality confidence and validation performance
   - zero contribution for missing modalities
5. Once enough matched multimodal patients exist, train:
   - gated attention fusion over modality tokens
   - optional transformer encoder over modality embeddings
   - calibration layer on validation data

Do not train a fusion model by mixing unrelated patients from different datasets.

## Users

### Patient

Needs:

- plain-language result summary
- downloadable patient-friendly report
- clear disclaimer and next-step guidance

Should not see:

- raw gene coefficients without explanation
- unreviewed model internals presented as diagnosis

### Doctor

Needs:

- run inference
- see modality availability
- see diagnosis/risk/confidence
- inspect XAI
- compare modalities
- download clinical report

### Lab Technician

Needs:

- upload genomics table
- validate schema
- view missing/invalid features
- train or update artifact with approved dataset
- verify QC before doctor review

## Current Implementation

Implemented now:

- artifact-backed genomics training and inference
- coefficient-based genomics XAI
- missing-modality-aware fusion endpoint
- HTML/PDF report generation
- FastAPI backend
- React/Vite frontend
- role-specific UI for patient, doctor, and lab technician

Pending real data:

- TCGA-HNSC/GDC data extraction
- label definition
- artifact training with real table
- validation metrics and calibration
