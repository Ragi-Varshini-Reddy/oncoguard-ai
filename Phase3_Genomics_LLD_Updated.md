# Phase 3 Genomics Module — Updated Low-Level Design (LLD)

**Project:** KR0468 — Explainable Multimodal AI System for Oral Cancer Diagnosis and Risk Assessment  
**Module Owner:** Phase 3 — Genomics  
**Document Type:** Low-Level Design  
**Version:** 2.0 Hackathon-Ready  
**Primary Goal:** Build an explainable genomics module that converts structured genomic/molecular patient features into a standardized embedding, standalone risk output, and explainability payload for multimodal fusion.

---

## 1. Module Purpose

The genomics module adds molecular-level evidence to the oral cancer diagnosis and risk assessment system.

Its job is to take structured genomic or molecular features, validate them, preprocess them, generate a compact embedding, produce a standalone genomic risk signal, and return a standardized output contract that can be consumed by the fusion layer.

This module is not responsible for raw sequencing workflows such as FASTQ processing, alignment, variant calling, or full bioinformatics pipelines. For the hackathon prototype, the module assumes model-ready genomic feature tables.

---

## 2. Scope of This Module

### 2.1 In Scope

- Accept genomic feature data in tabular form.
- Validate genomic input schema.
- Handle missing features safely.
- Normalize and transform genomic features.
- Select or use a fixed set of important genomic features.
- Generate a fixed-size genomics embedding.
- Produce standalone genomic prediction.
- Generate feature-level explanation.
- Return output in the common integration contract.
- Support missing-modality behavior.
- Support demo fallback mode if real model artifacts are not available.

### 2.2 Out of Scope

- Raw FASTQ processing.
- Whole-genome analysis.
- Variant calling from raw sequence reads.
- Clinical-grade biomarker validation.
- Regulatory approval.
- Hospital EHR integration.
- Real-time genomic report interpretation.

---

## 3. System Positioning

The genomics module sits after the patient/cohort alignment layer and before the multimodal fusion layer.

### 3.1 Upstream Inputs

The module receives:

- patient identifier
- optional sample identifier
- genomic feature table
- selected feature values
- metadata about dataset source
- label during training

### 3.2 Downstream Consumers

The module outputs to:

- multimodal fusion layer
- explainability layer
- clinical report generator
- dashboard UI
- audit/logging system

---

## 4. Hackathon Design Philosophy

Because this is a 17-hour hackathon context, the module prioritizes:

1. End-to-end integration readiness.
2. Explainability and visual usefulness.
3. Reliable handling of missing data.
4. Clean contracts with other modules.
5. Demonstrable molecular contribution in the final dashboard.
6. Simplicity over over-engineering.

The winning prototype should not depend on downloading large or difficult-access genomic models. A lightweight, transparent, and explainable genomic pipeline is preferred.

---

## 5. Recommended Prototype Strategy

### 5.1 Preferred Approach

Use a structured genomic feature table with a selected feature panel.

The recommended pipeline is:

1. Load patient genomic features.
2. Validate feature names and values.
3. Impute missing values.
4. Normalize values using saved preprocessing statistics.
5. Select top genomic features.
6. Pass features into a lightweight genomic encoder.
7. Generate a 128-dimensional embedding.
8. Generate standalone genomic risk prediction.
9. Generate top-feature explanation.
10. Return standardized output payload.

### 5.2 Recommended Modeling Choice

For hackathon delivery, use one of the following:

- XGBoost or Random Forest as a standalone genomic risk model.
- Small MLP encoder for embedding generation.
- Rule-based fallback for demo continuity if training is incomplete.

The final system should expose the same contract regardless of whether the module is running in real model mode, mock mode, or fallback mode.

---

## 6. Data Sources

### 6.1 Primary Source

**TCGA-HNSC through GDC Portal**

Use this as the primary genomics source because it contains head-and-neck cancer-related molecular and clinical information.

Useful data types:

- gene expression values
- mutation-related features
- copy number features, if available
- clinical survival or disease metadata

### 6.2 Secondary Source

**GEO datasets**

Use GEO only if time permits. It can support external validation or additional gene-expression experiments, but merging GEO with TCGA must be done carefully due to batch effects.

### 6.3 Demo Source

For hackathon reliability, prepare a small sample genomic CSV with selected molecular features.

Example feature categories:

- cancer-associated gene expression features
- mutation status features
- pathway-level scores
- molecular risk indicators

---

## 7. Recommended Feature Set

The module should use a compact and interpretable feature panel instead of thousands of genes.

### 7.1 Feature Categories

| Category | Examples | Purpose |
|---|---|---|
| Gene expression | TP53, EGFR, CDKN2A, PIK3CA | Molecular activity signal |
| Mutation flags | TP53_mut, NOTCH1_mut | Known cancer-associated variation indicators |
| Pathway scores | cell_cycle_score, apoptosis_score | Higher-level biological signal |
| Copy number features | EGFR_amp, CDKN2A_loss | Structural alteration signal |
| Clinical-linked molecular risk | HPV-related marker if available | Risk subgroup signal |

### 7.2 Hackathon-Safe Feature Panel

For the prototype, use approximately 10 to 50 features.

Suggested feature families:

- TP53-related signal
- EGFR-related signal
- CDKN2A-related signal
- PIK3CA-related signal
- NOTCH1-related signal
- cell-cycle score
- immune-response score
- apoptosis score
- proliferation score
- DNA-repair score

Do not claim these are clinically validated biomarkers unless externally supported. In the UI and report, describe them as **model-important molecular features**.

---

## 8. Input Data Contract

### 8.1 Required Input Fields

| Field | Type | Required | Description |
|---|---|---:|---|
| patient_id | string | Yes | Common patient identifier across system |
| sample_id | string | Optional | Genomic sample identifier |
| modality | string | Yes | Should be `genomics` |
| genomic_features | object/table | Yes if modality available | Key-value mapping of genomic features |
| source | string | Optional | TCGA, GDC, GEO, demo, manual |
| timestamp | string | Optional | Inference or upload time |

### 8.2 Example Logical Input

The input should conceptually contain:

- patient ID
- molecular feature names
- molecular feature values
- optional metadata

The exact implementation may use JSON, CSV, or internal dataframe representation, but it must map into the same logical structure.

---

## 9. Output Data Contract

Every genomics inference output must follow the shared module output contract.

### 9.1 Required Output Fields

| Field | Required | Description |
|---|---:|---|
| contract_version | Yes | Contract version used by this module |
| patient_id | Yes | Common patient ID |
| sample_id | Optional | Sample ID if available |
| modality | Yes | Must be `genomics` |
| status | Yes | available, missing, or error |
| mode | Yes | model, mock, rule_based, missing, or error_fallback |
| embedding | Yes if available | Fixed-size genomic embedding |
| embedding_dim | Yes | Must match feature contract |
| prediction | Optional | Standalone genomic prediction |
| confidence | Optional | Confidence score from genomic module |
| quality_flags | Yes | Input quality and preprocessing status |
| explanations | Optional | Feature importance explanation |
| warnings | Yes | List of warnings |

### 9.2 Embedding Requirement

The genomics module must output:

- encoder output dimension: **128**
- fusion projected dimension: **256** handled by projection layer

The genomics module itself should own the 128-dimensional output. The fusion module owns the projection to the common hidden dimension.

---

## 10. Missing Modality Contract

If genomic data is not provided, the module must not crash.

It must return:

- status: missing
- mode: missing
- embedding: null
- embedding_dim: 128
- prediction: null
- confidence: null
- warning explaining that genomics was not provided

The fusion layer will then create a modality mask.

For example:

- intraoral available
- histopathology available
- clinical available
- genomics missing

The fusion modality mask becomes:

- intraoral: 1
- histopathology: 1
- clinical: 1
- genomics: 0

Zero vectors may be used internally inside fusion only when accompanied by this mask.

---

## 11. Genomics Module Internal Components

### 11.1 Input Validator

Purpose:

- confirm patient ID exists
- confirm genomic payload is present when expected
- check required feature names
- check value types
- detect missing values
- detect invalid strings
- detect out-of-range values

Output:

- validation status
- missing feature list
- invalid feature list
- quality flags
- warnings

### 11.2 Feature Mapper

Purpose:

- map incoming feature names to canonical names
- handle aliases
- enforce fixed feature order
- create consistent model input vector

Example:

- `tp53` → `TP53_expr`
- `EGFR Expression` → `EGFR_expr`
- `cdkn2a` → `CDKN2A_expr`

### 11.3 Missing Value Handler

Purpose:

- impute missing numeric values
- track which features were imputed
- add warning if too many features are missing

Hackathon recommendation:

- use median imputation from training data
- if no artifact exists, use configured demo medians
- reject or downgrade confidence when missing rate is high

### 11.4 Normalizer

Purpose:

- apply saved scaling parameters
- keep train/test preprocessing consistent
- prevent data leakage

Recommended method:

- z-score standardization
- scaler fitted only on training data
- saved as preprocessing artifact

### 11.5 Feature Selector

Purpose:

- reduce high-dimensional genomics into a compact feature panel
- improve interpretability
- prevent overfitting

Recommended hackathon methods:

- fixed domain-informed gene panel
- top-k variance-filtered features
- mutual-information-selected features
- fallback configured feature list

### 11.6 Genomic Encoder

Purpose:

- convert selected genomic features into a 128-dimensional embedding

Recommended options:

- small MLP encoder
- lightweight dense network
- deterministic fallback embedding in mock mode

The encoder must not change the output dimension without updating the feature contract.

### 11.7 Standalone Genomic Predictor

Purpose:

- produce genomics-only risk or diagnosis prediction
- help with ablation studies
- help debug whether genomics contributes useful signal

Recommended outputs:

- diagnosis probability
- risk score
- risk class
- confidence

### 11.8 Genomic Explainability Engine

Purpose:

- explain which molecular features contributed most
- produce patient-level feature ranking
- provide report-ready explanation

Recommended methods:

- SHAP if available
- permutation importance fallback
- model coefficient importance fallback
- rule-based ranking fallback for demo

### 11.9 Output Formatter

Purpose:

- package all module results into the common integration contract
- include warnings and quality flags
- ensure fusion can consume the result directly

---

## 12. Data Flow

### 12.1 Available Genomics Flow

1. Receive patient genomic payload.
2. Validate patient ID and feature payload.
3. Map features to canonical names.
4. Reorder features according to feature registry.
5. Impute missing values.
6. Normalize using saved scaler.
7. Select configured features.
8. Generate genomic embedding.
9. Generate standalone genomic prediction.
10. Generate top-feature explanations.
11. Attach quality flags and warnings.
12. Return standardized module output.

### 12.2 Missing Genomics Flow

1. Receive patient payload without genomic data.
2. Mark genomics as missing.
3. Return missing-modality output.
4. Fusion continues with other available modalities.

### 12.3 Error Flow

1. Detect invalid input or processing failure.
2. Mark status as error.
3. Set mode as error fallback.
4. Return no embedding or fallback embedding based on configuration.
5. Add clear warning.
6. Fusion either ignores genomics or uses safe fallback depending on system configuration.

---

## 13. Configuration Requirements

The genomics module should be driven by configuration rather than hardcoded values.

### 13.1 Required Config Sections

| Section | Purpose |
|---|---|
| module | module name, version, mode |
| paths | artifact paths, feature registry paths |
| features | selected feature list and feature order |
| preprocessing | imputation and scaling settings |
| model | model type, embedding dimension, thresholds |
| explainability | explanation method and top-k count |
| fallback | demo fallback behavior |
| quality | missingness and confidence thresholds |

### 13.2 Useful Parameters

Recommended parameters:

- embedding dimension: 128
- projected dimension: 256
- top explanation features: 5
- maximum allowed missing feature rate: 0.3
- low confidence threshold: 0.6
- high risk threshold: 0.7
- medium risk threshold: 0.4
- feature normalization: z-score
- imputation strategy: median
- demo mode: enabled or disabled

---

## 14. Quality Flags

The module should return quality information with every prediction.

### 14.1 Required Quality Flags

| Flag | Description |
|---|---|
| input_valid | Whether payload passed validation |
| missing_feature_count | Number of missing genomic features |
| missing_feature_rate | Percentage of missing genomic features |
| used_imputation | Whether imputation was applied |
| out_of_distribution | Whether values appear outside expected range |
| low_confidence | Whether module confidence is low |
| artifact_loaded | Whether real model/preprocessor artifacts were loaded |

### 14.2 Why This Matters

Quality flags make the system look clinically responsible. They also help the fusion layer and UI decide when to show warnings.

Example warning logic:

- if missing rate is high, reduce confidence
- if model artifact is missing, mark mode as mock
- if out-of-distribution values exist, show caution in report

---

## 15. Explainability Design

### 15.1 Explanation Output

The module should return top molecular features with:

- feature name
- feature value
- importance score
- direction of contribution
- human-readable description if available

### 15.2 Direction Values

Allowed direction values:

- increases_risk
- decreases_risk
- neutral
- unknown

### 15.3 Explanation Methods

Preferred order:

1. SHAP local explanation.
2. Permutation importance fallback.
3. Model coefficient fallback.
4. Rule-based demo ranking fallback.

### 15.4 Report Language

Use cautious language:

- “Model-important molecular features”
- “Genomic risk signal”
- “Contributed to the model output”

Avoid overclaiming:

- Do not say “this gene proves cancer.”
- Do not say “this is a validated biomarker” unless the project has validated it.
- Do not say “diagnosis confirmed by genomics.”

---

## 16. Standalone Prediction Design

The genomics module should support standalone prediction even before fusion.

### 16.1 Standalone Outputs

- diagnosis probability
- risk score
- risk class
- confidence
- top contributing genomic features

### 16.2 Why Standalone Prediction Is Needed

It helps with:

- ablation studies
- debugging
- fallback operation
- demo visualization
- proving that the genomics module contributes value

### 16.3 Risk Class Mapping

Recommended mapping:

| Risk Score Range | Risk Class |
|---|---|
| 0.00 to 0.39 | low_risk |
| 0.40 to 0.69 | medium_risk |
| 0.70 to 1.00 | high_risk |

Thresholds should be configurable.

---

## 17. Fusion Integration Design

### 17.1 Genomics Output to Fusion

The fusion module expects:

- modality name: genomics
- status
- embedding
- embedding dimension
- confidence
- warnings
- explanation summary

### 17.2 Fusion Behavior

If genomics is available:

- use the 128-dimensional genomics embedding
- project to 256 dimensions in fusion projection layer
- include genomics in modality mask

If genomics is missing:

- mark genomics mask as 0
- ignore genomics attention contribution
- preserve warning in final report

### 17.3 Modality Order

The global modality order must be:

1. intraoral
2. histopathology
3. clinical
4. genomics

This order must not change unless the feature contract version changes.

---

## 18. UI and Report Requirements for Genomics

The final dashboard should show a genomics card.

### 18.1 Genomics Card Contents

- genomics status: available, missing, or error
- genomic risk score
- genomic risk class
- confidence
- top 5 molecular contributors
- missing feature warning if applicable
- explanation method

### 18.2 Suggested UI Copy

“Genomics module detected a high molecular risk signal. The most influential model features were TP53_expr, EGFR_expr, and CDKN2A_expr. These are model-derived explanations and should be interpreted as decision-support evidence, not standalone diagnostic proof.”

### 18.3 Report Contents

The generated report should include:

- genomic data availability
- genomic risk estimate
- top molecular features
- quality flags
- warnings
- disclaimer

---

## 19. Hackathon USP Features From Genomics

Your module can strongly impress judges if you highlight:

### 19.1 Molecular Evidence Layer

Most image classifiers stop at image prediction. Your system adds molecular-level reasoning.

### 19.2 Missing Genomics Support

The system does not fail if genomic data is unavailable.

### 19.3 Explainable Feature Contribution

The system shows top genomic features contributing to risk.

### 19.4 Confidence and Quality Flags

The system tells the user when genomic input is incomplete or low confidence.

### 19.5 Clinical Report Readiness

The genomic explanation is automatically converted into report-friendly language.

---

## 20. Failure Modes and Handling

| Failure Case | Expected Behavior |
|---|---|
| No genomic data uploaded | Return missing contract; fusion continues |
| Some features missing | Impute values; add warning; reduce confidence if needed |
| Too many features missing | Mark low confidence or unavailable depending threshold |
| Unknown feature names | Ignore or map if alias exists; warn user |
| Non-numeric values | Validation error or conversion if safe |
| Model artifact missing | Switch to mock or rule-based fallback mode |
| Explainer unavailable | Use fallback explanation and mark method clearly |
| Out-of-range values | Flag out-of-distribution warning |

---

## 21. Testing Requirements

### 21.1 Contract Tests

Verify that output always includes:

- contract version
- patient ID
- modality
- status
- mode
- embedding dimension
- quality flags
- warnings

### 21.2 Shape Tests

Verify:

- genomics embedding dimension is 128
- missing modality returns null embedding
- fusion projection receives expected shape

### 21.3 Missing Data Tests

Test cases:

- all features present
- one feature missing
- many features missing
- no genomic payload

### 21.4 Explanation Tests

Verify:

- top features are returned
- importance values exist
- explanation method is recorded
- fallback explanation works

### 21.5 Integration Tests

Verify:

- genomics module output plugs into fusion payload
- modality mask is correct
- dashboard can render genomics output
- report generator can consume genomics explanations

---

## 22. Acceptance Criteria

The genomics module is accepted if:

1. It accepts a sample genomic CSV or payload.
2. It validates and preprocesses input.
3. It handles missing genomic data safely.
4. It outputs a 128-dimensional embedding when available.
5. It returns the standard integration contract.
6. It produces standalone genomic risk output.
7. It returns top genomic feature explanations.
8. It supports model, mock, or fallback mode.
9. It integrates with fusion without changing other modules.
10. It provides report-ready genomic interpretation.

---

## 23. Recommended Deliverables for Phase 3

### 23.1 Documentation Deliverables

- Genomics LLD
- Genomics feature dictionary
- Genomics input/output contract
- Genomics README
- Known limitations section

### 23.2 Artifact Deliverables

- selected feature list
- preprocessing configuration
- scaler or normalization parameters
- trained or fallback genomic model artifact
- explanation metadata
- sample genomic CSV

### 23.3 Integration Deliverables

- standard module output sample
- missing-modality output sample
- error output sample
- dashboard-ready explanation sample

---

## 24. Recommended 17-Hour Execution Plan for This Module

### Hour 1–2: Freeze Contract

- finalize feature list
- finalize output contract
- finalize embedding dimension
- finalize missing-modality behavior

### Hour 3–5: Data Preparation

- create sample genomic CSV
- prepare selected feature registry
- define demo patient examples
- validate CSV loading and schema logic

### Hour 6–8: Preprocessing and Standalone Model

- imputation
- normalization
- simple standalone predictor
- risk score mapping

### Hour 9–11: Embedding and Explanation

- produce 128-dimensional embedding
- generate top-feature ranking
- generate explanation payload

### Hour 12–14: Integration

- plug output into fusion contract
- test missing-modality behavior
- verify dashboard payload

### Hour 15–16: Polish

- add quality flags
- improve report language
- prepare demo cases

### Hour 17: Final Demo Readiness

- test end-to-end flow
- prepare explanation for judges
- prepare backup mock mode

---

## 25. Key Design Justifications

### 25.1 Why Use Structured Genomic Features?

Because raw sequencing pipelines are too large for a hackathon and unnecessary for a prototype. Structured feature tables are fast, explainable, and integration-friendly.

### 25.2 Why Use a 128-Dimensional Embedding?

It is compact enough for fast fusion and large enough to represent molecular patterns. It also keeps the genomics module lighter than image encoders.

### 25.3 Why Use SHAP or Feature Ranking?

Judges and clinicians need to understand which molecular features influenced the prediction. A black-box genomic score alone is less impressive.

### 25.4 Why Support Missing Genomics?

In real clinics, genomics may not always be available. Missing-modality support makes the system realistic and robust.

### 25.5 Why Use Fallback Mode?

In hackathons, model artifacts may fail, data may be limited, or teammates may integrate late. Fallback mode keeps the demo alive while preserving transparency.

---

## 26. Final Genomics Module Summary

The Phase 3 genomics module should behave like a reliable molecular intelligence component.

It should not try to become a full bioinformatics platform. Instead, it should deliver:

- clean genomic feature ingestion
- model-ready preprocessing
- fixed-size molecular embedding
- standalone genomic risk prediction
- top-feature explanation
- missing-data robustness
- integration-safe output contract
- report-ready molecular interpretation

This gives the final system a strong “wow” factor because it goes beyond image classification and demonstrates a clinically realistic multimodal decision-support workflow.
