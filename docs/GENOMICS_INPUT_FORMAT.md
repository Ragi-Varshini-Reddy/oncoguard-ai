# Genomics CSV Input

Upload a CSV file with one row per sample. For inference, the app expects:

- `patient_id`
- `sample_id`
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

Training files additionally need the configured label column:

- `risk_label`

Accepted training labels follow the project config, where `high` is the positive label.
