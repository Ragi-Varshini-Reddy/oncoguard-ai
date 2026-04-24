# Phase 1 TODO

## Intraoral

- [ ] Choose dataset and document source/license.
- [ ] Build image manifest with patient/image IDs and labels.
- [ ] Add image preprocessing pipeline.
- [ ] Train or fine-tune pretrained image encoder.
- [ ] Save model artifact under `artifacts/phase1/`.
- [ ] Implement artifact-backed inference.
- [ ] Add Grad-CAM or heatmap explanation.
- [ ] Add tests for missing image, bad file, embedding shape, and output contract.

## Clinical

- [ ] Define clinical schema.
- [ ] Build clinical training CSV.
- [ ] Implement preprocessing fit only on training data.
- [ ] Train lightweight tabular model.
- [ ] Save artifact under `artifacts/phase1/`.
- [ ] Implement clinical XAI.
- [ ] Add tests for missing fields, categorical encoding, embedding shape, and output contract.

## Frontend

- [ ] Replace `#/phase-1` placeholder with upload/form workflow.
- [ ] Show modality output cards.
- [ ] Show Grad-CAM/clinical feature explanation.
- [ ] Send Phase 1 module outputs to fusion.
