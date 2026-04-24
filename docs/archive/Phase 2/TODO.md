# Phase 2 TODO

## Histopathology

- [ ] Choose dataset and document source/license.
- [ ] Decide whether input is patch image or whole-slide derived patches.
- [ ] Build histopathology manifest.
- [ ] Add preprocessing and quality checks.
- [ ] Train classifier head or extract pathology foundation embeddings.
- [ ] Save artifact under `artifacts/phase2/`.
- [ ] Implement artifact-backed inference.
- [ ] Add heatmap/attention explanation.
- [ ] Add tests for missing file, embedding shape, and output contract.

## Frontend

- [ ] Replace `#/phase-2` placeholder with histopathology upload workflow.
- [ ] Show patch/heatmap explanation.
- [ ] Send Phase 2 module output to fusion.
