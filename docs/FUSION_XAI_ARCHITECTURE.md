# Fusion + Interactive XAI Architecture

## Current V2

The current fusion engine is a gated evidence-weighted late-fusion model. It is intentionally deterministic and CPU-fast so it can operate before all modalities have patient-matched training data.

Inputs:

- contract-compatible module outputs
- modality mask
- optional disabled modalities from the UI

Weighting uses:

- configured modality prior
- module confidence
- modality availability
- quality penalties from `quality_flags`
- prediction strength
- agreement with the active-modality consensus
- a high-risk guardrail that prevents one high-confidence high-risk branch from being diluted away by lower-risk branches

Outputs:

- diagnosis class/probability
- risk class/score
- fusion confidence
- modality contributions
- modality evidence
- raw/gated weights, signal strength, and agreement factor per modality
- decision trace
- what-if leave-one-out deltas
- quality summary
- warnings

## Why Late Fusion First

This project currently has a real trained Genomics artifact, while Phase 1 and Phase 2 are still being built. Late fusion lets every phase integrate independently without fabricating matched multimodal training samples.

## Deferred Trainable Fusion

Do not train gated-attention or transformer fusion until we have patient-matched multimodal rows where embeddings from two or more modalities belong to the same patient.

Future trainable fusion path:

1. Freeze or lightly fine-tune modality encoders.
2. Export patient-matched embeddings and labels.
3. Train a gated-attention fusion model over modality tokens.
4. Calibrate predictions on validation data.
5. Use learned gates/attention plus leave-one-out ablations for XAI.

Never mix unrelated Kaggle/Mendeley/GDC/GEO patients as if they are matched multimodal records.
