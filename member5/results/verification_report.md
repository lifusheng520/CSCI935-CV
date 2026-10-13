# MDB-Net verification

Status: **COMPLETE_VERIFIED**
Metric validation: VERIFIED

VERIFIED = directly checked; PARTIALLY VERIFIED = evidence gaps; NOT VERIFIED = no evidence; INVALID = rejected.

Input-derived metrics do not prove executed formal training. Synthetic tests are not real experiments.

## Errors

## Limitations
- Evidence integrity is relative to external runner records, NOT independent proof of authorship/training or absence of near-duplicate/development leakage

## Warnings
- seed2_predictions.csv: 1 rounded/near-tied rows; exact logits/argmax cannot be reconstructed
- seed3_predictions.csv: 1 rounded/near-tied rows; exact logits/argmax cannot be reconstructed

Seed 0: head=3, fine-tune=30, inferred best epoch=22; VERIFIED (runner record, hash and epoch binding; weights not deserialized)

Seed 1: head=3, fine-tune=29, inferred best epoch=21; VERIFIED (runner record, hash and epoch binding; weights not deserialized)

Seed 2: head=3, fine-tune=28, inferred best epoch=20; VERIFIED (runner record, hash and epoch binding; weights not deserialized)

Seed 3: head=3, fine-tune=25, inferred best epoch=17; VERIFIED (runner record, hash and epoch binding; weights not deserialized)

Seed 4: head=3, fine-tune=28, inferred best epoch=20; VERIFIED (runner record, hash and epoch binding; weights not deserialized)

Segmentation: pseudo-mask agreement, NOT hand-annotated truth.
Metric tables exported: True. Partial exports are descriptive, NOT verified formal results.
