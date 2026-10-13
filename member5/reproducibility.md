# Member 5 reproducibility and provenance

## Scientific identity

- Scientific source commit: `0481c4f10e5608d059cd9c489c82fbc27076bb71`.
- Historical execution checkout: `edc7c8e3c2bfa7556617d712e61c42e99a10f8fd`.
- Delivery base main: `4909c354bbdac0c69682fab1d2f2004885b38bb9`.
- Completed cohort: `member5_mdb_formal_002`; five seeds: `0,1,2,3,4`.
- Backbone: ResNet18, torchvision `ResNet18_Weights.IMAGENET1K_V1`.
- Official weight SHA-256: `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.

The four shared scientific files are unchanged from the actual experiment version.
Baseline and data_pipeline changes were inspected and are not MDB-Net dependencies.
Executed Runner/monitor/statistics tool hashes are preserved in public provenance.
Public delivery path adaptations are identified separately with their own hashes;
they are not falsely attributed to the historical training run.

## Frozen settings

| Setting | Value |
|---|---|
| Images | 1,106 total; 769 White, 337 Field; five disease classes |
| Split | 70/15/15, jointly stratified by class and background; 774/166/166 |
| Seeds | 0, 1, 2, 3, 4 |
| Input/cache | 256 / 448 |
| Batch/workers | 16 / 2 |
| Head/fine-tuning | 3 frozen-encoder epochs / at most 30 epochs |
| Early stopping | Eight consecutive fine-tuning epochs without strict validation macro-F1 gain |
| Optimizer | AdamW |
| Head/backbone learning rates | 0.001 / 0.0002 |
| Weight decay/smoothing/dropout | 0.01 / 0.1 / 0.2 |
| Segmentation weight/pooling | 1.0 / mask |
| MIL top-k fraction | 0.2 (inactive under mask pooling) |
| Background swap/teacher forcing | 0.5 / 0.5 |
| Field training | cls |
| Mask statuses/exclude | ok,review / no excluded image |
| SupCon weight/temperature | 0.0 / 0.1 |
| Class weights/from scratch/allow missing | false / false / false |
| Save models | true |

The actual head/fine-tuning epoch counts were 3/30, 3/29, 3/28, 3/25, and 3/28;
best fine-tuning epochs were 22, 21, 20, 17, and 20, respectively. Checkpoints were
selected by mixed-validation fixed-five-class macro-F1, never by test scores.

## Data, preprocessing and masks

Raw images, cache files, masks, and QC were frozen and independently copied into the
cohort with per-file SHA-256 checks; no links shared writable files. The cache contains
1,106 images, and 769 white-background pseudo-masks exist (755 trusted under
ok,review). Field images contribute classification supervision; training-field-only
background pools contain 237 images per seed, with no validation/test images.
The original mask pipeline uses classical Lab/background segmentation and QC.
No mask was regenerated for the recovered formal experiment. Empty exclude files
preserve the approved no-exclusion behavior. Original data and all large artifacts
remain outside this submission.

## Environment and non-training verification commands

The executed environment used Python 3.12.10, torch 2.8.0+cu128,
torchvision 0.23.0+cu128, numpy 2.2.6, scikit-learn 1.6.1,
opencv-python-headless 4.13.0.92, and matplotlib 3.9.4 on an NVIDIA RTX 5070 Ti
Laptop GPU. Dependency versions are recorded in [requirements](requirements.txt).
Use an appropriate official PyTorch CUDA installation for the target hardware;
no weights or environment are bundled.

From the repository root, the following commands run synthetic tests without training:

```powershell
python -B -m unittest discover -s tests -p test_member5_mdb_live_monitor.py -v
python -B -m unittest discover -s tests -p test_member5_mdb_runner.py -v
python -B -m unittest discover -s tests -p test_member5_mdb_results.py -v
```

The monitor historical-replay test is optional on other machines: set
`MEMBER5_HISTORICAL_AMP_LOG` to the privately retained historical AMP JSONL if it is
available. The historical file itself is not distributed, and the test otherwise
reports a skip rather than inventing evidence. All other fixtures are temporary
synthetic unit-test data, not experimental results.

Full local evidence can be verified with the existing statistics CLI (portable
placeholders below must be replaced with real, private paths):

```powershell
python -B -m src.member5_mdb_results --input FORMAL_RESULTS_DIR --output NEW_AUDIT_DIR --expected-seeds 0 1 2 3 4 --approved-config APPROVED_CONFIG_JSON --evidence-manifest EVIDENCE_MANIFEST_JSON --require-complete
```

`NEW_AUDIT_DIR` must not exist. This command verifies artifacts and does not train.
The public provenance has symbolic cohort/repository references, not public download
links to the private records. Public files alone do not replace the complete schema-v1
local execution evidence or provide independent third-party training attestation.

## Runner portability and execution gates

Public Runner paths are configured with `MEMBER5_WORKSPACE`, `MEMBER5_DATA_SOURCE`,
and `MEMBER5_SELECTED_CONFIG`; use absolute portable paths outside the repository
for data/workspaces. Without overrides, sibling directories are used, never a
personal user directory. `scientific_configuration.json` is an
archival record, not an executable approval. It intentionally retains historical
Git pins, so the public main checkout is not silently accepted as that execution
checkout. Any future execution requires a separately reviewed selected configuration
and schema-v1 approvals bound to the actual Git branch/HEAD/tracking ref, unchanged
scientific hashes, real data/preprocessing/weights, and current tool-validation hashes.
An external selected configuration can explicitly provide its required branch.
Do not repoint or modify historical evidence to make a new checkout pass.

The PowerShell wrapper defaults to a Dry Run:

```powershell
powershell -NoProfile -File scripts/member5_mdb_formal_run.ps1 -ApprovedConfig APPROVED_CONFIG_JSON -QcApproval QC_POLICY_JSON
```

No training was run during submission preparation. A future explicit `-Execute`
request still cannot bypass any approval, source/hash, dataset, output, or lock gate.
Used output directories and reservations are never cleaned or reused automatically.

## Monitoring and evaluation

The executed AMP monitor was member5-amp-monitor-v2. It checks current-attempt
identity, sequence and finite values; cumulative skip rate is judged after the first
complete 48-batch training epoch, then continuously over the entire attempt.
Strictly more than 5% skipped updates or three consecutive skips requires review;
nonfinite loss, applied gradients, or weights fail immediately. Native isolated
GradScaler overflow is logged as a warning. No hard-coded PID termination is used.
The completed run had 7,440 batches, 7,434 optimizer updates, six isolated skips,
and no nonfinite loss/applied-gradient/weight events.

Each seed's 166 test images include 116 White and 50 Field images. Accuracy and
macro-F1 use the actual saved predicted classes, with all five classes and
zero_division=0. Mixed is recomputed from the full test split. Five per-seed metrics
are averaged with sample SD (ddof=1); pooled predictions are not substituted for
five-run means. Individual and summary CSVs use 0-1 values; README/report tables
use percentages. Three summed confusion matrices are descriptive count sums only.
Prediction-case rows appear once for their background scenario and once for Mixed.

## Limitations and recovery disclosure

Same-leaf/near-duplicate grouping was not independently verified. Historical MALG
and Pilot test exposure remains, although none of those metrics is included here.
Under frozen ok,review policy, class-relative QC did not change mask eligibility;
this does not prove complete absence of leakage. Masks are automatic pseudo-labels,
not manual truth. Baseline uses different seeds and split algorithms: this is not a
strictly paired comparison. The first formal attempt ended because of an engineering
monitor false positive (zero successful seeds). The recovered formal_002 completed
all five seeds with unchanged scientific settings. Full failure and successful
execution evidence remains local and unchanged; it is not fabricated or publicly
downloadable. Verification is COMPLETE_VERIFIED in the recorded-artifact scope.
