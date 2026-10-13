# Member 5: verified MDB-Net formal experiments

Member 5 is responsible for formal evaluation, independent metric verification,
execution-evidence auditing, and preparation of the experimental settings/results.
This directory contains only the final recovered MDB-Net five-seed formal results.

## Model and evaluation

MDB-Net uses an ImageNet-pretrained ResNet18 backbone (IMAGENET1K_V1). The dataset
contains 1,106 images across five rice-disease classes: 769 white-background and
337 field-background images. Seeds 0, 1, 2, 3, and 4 use distinct disease-by-background
stratified 70/15/15 splits, each with 774 training, 166 validation, and 166 test images.
White and Field are background subsets of each test split; Mixed is the full split.

Accuracy and fixed-five-class macro-F1 were independently recomputed from actual
per-image test predictions. The table reports percentages and sample standard
deviations over five runs, with ddof=1 (SD in percentage points).

| Scenario | Accuracy, mean +/- sample SD | Macro-F1, mean +/- sample SD |
|---|---:|---:|
| White | 81.8966 +/- 1.6128 | 80.3650 +/- 2.2705 |
| Field | 68.0000 +/- 8.2462 | 67.3267 +/- 9.0306 |
| Mixed | 77.7108 +/- 3.3541 | 76.5068 +/- 3.9284 |

## Files

- [English report sections](experimental_settings_results.tex): intended for Sections 6.1 and 6.2.
- [Reproducibility](reproducibility.md): settings, dependencies, commands, and provenance boundaries.
- [Individual runs](results/individual_runs.csv): 15 seed-by-scenario records.
- [Summary](results/summary_results.csv): five-run means and sample SD, in 0-1 units.
- [Result table](results/result_table.tex): LaTeX table in 0-1 units.
- [Confusion matrices](results/confusion_matrices/): 15 per-seed matrices and three descriptive sums.
- [Prediction cases](results/member6_prediction_cases.csv): safe relative image identifiers and verified cases.
- [Public provenance](results/experiment_provenance.public.json): actual source hashes and recovery provenance.
- [Verification report](results/verification_report.md): COMPLETE_VERIFIED in the recorded-artifact scope.

Reusable modules are [metric verification](../src/member5_mdb_results.py),
[Runner](../src/member5_mdb_runner.py), and [AMP monitor](../src/member5_mdb_live_monitor.py).
The [PowerShell wrapper](../scripts/member5_mdb_formal_run.ps1) defaults to no training.
Tests remain under `tests/`; no generated synthetic outputs are published.

## Provenance and limitations

All results derive from the actual formal_002 experiment, which completed all five
seeds. The first formal attempt was interrupted by an engineering monitor false
positive and contributed zero successful seeds; its raw artifacts are not uploaded.
Recovery retained the scientific settings and did not select parameters by earlier
validation/test scores. Full private execution evidence remains local.

Same-leaf and near-duplicate grouping was not independently verified. Historical
MALG and Pilot test exposure remains a limitation; none of their results is included.
Under the frozen ok,review QC policy, class-relative statistics did not change
training-mask eligibility. This does not prove complete absence of data leakage.
Leaf masks are automatically generated pseudo-labels, not manual ground truth;
Dice/IoU must be interpreted as agreement with pseudo-labels. Baseline and MDB-Net
use different seeds and splitting algorithms and must not be described as a
strictly paired comparison.
