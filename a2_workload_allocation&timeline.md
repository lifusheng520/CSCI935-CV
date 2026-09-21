Here is a polished English version of the **balanced 6-person task allocation**, with both workload and difficulty considered.

| Member                                                                 | Main Responsibility                                                          | Specific Tasks                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Workload | Difficulty |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------: | ---------: |
| **Member 1 – Dataset and Problem Analysis**                            | Dataset inspection, problem formulation, and data splitting                  | Inspect the five disease classes and the two background types; analyse class distribution and image characteristics; identify challenges such as lighting variation, complex field backgrounds, leaf orientation, similar disease symptoms, and class imbalance; design the train/validation/test splitting strategy; ensure there is no data leakage; write **Section 3: Problem Description**.                                                                                                                         |    ★★★★☆ |      ★★★☆☆ |
| **Member 2 – Literature Review and Method Design**                     | Related work, method comparison, and method justification                    | Find at least **4 relevant IEEE Xplore papers published in or after 2022**; summarise their methods, datasets, strengths, and limitations; compare possible approaches such as CNNs and transfer learning; work with Member 4 to determine the final method; justify the use of the selected model; manage IEEE-style references; write **Section 4: Related Work** and contribute to the method rationale in Section 5.                                                                                                 |    ★★★★☆ |      ★★★☆☆ |
| **Member 3 – Baseline Model and Data Pipeline**                        | Data preprocessing and baseline implementation                               | Implement the PyTorch Dataset/DataLoader; perform resizing, normalization, and basic preprocessing; implement the **Simple CNN baseline**; build the training, validation, and testing pipeline; run the baseline experiments for **white-background, field-background, and mixed-background scenarios**, using **5 different random splits**; save the predictions and performance metrics.                                                                                                                             |    ★★★★☆ |      ★★★★☆ |
| **Member 4 – Main Model and Method Implementation**                    | Main model development and algorithm description                             | Implement the **pretrained ResNet18** model; replace the final classification layer for the five disease classes; design and implement the fine-tuning strategy; determine training hyperparameters such as learning rate, optimizer, epochs, and checkpoint settings; run the ResNet18 experiments for the three required testing scenarios with **5 random splits**; prepare the algorithm description, pseudocode, and method flowchart; write most of **Section 5: Method and Implementation**.                      |    ★★★★★ |      ★★★★★ |
| **Member 5 – Improved Model and Evaluation**                           | Improved method, evaluation pipeline, and result aggregation                 | Implement the **ResNet18 + Data Augmentation** version, using techniques such as random flipping, rotation, cropping, or colour augmentation; run the improved-model experiments for the three required scenarios with **5 random splits**; implement the unified evaluation pipeline; calculate metrics such as Accuracy, Macro-F1, mean, and standard deviation; generate confusion matrices, tables, and figures; organise the final quantitative results for **Sections 6.1 and 6.2**.                               |    ★★★★★ |      ★★★★☆ |
| **Member 6 – Result Analysis, Reproducibility, and Final Integration** | Result discussion, failure analysis, reproducibility, and report integration | Analyse successful, failed, and difficult classification cases; interpret the confusion matrices; explain why certain diseases are confused; compare performance between white-background and field-background images; discuss limitations and possible method improvements; write **Section 6.3: Discussion of Results**; prepare the **User Manual**; independently test the final code to ensure the reported results are reproducible; write the **Executive Summary** and integrate and proofread the final report. |    ★★★★☆ |      ★★★★☆ |

The experimental work is deliberately distributed across **Members 3, 4, and 5** so that one person is not responsible for all model training and repeated runs. This is important because the assignment requires results to be averaged over **5 different random splits**, with both the mean and standard deviation reported, and it requires at least three testing scenarios: white background, field background, and mixed background. 

The model ownership can therefore be defined clearly as:

* **Member 3:** Simple CNN baseline
* **Member 4:** Pretrained ResNet18
* **Member 5:** ResNet18 + Data Augmentation / Improved Model

Each of these members is responsible for implementing, training, and evaluating their own model under the same experimental protocol.

The overall workload is approximately balanced as follows:

| Member   | Overall Effort |
| -------- | -------------: |
| Member 1 |           ~80% |
| Member 2 |           ~80% |
| Member 3 |           ~90% |
| Member 4 |          ~100% |
| Member 5 |          ~100% |
| Member 6 |           ~90% |

The difficulty is not identical across all members because the main-model implementation and formal evaluation are technically more demanding than literature review or dataset analysis. However, the workload is balanced by assigning additional responsibilities to Members 1, 2, and 6, such as data-splitting design, method justification, reproducibility checking, and final report integration.

For the **cover sheet**, you can write the contribution statement in this form:

> **Member 1:** Dataset analysis, problem formulation, image characteristic analysis, data-splitting strategy, and Problem Description.
> **Member 2:** Literature review, IEEE Xplore paper analysis, method comparison, method justification, and reference management.
> **Member 3:** Data preprocessing pipeline, Dataset/DataLoader implementation, baseline CNN development, and baseline experiments.
> **Member 4:** Pretrained ResNet18 implementation, transfer learning and fine-tuning, main-model experiments, algorithm design, pseudocode, flowchart, and Method section.
> **Member 5:** Improved ResNet18 model with data augmentation, evaluation pipeline, repeated experiments, statistical analysis, confusion matrices, tables, and quantitative result presentation.
> **Member 6:** Failure-case analysis, result discussion, method-improvement analysis, reproducibility testing, User Manual, Executive Summary, final report integration, and proofreading.

This also directly satisfies the assignment requirement that the cover sheet should state the **contributions of individual group members**. 







| Period           | Main Goal                            | Member 1                                       | Member 2                                                   | Member 3                                              | Member 4                              | Member 5                                                        | Member 6                                       |
| ---------------- | ------------------------------------ | ---------------------------------------------- | ---------------------------------------------------------- | ----------------------------------------------------- | ------------------------------------- | --------------------------------------------------------------- | ---------------------------------------------- |
| **21–27 Sep**    | Dataset + literature + project setup | Dataset inspection, statistics, split strategy | Find 4+ IEEE papers, compare methods                       | Set up repository, Dataset/DataLoader skeleton        | Set up ResNet18 environment           | Design augmentation + evaluation metrics                        | Create Overleaf structure, formatting template |
| **28 Sep–4 Oct** | Finish design & first working models | Finish Section 3                               | Finish Section 4 + method rationale                        | Finish Simple CNN                                     | Finish ResNet18 implementation        | Finish augmentation model + evaluation pipeline                 | Review Sections 3–4, prepare report structure  |
| **5–11 Oct**     | Pilot experiments                    | Check split correctness / leakage              | Help interpret literature vs method                        | Run first baseline experiments                        | Run first ResNet experiments          | Run first improved-model experiments                            | Start reproducibility checklist                |
| **12–18 Oct**    | **Formal experiments**               | Support dataset/split issues                   | Help method/report writing                                 | Complete 5-split CNN experiments                      | Complete 5-split ResNet18 experiments | Complete 5-split improved-model experiments + aggregate results | Start collecting failure cases                 |
| **19–25 Oct**    | Results + report writing             | Polish Section 3                               | Polish Section 4                                           | Provide baseline results + implementation description | Finish Section 5                      | Finish Sections 6.1–6.2, tables/figures                         | Write Section 6.3 + User Manual                |
| **26–29 Oct**    | Integration + reproducibility        | Proofread                                      | References check                                           | Code cleanup                                          | Code cleanup                          | Verify metrics/results                                          | Merge report, independently run final code     |
| **30–31 Oct**    | **Final freeze**                     | Final check                                    | Final check                                                | Final code check                                      | Final code check                      | Result consistency check                                        | Final PDF, naming, zip, submission check       |
| **1 Nov**        | Submission only                      | colspan                                        | **Do not run new experiments unless absolutely necessary** |                                                       |                                       |                                                                 |                                                |

## Week 1: 21–27 September

### Goal: understand the problem before locking the model

This week is mainly **M1 + M2**, but M3–M5 should already start coding.

**Member 1**

* Download/check dataset
* Count images per disease
* Count white vs field background
* Inspect representative images
* Check class imbalance
* Decide train/validation/test strategy
* Decide how the 5 random splits will be generated

By **27 Sep**, M1 should provide everyone something like:

```text
Classes: 5
Backgrounds: white / field
Train: xx%
Validation: xx%
Test: xx%
Seeds: [....]
```

This is important because **M3/M4/M5 must all use exactly the same experimental protocol**. Otherwise you cannot fairly compare the three models.

**Member 2**

* Find ≥4 IEEE Xplore papers from 2022+
* Create comparison table:

  * paper
  * model
  * preprocessing
  * dataset
  * result
  * advantage
  * limitation
* Recommend model design with M4

**M3/M4/M5**
Do not wait.

Already create:

```text
load_data()
split_data()
train()
evaluate()
```

and confirm PyTorch works.

**Member 6**
Create the Overleaf project immediately with:

```text
1 Coversheet
2 Executive Summary
3 Problem Description
4 Related Work
5 Method and Implementation
6 Experimental Results
    6.1 Experimental Settings
    6.2 Results
    6.3 Discussion
7 References
Appendix User Manual
```

---

# Week 2: 28 September–4 October

### Goal: all three models must run successfully at least once

By the end of this week:

### M3

Simple CNN should work:

> image → CNN → 5 classes

### M4

ResNet18 should work:

> pretrained ResNet18 → replace classifier → 5 classes → fine-tune

### M5

Improved model should work:

> ResNet18 + augmentation

For example:

```text
RandomHorizontalFlip
RandomRotation
RandomResizedCrop
ColorJitter
```

But don't pile on augmentation randomly. Pick a small, justified set.

At this point **do not start all 5 formal runs yet**.

First make sure one training run works from start to finish:

```text
train
↓
validation
↓
test
↓
accuracy
↓
macro F1
↓
confusion matrix
```

The most important milestone:

> **By 4 October, all three models must produce a valid test result.**

If one model still cannot run at this point, the group should help that member immediately.

---

# Week 3: 5–11 October

### Goal: pilot experiments

This week is for finding problems **before** the expensive 5-run experiments.

Run maybe:

```text
Seed 0
```

or 1–2 pilot runs first.

Check:

### 1. Training reasonable?

For example:

```text
Train Accuracy: 98%
Test Accuracy: 25%
```

Then something is probably wrong.

### 2. Data leakage?

Especially make sure the same image is not appearing in both train and test.

### 3. Background scenarios implemented correctly?

You need:

```text
White-background test

Field-background test

Mixed-background test
```

These three are explicitly required in the assignment. 

### 4. Hyperparameters fixed

By **11 October**, stop casually changing:

* learning rate
* batch size
* epochs
* optimizer
* image size
* augmentation
* pretrained setting

Otherwise you will keep invalidating previous results.

So I would define:

> **11 October = Method Freeze**

After this date, only fix bugs, don't redesign the whole model.

---

# Week 4: 12–18 October

## This is the most important week: formal experiments

Now M3/M4/M5 work in parallel.

### Member 3

Run:

```text
Simple CNN

Seed 1 → White / Field / Mixed
Seed 2 → White / Field / Mixed
Seed 3 → White / Field / Mixed
Seed 4 → White / Field / Mixed
Seed 5 → White / Field / Mixed
```

### Member 4

Same:

```text
ResNet18 × 5 splits × 3 scenarios
```

### Member 5

Same:

```text
ResNet18 + Augmentation × 5 splits × 3 scenarios
```

This directly follows your previous division of responsibilities. 

The assignment specifically says results should be averaged over **5 runs with different random splits**, and both mean and standard deviation must be reported. 

By **18 October**, the final raw results should be finished.

Something like:

```text
results/
├── cnn_seed1.csv
├── cnn_seed2.csv
├── ...
├── resnet_seed1.csv
├── ...
└── augmentation_seed5.csv
```

And M5 starts converting these into:

```text
mean ± std
```

---

# Week 5: 19–25 October

## Goal: stop training and write the actual report

This is where many groups make a mistake: they keep training until the final day.

I would set:

> **18 October = Experimental Deadline**

After that, unless a result is clearly broken, stop experimenting.

Then:

### M1

Finish:

**Section 3 Problem Description**

### M2

Finish:

**Section 4 Related Work**

and IEEE references.

### M3

Give M4/M6:

* Simple CNN architecture
* preprocessing description
* baseline results

### M4

Finish:

**Section 5 Method and Implementation**

Including:

```text
Input image
   ↓
Preprocessing
   ↓
Data augmentation
   ↓
Pretrained ResNet18
   ↓
Fine-tuning
   ↓
5-class classifier
   ↓
Prediction
```

Plus pseudocode/flowchart.

### M5

Finish:

**6.1 Experimental Settings**

and

**6.2 Results**

Create one main table:

| Model                   |      White |      Field |      Mixed |
| ----------------------- | ---------: | ---------: | ---------: |
| Simple CNN              | mean ± std | mean ± std | mean ± std |
| ResNet18                | mean ± std | mean ± std | mean ± std |
| ResNet18 + Augmentation | mean ± std | mean ± std | mean ± std |

And confusion matrices.

### M6

Now the workload really begins.

Write:

**6.3 Discussion**

Analyse:

* Which disease is easiest?
* Which disease is hardest?
* Which classes get confused?
* White vs field difference?
* Why does augmentation help/not help?
* Typical successful cases?
* Typical failure cases?

This corresponds exactly to the role you assigned M6 previously. 

---

# 26–29 October

## Reproducibility week

At this point **do not let each person submit their own code fragments**.

Merge everything into the final:

```text
group_name.py
```

Member 6 should act like someone who has never seen the project before.

Create a clean environment and follow the User Manual:

```bash
python group_name.py
```

Check whether it:

1. Finds `.\Dhan-Shomadhan\`
2. Loads the dataset
3. Creates the required splits
4. Trains or loads the model appropriately
5. Produces evaluation results

The assignment specifically warns that insufficient instructions for reproducing results may lead to a heavy penalty. 

I would make:

> **29 October = Code Freeze**

No more feature additions after that.

---

# 30–31 October

## Final two days: only final checks

Absolutely try not to use these two days for training.

Check:

**Report**

* page limits
* 12 pt
* single column
* margins
* figure numbers
* table numbers
* citations
* IEEE references
* member contributions
* no plagiarism

**Code**

* one `.py`
* no `/content/...`
* no personal absolute paths
* no missing packages
* no hardcoded local username
* no dataset included

Final structure:

```text
group_name.zip
├── group_name.py
└── group_name.pdf
```

---

# 1 November

Ideally:

> **Submit in the afternoon, not 11:20 pm.**

Your official deadline is **11:30 pm**. 

I would make the **internal deadline 31 October at 8 pm**.

That gives you almost a full day of buffer.

---

## The four deadlines I would put in the group chat

If you want something very concise to send to everyone, use these four milestones:

> **4 Oct — Implementation milestone:** All three models must run successfully.
> **11 Oct — Method freeze:** Finalise models, hyperparameters, splits and evaluation protocol.
> **18 Oct — Experiment freeze:** Complete all 5-run experiments and collect results.
> **25 Oct — Report draft:** All sections, tables and figures completed.
> **29 Oct — Code/report freeze:** Reproducibility check completed.
> **31 Oct — Internal submission deadline:** Final ZIP ready.

