# CSCI935 Group Project: Task Allocation and Timeline

Updated: 6 October 2026

This plan follows [Group project.pdf](<Group project.pdf>) and the team's current **Mask-guided Adaptive Local-Global MobileNetV2 (MALG-MobileNetV2)** method. The work covers one recommended method, its implementation, the required evaluation, and the consultation report.

The assignment requirements are taken from pages 2–5 of the brief: problem and report requirements on pages 2–3, experimental requirements on page 4, and reproducibility, implementation, formatting, and submission requirements on page 5. The model, metric choices, seeds, member assignments, and internal dates in this plan are team decisions. Quality is achieved by explaining and verifying the required work thoroughly, without adding separate comparison models or extra experiments.

**Official submission deadline: 11:30 pm, Sunday, 1 November 2026, via Moodle.** All earlier dates below are internal planning targets, not assignment requirements.

## 1. Six-person task allocation

Responsibilities apply to completing and documenting the shared project. Members should use the existing implementation and results where applicable. Cover-sheet contributions must reflect work actually completed.

| Member | Responsibility | Required tasks and handover |
| --- | --- | --- |
| **Member 1 — Dataset and Problem Analysis** | Problem formulation and data-splitting design | Translate the client's request into a five-class image-classification problem after inspecting the provided images; explain image characteristics and challenges that motivate the method; define the training/validation/test splitting scheme and prevent test samples from entering training. Write **Problem Description**, including the justification for using pretrained models with Member 2. Give the splitting specification to Members 3 and 5. |
| **Member 2 — Literature Review and Method Justification** | Related work, rationale, and references | Identify and read at least **four relevant IEEE Xplore papers published in or after 2022**, using the UoW library; summarise them in the context of the problem using original wording; explain the rationale for the selected method with Member 4; justify the use of pretrained models with Members 1 and 4. Write **Related Work** and maintain the **IEEE-style References**. |
| **Member 3 — Data Pipeline** | Data handling and preprocessing implementation | Complete or verify data loading, preprocessing, and the splitting scheme specified by Member 1; support the original dataset folder structure; check that training, validation, and test partitions are separate. Give Member 4 the data-pipeline implementation description and Member 6 the data setup and dependency instructions needed for the User Manual. |
| **Member 4 — Method and Training Implementation** | MALG-MobileNetV2 and training | Complete or verify the shared backbone, global/local/mask branches, mask generation and reliability handling, prediction fusion, and training procedure described in the current method design. Record the hyperparameters and pretrained weights used. Write **Method and Implementation**, explaining the rationale and algorithm step by step, with a numbered flowchart or pseudocode where appropriate. Provide the working training code and settings to Member 5, and weight download/setup instructions to Member 6. With Member 3, prepare the final `group_name.py` so that it runs without unsubmitted local source files. |
| **Member 5 — Experiments and Results** | Required evaluation and result reporting | Run the selected method with **five different random data splits**, using unchanged split percentages; evaluate **White, Field, and Mixed** test scenarios; calculate the mean and standard deviation of the chosen metrics. Write **Experimental Settings** and **Experimental Results**. Give Member 6 the results and successful, failed, and challenging predictions needed for the discussion. |
| **Member 6 — Discussion, User Manual, and Integration** | Result interpretation and final report | Discuss whether results meet expectations; explain typical successful, failed, and challenging cases; propose improvements to the method itself. Write **Discussion of the Results**, **Executive Summary**, and **Appendix: User Manual**. Collect actual member contributions for the cover sheet, integrate the report, check that its execution instructions reproduce the reported results, and prepare the final submission with the team. |

Members 3 and 4 support Member 5 when data or implementation issues prevent the required experiments. Member 5 owns the consolidated experimental results; there is no separate model-development assignment for each member.

## 2. Required experimental work

The assignment requires evaluation of the recommended method. It does not prescribe multiple model architectures. The team's current method is MALG-MobileNetV2.

1. **Define and record the settings.** State all hyperparameter values, the data-splitting scheme, and the experiments conducted. Record the actual preprocessing, training, and fine-tuning settings used. For MALG, include the mask thresholds and reliability settings, local-selection settings, fusion and loss settings, and pretrained weight versions as well as learning rates, batch size, epochs, and model-selection rules. The description must match the configuration that produced the reported results.
2. **Use five different random splits.** Keep the train/validation/test percentages unchanged across runs. State the selected percentages and random seeds in the report. Seeds 42, 43, 44, 45, and 46 are the team's current selection, not a requirement from the brief.
3. **Keep the test data separate.** Test samples must not be included in training. Use training/validation data to make development decisions.
4. **Evaluate the three required scenarios.** For each run, evaluate the white-background test subset, the field-background test subset, and the combined test set.
5. **Report mean and standard deviation.** Calculate these across the five runs for each chosen metric and each scenario. Accuracy and Macro-F1 are the team's current metric choices; the brief does not mandate these specific metrics.

The current protocol trains **one mixed-background model per random split** and evaluates that model on the three test scenarios. This means **five training runs**, each producing three sets of test metrics. Separate training for each background scenario is not required. Mixed results are computed on the combined test samples, not by taking an unweighted average of the White and Field metrics.

For each metric and scenario, calculate one value per run and then calculate the mean and standard deviation of those five values. State the standard-deviation convention used. Do not replace the required five-run statistics with the best run, validation performance, or a single pooled score. Keep the settings consistent across the five runs so that the reported variation reflects the repeated experiment.

## 3. Experimental subsection responsibilities

The current report uses **5.1–5.3** for these subsections; the assignment brief labels the corresponding content **6.1–6.3**. Use consistent numbering in the final report. The entire Experimental Results and Analysis section has a **two-page limit**.

### 5.1 Experimental Settings

**Lead: Member 5. Support: Members 1, 3, and 4.**

Explain how the experiments were conducted. Include all hyperparameter values, the training/validation/test splitting scheme, and a list of experiments performed. Define the chosen evaluation metrics and explain how the three test scenarios are evaluated.

- **Member 1:** Supply the splitting scheme.
- **Member 3:** Supply the preprocessing and data-handling settings.
- **Member 4:** Supply the model, training, and fine-tuning settings actually used.
- **Member 5:** Combine this information and write the subsection.

### 5.2 Experimental Results

**Lead: Member 5.**

Report the results of every experiment listed in Experimental Settings. Present the **mean and standard deviation over five runs with different random splits**, keeping the split percentages the same. Include **White, Field, and Mixed** test results and clearly explain the shared-model evaluation protocol.

Member 5 runs the experiments, calculates the statistics, and writes the subsection. Members 3 and 4 help resolve data or code issues affecting those runs.

### 5.3 Discussion of the Results

**Lead: Member 6. Results supplied by Member 5.**

Discuss whether the results meet expectations. Describe typical successful predictions, failed predictions, and challenging cases, explaining the reasons for each. Explain how **the method itself, rather than the dataset size, could be improved**.

Member 6 writes the subsection using the completed experimental results and examples supplied by Member 5.

The discussion should connect observations to the characteristics of the images and the behaviour of the method. Distinguish explanations supported by the results from plausible hypotheses. Do not claim that an individual branch improves accuracy unless the available evidence establishes that claim. Proposed improvements should address an observed limitation, rather than being an unrelated list of techniques.

## 4. Report ownership and page limits

The section numbers below follow the assignment brief; headings can be matched to the corresponding sections in the team's report. The brief allocates **25 marks** in total, including **8 marks for Method and Implementation** and **8 marks for Experimental Results and Analysis**.

| Required section | Lead | Marks | Page limit |
| --- | --- | --- | --- |
| 1. Coversheet: title, member names, and actual individual contributions | Member 6, with information from everyone | No separate mark stated | Not specified in the brief |
| 2. Executive Summary: concise summary of Problem Description through Experimental Results and Analysis | Member 6 | 1 | 1/2 page |
| 3. Problem Description | Member 1 | 5 | 1 page |
| 4. Related Work | Member 2 | 2 | 2 pages |
| 5. Method and Implementation | Member 4, with input from Members 2 and 3 | 8 | 3 pages |
| 6. Experimental Results and Analysis: Settings, Results, and Discussion | Member 5 for Settings/Results; Member 6 for Discussion | 8 for the whole section | 2 pages for all three subsections combined |
| 7. References | Member 2 | 1 | Not specified in the brief |
| 8. Appendix: User Manual | Member 6, with setup information from Members 3 and 4 | No separate mark stated | 2 pages |

Method and Implementation must explain the rationale, algorithm steps, and implementation clearly enough for a classmate to implement the method. Discuss the method's strengths and limitations. Do not paste Python code into the report as the implementation description; the brief prefers block diagrams or pseudocode.

The User Manual must explain the required packages and their installation, how to download and locate pretrained weights, and how to run the code to reproduce the reported results.

### Completion criteria for the required work

These criteria clarify the quality expected for the existing deliverables; they do not introduce additional experiments or report sections.

| Owner | The work is ready when... |
| --- | --- |
| **Member 1** | The report explains the client's five-disease classification task, recognises both background types, and uses observations from the provided images to justify the method. It does not claim to recognise healthy leaves or diseases absent from the dataset. The splitting specification is explicit and can be implemented by Member 3. |
| **Member 2** | At least four papers meet the IEEE Xplore and publication-year requirements; their findings are explained in relation to this problem in original wording. Every cited paper has a matching, accurate IEEE-style reference. The method rationale makes clear which ideas are adopted or adapted. |
| **Members 3 and 4** | The shared implementation follows the described method and uses the required dataset structure. The report explains both algorithm operation and implementation choices, including the pretrained components, without copying Python code. A reader with computer-vision knowledge could implement the stated steps. The submitted program and its documented dependencies are sufficient to run it. |
| **Member 5** | Every listed experiment has all five split results and all three test scenarios. Reported means and standard deviations can be traced to those runs, and the settings describe what was actually executed. Test samples are excluded from training; test results are not presented as validation results or vice versa. |
| **Member 6** | Discussion covers expectations, successes, failures, difficult cases, and improvements to the method, with reasons grounded in the results. The Executive Summary accurately reflects the completed report. The User Manual has been checked against the submitted program, and the integrated report respects every section's page limit. |
| **All members** | The final cover sheet records actual contributions. Claims, code, results, and instructions agree; unresolved implementation or evaluation limitations are stated honestly. |

## 5. Internal timeline

The schedule below starts from the current planning date. It assigns completion and review of the required work without assuming that earlier tasks have already been finished.

| Period | Internal target | Responsibilities |
| --- | --- | --- |
| **6–11 October** | Confirm the final method, data split, and runnable implementation | Member 1 confirms the problem analysis and splitting scheme. Member 2 completes the required literature coverage and method rationale. Members 3 and 4 complete or verify the shared pipeline. Member 5 records the experimental settings and confirms that all three test scenarios are supported. Member 6 assembles the report structure and setup instructions. |
| **12–18 October** | Complete the five required experimental runs | Member 5 runs the five random splits, records the three test scenarios, and calculates mean and standard deviation. Members 3 and 4 resolve implementation problems; Member 1 supports splitting issues. Member 2 checks method-related citations. Member 6 receives the results and cases needed for discussion. |
| **19–25 October** | Complete the report draft | Member 1 finishes Problem Description. Member 2 finishes Related Work and References. Member 3 supplies the data implementation and setup descriptions. Member 4 finishes Method and Implementation. Member 5 finishes Experimental Settings and Results. Member 6 finishes Discussion, Executive Summary, and User Manual, and combines the sections. |
| **26–29 October** | Verify reproducibility and submission requirements | Member 6 follows the User Manual to check the final program. Members 3 and 4 correct execution or packaging issues. Member 5 checks reported values against saved results. Members 1 and 2 verify the data description, rationale, and references. All members confirm their actual contributions. |
| **30–31 October** | Prepare the final report and ZIP | Member 6 coordinates final formatting, file naming, and packaging. Every member checks their own contribution. Aim to have the submission ready by **31 October** as an internal target. |
| **1 November** | Submit via Moodle | Submit the final ZIP before the official deadline: **11:30 pm, Sunday, 1 November 2026**. |

## 6. Required submission checks

These checks come from the assignment brief and apply to the submitted version.

- **Report format:** single column, 12-point font, margins of at least 1.25 cm, and page numbering starting at 1 after the cover sheet. Respect each section's page limit.
- **Writing and references:** use original wording and IEEE-style citations. Number, caption, and reference any diagrams included in the report.
- **Implementation:** use Python 3.12. The brief lists its standard libraries, NumPy, matplotlib, OpenCV 4.13, scikit-learn, scikit-image, and PyTorch as permitted packages. Other packages specifically required by the method may also be permitted, but must be available and have clear installation instructions. Members 3 and 4 supply the actual dependency list; Member 6 verifies it when checking the User Manual. Do not describe a run on a different Python version as verification of the required environment.
- **Dataset structure:** support the supplied `./Dhan-Shomadhan/` folder structure. The brief requires generated interim-data folders to be under the main dataset directory and a Python tool to generate their contents. The current development layout keeps generated files in `artifacts/`; Members 3 and 4 must reconcile this difference for the submitted version.
- **Pretrained models:** justify their use in Problem Description and Method and Implementation, and provide download and placement instructions in the User Manual.
- **Reproducibility:** the submitted program and User Manual must reproduce the results reported in the document.
- **Submission contents:** package `group_name.py` and `group_name.pdf` into `group_name.zip`. Exclude dataset images and interim data. Submit the ZIP via Moodle.

```text
group_name.zip
├── group_name.py
└── group_name.pdf
```
