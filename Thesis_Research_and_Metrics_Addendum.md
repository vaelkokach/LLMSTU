# Research and Benchmarking Addendum

**Project:** Deep Learning-Based Real-Time Student Behavior Analysis and Attention Loss Detection  
**Prepared:** 2026-08-01  
**Purpose:** Append to the thesis analysis and provide an implementation-oriented brief for the agent currently developing the project.

---

## 1. Executive conclusion

The most valuable next step is **not another generic image classifier**. The current evidence indicates four specific technical gaps:

1. **Temporal episode boundaries:** frame-level performance does not consistently convert into event recall or accurate onset/offset localization.
2. **Wide-shot head pose and gaze:** the current facial landmark channel has limited coverage and the remaining weak classes are orientation-related.
3. **Uncertainty and abstention:** an instructor-facing system should suppress uncertain alerts rather than force a mental-state prediction.
4. **Literature-compatible evaluation:** the project currently reports several strong internal metrics, but many cannot be compared directly with engagement-classification or behavior-detection papers.

The project should continue to describe its outputs as **observable behavioral cues**, not direct measurements of attention, comprehension, boredom, or curiosity. Different papers study different tasks: ordinal engagement classification, continuous engagement regression, object detection, framewise behavior classification, or temporal event localization. They must be benchmarked separately.

---

## 2. Highest-value papers, datasets, and repositories

### 2.1 Temporal action segmentation and event-boundary modeling — highest priority

#### Action Segment Refinement Framework (ASRF)

- Paper/project: https://yiskw713.github.io/asrf/
- Code: https://github.com/yiskw713/asrf
- Why it matters: ASRF explicitly separates frame classification from boundary regression. This matches the project’s observed problem: reasonably strong frame accuracy but substantially weaker human-gold event recall and inconsistent boundaries.
- Recommended experiment: retain the existing 556- or 570-dimensional per-frame features, add a boundary-probability head, and use predicted boundaries to refine cue segments.
- Metrics to reuse: segmental edit score and segmental F1 at multiple overlap thresholds.

#### ASFormer

- Paper: https://arxiv.org/abs/2110.08568
- Code: https://github.com/ChinaYi/ASFormer
- Why it matters: a transformer designed specifically for action segmentation, with temporal locality suited to relatively small datasets. It is a stronger temporal baseline than a generic transformer alone.
- Recommended experiment: compare the current temporal transformer with ASFormer on exactly the same sequences, split, features, seed policy, and event evaluator.
- Implementation warning: the official repository documents a small mask-code correction that should be applied in a new implementation.

#### MS-TCN / MS-TCN++

- Code: https://github.com/yabufarha/ms-tcn
- Why it matters: a standard multi-stage temporal-convolution baseline for action segmentation. It is simpler than ASFormer and therefore valuable as a sanity-check baseline.
- Recommended experiment: run MS-TCN on the current 570-dimensional features before designing a more complex custom architecture.

#### ActionFormer and OpenTAD

- ActionFormer code: https://github.com/happyharrycn/actionformer_release
- OpenTAD toolbox: https://github.com/sming256/OpenTAD
- Why they matter: they treat events as temporal instances with class, onset, and offset, and evaluate with temporal-IoU mAP. OpenTAD provides a modular toolbox containing multiple temporal action detection methods.
- Recommended use: use these after ASRF/ASFormer if the thesis needs explicit proposal-style event detection rather than only frame segmentation.
- Scope warning: these systems were designed for larger action-detection benchmarks, so adapting a full model may be excessive for only 24 human-gold episodes. Their evaluators and loss design may be more valuable than their full architectures.

### 2.2 Head pose and gaze — highest feature priority

#### DirectMHP

- Paper: https://arxiv.org/abs/2302.01110
- Code: https://github.com/hnuzhy/DirectMHP
- Why it matters: it performs direct, multi-person, full-range head-pose estimation. This is better aligned with a wide classroom frame containing small, partially occluded students than a face-landmark pipeline that first needs a confidently detected face.
- Recommended experiment: benchmark DirectMHP against the current MediaPipe-derived head-pose cache on a manually audited subset, measuring face/head coverage and yaw/pitch/roll error where labels are available.

#### 6DRepNet / 6DRepNet360

- Code: https://github.com/thohemp/6DRepNet
- Why it matters: robust unconstrained head-pose regression using a continuous 6D rotation representation and geodesic loss. It is easy to test through the provided package and pretrained models.
- Recommended experiment: use it as a second head-pose baseline. Compare downstream cue macro-F1 and human-gold event metrics, not only pose coverage.

#### UniGaze

- Paper/code: https://github.com/ut-vision/UniGaze
- Why it matters: recent cross-domain gaze estimation based on large-scale pretraining, with pretrained inference models and video prediction code. Cross-domain robustness is directly relevant because the current project exhibits severe setting shift between the computer lab, DIPSER, and SCB.
- Recommended experiment: extract normalized gaze vectors from accepted face crops and test whether personalized gaze deviation improves `looking_away` and `turned_to_peer` beyond the current head-pose approximation.
- License warning: the released model is non-commercial/responsible-AI licensed; confirm that thesis use complies with the license.

#### L2CS-Net and Gaze360

- L2CS-Net paper: https://arxiv.org/abs/2203.03339
- L2CS-Net code: https://github.com/Ahmednull/L2CS-Net
- Gaze360 paper: https://arxiv.org/abs/1910.10088
- Gaze360 code: https://github.com/erkil1452/gaze360
- Why they matter: L2CS-Net is a practical unconstrained gaze estimator; Gaze360 adds temporal gaze modeling and uncertainty. They are useful baselines if UniGaze is too heavy or incompatible with the existing environment.

#### OpenFace

- Code: https://github.com/TadasBaltrusaitis/OpenFace
- Why it matters: produces head pose, gaze, facial landmarks, and action units. CMOSE and several engagement studies use OpenFace-derived features, so an OpenFace feature baseline would improve methodological comparability with that literature.
- Recommended experiment: compare current handcrafted/head-pose features against an OpenFace-only temporal baseline on the same split.

### 2.3 Engagement and academic-emotion datasets

#### CMOSE and MocoRank

- Paper: https://arxiv.org/abs/2312.09066
- Dataset/features: https://huggingface.co/datasets/cwuau/CMOSE
- Contents: 12,193 online-learning video segments from 102 participants, four ordered engagement labels, OpenFace facial/gaze/head-pose features, I3D visual features, audio, and speech information.
- Published metrics: overall accuracy and average accuracy/balanced per-class accuracy. The paper reports 78.14% overall accuracy for its best configuration and 60.94% best average accuracy in its main architecture/loss comparison.
- Why it matters: it provides an ordinal-engagement benchmark and directly reusable feature families.
- Critical caveat: CMOSE uses online presentation-training videos and a random 70/20/10 segment split. These results are not directly comparable with the project’s leak-free video-wise six-cue classification.
- Recommended use: train a **separate ordinal engagement head** on CMOSE. Do not merge CMOSE’s internal-state labels into the visible-cue taxonomy.

#### DAiSEE

- Dataset: https://people.iith.ac.in/vineethnb/resources/daisee/
- Contents: 9,068 video clips from 112 users with four levels each for boredom, confusion, engagement, and frustration.
- Why it matters: it is a common literature reference and supports four-level affect/engagement classification.
- Caveat: webcam-style, single-person clips are substantially different from a wide in-person laboratory scene. Report cross-dataset transfer separately.

#### EngageWild / EmotiW Student Engagement

- Paper: https://arxiv.org/abs/1804.00858
- Typical target: engagement intensity represented as an ordered/continuous value.
- Typical metrics: mean squared error, Pearson correlation, classwise MSE, and weighted Cohen’s kappa for ordinal annotation agreement. The EmotiW 2018 challenge reported a best test MSE around 0.06 versus a 0.15 baseline.
- Why it matters: it provides the standard regression-style evaluation missing from a classification-only comparison.
- Recommended use: only when training a dedicated continuous/ordinal engagement estimator; do not calculate MSE over the project’s six nominal cue classes.

#### EngageNet

- Paper: https://arxiv.org/abs/2302.00431
- Code: https://github.com/engagenet/engagenet_baselines
- Contents: a large engagement dataset and baselines using action units, gaze, head pose, and learned visual representations.
- Why it matters: its repository can supply an additional reproducible feature/model baseline and a literature-oriented engagement experiment.

#### DIPSER

- Paper: https://arxiv.org/abs/2502.20209
- Processing code: https://bitbucket.org/rovitlib/dipser/
- Contents: in-person contextual and per-student camera views, 1–5 attention ratings, nine academic emotions, head-pose/gaze-related information, and wearable sensor data.
- Why it matters: this is the closest dataset to the thesis topic’s in-person, academic-emotion, and gaze requirements.
- Recommended use:
  1. evaluate head-pose/gaze estimators;
  2. train a separate academic-emotion or engagement model;
  3. retain the already discovered reversed cue–engagement relationship as evidence of context dependence.
- Caveat: the individual-camera geometry still differs sharply from the project’s wide-shot student crops.

#### Student Engagement Dataset and authentic-classroom engagement work

- Student Engagement Dataset paper: https://openaccess.thecvf.com/content/ICCV2021W/ABAW/html/Delgado_Student_Engagement_Dataset_ICCVW_2021_paper.html
- Authentic-classroom paper: https://arxiv.org/abs/2101.04215
- Why they matter: they focus on observable visual attention or real classroom engagement rather than only webcam MOOC settings. The authentic-classroom study reports AUC values of 0.620 and 0.720 in two grade groups and an average AUC improvement of 0.084 after limited person-specific personalization.
- Relevance to the current design: this supports the project’s personalized median-gaze baseline and suggests that student-specific calibration should be evaluated directly.

### 2.4 Classroom behavior detection

#### SCB-Dataset

- Paper: https://arxiv.org/abs/2304.02488
- Code/data: https://github.com/Whiffe/SCB-dataset
- Task: bounding-box detection for classroom behaviors.
- Published metrics: precision, recall, mAP@0.5, and higher-IoU mAP. The paper reports, for example, 74.0% mAP@0.5 and 56.8% higher-IoU mAP for one three-class subset, while explicitly omitting bow-head and turn-head results because they were unsatisfactory.
- Why it matters: it gives the standard metrics needed to compare an end-to-end classroom behavior detector.
- Caveats: the paper labels itself interim/in progress, annotation completeness varies by subset, and the lecture-hall behavior semantics differ from the computer-lab setting.

### 2.5 Calibration and safe alerting

#### netcal calibration framework

- Code: https://github.com/efs-opensource/calibration-framework
- Why it matters: measures and reduces neural-network miscalibration for classification and detection.
- Recommended metrics: expected calibration error, classwise ECE, negative log-likelihood, Brier score, reliability diagrams, and risk–coverage curves.

#### Temperature scaling

- Reference implementation: https://github.com/gpleiss/temperature_scaling
- Recommended experiment: fit temperature only on the validation set, apply it to untouched test predictions, and report calibration before/after without changing accuracy.

#### Selective prediction / abstention

An instructor-facing system should support an `uncertain` or `no alert` outcome. Evaluate selective performance by varying the confidence threshold and reporting:

- retained coverage;
- error/risk at that coverage;
- area under the risk–coverage curve;
- event recall and false alerts/hour after abstention.

This is preferable to forcing every student frame into an attention interpretation.

---

## 3. Metrics used in related literature and what the project should calculate

### 3.1 Important comparability rule

Do **not** create a single leaderboard mixing the following tasks:

1. six-class visible-cue classification;
2. four-level ordinal engagement classification;
3. continuous engagement regression;
4. behavior bounding-box detection;
5. phrase-to-student grounding;
6. temporal episode localization.

A 78% engagement accuracy, 0.06 MSE, 74% mAP, 0.44 macro-F1, and 0.63 R@1 answer different questions.

### 3.2 Frame-level visible-cue classification

Use these for the project’s six-class temporal model:

| Metric | Purpose | Priority |
|---|---|---|
| Accuracy | Overall fraction correct; can be dominated by `screen_oriented` | Secondary |
| Balanced accuracy / average accuracy | Mean recall across classes; used in CMOSE-style imbalanced evaluation | Primary |
| Macro-F1 | Equal weight to each class | Primary |
| Per-class precision, recall and F1 | Shows which cues fail | Primary |
| Weighted-F1 | Descriptive overall performance | Secondary |
| Macro-AUPRC | Better than AUROC when positive classes are rare | Primary |
| One-vs-rest AUROC | Ranking quality for each class | Secondary |
| Confusion matrix | Reveals semantic substitutions | Primary |
| ECE, Brier score, NLL | Confidence reliability | Primary for dashboard deployment |

Statistical reporting:

- calculate 95% confidence intervals with **video- or track-level bootstrap resampling**, not frame-level resampling;
- report results across at least three seeds for small improvements;
- keep the test split untouched until the final table.

### 3.3 Ordinal engagement classification

Use only for a separate DAiSEE/CMOSE/DIPSER engagement head:

- overall accuracy;
- balanced/average accuracy;
- macro-F1;
- mean absolute error between ordered labels;
- quadratic weighted kappa;
- Spearman rank correlation;
- confusion matrix that preserves label order.

CMOSE primarily reports overall and average accuracy. Its best reported overall accuracy is 78.14%, while its best average accuracy in the main comparison is 60.94%. These are dataset-specific reference points, not targets for the six-cue model.

### 3.4 Continuous engagement regression

Use only when the target is an ordered continuous score such as 0–1 or 1–5:

- MSE and RMSE;
- MAE;
- Pearson correlation coefficient;
- Spearman correlation;
- concordance correlation coefficient, when appropriate;
- class/bin-wise error to expose majority-range bias.

EngageWild/EmotiW uses MSE as a principal benchmark; correlation should also be reported because a low MSE can result from regressing toward the mean.

### 3.5 Bounding-box behavior detection

Use for end-to-end student behavior boxes:

- precision and recall;
- AP per behavior class;
- mAP@0.5;
- COCO-style mAP@[0.5:0.95];
- AP by object scale if enough examples exist;
- false positives per image or per minute.

Retain R@1/R@5/R@10 only for phrase-to-region grounding. R@k is not a replacement for mAP and should not be compared with SCB results.

### 3.6 Temporal event localization — the most important missing metric family

The project already reports event matches, miss rate, onset error, duration error, and false alerts/hour. Add:

| Metric | Why it is needed |
|---|---|
| Segmental F1@10, F1@25, F1@50 | Standard action-segmentation evaluation at increasing overlap strictness |
| Segmental edit score | Penalizes fragmentation and over-segmentation |
| Event precision, recall and F1 at tIoU 0.10/0.25/0.50 | Separates under-firing from false alerts |
| AP/mAP over temporal IoU thresholds | Threshold-independent event ranking |
| Onset MAE and offset MAE | Separates start and end errors |
| Duration MAE | Measures episode-length error |
| Detection delay | Critical for live intervention |
| False alerts per hour | Instructor nuisance metric |

For model comparisons, report boundary error in two ways:

1. over every event matched by that model;
2. over the **common subset of gold events matched by both models**.

The second prevents a higher-recall model from appearing to have worse boundaries merely because it detects additional difficult events.

### 3.7 Human-label agreement

For future annotation rounds:

- ICC(2,1) for multiple raters assigning continuous/ordinal scores;
- quadratic weighted Cohen’s kappa for pairwise ordinal agreement;
- Krippendorff’s alpha when multiple raters and missing labels are present;
- cue-wise agreement and acceptance/rejection rates for visible-cue annotations.

CMOSE reports ICC(2,1)=0.84. EngageWild used quadratically weighted Cohen’s kappa when evaluating annotators.

### 3.8 Real-time system metrics

FPS alone is insufficient. Report:

- end-to-end p50 and p95 latency;
- throughput/FPS versus number of visible students;
- detector, feature extraction, temporal inference, event generation, and dashboard latency separately;
- GPU memory and hardware;
- alert delay after a gold episode begins;
- performance with face blurring enabled;
- dropped-frame rate.

---

## 4. Where the current project stands

### 4.1 Grounding and attribution

The controlled ordinal-versus-Hungarian experiment is a strong contribution: R@1 rises from 0.3230 to 0.4954 while R@10 changes much less. This supports an attribution-quality interpretation. The exact-correspondence main model reaches R@1 0.6343 on a leak-free video split.

Literature position:

- strong internal causal ablation;
- not directly comparable with engagement-classification papers;
- not directly comparable with SCB mAP;
- should be supplemented with standard behavior-box AP only if an end-to-end detection comparison is claimed.

### 4.2 Six-class temporal cue model

The current 570-dimensional configuration has the best reported DDP macro-F1 and frame accuracy, but the absolute 552/556/570 values must be rerun with one single-process evaluator before being placed beside literature results. The 556-to-570 macro-F1 gain is small and the two added feature families are confounded.

Required before comparison:

1. single-process evaluation for all three checkpoints;
2. balanced accuracy and macro-AUPRC;
3. bootstrap confidence intervals;
4. at least three seeds for the small 556-to-570 difference;
5. isolated `+expression` and `+dynamic/gaze` ablations.

### 4.3 Event detection

The project is unusually strong in one respect: it already evaluates events against human gold and reports false alerts/hour. Many student-engagement papers stop at clip accuracy or MSE.

Current weakness:

- event recall remains far below the teacher reference;
- frame accuracy does not transfer reliably to episode recall;
- boundary conclusions are sensitive to which episodes each model matches.

The highest-value model experiment is therefore ASRF-style boundary regression or an ASFormer/MS-TCN baseline, not another static feature extractor.

### 4.4 Facial expressions and academic emotion

The measured boredom AUROC of 0.544 indicates that seven basic facial expressions do not reliably provide academic-emotion recognition. The thesis should treat this as a measured limitation.

Literature position:

- do not compare 0.544 with four-level engagement accuracy;
- do not claim boredom detection from the current basic-expression features;
- use DIPSER academic-emotion labels for a separate supervised task or report the channel only as auxiliary features.

### 4.5 External validity

The reversed DIPSER correlation and the confounded SCB transfer result support a defensible conclusion: visual cues are context-dependent. This is valuable evidence against universal mental-state claims.

A clean external experiment should either:

- evaluate a directly shared target such as head-pose/gaze angular error; or
- train a separate target-dataset engagement head and report within-dataset metrics;
- avoid interpreting incomplete behavior boxes as exhaustive person annotations.

### 4.6 Runtime

The measured 7.1 FPS real-scene result supports a near-real-time prototype, while 2 FPS with 30 synthetic students shows a scaling limitation. Add latency percentiles and student-count curves before claiming real-time operation.

---

## 5. Prioritized task list for OPUS

### P0 — required for a defensible final thesis table

1. **Create one unified single-process evaluator** for 552-, 556-, and 570-dimensional models.
2. Report accuracy, balanced accuracy, macro-F1, macro-AUPRC, per-class metrics, confusion matrices, ECE, Brier score, and NLL.
3. Add video/track-level bootstrap 95% confidence intervals.
4. Add temporal metrics: edit score, F1@10/25/50, event precision/recall/F1 at tIoU 0.10/0.25/0.50, onset/offset/duration MAE, detection delay, and false alerts/hour.
5. Recompute boundary errors on the common matched-event subset.
6. Run isolated feature ablations:
   - 552 base;
   - 556 base + head pose;
   - 563 base + head pose + expression;
   - 563 base + head pose + dynamic/gaze;
   - 570 full.
7. Resolve the final-results register so stale DDP, matching, dashboard, and open-item statements cannot enter the thesis.

### P1 — highest-value model improvements

8. Implement an MS-TCN baseline using the current feature sequences.
9. Implement ASRF-style boundary regression or adapt ASFormer.
10. Benchmark DirectMHP and 6DRepNet against the current head-pose extractor.
11. Benchmark UniGaze or L2CS-Net for gaze; retain the student-specific median deviation design.
12. Calibrate cue probabilities with temperature scaling and evaluate abstention/risk–coverage for dashboard alerts.
13. Add repeated-seed evaluation for all small feature improvements.

### P2 — literature alignment and optional external benchmarks

14. Train a **separate** four-level ordinal engagement model on CMOSE and report accuracy, average accuracy, macro-F1, MAE, and quadratic weighted kappa.
15. Optionally evaluate continuous engagement on EngageWild/EmotiW with MSE, MAE, Pearson, and Spearman correlation.
16. Add standard mAP@0.5 and mAP@[0.5:0.95] only if an end-to-end behavior-box detector is evaluated.
17. Benchmark end-to-end latency, p50/p95, GPU memory, and FPS across student counts.

---

## 6. Recommended final literature-comparison table structure

Do not put every result in one table. Use separate tables:

### Table A — Visible-cue classification on the project dataset

Columns: model, features, split, accuracy, balanced accuracy, macro-F1, macro-AUPRC, ECE, parameters/FPS.

### Table B — Human-gold temporal events

Columns: model, event precision, event recall, event F1, F1@10/25/50, edit, onset MAE, offset MAE, false alerts/hour.

### Table C — Phrase-to-student grounding

Columns: binding method, training regions, R@1, R@5, R@10, calibration/coverage.

### Table D — External engagement literature

Columns: paper/dataset, task definition, label scale, split unit, principal metric, reported result, comparability caveat.

Suggested rows:

- CMOSE/MocoRank — four-level ordinal engagement; random segment split; accuracy/average accuracy; 78.14/60.94.
- DAiSEE — four-level boredom/confusion/engagement/frustration; clip classification; report the exact result from the chosen baseline paper.
- EngageWild/EmotiW — continuous/ordinal engagement; MSE and correlation; challenge best MSE around 0.06.
- Authentic classroom engagement — binary/continuous engagement from facial video; AUC 0.620/0.720; personalized improvement 0.084.
- SCB — behavior-box detection; P/R and mAP; not comparable with cue macro-F1 or grounding R@k.
- Current project — six observable cues plus event localization; video-wise split; report only finalized single-process and human-gold metrics.

---

## 7. Recommended thesis wording

> Prior student-engagement studies do not evaluate a single uniform task. Some assign four ordered engagement levels to short video clips, some regress a continuous engagement score, some detect classroom behaviors as bounding boxes, and others classify visible cues frame by frame. Consequently, their reported accuracy, mean squared error, mean average precision, or temporal F1 values are not interchangeable. This thesis therefore reports task-specific metric families: retrieval metrics for description-to-student grounding, classification and calibration metrics for visible behavioral cues, temporal-overlap metrics for sustained episodes, and runtime metrics for deployment. External literature values are presented as contextual references rather than as a single cross-dataset leaderboard.

> The project’s main empirical contribution is not a claim that internal attention can be read directly from video. It is a leak-free, human-audited pipeline for detecting, assigning, and temporally aggregating observable student behavioral cues in a computer-laboratory setting. External experiments show that the interpretation of those cues changes with the learning environment, which limits universal attention inference and motivates uncertainty-aware, human-in-the-loop use.

---

## 8. Final recommendation

For the remaining thesis schedule, prioritize **evaluation consistency and event-boundary modeling** over dataset expansion. A single-process, confidence-calibrated, boundary-aware comparison with proper temporal metrics will strengthen the thesis more than adding another broad engagement dataset without a matching task definition.
