# Master's Thesis Defensibility Review

**Project reviewed:** *Deep Learning-Based Real-Time Student Behavior Analysis and Attention Loss Detection*

**Repository reviewed:** commit `03e73f2b5c3f83d7544439a8d9ff71cd8115987d`

**Review date:** 8 August 2026

**Scope:** scientific claim, implementation, experiments, preserved evidence, reproducibility, ethics, and likely defence exposure

## Executive judgement

**This is conditionally defensible as a master's thesis.** The work is strongest when presented as a setting-specific, near-real-time system for detecting, attributing, and temporally aggregating **observable student behavioural cues** in a computer laboratory. It is not defensible as a system that measures a student's internal attention, comprehension, boredom, curiosity, or learning outcome.

The corrected project contains enough technical depth for a master's thesis:

- a data-recovery and leak-free video-splitting pipeline;
- a controlled caption-to-student binding experiment;
- a Grounding-DINO/LLMDet grounding model;
- a six-class temporal cue model with transformer, MS-TCN, and ASRF-style comparisons;
- isolated feature ablations over three seeds;
- calibration and selective alerting;
- human-referenced temporal-event evaluation;
- external analyses on DIPSER, CMOSE, and SCB;
- an instructor dashboard and measured deployment optimisations; and
- a unusually valuable audit of silent experimental failures.

The thesis is nevertheless exposed in four places: the repository does not preserve most raw HPC evidence, the human-gold event set is small and single-annotator *(closed 2026-09-11 — see C; a second annotator gives κ 0.800 over fields and 0.711 on the derived cue, and in doing so shows the label ceiling is annotator-dependent by ~0.16 macro-F1)*, the exact pseudo-labelling model release is unknown, and no ethics/consent approval record is present. The first two weaken the empirical claims; the last may become an administrative blocker depending on institutional rules.

My practical judgement is:

| Area | Judgement | Defence implication |
|---|---|---|
| Technical scope | Strong for a master's project | More than sufficient implementation breadth |
| Novelty | Moderate, with one strong controlled result | Phrase-to-student binding and failure-aware evaluation are the clearest contributions |
| Internal validity | Strong after the August rebuild | Video-wise splits, pre-registered test use, paired cluster bootstrap, and three seeds are credible |
| Construct validity | Acceptable only under the visible-cue framing | Internal attention or emotion claims would make the thesis vulnerable |
| External validity | Limited and honestly measured | One computer laboratory; external results show the cue meaning changes by setting |
| Human validation | Weak-to-moderate | Sixteen distinct episodes over ten tracks, annotated by one person |
| Runtime claim | Near-real-time only | 5.77 FPS at about six students on one A100; not real-time by a 10 FPS budget |
| Reproducibility from this clone | Weak | Code and summaries are present, but most checkpoints, predictions, tables, and logs remain on the HPC |
| Ethics/data governance | Unresolved | No approval, consent, retention, or dataset-rights record was found in the clone |

## What is genuinely defensible

### 1. The scientific claim is measurable

The supervisor's reframing is the key epistemic correction: report that a student's head is down or that a phone-use cue persists, not that the student is inattentive. This makes the output visually auditable and prevents the model from turning uncertain pixels into an unsupported mental-state claim.

The DIPSER analysis supports this decision empirically. An a-priori head-pose proxy correlates weakly but significantly in the opposite direction to the naive disengagement hypothesis (`rho = +0.172`, permutation `p < 0.0001`, 1,825 pairs). In a lecture hall, head movement can reflect note-taking or following an instructor; in a computer lab, stillness toward a monitor may be the on-task pattern. This result limits generalisation but strengthens the visible-cue construct.

### 2. The caption-to-student binding ablation is a strong contribution

The cleanest causal experiment holds frames, captions, boxes, schedule, learning rate, seed, and evaluation data fixed. Changing only the description-to-box binding from detector-confidence order to content-based Hungarian assignment raises validation R@1 from `0.3230` to `0.4954`, an absolute gain of `0.1724` or 53% relative. R@10 changes by only `0.0141`, supporting the interpretation that the main gain is correct attribution rather than person localisation.

The confidence-filter arm is also useful because it is negative: retaining only high-confidence assignments reduces R@1 to `0.4787`. The experiment therefore answers a real methodological question instead of reporting only a winning model.

### 3. The corrected Branch-B evaluation is substantially better than the early project

The final temporal evaluation uses a shared 73/27/27 video split, three seeds, one evaluator, stored metric definitions, video-level cluster bootstrap, paired comparisons, and a written test protocol. The principal result is not a tiny feature delta: changing the temporal architecture from the generic transformer to MS-TCN improves test macro-F1 by `0.0933`, significant in all three seed-matched comparisons. Head pose adds a smaller but repeatable `0.0296` on test.

The study also reports null results: isolated expression and body-language/gaze blocks do not improve macro-F1, and the combined block is nominally harmful on test. This restraint will read well in a defence if the invalid earlier claims are not reintroduced.

### 4. The event and calibration layers match the deployment question

The project does not stop at frame accuracy. It reports segmental F1, edit score, event precision/recall, temporal overlap, boundary error, false alerts per hour, and selective-alert thresholds. MS-TCN-570 raises edit score to `59.9` from the transformer's `28.8–38.2`, showing reduced fragmentation. Event recall remains low (`0.312` at best versus `0.625` for the pseudo-labelling teacher), and the thesis correctly treats this as an unresolved gap.

Temperature scaling and abstention are appropriate for an instructor-facing prototype because confidence controls whether an alert is shown without changing the underlying argmax. The latest deployment configuration freezes a display threshold of `0.46` and alert threshold of `0.66` from validation data.

### 5. The post-mortem is thesis-worthy methodology, not an embarrassment

The March experiments failed through temporal leakage, duplicate checkpoints, empty metric fallbacks, zero-support classes, incorrect metric aggregation, and broken inference. The valuable finding is that most defects produced plausible outputs. The rebuilt project responds with video-wise splits, fail-loud assertions, checkpoint provenance, class-support checks, unified evaluation, and an explicit non-citable-results register.

This material should be presented as a reproducibility and measurement-validity contribution. It should not dominate the thesis, but it demonstrates mature scientific judgement.

## Claims that are not defensible

The following claims should not appear, even informally:

- “The system detects attention loss” without immediately defining it as sustained visible off-task cues.
- “The system measures engagement, comprehension, boredom, curiosity, or emotion.” The measured boredom AUROC is `0.544`, barely above chance.
- “The model generalises to classrooms.” DIPSER reverses the cue interpretation, and the SCB transfer setup is confounded by incomplete person annotation and semantic mismatch.
- “The system operates in real time.” The adopted configuration reaches `5.77 FPS` at roughly six students on an A100 and misses a 10 FPS budget.
- “The end-to-end system has the human-gold event scores.” The reported event study excludes detector and tracker errors by design.
- “The 570-dimensional model is the best.” The isolated feature study rejects that conclusion.
- Any March detector or temporal result, the contaminated `0.9231` matching accuracy, shard-local DDP macro-F1, or SCB `R@1 = 0.0322` as a clean transfer score.

## Material threats before submission or defence

### A. Ethics, consent, and data governance

The repository contains student video/crops but no ethics approval, consent form, participant-information sheet, data-management plan, retention schedule, or publication permission. The early system diagram says ethics approval is a prerequisite, and the supervisor's notes state that approval was forgotten. This cannot be repaired through wording.

**Action:** confirm the university's requirements with the supervisor/ethics office and preserve the decision or approval document in a restricted administrative bundle. The thesis must accurately state whether the data were collected under consent, an existing institutional basis, retrospective approval, exemption, or another authorised process. Do not invent an approval number.

### B. Raw experiment archival

The final register contains 207 metric entries, 134 marked citable. In this clone, most direct evidence paths under `work_dirs/` do not exist. The code, register, narrative tables, test protocols, split manifest, and human annotation JSONL survive, but the majority of checkpoints, prediction archives, run records, raw table JSON, calibration outputs, profiling JSON, and logs remain external to the clone.

**Action:** recover the final HPC thesis bundle before defence. At minimum archive:

1. Branch-A final and ablation checkpoints;
2. all seed checkpoints used in Table A;
3. prediction archives for validation and test;
4. Table A/B source JSON and paired-comparison JSON;
5. human-gold feature cache and derived event JSONL;
6. calibration/coverage-risk outputs;
7. runtime profiling and stride-equivalence reports;
8. CMOSE and DIPSER result summaries;
9. each run's `command.txt`/`run_record.json`; and
10. SHA-256 hashes plus the git commit and environment specification.

Without this bundle, the thesis can be defended as a documented completed project, but not independently reproduced from the submitted repository.

### C. Single-annotator human gold — **CLOSED 2026-09-11**

The raw dense bundle verifies 984 annotations, of which 754 are accepted and 230 rejected. The corrected event denominator is only 16 distinct episodes over ten tracks. No second-annotator file is present, despite the proposed design diagram and supervisor notes calling for two annotators.

**Action:** if time permits, have a second person independently annotate 100–200 sampled frames or several complete tracks and report per-cue agreement/Cohen's kappa. If this cannot be done, state plainly that the human-gold set reflects one annotator's operational interpretation and use “diagnostic set,” not “ground truth” or “ceiling.”

**Done, 2026-09-11 (FINDINGS §19).** A second annotator independently labelled a
250-crop shared subset, stratified across the six cue classes — above the
100–200 the action asked for. Results:

| | |
|---|---|
| mean Cohen's κ, 10 annotation fields | **0.800** |
| κ on the derived `cue6` class | **0.711** (76.0% raw agreement) |
| κ on the derived `cue9` class | 0.713 (75.2%) |
| strongest / weakest field | `laptop_visible` 0.992 / `engagement_level` 0.511 |

Both derived-cue figures are "substantial" on Landis–Koch. `engagement_level` is
the weakest field and is read by no cue rule except as an `unknown` test, so it
costs the cue labels little.

**But the second half of the action still stands, for a reason the measurement
itself uncovered.** The same pass shows the "ceiling" is a property of the
annotator, not only of the pseudo-labeller: on the *same 250 crops* it is 0.8516
against the first annotator and **0.6880** against the second, and a control
isolating the sample shows stratification explains only −0.013 of that gap while
the annotator explains −0.164. The cause is that the annotation UI pre-fills the
pseudo-label and the two annotators deferred to it at very different rates
(73.9% vs 47.6% of crops kept entirely).

So the review's fallback wording should be adopted anyway: quote the ceiling as
a **bracket, 0.69–0.85**, prefer the annotator-independent human–human agreement
(κ 0.711), and do not call any single figure "the ceiling". Both passes saw the
same pre-fill, so even 0.711 is an upper bound on independent agreement.

### D. Pseudo-label provenance

The structured label records do not identify the exact VLM checkpoint. Repository documents conflict between “Qwen3-VL” and “Qwen3.5-VL.” The evidence only supports “Qwen3-VL-family model.”

**Action:** recover the job submission/configuration or model-cache record. If unavailable, keep the family-level description and list the exact release as unrecoverable provenance.

### E. Evidence-register freshness

`outputs/FINAL_RESULTS_REGISTER.*` was generated on 1 August, but deployment was changed on 3 August. It still contains older values such as the `0.64` alert threshold and approximately `3 FPS` deployment figure, while the current deployment config uses `0.66` and detector/temporal striding reaches `5.77 FPS`. The Branch-A protocol file also records code commit `7c563f9`, whereas the protocol itself was committed at `87bb2db`; this is explainable but should be normalised.

**Action:** rebuild the final register from the recovered HPC artifacts after incorporating the August 3 deployment results. Use one evidence cutoff in the submitted thesis.

### F. Branch-B test-protocol scope

The written Branch-B protocol lists five test systems: transformer-552, transformer-556, MS-TCN-556, ASRF-556, and ASRF-570. The final test table additionally contains transformer-563-expression, transformer-563-dynamics, transformer-570, and MS-TCN-570. The protocol's reproduction commands evaluate whole experiment roots, which helps explain how the extra rows were produced, but it conflicts with the section titled “Exactly what will be run.”

**Action:** retain the pre-registered status only for the five listed systems and their planned contrasts. Label the additional test configurations exploratory/descriptive, or document a time-stamped pre-run amendment if one exists. The central architecture contrast and head-pose contrast remain inside the listed protocol; the transformer 570-vs-556 test null should not be presented as pre-registered.

## Recommended thesis contribution statement

The thesis should claim four contributions:

1. **A leak-free, human-audited data and evaluation pipeline** for wide-shot computer-lab video, including recovered video identity and video-wise split control.
2. **A controlled study of pseudo-label attribution**, demonstrating that content-based bipartite binding materially improves phrase-to-student grounding while aggressive confidence filtering does not.
3. **A temporal visible-cue study**, showing that task-appropriate temporal convolution/refinement architectures matter more than the tested expression and motion/gaze feature additions.
4. **An uncertainty-aware near-real-time prototype**, with temporal-event evaluation, calibrated abstention, privacy-oriented UI choices, and measured runtime/accuracy trade-offs.

The failure audit and external context-dependence experiments support these contributions as methodological evidence rather than forming separate headline claims.

## Minimum defence preparation

Before the defence, prepare concise answers to these questions:

1. Why is this not a mental-state detector?
2. What exactly is novel beyond combining existing models?
3. Why is R@1 the correct Branch-A metric, and why is it not mAP?
4. How was leakage prevented, and when was the test split first read?
5. Why did MS-TCN outperform the transformer with fewer parameters?
6. Why do facial-expression and gaze-dynamic features remain in the implementation if they did not improve macro-F1?
7. Why are the event metrics based on only 16 episodes?
8. Why does the human event evaluation omit detector/tracker errors?
9. Why is 5.77 FPS described as near-real-time?
10. What ethical authority permitted collection and processing of student video?
11. Can every table be regenerated from the submitted artifact bundle?

## Final verdict

**Defensible now as a carefully scoped engineering/research prototype; considerably stronger after closing the ethics, archival, and second-annotator gaps.** The thesis should lead with the corrected scientific discipline and controlled experiments, not with a broad promise to read attention from faces. Its strongest characteristic is not a state-of-the-art score. It is that the rebuilt evaluation now measures what the project says it measures, and documents where it does not.
