# Draft chapters — Experimental Setup, Results, Discussion, Limitations

**Status:** draft prose for the thesis, 2026-08-03. Every number is taken from a
frozen evidence file; the source is named in the margin note under each table so
it can be checked or regenerated. Caveats that must travel with a number are
written into the sentence that states it, not relegated to a footnote — if a
caveat is cut, the claim must be cut with it.

**Numbers deliberately absent from this draft**, because they exist in the
repository but must not be cited: the 92.3% caption–box matching value
(ground-truth leak via the spatial prior), any DDP shard-averaged macro-F1, the
SCB zero-shot 0.0322 as a transfer score, and the pre-2026-07-29 pipeline in
its entirety. See `outputs/FINAL_RESULTS_REGISTER.md`, where each carries its
reason.

---

## 4. Experimental setup

### 4.1 Data and splits

The internal corpus (LLMSTU) comprises 283,913 per-student crops sampled at 1 fps
from 128 recordings of a university computer laboratory, each crop carrying a
caption and ten structured fields produced by a Qwen3-VL-family vision–language
model. One recording carries no pseudo-labels at all and is excluded upstream of
every split, leaving **127 videos**.

Video identity was recoverable from filenames for only 2.3% of frames, which
initially blocked leak-free splitting; it was recovered by appearance-chain
tracking, verified at 4,660 of 4,660 anchors.

All splits are **video-wise**: every crop of a recording lands in exactly one
split, so the near-duplicate frames that 1 fps sampling produces cannot leak
across the boundary. Both branches of the system share one partition of
**73 / 27 / 27 videos**. For the temporal model this yields 4,390 / 1,085 / 1,056
sequences and 185,050 / 42,702 / 43,733 frames.

> This shared partition is recent. The sequence builder originally drew its own
> 80/20 shuffle independent of the detector's, which placed 23 of the detector's
> 27 held-out videos inside the cue model's training set. No test figure for the
> temporal model was defensible until the sequences were re-partitioned.

### 4.2 Taxonomy

Six observable cue classes: `screen_oriented`, `looking_away`, `head_down`,
`turned_to_peer`, `phone_use`, `uncertain`. The taxonomy describes **visible
behaviour** and never asserts a mental state: the system reports that a
student's head is down, not that the student is inattentive. `uncertain` is a
first-class outcome for crops whose orientation cannot be verified from pixels,
rather than a residual bucket.

### 4.3 Models

The per-frame feature vector is 570-dimensional and blockwise:

| columns | block | contents |
|---|---|---|
| 0–551 | base | CLIP (512) + bbox geometry (8) + colour statistics (24) + posture proxies (8) |
| 552–555 | head pose | yaw, pitch, roll, `face_found` |
| 556–562 | expression | seven basic-expression probabilities |
| 563–569 | dynamics | motion statistics and personalised gaze deviation |

Because the blocks are contiguous columns of one stored array, every feature
configuration in Section 5.2 is a **column slice of the same data**. No feature
is re-extracted between conditions, so the ablation cannot be confounded by
extraction drift.

Three temporal architectures are compared over identical sequences: the
project's original four-layer transformer encoder, a multi-stage dilated
temporal convolutional network (MS-TCN), and an action-segment refinement
framework (ASRF) that adds a boundary-regression head and relabels each
predicted segment by its mean posterior.

### 4.4 Evaluation protocol

All reported metrics come from a single evaluator run in one process. This
matters: three separate harnesses previously produced numbers that entered the
same table, and the distributed trainer computed validation metrics on one
rank's shard, which reversed the ordering of the feature ladder.

Every configuration is trained with **three seeds**. Confidence intervals use a
**video-level cluster bootstrap**: frames within a track are strongly
autocorrelated by construction — a sustained cue is the premise of the event
layer — so resampling frames would treat ~271k dependent observations as
independent. Comparisons between configurations use **paired** resampling, so
the interval is on the difference. A difference is described as real only when
it holds in every seed pair.

Checkpoints are selected by the highest mean validation macro-F1 over a trailing
five-epoch window. Single-epoch validation macro-F1 in this setting swings by
about ±0.05 between adjacent epochs, so an `argmax` over ninety epochs would
select substantially on noise.

The test split was evaluated **once**, under a protocol registered in the
repository before any model read it.

---

## 5. Results

### 5.1 Description-to-student grounding

| binding used to build training labels | training regions | R@1 | R@5 | R@10 |
|---|---|---|---|---|
| ordinal (detector-confidence order) | 165,013 | 0.3230 | 0.8978 | 0.9811 |
| **Hungarian (CLIP cost)** | 165,017 | **0.4954** | 0.9595 | 0.9952 |
| Hungarian + confidence filter | 98,526 | 0.4787 | 0.9441 | 0.9913 |
| exact correspondences — validation | 141,522 | 0.6343 | 0.9808 | 0.9963 |
| **exact correspondences — TEST** | 141,522 | **0.6462** | **0.9891** | **0.9982** |

*Evidence: `outputs/FINAL_RESULTS_REGISTER.md`, Branch A. All arms share frames,
captions, boxes, schedule, learning rate and seed; only the binding differs.*

Replacing ordinal binding with Hungarian assignment raises R@1 by **+0.1724
(+53% relative)** while R@10 moves by only 0.0141. That divergence between
top-1 and top-10 on identical inputs is the evidence that the gain is one of
**attribution** — which description belongs to which student — rather than of
detection. Content-only matching accuracy is 0.7896 for Hungarian against
0.2607 for ordinal.

The confidence filter is a **negative result**: discarding 40% of assignments to
retain only high-confidence pairs costs 0.0167 R@1, losing more to reduced data
volume than it gains in label purity.

Test exceeds validation by 0.0119. This is ordinary between-split variance and
is not claimed as an improvement; its value is as evidence that the many
validation-guided choices did not overfit.

### 5.2 Visible-cue classification

**Feature ablation** (temporal transformer, three seeds, paired video-level
bootstrap on validation):

| added block | Δ macro-F1 | seeds significant | verdict |
|---|---|---|---|
| head pose (556 − 552) | **+0.0294** | **3/3** | supported |
| facial expression, isolated | +0.0009 | 0/3 | not supported |
| body-language/gaze dynamics, isolated | −0.0105 | 0/3 | not supported |
| expression + dynamics together | −0.0031 | 0/3 | not supported |

*Evidence: `work_dirs/thesis/tables/table_a_val.md`.*

Only the head-pose block earns its place. The expression and dynamics blocks
were previously added **together**, which made their individual contributions
unidentifiable; isolating them shows neither improves aggregate macro-F1, and
the result holds under all three architectures. On the test split the combined
block is **−0.0167** macro-F1, i.e. nominally harmful.

Decomposing the block that does work: `face_found` alone contributes **+0.0236**
of the +0.0294, while the metric yaw/pitch/roll angles contribute **+0.0053**
(0/3 significant) and add nothing measurable on top of the flag (+0.0058, 0/3).
Roughly four fifths of the head-pose contribution is the single bit recording
whether a face was detectable at all.

**Architecture comparison** (556-dimensional features held identical):

| contrast | validation Δ macro-F1 | test Δ macro-F1 | seeds significant |
|---|---|---|---|
| MS-TCN − transformer | +0.1007 | **+0.0933** | **3/3** |
| ASRF − transformer | +0.1043 | **+0.0705** | **3/3** |
| ASRF − MS-TCN | +0.0036 | — | 0/3 (tie) |

Per class, ASRF over the transformer: `looking_away` +0.156, `phone_use` +0.147,
`uncertain` +0.123, `turned_to_peer` +0.110, `head_down` +0.076.

**Table A — test split**, mean ± sd over three seeds:

| model | dims | accuracy | balanced acc. | macro-F1 | macro-AUPRC | ECE |
|---|---|---|---|---|---|---|
| transformer | 552 | 0.657 ± 0.003 | 0.396 | 0.377 ± 0.007 | 0.357 | 0.065 |
| transformer | 556 | 0.699 ± 0.011 | 0.424 | 0.407 ± 0.004 | 0.392 | 0.068 |
| transformer | 570 | 0.660 ± 0.011 | 0.424 | 0.390 ± 0.005 | 0.388 | 0.075 |
| ASRF | 556 | 0.710 ± 0.022 | 0.513 | 0.477 ± 0.016 | 0.473 | 0.033 |
| ASRF | 570 | 0.735 ± 0.023 | 0.521 | 0.492 ± 0.019 | 0.482 | 0.031 |
| **MS-TCN** | **556** | 0.759 ± 0.002 | 0.519 | **0.500 ± 0.011** | 0.489 | 0.032 |
| MS-TCN | 570 | 0.765 ± 0.018 | 0.481 | 0.488 ± 0.023 | 0.477 | 0.025 |

*Evidence: `work_dirs/thesis/tables/table_a_test.{json,md}`. A majority-class
control reaches macro-F1 0.144.*

The central finding of this section is that **the temporal inductive bias
matters far more than the feature engineering**. The architecture change is
worth roughly three times the only feature block that works, at one fifth of the
parameters (2.7 M against 12.6 M), and its gains concentrate on precisely the
two orientation classes — `looking_away` and `turned_to_peer` — that resisted
every feature added to the transformer.

A note on model selection, recorded because pre-registration only means
something when it is honoured: **ASRF-556 was selected on validation**, where it
was statistically tied with MS-TCN. On the test split MS-TCN-556 scores higher
(0.500 against 0.477). The selection is reported as made, not revised.

### 5.3 Temporal episodes

Model, teacher and control are scored against the same densely annotated human
gold, with the detector and tracker excluded by design so the measurement
isolates the cue and event layers.

| system | frame acc. | F1@25 | edit | event recall | event precision | FA/h |
|---|---|---|---|---|---|---|
| transformer 552 | 0.738 | 0.244 | 30.9 | 0.292 | 0.560 | 9.4 |
| transformer 556 | 0.782 | 0.265 | 38.2 | 0.208 | **0.622** | **5.1** |
| transformer 570 | 0.740 | 0.237 | 28.8 | 0.167 | 0.450 | 9.4 |
| MS-TCN 556 | 0.788 | 0.291 | 48.5 | 0.271 | 0.554 | 11.1 |
| **MS-TCN 570** | 0.771 | **0.362** | **59.9** | **0.312** | 0.454 | 15.3 |
| ASRF 570 | 0.785 | 0.291 | 53.4 | 0.229 | 0.506 | 9.4 |
| teacher (pseudo-labeller) | 0.874 | 0.700 | 76.5 | 0.625 | 0.588 | 17.9 |
| majority control | 0.341 | 0.067 | 34.9 | 0.000 | 0.000 | 0.0 |

*Evidence: `work_dirs/thesis/tables/table_b_gold_dedup.md`, three seeds, tIoU
0.30, **16 distinct gold episodes over 10 tracks**.*

Three observations, none flattering in isolation.

**Over-segmentation is the failure mode, and boundary-aware models address it.**
The segmental edit score — which penalises the fragmentation that frame accuracy
is blind to — rises from 28.8–38.2 under the transformer to 48.5–59.9 under
MS-TCN, against a teacher at 76.5. Segmental F1@25 rises from 0.265 to 0.362.

**Frame accuracy does not convert into episode recall.** MS-TCN-570 reaches 88%
of the teacher's frame accuracy but only 50% of its episode recall. The gap
narrowed; it did not close.

**The models are conservative, which is the preferable failure direction here.**
The highest-precision configuration fires at 5.1 false alerts per hour against
the teacher's 17.9. For a system that interrupts an instructor, under-firing
accurately is better than over-firing.

> **The gold set has 16 distinct episodes, not 24.** The event layer defined an
> `inactivity` channel identical to `head_down`, so every head-down episode was
> emitted twice, and one zero-length marker appeared twice in the gold and was
> therefore unmatchable by one-to-one assignment. The duplicate channel has been
> removed. All figures in this table use the de-duplicated denominator.

> **Sample size governs what may be said here.** One episode is 6.25 percentage
> points of recall and seed spreads are ±0.04–0.07. These are **diagnostic**
> measurements from two densely annotated segments and are not an estimate of
> classroom-wide performance.

**Boundary localisation, like-for-like.** Boundary errors computed over each
system's own matched events are not comparable across systems with different
recall — a model that finds additional, harder episodes is charged for their
looser boundaries. Restricted to episodes both systems matched, the 556-dim
model's onset error is 1.44 s against the 552-dim model's 8.92 s, reversing the
apparent degradation seen in the unrestricted comparison. ASRF has the tightest
boundaries of any system measured, including the teacher (onset 1.00 s, offset
1.44 s), on three commonly matched episodes.

### 5.4 Calibration and selective alerting

Temperature fitted on validation and applied unchanged to test. Temperature
scaling cannot alter an argmax, so accuracy, macro-F1 and the confusion matrix
are unchanged by construction; only confidence moves.

| model | T | ECE before → after | NLL before → after |
|---|---|---|---|
| transformer 556 | 1.311 | 0.0739 → **0.0191** | 0.9660 → 0.9121 |
| ASRF 556 | 0.759 | 0.0570 → **0.0306** | 0.7854 → 0.7791 |
| MS-TCN 556 | 0.924 | 0.0287 → **0.0155** | 0.7033 → 0.7043 |

The transformer is over-confident (T > 1); the boundary-aware models are
slightly under-confident and were already better calibrated before scaling.

Two abstention thresholds are read from the validation coverage–risk curve and
frozen: a **display** threshold retaining ≥ 90% of frames (0.46; 91.1% coverage
at 76.4% selective accuracy) and an **alert** threshold requiring ≥ 85%
selective accuracy (0.66; 64.6% coverage at 85.3%), against 75.6% accuracy at
full coverage. Below the display threshold the interface reads `uncertain`;
below the alert threshold no sustained-episode alert may fire. The raw
prediction is retained either way — abstention withholds an alert, never
evidence.

### 5.5 Runtime

Deployment is single-GPU. The four A100s used for training are not the intended
classroom hardware and their throughput is not reported as such.

| configuration | FPS | p50 | p95 |
|---|---|---|---|
| transformer, no head pose (earlier measurement) | 5.32 | — | — |
| MS-TCN 556, FaceMesh head pose | 2.84 | 342.6 ms | 433.7 ms |
| MS-TCN 553, face detector | 3.69 | 270.7 ms | 295.9 ms |
| **MS-TCN 553, detector, strided 3:2** | **5.77** | **157.6 ms** | 291.7 ms |
| MS-TCN 553, detector, strided 5:3 | 7.03 | 117.9 ms | 257.5 ms |

Two optimisations, each validated against the unoptimised pipeline rather than
assumed:

Since about four fifths of the head-pose block's value is the `face_found` bit
(Section 5.2), the full facial-landmark mesh was replaced by a plain face
**detector**. Downstream accuracy is a statistical tie (−0.0120 macro-F1,
significant in 1 of 3 seed pairs) for a 30% throughput gain.

Because seated students are stationary — box centres jitter by 4.4% of their
diagonal across 115 tracks — the detector runs every third frame and the
temporal head every second, with the tracker holding boxes in between. Replaying
the same video at stride 1 and 3:2 and matching students across runs by box
overlap gives **94.7% per-frame cue agreement at 100% track coverage**.

Scaling by visible students, at 3:2, the per-student feature stage dominates
beyond roughly five students. **The system is near-real-time, not real-time**:
even at 7.03 FPS every frame exceeds a 10 fps budget, and striding improves
throughput without improving the tail — detection frames still cost ~270 ms, so
latency is bimodal.

### 5.6 External evidence

**Cue meaning is setting-dependent.** On DIPSER, an independent in-person corpus
with expert engagement ratings, the correlation between an a-priori off-task
head-pose proxy and expert rating is **ρ = +0.172 (p < 0.0001, 25 subjects,
1,825 paired observations)** — significant, and in the **opposite** direction to
the naive hypothesis. In a lecture hall, head movement accompanies note-taking
and tracking the instructor; in a computer laboratory, stillness facing a
monitor is the on-task state. The same pixels carry opposite meaning. The
correlation is weak and the ratings are skewed, so the defensible statement is
that the relationship is significantly non-zero and reversed, not that
engagement is predicted by head movement.

**Basic facial expressions do not deliver academic emotions.** Against DIPSER's
expert boredom annotations, a logistic combination of the seven basic-expression
probabilities reaches **AUROC 0.544** on held-out subjects (1,176 paired
observations). Only `happy` is individually significant, and inversely. The
thesis names boredom, perplexity and curiosity — Pekrun's academic emotions —
whereas an off-the-shelf recogniser outputs Ekman's basic expressions. These are
different constructs, and the measurement says so.

**Split policy is worth quantifying.** A separate four-level ordinal engagement
model trained on CMOSE reproduces the published range under the dataset's own
split (accuracy 0.718, average accuracy 0.601, against 78.14% and 60.94%
reported). However, **101 of its 103 subjects appear in more than one official
split**, and 100 are shared between train and test — the release partitions
clips, not people. Re-splitting by whole subject, everything else identical,
costs **quadratic weighted kappa 0.537 → 0.317** and average accuracy 0.601 →
0.435. This is not a criticism of the dataset, which states its protocol; it is
a measurement of what that protocol is worth, and independent corroboration of
this project's own leak-free design.

---

## 6. Discussion

### 6.1 What the evidence supports

The system's demonstrated contribution is a leak-free, human-audited pipeline
for detecting, **attributing** and temporally aggregating observable behavioural
cues in a computer laboratory — not a claim that internal attention can be read
from video.

Within that, three results are supported at the level of statistical control
this thesis applies. Attribution quality drives grounding quality: +0.1724 R@1
from the binding alone, with detection held constant. Temporal architecture
dominates feature engineering for cue classification: +0.0933 test macro-F1 from
MS-TCN over the transformer, against +0.0296 for the only feature block that
survives isolation. And boundary-aware modelling substantially reduces
over-segmentation, closing roughly half the edit-score gap to the pseudo-label
teacher.

### 6.2 What the evidence overturns

Two claims made earlier in the project do not survive controlled measurement,
and are reported here because a thesis that only reports its successes cannot be
checked.

The 570-dimensional configuration was described as the best model on the basis
of a distributed metric computed per shard. Measured in one process, it is
statistically tied with the 556-dimensional model and nominally lower, and on
the test split the extra fourteen dimensions are worth −0.0167 macro-F1.

The improvement in `turned_to_peer` was attributed to the personalised
gaze-deviation feature "doing exactly what it was designed for". Under an
isolated ablation with three seeds, that block's effect on `turned_to_peer` is
**−0.0120, CI [−0.050, +0.032], significant in 0 of 3 seeds**. The original
observation was a single seed on a different split measured with shard-averaged
metrics. The feature families remain implemented, and the thesis can state that
the indicator channels named in the proposal exist and were measured; it cannot
state that they improved accuracy.

### 6.3 A recurring failure mode

Four separate defects in this project shared one signature: they produced
plausible numbers. A distributed trainer averaging metrics per shard; an
evaluator returning one label per sequence where the caller expected one per
frame; features extracted from a pre-cropped image so sixteen dimensions went
constant; and a deployment path that loaded a mismatched checkpoint, printed one
line about it and served predictions from a randomly initialised network. None
crashed. Each was found by comparing an expected relationship against a measured
one, not by inspection.

The methodological response adopted here — one evaluator, stored prediction
archives, interval estimates before claims, tests that assert the property a bug
would violate, and a written register recording which numbers may not be cited
and why — is itself a contribution of this work, and is what makes the results
above checkable.

---

## 7. Limitations

**Labels.** Every training and test label is a pseudo-label from a
vision–language model. The test split measures generalisation to unseen
*videos*, not agreement with human judgement. Measured against human annotation,
the teacher agrees on 87.4% of frames and recovers 0.625 of episodes; it is an
**empirical benchmark, not a ceiling**, and the exact model release is not
recorded in the pipeline outputs — a reproducibility limitation stated rather
than papered over.

**Human-referenced evaluation is small and selected.** Sixteen distinct episodes
over ten tracks, drawn from two densely annotated segments. `looking_away` has
**no** gold episodes at all, so the class where the architecture change gained
most is unmeasured at the episode level.

**Setting.** The cue-to-attention mapping is calibrated to one computer
laboratory. The DIPSER reversal and the degradation under zero-shot transfer to
a lecture-hall corpus are two independent indications that it should not be
presented as a general classroom attention model.

**Scope of the event measurement.** Episode metrics exclude the detector and
tracker by design, isolating the cue and event layers; end-to-end episode
performance including detection and tracking errors is not measured.

**Runtime.** 5.77 FPS at ~6 students on one A100, with every frame exceeding a
10 fps budget and bimodal latency under striding. Near-real-time.

**What is not claimed.** Boredom, perplexity or curiosity detection. Any
inference of a student's internal state from a visible cue. Comparability
between this system's macro-F1 and published engagement accuracies, behaviour
detection mAP, or engagement regression error — those answer different questions
about different tasks and are reported in separate tables for that reason.
