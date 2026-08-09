# Branch C evaluation protocol — frozen

**Status:** FROZEN 2026-08-09. **Protocol version:** 1.0.0
**Frozen at commit:** see `git log` for the commit that introduced this file on `branch-c/overt-cue`, base `ce1add2`.
**Fold manifest:** `outputs/branch_c/splits/branch_c_folds.json`, `manifest_sha256 = fd913e7c5a52839ee5115224ffdb741c09cc1104eeb1fee5d32b9595e2388314`
**Producer:** `python tools/branch_c/build_folds.py` (deterministic; re-running reproduces the hash)
**Leakage tests:** `LLMDet/attention/tests/test_branch_c_splits.py` — 8 tests, all passing

This document is written **before any Branch-C model reads a held-out result**. It
is immutable in the sense that matters: any change to it after the first outer
fold is opened must be recorded as a dated amendment at the bottom, with the
reason, and every number produced under the old version stays in the record. It
is not a document to be quietly edited into agreement with the results.

---

## 1. Why this protocol exists in this form

### 1.1 There is no external holdout, and none can be manufactured

The specification's preferred design is to freeze genuinely unused
videos/subjects/sessions as an external holdout. That option does not exist here.

| source | count | evidence |
|---|---|---|
| videos with frames on disk | 128 | `grounding_data/llmstu_tools/outputs/frame_to_video.json`, 104,069 frames |
| videos in the Branch-A/B split | 127 | `grounding_data/llmstu_tools/outputs/splits.json` (73/27/27) |
| videos in neither | **1** | `video_0162_0_10_20251026030716_20251026032847`, **4 frames** |

Four frames is not a test set. The specification's fallback therefore applies and
is mandatory, not discretionary: **nested grouped cross-validation**.

### 1.2 The old Branch-B test split is spent

`BRANCH_B_TEST_PROTOCOL.md` registered a single permitted opening of the 27-video
test split, and FINDINGS §11.9 reports it. That number is real and stays in the
record. It cannot become an untouched test for a new architecture, and no
Branch-C selection decision may consult it.

### 1.3 The grouping unit is `video_id`, and the other two candidates are unavailable

- **Camera** grouping is *vacuous*. The entire corpus is one fixed corner-mounted
  oblique camera in one room (FINDINGS §12.1). There is no second viewpoint to
  hold out, so no claim about cross-camera generalisation may be made.
- **Subject** grouping is *unrecoverable*. Students are not identified across
  recordings; `seat_id` indexes a chair, not a person. The same student almost
  certainly appears in multiple recordings of the same course, and nothing in the
  data lets us detect it.

`video_id` is therefore the only defensible grouping unit, and **subject leakage
across folds cannot be excluded**. This is a limitation of the corpus, it is
stated here, it will be stated in the thesis, and it will not be discovered later
and quietly omitted.

---

## 2. The registered design

### 2.1 Structure

Nested grouped cross-validation over all 127 videos:

- **Outer:** 5 folds, whole videos, prevalence- and support-balanced. Purpose:
  estimate generalisation. Fold sizes 25–26 videos, 1,284–1,333 sequences.
- **Inner:** within each outer-fold's 101–102 remaining videos, a single
  prevalence-balanced grouped holdout of 20 videos. Purpose: **all** selection —
  architecture, loss coefficients, thresholds, calibration, early stopping,
  checkpoint choice.

A single inner holdout rather than inner k-fold is a deliberate compute
trade-off: the screening stage evaluates on the order of a hundred
configurations, and an inner k-fold multiplies that by k for a decision a
20-video grouped holdout already resolves. The consequence — a slightly noisier
selection signal — is accepted and recorded here rather than hidden.

### 2.2 Realised balance

| fold | test videos | test sequences | per-class sequence support (6 cues) |
|---|---|---|---|
| 0 | 26 | 1308 | 1096 / 40 / 46 / 13 / 53 / 60 |
| 1 | 26 | 1302 | 1092 / 40 / 45 / 13 / 53 / 59 |
| 2 | 25 | 1333 | 1116 / 41 / 47 / 13 / 54 / 62 |
| 3 | 25 | 1284 | 1076 / 40 / 45 / 12 / 52 / 59 |
| 4 | 25 | 1304 | 1094 / 40 / 46 / 12 / 53 / 59 |

Order: `screen_oriented, looking_away, head_down, turned_to_peer, phone_use, uncertain`.

**`turned_to_peer` has 63 sequences in the entire corpus** — 12–13 per fold. Any
per-fold `turned_to_peer` F1 is a near-meaningless statistic and will be reported
only pooled across outer folds, with its support printed beside it. The class is
retained because dropping a class after seeing that it is hard is exactly the
kind of after-the-fact metric change this protocol exists to prevent.

### 2.3 Seeds

Fixed seeds **42, 43, 44** for every shortlisted model, matching Branch B so that
seed-matched paired comparisons remain possible. Extended to 42–46 for the final
two configurations only if wall-clock permits; the decision to extend must be
made before the results of 45/46 are read, and applies to *both* configurations
or neither.

### 2.4 What may touch what

| stage | may read | must never read |
|---|---|---|
| feature/cache generation | all videos (unsupervised; no labels) | — |
| architecture + hyper-parameter screening | inner train + inner val | outer test, legacy Branch-B test, human-gold event set |
| loss coefficient selection | inner train + inner val | outer test |
| calibration (temperature) and event thresholds | **inner val only** | outer test, inner train |
| final scoring | outer test, once, after freeze | — |

Feature caches are built over all videos because they are label-free
transformations of pixels. This is recorded explicitly because it *looks* like
leakage and is not: no label, no fold assignment, and no fitted statistic crosses
a fold boundary. Any cache step that fits a normalisation statistic must fit it
on inner-train only; a test asserts this.

**Outer-test access is opened once, by the lead, after the shortlist is frozen.**
Implementation agents do not have access before that point. Re-opening a fold to
try a variant invalidates the estimate and must be recorded as such.

---

## 3. Metrics, registered before the experiments

### 3.1 Primary

**Six-cue macro-F1 over outer-fold predictions pooled before the metric is
computed.** One number. Predictions from all five outer folds are concatenated,
then macro-F1 is computed once — not averaged over per-fold macro-F1, which
would weight a 25-video fold's rare-class noise equally with its signal.

### 3.2 Secondary — all reported, none promotable to primary after the fact

- balanced accuracy; per-class F1 with support
- NLL, Brier, ECE — **both raw and temperature-calibrated**
- risk–coverage curve and AURC; the registered display operating point
- segmental F1 @ 10/25/50, segmental edit score
- event precision / recall / F1, onset error, duration error, false alerts per hour
- alert coverage (FINDINGS §11.21 shows it does *not* rank like macro-F1 — both are reported)

### 3.3 Detector / tracker metrics

mAP and recall for detection; HOTA, IDF1, identity switches, fragmentation, track
coverage for tracking — **only where annotations support them**. This corpus has
no tracking ground truth, so tracker comparisons are **proxy metrics** and will be
labelled as proxies in every table. A tracker that "wins" on a proxy has not been
shown to win.

### 3.4 Reporting rules

- Every pseudo-label metric is labelled **teacher agreement**, never accuracy.
- Retrieval R@1, detection mAP, cue macro-F1, ordinal engagement accuracy and
  event recall never appear in one undifferentiated leaderboard.
- Branch-C numbers and legacy Branch-B numbers never share a table. The legacy
  test numbers come from a different partition and are **not comparable**;
  they appear only in a clearly separated retrospective section.
- Uncertainty: seed mean ± sd, *and* paired per-video cluster bootstrap CI on
  differences. Reuse `attention.thesis_eval.bootstrap` — do not reimplement.

### 3.5 Multiple comparisons

The confirmatory family is the registered arm list in §4. Within it, paired
comparisons against the primary baseline are corrected by **Holm–Bonferroni** at
family-wise α = 0.05. Screening-stage comparisons are exploratory, uncorrected,
and may never be reported as confirmatory findings.

---

## 4. Registered arms

Every arm is trained under the identical protocol above. Arms 1–2 are **retrained
under Branch-C folds**; the published Branch-B numbers (MS-TCN-556 test macro-F1
0.500 ± 0.011, FINDINGS §11.9) were measured on a different partition and are
**not** a valid comparator for a Branch-C number.

| # | arm | tests |
|---|---|---|
| 1 | MS-TCN-553 (`face_found`) — deployment baseline | the deployed system |
| 2 | MS-TCN-556 (full head-pose block) — strongest existing | the primary baseline |
| 3 | OVERT appearance-only (552) | does any of this beat plain appearance |
| 4 | + head quality/presence flags, **no angles** | the control that killed the last pose claim |
| 5 | + absolute head pose (camera frame) | is canonicalisation doing the work |
| 6 | + **seat/task-relative** pose | **the critical arm** |
| 7 | + body/action expert (frozen features) | does action add anything |
| 8 | uniform fusion vs learned reliability fusion | is the gate worth it |
| 9 | learned fusion **without** quality-order/counterfactual losses | are the losses worth it |
| 10 | full model | the headline |
| 11 | full model, each modality dropped at inference | graceful degradation |
| 12 | full model under head blur / occlusion / downsampling / track gaps / box jitter | robustness |
| 13 | scene- vs torso- vs causal-learned reference frame | which R_ref |
| 14 | WHENet vs best full-range head model | coverage, stability, runtime, downstream |
| 15 | ByteTrack vs BoT-SORT vs current tracker, identical detections | tracker choice |
| 16 | LLMDet vs YOLO, identical tracking and evaluation | detector choice |

Arms that fail stay in the results table. A registered variant that is dropped
from the report is a protocol violation.

---

## 5. The critical success condition, and what it is really testing

> **Seat/task-relative pose (arm 6) must outperform BOTH the quality/presence-only
> control (arm 4) and the absolute-pose control (arm 5), on pooled outer folds,
> with a paired per-video bootstrap 95 % CI on the difference that excludes zero.
> If it does not, no pose contribution is claimed — even if the full stack improves.**

This is not an arbitrary hurdle. FINDINGS §11.10 established that ~80 % of the
existing head-pose gain is the binary `face_found` flag, that the metric angles
add +0.0058 with 0/3 seeds significant, and that MediaPipe finds a face on 92 %
of `screen_oriented` crops versus 8 % of `head_down` crops.

That 92-vs-8 contrast is **a missingness artifact of this specific camera
geometry**, not a behavioural signal. In this room students face their monitors
with their backs to the camera, so a face becomes detectable largely *because*
the student has turned away from the task. The existing model's best feature is a
proxy for the label that works only as long as the camera never moves.

Branch C's falsifiable prediction follows directly:

- **H1.** A full-range head-pose estimator (DirectMHP / 6DRepNet360) that succeeds
  on rear- and downward-facing heads will *flatten* the coverage contrast — it will
  destroy `face_found` as a feature. FINDINGS §11.10 predicted exactly this and
  called the upgrade "not a straightforward improvement".
- **H2.** With the shortcut removed, seat-relative pose angles must recover the
  lost signal from real geometry. Camera-frame yaw cannot: the yaw corresponding
  to `screen_oriented` differs systematically between a front-left seat and a
  right-hand seat, so camera-frame pose confounds cue with seat identity.
  Seat-frame pose does not.

H1 and H2 are measured separately. **H1 failing is itself a publishable result**
(the shortcut is robust); **H2 failing after H1 succeeds is the honest negative
result** the specification anticipates.

### 5.1 Mandatory shortcut audit

Independent of the arms above, and reported whatever it shows:

1. Train a classifier on **quality/missingness signals alone** — no appearance, no
   pose angles — and report its cue macro-F1. This upper-bounds how much of any
   model's performance is available from availability patterns.
2. Re-evaluate under a **changed missingness distribution** (simulated: equalise
   face/head visibility across cue classes).
3. Report performance with **label-flipped** and **equalised** face-visibility
   conditions.

A gate that predicts cues from missingness alone is a finding to report, not a
defect to hide.

---

## 6. Causality

Every online feature, reference-frame estimator, temporal model, calibration rule
and event decision uses **past and present frames only**. Specifically:

- `R_ref` is estimated causally, from past high-quality frames of that seat only.
- No feature may use a whole-track statistic computed over future frames. The
  existing handcrafted `dynamic` block (FINDINGS §11.4) is computed over the whole
  track and is therefore **not causal**; it may not be reused as an online feature.
- Any acausal/offline result is reported in a separately labelled table and never
  compared directly with an online one.

A test asserts that shifting the input tail does not change earlier outputs.

---

## 7. Go / no-go gates

### 7.1 Scientific gate — all must hold

1. Pooled outer macro-F1 of the full model exceeds the strongest properly
   comparable baseline (arm 2, retrained under these folds) by **≥ 0.02
   absolute**, with a paired per-video bootstrap 95 % CI excluding zero.
2. Arm 6 beats arms 4 and 5 (§5).
3. Reliability fusion beats uniform fusion under **≥ 2** predefined
   missing/corrupted-modality conditions, without materially degrading clean
   performance ("materially" = a drop whose CI excludes zero).
4. Directionally consistent across all three seeds.
5. Calibration and event fragmentation do not worsen enough to erase the
   frame-level gain: calibrated ECE must not worsen by > 0.01 absolute, and
   segmental edit score must not fall.

If these fail: report the negative result, keep Branch C as an engineering
extension, and use the corresponding thesis wording already drafted in the
specification. Do not change the metric after the fact.

### 7.2 Runtime / demo gate

Measured on **one** A100 at ~6 visible students, against the recorded baseline
envelope (FINDINGS §11.17): ≥ **5.77** processed FPS, p95 latency ≤ ~**300 ms**,
complete track coverage, non-degenerate outputs, working abstention, stage-level
p50/p95/p99 and peak GPU memory reported, strict checkpoint/schema loading.
Preprocessing time is included, not hidden.

### 7.3 Compatibility gate

Legacy Branch-B dashboard replay reproduces its frozen fixture within predefined
tolerances; old configs and checkpoints still load strictly; mode switching leaks
no state; a one-command rollback is documented and tested. **Legacy Branch B
remains the dashboard default until all three gates pass.**

---

## 8. Claim scope

The contribution is narrowed, on evidence, before any experiment (FINDINGS §12.1,
confirmed with the researcher 2026-08-09):

- The word **"overhead" is dropped**. The setting is a *fixed oblique
  corner-mounted classroom camera*.
- **Viewpoint invariance is proved and unit-tested, and probed synthetically
  only.** With one camera pose in the corpus there is no empirical test of
  cross-camera generalisation, and none will be claimed.
- Rotation invariance does not imply homography invariance. Synthetic global
  rotations and projective perturbations are tested and reported **separately**.

Permitted claim wording is fixed in advance by outcome, per the specification's
four cases (full success / invariance only / engineering only / negative). The
wording is selected by the gates in §7, not by preference.

---

## 9. Compute and execution

No cluster scheduler exists on this host (`sinfo`/`squeue`/`sbatch` absent). Jobs
run through `LLMDet/attention/thesis_eval/launch_sweep.py`, extended for Branch C:
independent single-GPU processes dispatched to whichever GPU frees first.

**Concurrency is capped at 4 of the 8 A100s** by standing instruction — the box is
shared and GPUs 0 and 4 already carried other users' processes at audit time.

Two-stage search, per the specification: cheap screening on inner validation with
frozen caches and one seed; then full evaluation of the shortlist across three
seeds and five outer folds. No large unconstrained hyper-parameter sweep.

Every run appends to `outputs/branch_c/RUNS.jsonl`: git commit, dirty-diff hash,
environment lock hash, dataset/split hash, feature-schema hash, checkpoint hash,
command, seed, host/GPU, wall time, peak memory. Atomic writes and completion
sentinels; a partial cache must never appear valid. Failed-run evidence is never
deleted.

---

## 10. Amendments

None. Any amendment must be appended below with date, reason, and the list of
results produced under the previous version.
