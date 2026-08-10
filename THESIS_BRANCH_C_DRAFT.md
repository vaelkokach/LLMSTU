# Branch C — thesis-ready Methods, Protocol, Results, Limitations, Contribution

**Status:** results frozen 2026-08-10. Inner validation only; the five outer folds
have never been opened. Every number here is reproducible from
`outputs/branch_c/` and cites its producing artifact.

**Read alongside:** `BRANCH_C_PROTOCOL.md` (registered before any result),
`FINDINGS.md` §12 (chronological evidence), `outputs/branch_c/ABLATIONS.md`,
`docs/branch_c/NOVELTY_AUDIT.md`, `docs/branch_c/INDEPENDENT_REVIEW_LOG.md`.

---

## 1. Contribution statement

Branch C set out to test whether representing head orientation relative to a
student's seat/task frame, and routing multimodal evidence by measured
observability, improves visible-cue segmentation on this corpus. **It does not.**
Four registered hypotheses returned null. The contribution is therefore a set of
controlled negative results plus one measurement finding and one unregistered
positive:

> We show that the head-orientation channel in the deployed 556-dimensional feature
> block was never measuring head orientation: its angles are uncorrelated
> (|circular r| ≤ 0.035) with a validated full-range estimator, and exist on only
> 63 % of frames. Replacing them with 6DRepNet360 at 100 % coverage — an estimator
> that recovers applied rotation to ~1° and separates `head_down` from
> `screen_oriented` at AUC 0.830 as a single scalar — yields **no** improvement in
> cue macro-F1 (−0.0022, 95 % CI [−0.0119, +0.0085]), because the appearance
> representation already recovers head orientation more accurately (AUROC 0.915)
> than the pose model measures it. Seat-relative canonicalisation of that
> orientation fails its pre-registered gate, and observability-weighted fusion of
> head and motion experts adds +0.0015 (95 % CI [−0.0007, +0.0041]). Separately and
> unexpectedly, a change of training configuration alone — identical architecture,
> fewer inputs — improves the deployed model by **+0.0406 macro-F1**
> (95 % CI [+0.0289, +0.0521]), twice the effect size the study was powered to detect.

This is the specification's fourth wording case, strengthened: the negative is not
attributable to a weak instrument, because the instrument was validated first.

---

## 2. Setting, and a correction to the premise

The work was commissioned for "fixed overhead multi-student video". **The corpus is
not overhead.** Six recordings sampled across four capture dates show one room and
one fixed corner-mounted camera at ~20–30° elevation, students at desks facing
monitors with their backs to the lens (FINDINGS §12.1). The project's own system
diagram had described it correctly as "corner-camera classroom/lab videos"; only the
Branch-C commission called it overhead.

| property | value |
|---|---|
| recordings | 128 (127 in the split; the 128th has 4 frames) |
| cameras | **1** |
| rooms | **1** |
| resolution | 2812×1050, constant |
| students/video | median 7 (2–13) |
| occluded crops | 31.0 % |

Two consequences bind everything downstream:

1. **Viewpoint invariance cannot be validated empirically.** One camera pose exists.
   The relative-rotation invariance is *proved* and unit-tested over 256 random
   `(Q, R_ref, R_head)` triples to 1e-10, and probed with synthetic rotations — but
   no cross-camera claim is made anywhere in this thesis.
2. **Grouping is by video alone.** Camera grouping is vacuous; students are not
   identified across recordings, so subject grouping is unrecoverable and **subject
   leakage across folds cannot be excluded.**

---

## 3. Experimental protocol

Registered in `BRANCH_C_PROTOCOL.md` and committed **before any held-out result was
read** (commit `f511d49`).

**No external holdout exists.** 128 videos on disk, 127 in the split, the remainder
carrying four frames. The specification's fallback is therefore mandatory: nested
grouped cross-validation, 5 outer folds by video, a 20-video prevalence-balanced
inner holdout per fold for every selection decision. Fold manifest
`outputs/branch_c/splits/branch_c_folds.json`, `sha256 fd913e7c…`, deterministic,
write-once, with 8 leakage tests (`test_branch_c_splits.py`).

Folds are balanced to 1,284–1,333 sequences with per-class support within 2× across
folds. `turned_to_peer` has **63 sequences in the entire corpus** (12–13/fold), so
its per-fold F1 is not interpretable and is reported pooled only.

Primary metric: six-cue macro-F1, seeds 42/43/44, paired over 15 matched (fold,
seed) pairs, 20 000-draw bootstrap. Go gate: **+0.02 absolute with a CI excluding
zero.**

**Selection rule.** Two trainers were used and they select differently — the project
trainer on a trailing 5-epoch mean (deliberately, to avoid argmax-on-noise bias),
Branch C's fusion trainer on argmax. All arms were therefore **re-scored under the
project rule** from stored per-epoch histories. The bias is +0.0022–0.0029, and
every number in §4 uses the project rule, making them comparable to Branch B's.

---

## 4. Results

180 runs, 12 arms × 5 folds × 3 seeds, zero training failures.
Raw: `outputs/branch_c/inner_val_results.json`, `outputs/branch_c/eval/*/metrics.json`.

### 4.1 Arms

| arm | features / model | macro-F1 | sd |
|---|---|---|---|
| 0b | 5 pipeline-measured quality signals only | 0.3208 | 0.028 |
| 0 | + 3 teacher-side annotation fields | 0.4602 | 0.023 |
| 1 | appearance 552 + `face_found` (deployed) | 0.4837 | 0.027 |
| 2 | + MediaPipe yaw/pitch/roll | 0.4887 | 0.033 |
| 5 | + 6DRepNet360 full-range rotation | 0.4865 | 0.027 |
| 3b | appearance 552, **Branch-C training config** | 0.5243 | 0.032 |
| 3 | as 3b + deeper stack | 0.5288 | 0.029 |
| 8 | + head/motion experts, uniform fusion | 0.5311 | 0.025 |
| 9c | + learned reliability fusion | 0.5326 | 0.024 |
| 10c | + observability losses | 0.5360 | 0.023 |

### 4.2 Registered contrasts

| contrast | Δ | 95 % CI | wins | verdict |
|---|---|---|---|---|
| pose: arm5 − arm2 | −0.0022 | [−0.0119, +0.0085] | 5/15 | **null** |
| pose vs no angles: arm5 − arm1 | +0.0028 | [−0.0072, +0.0139] | 8/15 | **null** |
| fuse at all: arm8 − arm3 | +0.0019 | [−0.0025, +0.0057] | 11/15 | **null** |
| learned vs uniform: arm9c − arm8 | +0.0015 | [−0.0007, +0.0041] | 10/15 | **null** |
| observability losses: arm10c − arm9c | +0.0041 | [+0.0010, +0.0070] | 11/15 | significant, ⅕ of gate |
| whole fusion stack: arm10c − arm3 | +0.0072 | [+0.0026, +0.0116] | 12/15 | below gate |
| **training config: arm3b − arm1** | **+0.0406** | **[+0.0289, +0.0521]** | **14/15** | **2× gate** |

**Scientific go gate (protocol §7.1): FAILED on every registered clause.**

### 4.3 Why pose fails — measured, not asserted

Full-range rotation is *not* weak. A single scalar (geodesic distance from the
corpus-mean rotation) separates `head_down` from `screen_oriented` at **AUC 0.830**
and `uncertain` at **0.852**, where the legacy MediaPipe yaw gives 0.529 and 0.554.
Coverage rises 63 % → 100 %.

But the appearance model already has that information, and more:

| cue | arm 1 AUROC (**no pose at all**) | standalone pose scalar |
|---|---|---|
| head_down | **0.915** | 0.830 |
| uncertain | **0.930** | 0.852 |
| turned_to_peer | **0.825** | 0.625 |
| phone_use | **0.889** | 0.519 |

A 552-dim CLIP representation of a student crop encodes "this head is down" more
reliably than a dedicated pose network reading the same pixels. Pose is **redundant
with appearance, not absent from it.**

Corroborating: arm 2 and arm 5 agree on **81.9 %** of frames, while two *seeds of
the same arm* agree on **82.7 %** — swapping the estimator changes predictions less
than changing the random seed does (`ERROR_ANALYSIS.md` §2).

### 4.4 The shortcut audit, and a corrected claim

Protocol §5.1 requires a classifier on availability signals alone. Eight such
signals reach 0.4602 — but three (`occluded`, `head_kpts`, `face_kpts`) come from
the same annotation record the cue label is derived from. Restricted to the five
genuinely pipeline-measured signals, the shortcut reaches **0.3208**
(arm0 − arm0b = +0.1395, 15/15).

**An earlier draft of this finding claimed "95 % of performance without looking at
the student". That claim is withdrawn.** The genuine inference-time shortcut is
0.3208, 66 % of the deployed model, and appearance is worth **+0.163**, not +0.023.
The separation was registered before the number was reported, which is why the error
was caught.

### 4.5 A leak, caught by that same audit

Learned reliability fusion initially measured **+0.0902** over uniform fusion
(15/15, CI excluding zero) — implausible for a two-value gate. Zeroing the three
teacher-side quality columns at inference left uniform fusion unchanged (0.5694 →
0.5694) and collapsed learned fusion **0.6566 → 0.2567**. The gate was recovering
the label. Retrained leakage-free, the effect is +0.0015 and null.

`fusion.py` is structurally correct — the gate provably cannot see *content*
features, and a test enforces it. But "not content" is not "not label-bearing", and
three columns of the quality vector were label-bearing. **The API guarantee held;
the data contract behind it did not.** This is the single most transferable
methodological lesson of the branch.

---

## 5. Limitations

1. **Inner validation only.** The five outer folds are unopened. Nothing here is a
   generalisation estimate. They were deliberately preserved: the go gate failed by
   a factor of three on inner validation, and spending an unrepeatable resource to
   confirm a null is poor economy.
2. **Subject leakage cannot be excluded** (§2).
3. **One camera, one room.** No viewpoint-generalisation claim is possible.
4. **All labels are pseudo-labels** from a single teacher. Every metric is *teacher
   agreement*, not human accuracy.
5. **The +0.0406 recipe effect is not attributed.** Six hyper-parameters differ at
   once; the responsible factor is unidentified. Six one-factor runs would settle it.
6. **`turned_to_peer` is under-supported** — 63 sequences corpus-wide.
7. **Robustness arms 11/12/17 were not completed.** They crashed on a
   generator-device bug (fixed, verified) and were not rerun. Given arm9c − arm8 is
   null, the arm-17 permutation diagnostic would confirm a gate already shown to do
   nothing, but it is a registered arm and its absence is recorded rather than
   omitted.

---

## 6. What a reader should take from this

The task, as posed by these pseudo-labels on this corpus, is **substantially
determined by whether a student is cleanly observable**, and appearance carries
almost all of the rest. Head-pose geometry, action evidence and observability-weighted
fusion each add nothing detectable once appearance is present. The practical lever
is not a new modality — it is the training configuration, which is worth twice what
any of the modelling hypotheses were.

Negative evidence of this kind is worth reporting precisely because the naive version
of it is not: "we added head pose and it did not help" is always answerable with
"your estimator was bad". Here the estimator was validated against a known quantity
before use, given complete coverage, and shown strongly discriminative in isolation —
and it still added nothing. The result is about the task and the feature space, not
the instrument.

---

## 7. Recommended next work, in order of expected value

1. **Isolate the +0.0406.** Six one-factor-at-a-time runs (~15 min each on 8 GPUs).
   Dropout 0.1 → 0.5 is the leading hypothesis on priors. This improves the
   *deployed* model and is the highest-value hour available.
2. **Complete arms 11/12/17** (~13 min) to close the registered set.
3. **Second annotator** on the gold set — still the largest defence exposure
   (FINDINGS §9 item 13), and unaffected by anything here.
4. Do **not** pursue head pose further on this corpus without changing the labels or
   the camera. The redundancy result in §4.3 is the reason.
