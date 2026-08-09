# Branch C architecture

**Status:** the pose-canonicalisation component described in §4 was **tested and
dropped** by its pre-registered gate (FINDINGS §12.15). This document records what
was built, what the evidence says about each piece, and what survives. It is not a
proposal.

Read with: `BRANCH_C_PROTOCOL.md` (frozen protocol and gates),
`docs/branch_c/NOVELTY_AUDIT.md` (what is and is not new),
`FINDINGS.md` §12 (the chronological evidence).

---

## 1. What the corpus actually is

Everything below follows from this, and getting it wrong is how the driving
specification came to be named for a viewpoint the data does not have.

| property | value | evidence |
|---|---|---|
| camera | **one** fixed, corner-mounted, ~20–30° elevation | FINDINGS §12.1, six recordings across four dates |
| room | **one** computer lab | same |
| recordings | 128 (127 in the split, 1 with 4 frames) | `frame_to_video.json`, `splits.json` |
| resolution | 2812×1050, constant | `DATA_INVENTORY.md` |
| subjects | seated at desks, facing monitors, **backs to the lens** | §12.1 |
| students/video | median 7 (2–13) | `DATA_INVENTORY.md` |
| occlusion | 31.0% of crops | audit §5.2 |

Two consequences bind the whole design:

1. **Viewpoint invariance cannot be validated.** One camera pose exists. The
   invariance in §4 is proved and unit-tested; it is probed with synthetic
   rotations only, and no cross-camera claim is made anywhere.
2. **Grouping is by video alone.** Camera grouping is vacuous; subject grouping is
   unrecoverable because students are not identified across recordings. Subject
   leakage across folds therefore cannot be excluded, and the protocol says so.

---

## 2. The problem the branch was built to attack

FINDINGS §11.10 established that ~80% of the existing head-pose gain is the binary
`face_found` flag, and the three metric angles add **+0.0058 macro-F1, 0/3 seeds
significant**.

Branch C's premise was that this is a *measurement* failure, not a fact about
head orientation. Two hypotheses, registered before any training:

- **H1** — a full-range estimator flattens the coverage contrast and recovers real
  geometric signal that MediaPipe never saw.
- **H2** — with the missingness shortcut gone, orientation must be expressed
  relative to the student's own seat/task frame, because camera-frame yaw confounds
  cue with seat.

**H1 confirmed. H2 rejected.** §6 gives the numbers.

---

## 3. Stage 1–2: detection and tracking — deliberately not rebuilt

The specification proposed swapping in YOLO11 and BoT-SORT. Both were rejected
before implementation, on evidence rather than preference:

- **Ultralytics YOLOv8/YOLO11 is AGPL-3.0, weights included.** Vendoring it imposes
  network-disclosure obligations on the entire combined work. `LICENSE_AUDIT.md`.
- **The corpus has no tracking ground truth.** Any tracker comparison would be a
  proxy metric, and a tracker that "wins" on a proxy has not been shown to win.
- The camera never moves, so BoT-SORT's camera-motion compensation is inert here.

The existing LLMDet detector and `IoUTracker` are retained unchanged. This is a
rejected-option record, not an omission.

---

## 4. Stage 4: pose canonicalisation — built, proved, and dropped

`LLMDet/attention/branch_c/canonical.py`.

### 4.1 The mathematics

With `R_head` the head rotation in camera coordinates and `R_ref` a causal
reference in the same coordinates:

```
R_rel = R_refᵀ · R_head
```

Under a global change of camera coordinates by `Q ∈ SO(3)`, both rotate together:

```
R_rel′ = (Q·R_ref)ᵀ (Q·R_head) = R_refᵀ Qᵀ Q R_head = R_rel
```

**This is true and it is tested** — `test_relative_rotation_is_invariant_to_global_camera_rotation`
over 256 random `(Q, R_ref, R_head)` triples to 1e-10, extended through the whole
feature assembler by `test_invariance_holds_for_the_derived_features_too`.

It is also **deliberately bounded**. `test_rotation_invariance_does_not_imply_homography_invariance`
asserts that a projective perturbation *does* change `R_rel`. That test fails if
anyone later reads the invariance as robustness to a general camera change.

### 4.2 Representation choices, and why

- **6D continuous rotation** (Zhou et al.) and **SO(3) log**, never Euler angles
  internally. Euler is discontinuous as a regression target and has a wraparound
  seam.
- **Geodesic differences, never angle subtraction.** 179° → −179° is a 2° turn.
  `test_yaw_wraparound_is_a_small_angle_not_a_large_one` pins this.
- **Backward differences only** for velocity and acceleration, so frame *t* uses no
  frame after *t*.

### 4.3 Three causal reference estimators

| estimator | source | causality | circularity risk |
|---|---|---|---|
| `SceneReferenceEstimator` | per-seat monitor direction, fixed | constant in time | none |
| `TorsoReferenceEstimator` | the student's own torso | frame *t* only | none |
| `CausalSeatReferenceEstimator` | running tangent-space mean of that seat's past | reads state **before** folding in frame *t* | **real** — documented, and the reason arm 13 existed |

Causality is enforced structurally and tested structurally:
`test_causal_seat_reference_never_uses_the_current_or_future_frame` runs the
estimator over 40 frames and over an 18-frame prefix and requires the overlap to be
identical.

### 4.4 Why this component is dropped

The registered gate (protocol amendment A1) measured whether canonicalisation
improves cue separation over raw rotation. **Mean AUC delta over the face-visible
contrasts: +0.0083.** Null.

The mechanism is visible in the data and worth recording, because it is the reason
a plausible idea failed:

- The **causal running reference** made separation *worse on all four contrasts*
  (e.g. `looking_away` 0.613 → 0.582). A reference averaged over a track's own past
  absorbs the signal, because most of that past is `screen_oriented`. This is
  exactly the circularity §4.3 flags, showing up as measurement.
- The **scene reference** is null because seat ID does not identify a stable
  monitor direction here: per-seat mean yaw spans 5.5–20.3° across seats while the
  per-video spread *within* one seat is 5–15°. The 42% of yaw variance that seat
  explains is mostly per-person idiosyncrasy, not seat geometry.

The code, proof and tests remain in the repository as the evidence for a reported
negative result.

---

## 5. Stage 3: the head expert — the part that worked

`LLMDet/attention/branch_c/head_pose_fullrange.py`.

**6DRepNet360**, ResNet-50 backbone with a 6D rotation head, MIT-licensed.
DirectMHP was rejected: GPL-3.0 via YOLOv5.

Three things were checked rather than assumed, each because this project has
previously been burned by the cheaper alternative:

1. **Architecture read off the checkpoint**, not from the model's name — 320
   tensors, bottleneck [3,4,6,3], `linear_reg (6, 2048)`. Loads `strict=True`.
2. **The 6D→matrix convention checked numerically** against upstream's
   `compute_rotation_matrix_from_ortho6d`: max deviation 3.5e-6 over 512 draws.
3. **The estimator validated convention-free**: rotate the input in-plane by a
   known angle, measure the geodesic between predicted rotations. Recovers
   9.7°/19.3°/39.6° for applied 10/20/40°, IQR 2–7.

**Head localisation.** The corpus stores *person* crops; a person crop is not a head
crop. `head_span_px` cannot bridge it — floored at exactly 120 px, median 0.43× the
person-box height. A documented geometric crop is used instead (top 42% of the
person box, squared), justified by the rotation-recovery test above rather than by
agreement with MediaPipe — because §12.14 shows MediaPipe cannot serve as a
reference here.

---

## 6. What the evidence says

### 6.1 The cached MediaPipe angles carry almost no orientation signal

On 800 development crops where MediaPipe *itself* reports a face, the maximum
absolute circular correlation between the two estimators, over all nine axis
pairings in both conventions, is **0.035**. One of the two is validated against a
known quantity (§5); the other has independently been measured as contributing
nothing (§11.10).

This does not retract §11.10. It changes what its null *means*: not "head
orientation does not help", but "head orientation was never measured".

### 6.2 Full-range rotation is strongly class-dependent

Development data, 184,304 frames. Coverage 63% → **100%**.

| cue | mean geodesic from corpus mean | AUC vs `screen_oriented` (raw geodesic) | AUC (MediaPipe yaw) |
|---|---|---|---|
| `head_down` | 78.0° | **0.830** | 0.529 |
| `uncertain` | 74.8° | **0.852** | 0.554 |
| `turned_to_peer` | 41.3° | 0.625 | 0.513 |
| `looking_away` | 29.2° | 0.524 | 0.612 |
| `phone_use` | 29.6° | 0.519 | 0.510 |
| `screen_oriented` | 29.4° | — | — |

A single scalar separates `head_down` and `uncertain` from `screen_oriented` at
0.83/0.85, where the legacy angles do nothing.

### 6.3 The controlled arm comparison

Arms 2 and 5 are byte-identical in 553 of 556 columns, with `face_found` changed on
**0.0%** of frames (`patch_pose_columns` verified). Any difference between them is
attributable to how head orientation was measured, and to nothing else. Inner-
validation results: `outputs/branch_c/inner_val_results.json`.

---

## 7. Stage 5–6: fusion and losses — implemented, not yet earned

`fusion.py`, `losses.py`. Implemented to protocol, unit-tested (31 tests), and
**not claimed**. The novelty audit found the reliability-gating pattern anticipated
by Multi-QuAD and PRIME, and the quality-ordering loss essentially reproduced by
RAC (arXiv:2605.16999) — so C3 is cited, not claimed.

Arm 17, the reliability-permutation diagnostic (protocol §4.1), is the registered
falsification test: permute `r_m` with model and inputs fixed, and if pooled
macro-F1 does not drop, the gate is decorative and no fusion contribution is
claimed. It was added *because* the literature (arXiv:2606.26473) shows this design
pattern frequently does nothing measurable.

Two defects the tests caught before any of this ran:

- `matrix_to_rot6d` broke its own round trip (row-major flatten interleaved the two
  columns `rot6d_to_matrix` expects contiguously). Silent — it would have surfaced
  as a pose feature that meant nothing.
- A single missing expert turned the whole fused output to NaN (`0 * NaN` is NaN).
  One missing head-pose estimate would have poisoned every cue logit on that track.

---

## 8. Honest contribution statement

Per the specification's wording cases, and by the gates rather than by preference:

> We contribute an observability-aware multimodal extension for fixed oblique
> classroom video. We do **not** claim that seat/task-relative pose canonicalisation
> is responsible for any gain: its pre-registered gate returned a null result and
> the component was dropped. We do show that the head-orientation signal the
> existing system was measured to lack was a **measurement** artifact — a
> face-detection-gated estimator with 63% coverage whose angles are uncorrelated
> with a validated full-range estimator — and that replacing it recovers a large,
> previously invisible geometric signal.

The mathematical invariance property is proved and unit-tested. It is not
empirically validated across viewpoints, because this corpus contains one.
