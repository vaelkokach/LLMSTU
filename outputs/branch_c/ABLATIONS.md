# Branch C ablations — controlled comparison and paired analysis (inner validation)

**Generated:** 2026-08-09 · **Evaluator:** `thesis_eval/1.0.0` · Companion: `outputs/branch_c/RESULTS_REGISTER.md`,
`outputs/branch_c/RESULTS_REGISTER.json`

**INNER VALIDATION ONLY. The outer folds have not been opened.** Every number below is independently recomputed
from the 45 raw `outputs/branch_c/eval/<eid>/metrics.json` files (46 directories exist; `eval/arm5_f0_s42` is a
superseded duplicate of `eval/arm5_mstcn_556_fr_f0_s42` and is excluded — see `RESULTS_REGISTER.md`
"Data-integrity findings"). Not comparable to Branch-B test numbers (different partition, different protocol).

**Method.** 15 matched (fold, seed) pairs per contrast (5 outer-fold-specific inner-val splits × seeds
42/43/44). Mean paired difference = mean over the 15 pairs of `metric(arm_hi) − metric(arm_lo)`. 95% CI:
non-parametric bootstrap, resampling the 15 matched pairs with replacement, **20,000 draws, seed 0**, percentile
interval on the bootstrap distribution of the mean. Two-sided p: `2·min(P(boot_mean≤0), P(boot_mean≥0))`, capped
at 1. Wins = number of the 15 pairs where the higher-numbered arm's metric exceeds the lower-numbered arm's
(ties: 0). Holm–Bonferroni applied across the full 33-comparison confirmatory family defined below.

## What is the "confirmatory family" here

Protocol §3.5 defines the confirmatory family as the registered arm list in §4, corrected against the primary
baseline (arm 2). In practice, after the amendment-A1 gate (FINDINGS §12.15) dropped arms 6/13 before training
and no fusion arm (7–13) was ever built, **the only three arms with any trained-and-evaluated result are 1, 2 and
5**, so all three pairwise contrasts among them are reported and treated as one 33-member confirmatory family:
3 contrasts × 11 metrics (macro-F1, balanced accuracy, NLL, Brier, ECE, 6 per-class F1). This is a deliberately
conservative reading — correcting across all 33 rather than only the arm-vs-primary-baseline subset — stated
explicitly so the correction is not mistaken for a narrower, more lenient one.

---

## Controlled comparison table (mean over 15 runs each; also in `RESULTS_REGISTER.md`)

| arm | features | macro-F1 | sd | balanced acc. | NLL (raw) | Brier (raw) | ECE (raw) |
|---|---|---|---|---|---|---|---|
| 1 | base + `face_found`, no angles | 0.4837 | 0.0270 | 0.5208 | 0.7653 | 0.3867 | 0.0471 |
| 2 | + MediaPipe yaw/pitch/roll | 0.4888 | 0.0335 | 0.5227 | 0.7402 | 0.3742 | 0.0349 |
| 5 | + 6DRepNet360 full-range rotation | 0.4865 | 0.0274 | 0.5258 | 0.7613 | 0.3876 | 0.0380 |

## Paired contrasts — macro-F1, balanced accuracy, calibration

| contrast | metric | mean Δ | bootstrap 95% CI | raw p | Holm-corrected p | wins | sig (raw) | sig (Holm) |
|---|---|---|---|---|---|---|---|---|
| arm2 − arm1 | macro_f1 | +0.0051 | [−0.0086, +0.0183] | 0.459 | 1.000 | 8/15 | no | no |
| arm5 − arm1 | macro_f1 | +0.0028 | [−0.0070, +0.0139] | 0.626 | 1.000 | 7/15 | no | no |
| **arm5 − arm2** | **macro_f1** | **−0.0023** | **[−0.0123, +0.0084]** | 0.643 | 1.000 | 5/15 | no | no |
| arm2 − arm1 | balanced_accuracy | +0.0019 | [−0.0260, +0.0301] | 0.882 | 1.000 | 7/15 | no | no |
| arm5 − arm1 | balanced_accuracy | +0.0050 | [−0.0187, +0.0303] | 0.691 | 1.000 | 7/15 | no | no |
| arm5 − arm2 | balanced_accuracy | +0.0031 | [−0.0213, +0.0302] | 0.850 | 1.000 | 7/15 | no | no |
| arm2 − arm1 | nll (raw) | −0.0251 | [−0.0697, +0.0159] | 0.249 | 1.000 | 6/15 | no | no |
| arm5 − arm1 | nll (raw) | −0.0041 | [−0.0482, +0.0420] | 0.854 | 1.000 | 7/15 | no | no |
| arm5 − arm2 | nll (raw) | **+0.0210** | [−0.0343, +0.0784] | 0.469 | 1.000 | 8/15 | no | no |
| arm2 − arm1 | brier (raw) | −0.0125 | [−0.0397, +0.0122] | 0.351 | 1.000 | 7/15 | no | no |
| arm5 − arm1 | brier (raw) | +0.0009 | [−0.0273, +0.0307] | 0.956 | 1.000 | 7/15 | no | no |
| arm5 − arm2 | brier (raw) | **+0.0134** | [−0.0196, +0.0477] | 0.444 | 1.000 | 8/15 | no | no |
| arm2 − arm1 | ece (raw) | −0.0122 | [−0.0242, −0.0006] | **0.039** | 1.000 | 6/15 | **yes** | no |
| arm5 − arm1 | ece (raw) | −0.0091 | [−0.0212, +0.0050] | 0.197 | 1.000 | 4/15 | no | no |
| arm5 − arm2 | ece (raw) | **+0.0031** | [−0.0084, +0.0142] | 0.591 | 1.000 | 8/15 | no | no |

The single row that is nominally significant before correction — `arm2 − arm1` ECE, raw p = 0.039 — **does not
survive Holm–Bonferroni across the 33-member family** (adjusted p = 1.000, since the smallest observed p in the
family, 0.039, must clear 0.05/33 ≈ 0.0015 to remain significant at step 1). No comparison in this ablation is
significant after correction. This is the only raw-significant cell in the entire family; reported precisely so
it is not lost, and precisely so its non-survival under correction is not lost either.

## Paired contrasts — per-class F1

| cue | contrast | mean Δ | bootstrap 95% CI | raw p | Holm p | wins |
|---|---|---|---|---|---|---|
| screen_oriented | arm2−arm1 | +0.0017 | [−0.0115, +0.0170] | 0.870 | 1.000 | 8/15 |
| screen_oriented | arm5−arm1 | −0.0056 | [−0.0247, +0.0122] | 0.565 | 1.000 | 8/15 |
| screen_oriented | arm5−arm2 | −0.0073 | [−0.0290, +0.0133] | 0.514 | 1.000 | 7/15 |
| looking_away | arm2−arm1 | +0.0096 | [−0.0222, +0.0348] | 0.489 | 1.000 | 10/15 |
| looking_away | arm5−arm1 | −0.0016 | [−0.0184, +0.0155] | 0.854 | 1.000 | 7/15 |
| looking_away | arm5−arm2 | −0.0112 | [−0.0311, +0.0139] | 0.336 | 1.000 | 3/15 |
| head_down | arm2−arm1 | +0.0026 | [−0.0155, +0.0232] | 0.824 | 1.000 | 6/15 |
| head_down | arm5−arm1 | +0.0048 | [−0.0204, +0.0288] | 0.683 | 1.000 | 9/15 |
| head_down | arm5−arm2 | +0.0022 | [−0.0200, +0.0221] | 0.809 | 1.000 | 9/15 |
| turned_to_peer | arm2−arm1 | +0.0061 | [−0.0265, +0.0394] | 0.714 | 1.000 | 7/15 |
| turned_to_peer | arm5−arm1 | +0.0087 | [−0.0077, +0.0280] | 0.335 | 1.000 | 7/15 |
| turned_to_peer | arm5−arm2 | +0.0026 | [−0.0285, +0.0346] | 0.874 | 1.000 | 7/15 |
| phone_use | arm2−arm1 | +0.0137 | [−0.0164, +0.0484] | 0.424 | 1.000 | 7/15 |
| phone_use | arm5−arm1 | +0.0174 | [−0.0133, +0.0502] | 0.282 | 1.000 | 8/15 |
| phone_use | arm5−arm2 | +0.0037 | [−0.0120, +0.0205] | 0.647 | 1.000 | 8/15 |
| uncertain | arm2−arm1 | −0.0030 | [−0.0157, +0.0098] | 0.652 | 1.000 | 7/15 |
| uncertain | arm5−arm1 | −0.0070 | [−0.0203, +0.0066] | 0.320 | 1.000 | 7/15 |
| uncertain | arm5−arm2 | −0.0040 | [−0.0190, +0.0133] | 0.612 | 1.000 | 4/15 |

**Every one of these 18 per-class CIs includes zero.** Confirms claim (b) exactly (see `RESULTS_REGISTER.md`
Verification). `turned_to_peer` (support 5,409 pooled frames, 63 sequences corpus-wide) has the widest CIs
relative to its effect size, consistent with protocol §2.2's warning that it is "a near-meaningless statistic"
per fold — retained per protocol §4's rule that failing/marginal classes stay in the table rather than being
dropped after the fact.

---

## Arms registered but not run (protocol §4)

| # | arm | status | reason |
|---|---|---|---|
| 3 | OVERT appearance-only (552) | **not reached** | Screening never proceeded past the A1 pose gate; no appearance-only ablation was trained under Branch-C folds. |
| 4 | + head quality/presence flags, no angles | **run, as arm1** | Protocol note: "Also serves as protocol arm 4 (presence-only control)." Reported throughout as `arm1_mstcn_553_ff`. |
| 6 | + seat/task-relative pose | **gate-dropped** | FINDINGS §12.15 (protocol amendment A1): the seat-relative canonicalisation pre-test on full-range rotation gave a null mean AUC delta (+0.0083) over the four face-visible-class contrasts, the registered stopping condition. Not trained. |
| 7 | + body/action expert | **not reached** | Depends on a positive result from arms 3–6, which did not occur. |
| 8 | uniform vs learned reliability fusion | **not reached** | No fusion model exists to compare. |
| 9 | learned fusion w/o quality-order losses | **not reached** | Depends on arm 8. |
| 10 | full model | **not reached** | This is the model the §7.1 gate is written against; it was never built because arm 5 (its pose-measurement component) failed the A1 gate before the fusion stage was scheduled. |
| 11 | full model, modality-dropped | **not reached** | Depends on arm 10. |
| 12 | full model, robustness perturbations | **not reached** | Depends on arm 10. |
| 13 | scene/torso/causal reference frame | **gate-dropped** | Dropped alongside arm 6 by the same A1 gate; FINDINGS §12.15 trains arm 5 anyway as the necessary control for judging what arm 6/13 would have been controlling for (a judgement call documented and flagged for independent review in FINDINGS §12.15, "A protocol reading that must not be made silently"). |
| 14 | WHENet vs full-range head model | **not reached** | Superseded — 6DRepNet360 was the only full-range backend built (FINDINGS §12.14); no second full-range backend exists to compare. |
| 15 | tracker choice (ByteTrack/BoT-SORT/current) | **not reached** | Not part of the pose-measurement question the branch stopped on. |
| 16 | detector choice (LLMDet vs YOLO) | **not reached** | Same. |
| 17 | reliability-permutation diagnostic | **not reached** | Requires a trained reliability-gated fusion model (arm 8/9), which does not exist. |
| — | `arm0_mstcn_quality` (protocol §5.1 mandatory shortcut audit) | **started, not completed** | Not a numbered arm in §4's table but separately mandated by §5.1. `outputs/branch_c/arm0_sweep.log` dispatched 15 runs (5 folds × 3 seeds); only fold 0's 3 seeds produced any artifact, and those three stop mid-training at epoch ~20/90 with no evaluation. See `RESULTS_REGISTER.md` for detail. This is an open protocol-compliance gap, listed here rather than omitted. |

Arms 1, 2 and 5 are the only ones with a trained-and-evaluated result; they are reported in full above. Every
other registered arm is listed here with its specific reason, per protocol §4's requirement that a dropped
variant stay in the table rather than disappear from the report.

---

## Go / no-go gate outcome (protocol §7.1)

**NO-GO.**

Protocol §7.1's five conditions are written against a "full model" (arm 10) that was never built, because its
pose-measurement component (arm 5) failed the upstream amendment-A1 gate first. Strictly, most of §7.1 cannot be
evaluated — there is no full model, no fusion arm, and no outer-fold estimate. What **can** be evaluated, and is
the closest available proxy for §7.1 condition 1 (does the upgrade beat the strongest comparable baseline by
≥0.02 absolute with a CI excluding zero), is the arm5-vs-arm2 contrast — replacing the near-noise legacy
MediaPipe angles with a validated, 100%-coverage, full-range pose estimator, holding everything else fixed:

- **Required:** ≥ +0.02 absolute macro-F1, paired bootstrap 95% CI excluding zero.
- **Observed (this audit, independently recomputed):** mean Δ = **−0.0023**, CI **[−0.0123, +0.0084]**, 5/15
  wins, raw p = 0.643, Holm-corrected p = 1.000.
- **Verdict: fails, by roughly an order of magnitude, in the wrong direction.**

Condition 4 (directional consistency across seeds) also fails under a stricter, less fold-noisy estimator: pooling
predictions across the 5 folds before scoring each seed (summing confusion matrices, then computing macro-F1
once per seed) gives arm2 = [0.5001, 0.4905, 0.4891] vs arm5 = [0.4924, 0.4917, 0.4832] for seeds
[42, 43, 44] — arm5 trails arm2 in 2 of 3 seeds. Conditions 2 and 3 are not assessable (arm 6 was gate-dropped
before training; no fusion arm exists). Condition 5 (calibration must not worsen ECE by >0.01) would pass in
isolation (arm5−arm2 ECE = +0.0031, well under the 0.01 budget) but is moot given condition 1's failure — there
is no frame-level gain for a calibration budget to protect.

**This matches FINDINGS §12.16's own conclusion** ("The gate fails, and it fails by a wide margin") — independent
recomputation from the raw per-run files agrees with the narrative document on the substantive result. The
disagreements found by this audit (detailed in `RESULTS_REGISTER.md`) are in bookkeeping — a stray duplicate eval
directory, 4 missing provenance-ledger entries, an incomplete mandatory shortcut audit, and a ~1e-4-level
recomputation discrepancy in `inner_val_results.json`'s macro-F1 field — not in the scientific conclusion.
