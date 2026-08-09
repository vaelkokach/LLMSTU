# Branch C results register — inner validation only

**Generated:** 2026-08-09 · **Evaluator:** `thesis_eval/1.0.0` · **Protocol:** `BRANCH_C_PROTOCOL.md` v1.0.0, amendment A1 · **Fold manifest sha256:** `fd913e7c5a52839ee5115224ffdb741c09cc1104eeb1fee5d32b9595e2388314`

**THE OUTER FOLDS OF `outputs/branch_c/splits/branch_c_folds.json` HAVE NOT BEEN OPENED.** Every number in this
file comes from the **inner-validation** holdout only (a single prevalence-balanced 20-video grouped holdout inside
each outer fold's 101–102 remaining videos, per protocol §2.1). These are **selection-stage statistics**, not
generalisation estimates, and per protocol §3.4 they are **NOT comparable** to the legacy Branch-B test numbers in
`outputs/FINAL_RESULTS_REGISTER.md` (macro-F1 ≈ 0.500 etc.) — those come from a different partition under a
different protocol. Never place a Branch-C inner-val number and a Branch-B test number in the same comparison.

Every number below was **independently recomputed by this audit** from the 45 raw
`outputs/branch_c/eval/<eid>/metrics.json` files — not copied from `outputs/branch_c/inner_val_results.json` or
from FINDINGS.md. Where the two disagree, it is reported under "Verification" at the bottom, loudly.

Companion file: `outputs/branch_c/RESULTS_REGISTER.json` (machine-readable, same numbers, one entry per
metric×arm plus full verification notes and the go-gate verdict).

**Value = arithmetic mean over 15 runs (5 outer-fold-specific inner-val splits × 3 seeds 42/43/44), each run's
metric read directly from its own `metrics.json`.** This is *not* the frame-pooled macro-F1 protocol §3.1 defines
for the (unopened) outer folds; a fold-pooled-per-seed robustness check is reported in Verification and does not
change any conclusion.

## Arms covered

| arm | features | input_dim | sequence_root | n runs |
|---|---|---|---|---|
| `arm1_mstcn_553_ff` | base + `face_found`, no angles (protocol arm 1 = arm 4, presence-only control) | 553 | `grounding_data/llmstu_sequences_full` | 15 |
| `arm2_mstcn_556_mp` | + MediaPipe yaw/pitch/roll (legacy, protocol arm 2, primary baseline) | 556 | `grounding_data/llmstu_sequences_full` | 15 |
| `arm5_mstcn_556_fr` | + 6DRepNet360 full-range rotation (protocol arm 5) | 556 | `grounding_data/llmstu_sequences_fullrange` | 15 |

Arms 2 and 5 report the identical `feature_config` label (`556_hp`) — the distinguishing field is
`sequence_root`, not `feature_config`; a register built from `feature_config` alone would silently conflate them.
`face_found` changed on 0.0% of frames between arms 2 and 5 (FINDINGS §12.16), so the arm5−arm2 contrast isolates
the pose-measurement method and nothing else.

Model/training config (identical across all 45 runs except as noted): `model=mstcn`, `epochs=90`, `batch_size=32`,
`lr=3e-4`, `weight_decay=0.01`, `dropout=0.1`, `boundary_loss_weight=1.0`, `smoothing_loss_weight=0.15`,
`select_window=5`, `use_class_weights=true`. Producing command (per run, `<eid>` = `<arm>_f<fold>_s<seed>`):

```
python -m attention.thesis_eval.run_eval --ckpt outputs/branch_c/runs/<arm>/<eid>/checkpoints/best.pth \
  --split val --out outputs/branch_c/eval/<eid> --batch-size 1 --cluster video
```

Raw artifact path per entry: `outputs/branch_c/eval/<arm>_f{0,1,2,3,4}_s{42,43,44}/metrics.json` (15 files).
Fold-manifest hashes: top-level `fd913e7c5a52839ee5115224ffdb741c09cc1104eeb1fee5d32b9595e2388314`; per-fold
`fold_0` `5813913b…`, `fold_1` `3c1dee2b…`, `fold_2` `4f0fec93…`, `fold_3` `0532432f…`, `fold_4` `4e98d41e…`
(full hashes in the JSON companion).

---

## C · scalar metrics (inner validation)

All rows below: split = inner-val, n runs = 15, evaluator = `thesis_eval/1.0.0`, citable = ✅ (as an
inner-validation selection statistic; **not** citable as a generalisation number — see caveat on every row).

| metric | arm1 (553, no angles) | arm2 (556, MediaPipe) | arm5 (556, full-range) | caveat |
|---|---|---|---|---|
| macro_f1 | 0.4837 (sd 0.0270) CI [0.4708,0.4972] | 0.4888 (sd 0.0335) CI [0.4728,0.5055] | 0.4865 (sd 0.0274) CI [0.4737,0.5002] | Inner-val selection statistic; NOT a generalisation estimate; NOT comparable to Branch-B test (0.500). |
| balanced_accuracy | 0.5208 (sd 0.0409) CI [0.5004,0.5401] | 0.5227 (sd 0.0437) CI [0.5003,0.5428] | 0.5258 (sd 0.0367) CI [0.5075,0.5432] | same |
| macro_auprc | 0.4815 (sd 0.0362) CI [0.4640,0.4993] | 0.4918 (sd 0.0445) CI [0.4705,0.5134] | 0.4903 (sd 0.0404) CI [0.4713,0.5105] | same |
| macro_auroc | 0.8584 (sd 0.0137) CI [0.8517,0.8649] | 0.8651 (sd 0.0183) CI [0.8561,0.8738] | 0.8649 (sd 0.0124) CI [0.8590,0.8711] | same |
| nll (**raw, uncalibrated**) | 0.7653 (sd 0.0810) CI [0.7261,0.8056] | 0.7402 (sd 0.0749) CI [0.7033,0.7766] | 0.7613 (sd 0.1022) CI [0.7111,0.8105] | ⚠ No temperature-calibrated NLL exists for Branch C anywhere on disk (see below). Protocol §3.2 requires both raw and calibrated; only raw is reportable. |
| brier (**raw, uncalibrated**) | 0.3867 (sd 0.0462) CI [0.3641,0.4093] | 0.3742 (sd 0.0442) CI [0.3527,0.3961] | 0.3876 (sd 0.0602) CI [0.3583,0.4167] | same calibration caveat |
| ece (**raw, uncalibrated**) | 0.0471 (sd 0.0185) CI [0.0384,0.0564] | 0.0349 (sd 0.0142) CI [0.0282,0.0420] | 0.0380 (sd 0.0197) CI [0.0293,0.0487] | same calibration caveat |

**Calibration status, stated explicitly because the protocol requires it:** every `metrics.json` carries
`"refined": false` and there is no temperature-scaling artifact anywhere under `outputs/branch_c/`. **All NLL /
Brier / ECE values in this register are raw.** No calibrated counterpart exists to report. This is a protocol
compliance gap against §3.2 ("both raw and temperature-calibrated"), not a withheld number.

## C · per-class F1 with support (inner validation)

Support = pooled frames, summed once per fold's inner-val split across the 5 distinct folds (**not** ×3 for
seeds — the same frames are scored three times, once per seed's model). Identical support across arms 1/2/5
(verified: same labels, same split, only features/model differ).

| cue | support (pooled frames) | arm1 F1 | arm2 F1 | arm5 F1 |
|---|---|---|---|---|
| screen_oriented | 168,028 | 0.8482 (sd 0.0239) CI [0.8364,0.8595] | 0.8499 (sd 0.0255) CI [0.8371,0.8621] | 0.8426 (sd 0.0362) CI [0.8246,0.8598] |
| looking_away | 14,498 | 0.3020 (sd 0.0469) CI [0.2784,0.3235] | 0.3116 (sd 0.0626) CI [0.2783,0.3384] | 0.3004 (sd 0.0365) CI [0.2827,0.3182] |
| head_down | 12,302 | 0.5360 (sd 0.0918) CI [0.4908,0.5812] | 0.5385 (sd 0.0857) CI [0.4979,0.5819] | 0.5408 (sd 0.0949) CI [0.4956,0.5886] |
| turned_to_peer | 5,409 | 0.1986 (sd 0.0625) CI [0.1682,0.2294] | 0.2047 (sd 0.0678) CI [0.1705,0.2375] | 0.2073 (sd 0.0532) CI [0.1815,0.2339] |
| phone_use | 10,135 | 0.5187 (sd 0.0688) CI [0.4875,0.5543] | 0.5324 (sd 0.0663) CI [0.5025,0.5666] | 0.5361 (sd 0.0619) CI [0.5073,0.5668] |
| uncertain | 10,283 | 0.4986 (sd 0.0373) CI [0.4808,0.5166] | 0.4956 (sd 0.0377) CI [0.4771,0.5136] | 0.4916 (sd 0.0356) CI [0.4741,0.5086] |

Caveat on every row: inner-val selection statistic, not a generalisation estimate, not comparable to Branch-B
test. `turned_to_peer` has the smallest support and protocol §2.2 flags it as "a near-meaningless statistic" per
fold — retained here (support-labelled) rather than dropped, per protocol §4's rule that failing/marginal arms
stay in the table.

## C · per-class AUROC (supporting the redundancy finding, FINDINGS §12.16)

Reported here because it is load-bearing for the branch's central mechanism claim and is fully reproducible from
the same 45 files (see Verification, claim (c), for what is and is not independently confirmed).

| cue | arm1 AUROC (no angles at all) | arm5 AUROC (full-range) |
|---|---|---|
| head_down | 0.9150 | 0.9169 |
| uncertain | 0.9300 | 0.9383 |
| turned_to_peer | 0.8249 | 0.8262 |
| phone_use | 0.8893 | 0.8951 |
| looking_away | 0.7672 | 0.7826 |

Citable ✅ as inner-validation statistics computed from the 45 registered runs. The "standalone pose scalar"
(0.830/0.852/0.625/0.519/0.524) that FINDINGS §12.16 places beside this table is **not** from these 45 runs — see
Verification claim (c) below before citing that comparison as controlled.

## C · mandatory shortcut audit (protocol §5.1) — INCOMPLETE, not citable

| metric | arm | value | citable | caveat |
|---|---|---|---|---|
| macro_f1 | `arm0_mstcn_quality` (quality/missingness-only features) | — | ⛔ | **NOT CITABLE — the audit is incomplete.** `outputs/branch_c/arm0_sweep.log` shows 15 runs dispatched (5 folds × 3 seeds), but only fold 0's 3 seeds produced any artifact, and all three stop mid-training (`outputs/branch_c/runs/logs/arm0_mstcn_quality_f0_s{42,43,44}.log`, 3 lines each, last entry "ep 20 ... macroF1 0.4180"). No `run_record.json`, no `metrics.json`, no final checkpoint evaluation exists for any of the 15 dispatched runs — only `checkpoints/{best,last}.pth` for the 3 that started. Protocol §5.1 requires this audit be run and reported "whatever it shows," independent of the arms-1/2/5 gate outcome. It was not completed. Do not infer a shortcut-audit finding (positive or negative) from the partial log — 0.42 macro-F1 at epoch 20/90 on an unconverged, unevaluated checkpoint is not a result. |

---

## Data-integrity findings (this audit)

1. **`outputs/branch_c/eval/` contains 46 directories, not 45.** `eval/arm5_f0_s42` is a byte-identical-metrics
   duplicate of `eval/arm5_mstcn_556_fr_f0_s42` (differs only in the recorded `checkpoint` path — relative vs
   absolute — and is timestamped ~4 minutes earlier: 19:34:33 vs 19:38:14 on 2026-08-09). It is a superseded prior
   invocation left on disk. **Excluded from every count and aggregate in this register.** A naive `glob("eval/*")`
   over the directory over-counts runs by one; anyone re-deriving "45" from a directory listing should filter by
   the canonical `<arm>_f<fold>_s<seed>` name, not by directory count.
2. **`outputs/branch_c/RUNS.jsonl` (the protocol-§9 append-only provenance ledger) is missing 4 of the 45
   `train`-stage entries.** It has 41 train entries + 3 cache entries = 44 total; the 4 missing are exactly
   `arm1_mstcn_553_ff_f0_s42`, `f0_s43`, `f0_s44`, `f1_s42`. Cross-checked against each run's own
   `run_record.json`: those same 4 runs (and no others, out of all 45) were built at git commit
   `bf649f5eb6…`, while every other run — the rest of arm1 and all of arm2/arm5 — was built at `ea2a8ad2…`. The
   tracked diff between the two commits touches only fold-manifest data files and two orchestration scripts
   (`tools/branch_c/aggregate_arms.py`, `launch_arm_sweep.py`), not training/eval code, so this is unlikely to be
   a substantive confound for the paired comparisons in `ABLATIONS.md` — but `run_record.json` (unlike
   `RUNS.jsonl`'s cache-stage entries) carries no dirty-diff hash, so the uncommitted state at the time of those 4
   runs cannot be fully audited from what's on disk.
3. **The mandatory shortcut audit (protocol §5.1) is incomplete** — see the table above. This is an open
   protocol-compliance item, not a result.
4. **Every one of the 45 per-run `macro_f1` values in `inner_val_results.json` differs from the corresponding
   `eval/<eid>/metrics.json` `macro_f1` field**, by amounts ranging ~3e-6 to ~3.5e-4 (both directions, no
   systematic sign). Per-arm means still agree to 4 decimals (0.4837 / 0.4888 / 0.4865 both ways), and one pair
   (`arm1 f0_s42`) rounds differently at 4 decimals (0.5436 vs 0.5437). NLL/Brier/ECE fields show **no** such
   discrepancy — they match `inner_val_results.json` exactly wherever checked. This is consistent with
   `inner_val_results.json`'s macro-F1 having been recomputed via a slightly different code path (e.g. from
   `predictions.npz` rather than the stored field) rather than with any run using different data or labels. See
   `RESULTS_REGISTER.json` → `verification_notes` for the full per-run diff table.

---

## Verification of claims (independently recomputed from the 45 raw `metrics.json` files)

**(a) "arm5 − arm2 = −0.0022, CI [−0.0119, +0.0085], 5/15 wins."**
Recomputed (own bootstrap, 20000 draws, seed 0, resampling the 15 matched fold/seed pairs): mean diff **−0.0023**,
CI **[−0.0123, +0.0084]**, wins **5/15**. **AGREES** within bootstrap/rounding noise; wins match exactly. The
small residual (−0.0023 vs −0.0022) traces to integrity finding 4 above.

**(b) "every per-class CI includes zero."** **CONFIRMED.** All 18 per-class paired-difference CIs (3 contrasts ×
6 classes, own bootstrap) include zero — see `ABLATIONS.md` for the full table.

**(c) "arm1 AUROC 0.915 head_down / 0.930 uncertain," and arm1 beats the standalone pose scalar (0.830/0.852) on
every class.** The arm1/arm5 AUROC figures are **CONFIRMED** exactly from the 45 runs (table above). The
"standalone pose scalar" comparison is **not independently verifiable by this audit**: those five numbers
(0.830/0.852/0.625/0.519/0.524) come from FINDINGS §12.15's separate dev-only pre-test (184,304 frames, fold-0
inner-train only, one geodesic-distance scalar, no seeds, no held-out split) and no saved artifact of that
computation exists under `outputs/branch_c/` — FINDINGS §12.15 cites only "this session's transcript." This audit
neither confirms nor contradicts those five numbers; it flags the comparison as spanning two different data
partitions and evaluation protocols, which the thesis text should not present as a single controlled contrast.

**(d) "NLL +0.0210, Brier +0.0134, ECE +0.0031 (arm5 vs arm2)."** **CONFIRMED EXACTLY** to 4 decimals: recomputed
+0.0210 / +0.0134 / +0.0031.

**(e) "45 runs, 0 failures, all 5 folds × 3 seeds × 3 arms present."** **Substantively confirmed, with three
integrity caveats that must accompany the claim** — see Data-integrity findings 1–3 above. All 45 (arm, fold,
seed) combinations for arms 1/2/5 have a complete `run_record.json` and a valid `metrics.json`; no training log
shows an error. But the eval directory over-counts to 46 (one stray duplicate), the provenance ledger is missing
4 of 45 entries (and those 4 happen to be the only runs built off a different — probably immaterial — commit),
and a 4th, separately-registered arm (`arm0`, the mandatory shortcut audit) was dispatched but not completed. "0
failures" is true of the 45 counted here; it is not true that the full experimental program the protocol
describes ran to completion.

Full go/no-go gate verdict and the paired-contrast tables are in `ABLATIONS.md`.
