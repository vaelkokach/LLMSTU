# Implementation report — addendum execution

**Date:** 2026-08-01 · **Base commit at start:** `bb8fe0c` · **Protocol commit:** `8eaf5bb`
**Companions:** `outputs/thesis_audit_report.md` (Phase 0), `BRANCH_B_TEST_PROTOCOL.md`,
`outputs/FINAL_RESULTS_REGISTER.md`, `outputs/thesis_tables_C_D.md`,
`FINDINGS.md §11`.

---

## 1. Repository audit

### What existed and was valid

* **The historic 552/556/570 ablation was genuinely controlled.** The three
  sequence datasets were verified bit-identical in their shared feature columns,
  labels, timestamps, video split, sequence count (6,531) and frame count
  (271,485). Nothing about the feature comparison was confounded by the data.
* **Branch A is sound.** Leak-free video-wise 73/27/27 split, a pre-registered
  single test run (R@1 0.6462), a clean single-variable binding ablation
  (+0.1724 R@1), and an honest retraction of the contaminated 0.9231 matching
  value in favour of 0.7896.
* **The evaluation harnesses that mattered were correct where it counted.**
  `eval_baseline_chain.py` and `eval_events_vs_human.py` are genuinely
  single-process, genuinely per-frame, and free of the mean-of-batch-means error
  that cost the March pipeline its credibility.
* **The human-gold event evaluation is unusual and valuable.** Most
  student-engagement papers stop at clip accuracy; this project measures episodes
  against human annotation and reports false alerts per hour.

### What was inconsistent

| # | Finding | Consequence |
|---|---|---|
| A0-2 | **Branch B had no test split.** `sequence_builder.py:213` draws its own 80/20 video shuffle and never reads `splits.json`; 23 of the detector's 27 test videos were in the cue model's training set. | No Branch-B number could legitimately be called a test result. |
| A0-3 | **`attention/events.py` aliases `inactivity` onto `head_down`.** The "24 human-gold episodes" are **16 distinct** ones (7 duplicated + 1 duplicated `return_to_task` marker that greedy one-to-one matching can never cover). | `head_down` double-weighted in every aggregate; one guaranteed miss for every system. |
| A0-4 | The DDP trainer's `validate()` has no all-gather; `best.pth` was selected on **rank 0's shard**. | Confirms `FINDINGS §6f`; the 0.4096/0.4364/0.4400 figures are not absolutes. |
| A0-5 | **Checkpoint selection was `argmax` over a metric that swings ±0.05 between adjacent epochs**, over 90 epochs, while the difference being argued about was 0.0018. | The reported "best" values carry a selection-on-noise upward bias. |
| A0-7 | The pseudo-label model identity is **not recorded anywhere** in the pipeline outputs. | Must be stated as a reproducibility limitation, not asserted as "Qwen3.5-VL". |
| A0-9 | `FINDINGS §9` listed four completed items as blocking, and listed the dashboard as dropped after §6d.3 reports it rebuilt and verified. | Corrected. |

**Resolved:** the 128-vs-127 video discrepancy is `video_0162_0_10_20251026030716_20251026032847`
— 1,358 recovered frames, **zero** LLMSTU label records, the unique zero-coverage
video (next lowest is 27.4%; median 99.7%). It never enters `labels_tracked`,
`splits.json` or any sequence build. Not a split bug.

---

## 2. Implemented changes

### New package: `LLMDet/attention/thesis_eval/`

| module | purpose |
|---|---|
| `data.py` | sequence loading; the five feature rungs as **column slices** of one 570-dim array |
| `metrics.py` | accuracy, balanced accuracy, macro/weighted F1, per-class P/R/F1, AUPRC, AUROC, ECE, classwise ECE, Brier, NLL, reliability bins, confusion matrix |
| `bootstrap.py` | video/track-level cluster bootstrap, multi-statistic and **paired** variants |
| `segmentation.py` | segmental F1@10/25/50, edit score, event P/R/F1 at multiple tIoU, onset/offset/duration MAE, detection delay, FA/h, gold de-duplication, **common-matched-subset** boundary errors |
| `models.py` | shared model contract + **MS-TCN** + **ASRF** (boundary regression and boundary-driven refinement) |
| `train.py` | single-process trainer; pins split-manifest hash, feature columns, seed, schedule, update count and selection rule into a run record |
| `run_eval.py` | authoritative evaluation → `metrics.json`, per-class CSV, confusion matrix, reliability CSV, **prediction archive**, `command.txt` |
| `eval_events.py` | all systems vs the same human gold, raw and de-duplicated |
| `calibrate.py` | temperature scaling, vector scaling, coverage–risk curves, AURC |
| `aggregate.py` / `aggregate_events.py` / `compare.py` | Tables A and B, paired contrasts across sweeps |
| `launch_sweep.py` / `eval_all.py` | run orchestration, capped at 4 GPUs |
| `build_gold_cache.py` | one 570-dim feature cache for the dense gold, replacing a ~1 h CLIP pass per model |
| `cmose.py` | separate four-level ordinal engagement task + its split-leakage audit |
| `build_register.py` | `FINAL_RESULTS_REGISTER.{json,md}` |

`attention/tests/test_thesis_eval.py` — **69 unit tests** on small synthetic cases
with hand-derivable answers, targeting exactly the properties past bugs violated:
class ordering, padding invariance, batch-size invariance, whole-set aggregation,
event matching one-to-one-ness, zero-length markers, segmental F1 thresholds, edit
score's length-independence, bootstrap clustering, temperature scaling's argmax
preservation, and the ordinal family (QWK's quadratic penalty, ordinal MAE,
average accuracy vs accuracy under imbalance, subject-disjoint splitting).

### Data

`grounding_data/llmstu_seq_split_manifest.json` re-partitions all 6,531 sequences
onto the detector's leak-free video-wise split by **metadata alone** — the split is
a property of `video_id`, which `meta.json` already records — so no feature was
re-extracted and no extraction drift could enter.

### Profiling

`profiling/profile_pipeline.py` extended with p90/p99 stage and end-to-end
percentiles, GPU reserved memory, host peak RSS, mean track count, and
dropped-frame rate against 10 fps and 25 fps budgets.

---

## 3. Experimental results

All Branch-B numbers below: one evaluator, one split, **3 seeds**, video-level
cluster bootstrap. Full tables in `work_dirs/thesis/tables/`.

### 3.1 Architecture — the largest effect measured in this project

Validation, 556-dim features held identical, paired seed-matched bootstrap:

| contrast | Δ macro-F1 | Δ balanced acc | seeds significant |
|---|---|---|---|
| **ASRF − transformer** | **+0.1043** | +0.1040 | **3/3** |
| **MS-TCN − transformer** | **+0.1007** | +0.0940 | **3/3** |
| ASRF − MS-TCN | +0.0036 | +0.0100 | 0/3 (tie) |

Per class, ASRF − transformer: `looking_away` **+0.156 (3/3)**, `phone_use` +0.147
(2/3), `uncertain` +0.123 (3/3), **`turned_to_peer` +0.110 (3/3)**, `head_down`
+0.076 (3/3).

MS-TCN reaches this with **2.7 M parameters against the transformer's 12.6 M**. The
two orientation classes that resisted every feature added over two months move
substantially the moment the temporal inductive bias matches the task.

### 3.2 Isolated feature ablation

| contrast | Δ macro-F1 | seeds significant | verdict |
|---|---|---|---|
| **+ head pose** (556 − 552) | **+0.0294** | **3/3** | ✅ real |
| + facial expression, **isolated** | +0.0009 | 0/3 | ✗ none |
| + dynamics/gaze, **isolated** | −0.0105 | 0/3 | ✗ none |
| + both (the historic 570 − 556) | −0.0031 | 0/3 | ✗ none |
| + both, under MS-TCN | +0.0012 | 0/3 (balanced acc **−0.0327, 3/3**) | ✗ none |
| + both, under ASRF | +0.0008 | 0/3 | ✗ none |

### 3.3 Temporal events (16 distinct gold episodes, 10 tracks, 3 seeds)

Headline: edit score — the fragmentation penalty frame accuracy is blind to — rises
from **28.8–38.2 (transformer) to 48.5–59.9 (MS-TCN)** against a teacher at **76.5**;
segmental F1@25 rises 0.265 → 0.362. Event recall improves (0.208 → 0.312) but
remains far below the teacher's 0.625.

### 3.4 Boundary errors on the **common matched subset**

`FINDINGS §6.0a` warned that the 556 model's apparent onset blow-up (1.6 s → 10.4 s)
was not like-for-like because the two models matched different episodes. Now
computed properly (3 common events per seed, seed-matched):

| pair | system | onset MAE | offset MAE | duration MAE |
|---|---|---|---|---|
| 552 vs 556 | transformer 552 | 8.92 s | 0.00 s | 8.92 s |
| | **transformer 556** | **1.44 s** | 6.10 s | 7.10 s |
| 556 vs ASRF | transformer 556 | 1.44 s | 6.10 s | 7.10 s |
| | **ASRF 556** | **1.00 s** | **1.44 s** | **2.44 s** |
| 556 vs MS-TCN 570 | transformer 556 | 1.44 s | 6.10 s | 7.10 s |
| | MS-TCN 570 | 7.53 s | 9.15 s | 16.68 s |

**The apparent degradation reverses.** On the episodes both models found, the 556
model's onset error is 1.44 s against the 552 model's 8.92 s — it was being charged
for the four extra, harder episodes only it detected. ASRF has the tightest
boundaries of any system including the teacher.

⚠️ Three common events. Directional evidence, not an estimate.

### 3.5 ★ TEST SPLIT — every conclusion replicates

Single pre-registered run, 27 held-out videos, 43,733 frames, 3 seeds.

| model | dims | accuracy | balanced acc | **macro-F1** | val macro-F1 |
|---|---|---|---|---|---|
| transformer | 552 | 0.657 | 0.396 | 0.377 | 0.373 |
| transformer | 556 | 0.699 | 0.424 | 0.407 | 0.402 |
| transformer | 570 | 0.660 | 0.424 | 0.390 | 0.399 |
| ASRF | 556 | 0.710 | 0.513 | 0.477 | 0.506 |
| ASRF | 570 | 0.735 | 0.521 | 0.492 | 0.507 |
| **MS-TCN** | **556** | 0.759 | 0.519 | **0.500** | 0.503 |
| MS-TCN | 570 | 0.765 | 0.481 | 0.488 | 0.504 |

Paired bootstrap on test: MS-TCN − transformer **+0.0933 (3/3)**; ASRF −
transformer **+0.0705 (3/3)**; head pose **+0.0296 (3/3)**; 570 − 556 **−0.0167**
(1/3, *against* the extra features).

The transformer rungs agree with validation to within 0.005 macro-F1 —
independent evidence the partition is leak-free and that validation-guided
choices did not overfit. The pre-registered primary model was **ASRF-556**
(validation-tied with MS-TCN); on test **MS-TCN-556 scores higher** (0.500 vs
0.477). The selection is not revised — that is what pre-registration is for.

### 3.6 ★ The head-pose gain is `face_found`, not the angles

| configuration | dims | macro-F1 | Δ vs base | seeds significant |
|---|---|---|---|---|
| base | 552 | 0.373 | — | — |
| **+ `face_found` only** | 553 | **0.396** | **+0.0236** | 1/3 (balanced acc +0.0346, 2/3) |
| + yaw/pitch/roll only | 555 | 0.378 | +0.0053 | 0/3 |
| + full head-pose block | 556 | 0.402 | +0.0294 | 3/3 |
| full block − `face_found` only | — | — | +0.0058 | 0/3 |

~80% of the contribution is the binary detection flag. **This redirects the
DirectMHP/6DRepNet upgrade**: better angles are the part that does not help, and
a full-range estimator that succeeds on downward-facing heads would erode the
92%-vs-8% coverage contrast the useful feature depends on. It must be judged on
downstream cue macro-F1 with `face_found` recomputed, not on pose coverage or
angular error.

### 3.7 ★ CMOSE — separate ordinal engagement task, and a leakage finding

A four-level ordinal head over CMOSE's released 1024-d I3D embeddings (11,902 of
12,197 clips; 295 with empty embeddings dropped, not zero-filled), 3 seeds.

Under CMOSE's own split our simple baseline reproduces the paper's range:
accuracy **0.718** / average accuracy **0.601** vs the paper's 78.14% / 60.94% —
on I3D alone, without audio, text or OpenFace.

Then the finding: clip names encode the subject, and **101 of 103 subjects
appear in more than one official split** (100 shared between official train and
test). Re-splitting by whole subjects, everything else identical:

| metric | official split | **subject-disjoint** | change |
|---|---|---|---|
| accuracy | 0.7179 ± 0.0030 | 0.6006 ± 0.0058 | **−0.117** |
| average accuracy | 0.6007 ± 0.0086 | 0.4347 ± 0.0084 | **−0.166** |
| macro-F1 | 0.5733 ± 0.0027 | 0.4111 ± 0.0051 | **−0.162** |
| MAE (ordered levels) | 0.3123 ± 0.0029 | 0.4459 ± 0.0123 | **+0.134** |
| quadratic weighted kappa | 0.5369 ± 0.0039 | 0.3167 ± 0.0319 | **−0.220** |
| Spearman ρ | 0.5354 ± 0.0086 | 0.3221 ± 0.0305 | **−0.213** |

Not a criticism of CMOSE — the paper states it uses a random segment split — but
independent, quantified corroboration of this project's own most expensive
lesson, and a mandatory caveat on any CMOSE-vs-this-project comparison.

### 3.8 Calibration

Temperature fitted on **validation** predictions and applied unchanged to
**test** — the deployment protocol, not an in-sample estimate:

| model | T | ECE before → after | NLL before → after | accuracy |
|---|---|---|---|---|
| transformer 556 | 1.311 | 0.0739 → **0.0191** | 0.9660 → 0.9121 | unchanged |
| ASRF 556 | 0.759 | 0.0570 → **0.0306** | 0.7854 → 0.7791 | unchanged |
| MS-TCN 556 | 0.924 | 0.0287 → **0.0155** | 0.7033 → 0.7043 | unchanged |

The transformer is over-confident (T > 1); the boundary-aware models are
slightly *under*-confident and already far better calibrated before scaling.

### 3.9 Runtime

Detector is a fixed ~122 ms floor. FPS 7.0 (1 student) → 2.6 (30 students); p99
152 ms → 456 ms. **100% of frames miss a 10 fps budget at every student count.**
Single-GPU deployment; the 4 A100s are a training resource.

---

## 4. Negative results

1. **Facial expression features do not improve cue classification.** +0.0009
   macro-F1 isolated, 0/3 seeds significant, under three architectures.
2. **Body-language / personalised-gaze dynamics do not improve cue
   classification.** −0.0105 isolated, 0/3 significant; costs balanced accuracy
   (−0.0327, 3/3) under MS-TCN.
3. **`FINDINGS §6e`'s `turned_to_peer` claim does not replicate.** It read
   *"turned_to_peer improved 47% relative … almost all from the dynamic block …
   the personalised gaze-deviation feature doing exactly what it was designed
   for."* Isolated: **−0.0120, CI [−0.050, +0.032], 0/3 significant** — nominally
   negative. The original was a single seed, a different split, and shard-averaged
   metrics.
4. **The 570-dim model is not the best model.** Under correct measurement it is
   statistically tied with 556 and nominally lower — `FINDINGS §6f` reached this
   with one seed; it is now established with three seeds and paired tests.
5. **ASRF's boundary head does not beat MS-TCN on frame metrics** (+0.0036, 0/3).
   It earns its place through boundary tightness and calibration, not accuracy.
6. **Event recall did not reach the teacher** for any architecture (best 0.312 vs
   0.625). The principal gap is narrowed, not closed.
7. **"Correcting" the gradient-clipping order collapses training.** Clipping the
   *scaled* gradients is load-bearing global gradient normalisation; unscaling
   first, or removing clipping, gives majority-class collapse (macro-F1 0.144).
8. **Metric head-pose angles contribute nothing beyond the detection flag**
   (+0.0053 alone, 0/3; +0.0058 on top of `face_found`, 0/3). The planned
   DirectMHP / 6DRepNet upgrade targets the part that does not help.
9. **The pre-registered primary model was not the best on test.** ASRF-556 was
   selected on validation (tied with MS-TCN); MS-TCN-556 scored higher on test
   (0.500 vs 0.477). Reported, not revised.

---

## 5. Literature positioning

**Directly comparable:** nothing across datasets. Within this project, all Branch-B
configurations are directly comparable (one evaluator, one split, one seed policy,
paired tests).

**Comparable in *kind*, not in value:** MS-TCN/ASRF/ASFormer report segmental
F1@{10,25,50} and edit score on 50Salads/GTEA/Breakfast. This project now reports
the same metric family, so the *methodology* is aligned and the numbers can be
described as "computed with the standard action-segmentation protocol" — but the
values must never be placed beside those benchmarks.

**Not comparable:** CMOSE (78.14% overall / 60.94% average accuracy, random segment
split, ordinal engagement); EmotiW/EngageWild (MSE ≈ 0.06, continuous regression);
SCB (74.0% mAP@0.5, box detection); this project's R@1 0.6462 (retrieval). Table D
in `outputs/thesis_tables_C_D.md` states each row's task, label definition, split
unit and comparability limitation.

**Reproduced externally, and informative:** DIPSER cue↔engagement ρ = +0.172
(p < 0.0001) — significant and **reversed**; and basic-expression → expert boredom
AUROC 0.544 — barely above chance.

---

## 6. Thesis implications

### May claim

* A leak-free, human-audited pipeline for detecting, **attributing** and temporally
  aggregating observable student behavioural cues in a computer laboratory, with
  both branches on one video-wise split.
* **Attribution quality drives grounding quality**: +0.1724 R@1 from binding alone,
  with R@10 moving +0.0141, under single-variable control. R@1 0.6462 on a
  pre-registered untouched test split.
* **Temporal architecture, not feature engineering, is the lever for visible-cue
  classification**: +0.10 macro-F1 from MS-TCN/ASRF over the transformer, 3/3 seeds
  significant, at one fifth of the parameters, with the gain concentrated on the
  orientation classes.
* **Head pose is the one feature family that earns its place** (+0.0294, 3/3).
* Boundary-aware modelling substantially reduces over-segmentation (edit 38 → 60
  against a teacher at 77).
* Calibrated, abstaining alerts: ECE 0.053 → 0.032 with accuracy unchanged, plus
  coverage–risk curves for threshold selection.
* **Cue meaning is setting-dependent** (DIPSER reversal), which bounds the
  generalisation claim and justifies the visible-cue framing.
* **Split policy matters, and by how much**: on CMOSE, closing a subject leak
  present in the published split costs quadratic weighted kappa 0.537 → 0.317
  and average accuracy 0.601 → 0.435. Independent, quantified support for the
  leak-free protocol this project adopted after March.

### Must qualify

* Every Branch-B label is a **pseudo-label**. Test measures generalisation to unseen
  videos, not agreement with humans. The human-referenced numbers come from **16
  distinct episodes over 10 tracks** — a diagnostic set, where one episode is 6
  points of recall.
* The teacher is an **empirical benchmark**, not a ceiling.
* The pseudo-label model's exact identity is unrecorded.
* "Near-real-time": 2.6–7.0 FPS, and 100% of frames miss a 10 fps budget.
* The event evaluation excludes the detector and tracker by design.

### Must not claim

* ✗ Boredom, perplexity or curiosity detection. Measured AUROC 0.544.
* ✗ That facial expression or gaze-dynamic features improved accuracy.
* ✗ That the 570-dim model is best.
* ✗ Any DDP shard-averaged macro-F1 as an absolute.
* ✗ The 92.3% matching value.
* ✗ SCB zero-shot 0.0322 as a transfer score.
* ✗ Real-time operation, or 4-A100 throughput as classroom performance.
* ✗ Any inference of internal attention from a cue.
* ✗ That the CMOSE engagement result and the visible-cue macro-F1 are comparable.
  They are separate tasks, separate label semantics, separate tables.

---

## 7. Reproduction

```bash
cd LLMDet

# tests (69)
python -m pytest attention/tests/test_thesis_eval.py -q

# feature ablation ladder: 5 configs x 3 seeds
python -m attention.thesis_eval.launch_sweep --sweep ladder \
    --out-root work_dirs/thesis/ladder --gpus 0,1,2,3 --threads 12

# boundary-aware architectures: MS-TCN + ASRF x 2 feature blocks x 3 seeds
python -m attention.thesis_eval.launch_sweep --sweep arch \
    --out-root work_dirs/thesis/arch --gpus 0,1,2,3 --threads 12

# authoritative evaluation
python -m attention.thesis_eval.eval_all --root work_dirs/thesis/ladder --split val
python -m attention.thesis_eval.eval_all --root work_dirs/thesis/arch --split val --refine-asrf

# Table A + paired contrasts
python -m attention.thesis_eval.aggregate --root work_dirs/thesis/ladder --split val \
    --out work_dirs/thesis/tables

# cross-sweep architecture comparison
python -m attention.thesis_eval.compare \
    --a s42=work_dirs/thesis/arch/asrf_556_hp_s42/eval_val/predictions.npz \
    --a s43=work_dirs/thesis/arch/asrf_556_hp_s43/eval_val/predictions.npz \
    --a s44=work_dirs/thesis/arch/asrf_556_hp_s44/eval_val/predictions.npz \
    --b s42=work_dirs/thesis/ladder/transformer_556_hp_s42/eval_val/predictions.npz \
    --b s43=work_dirs/thesis/ladder/transformer_556_hp_s43/eval_val/predictions.npz \
    --b s44=work_dirs/thesis/ladder/transformer_556_hp_s44/eval_val/predictions.npz \
    --out work_dirs/thesis/tables/cmp_asrf556_vs_transformer556.json

# human-gold feature cache (once) and event evaluation
python -m attention.thesis_eval.build_gold_cache \
    --manifest ../grounding_data/llmstu_tools/outputs/dense_event_manifest.jsonl \
    --annotations ../event_gold_bundle/gold_annotations_Admin.jsonl \
    --frames-root ../grounding_data/stu_img/frames \
    --affect-cache ../grounding_data/llmstu_tools/outputs/affect_cache.npz \
    --out ../grounding_data/llmstu_tools/outputs/gold_event_features.npz
python -m attention.thesis_eval.eval_events \
    --cache ../grounding_data/llmstu_tools/outputs/gold_event_features.npz \
    --gold-events ../grounding_data/llmstu_tools/outputs/gold_events.jsonl \
    --ckpt "asrf_556=work_dirs/thesis/arch/asrf_556_hp_s42/checkpoints/best.pth" \
    --out work_dirs/thesis/events/summary_s42.json
python -m attention.thesis_eval.aggregate_events \
    --inputs work_dirs/thesis/events/summary_s4{2,3,4}.json \
    --tag gold_dedup --out work_dirs/thesis/tables

# calibration
python -m attention.thesis_eval.calibrate \
    --val-predictions  work_dirs/thesis/arch/asrf_556_hp_s42/eval_val/predictions.npz \
    --eval-predictions work_dirs/thesis/arch/asrf_556_hp_s42/eval_test/predictions.npz \
    --out work_dirs/thesis/calibration/asrf_556_hp_s42

# runtime
python -m profiling.profile_pipeline --config configs/attention_temporal.yaml \
    --video 0325.mp4 --max-frames 160 --scaling 1,5,10,20,30 \
    --out work_dirs/profiling/report_thesis_scaling.json

# head-pose decomposition (face_found vs metric angles)
python -m attention.thesis_eval.launch_sweep --sweep headpose \
    --out-root work_dirs/thesis/headpose --gpus 0,1,2,3
python -m attention.thesis_eval.eval_all --root work_dirs/thesis/headpose --split val

# CMOSE (separate ordinal engagement task); 13.6 GB, CC-BY-SA-4.0
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('cwuau/CMOSE', repo_type='dataset', \
    local_dir='../grounding_data/external/CMOSE')"
python -m attention.thesis_eval.cmose --root ../grounding_data/external/CMOSE \
    --out work_dirs/thesis/cmose

# registers
python -m attention.thesis_eval.build_register --out ../outputs
```

Every `run_eval` output directory contains its own `command.txt`, and every training
run a `run_record.json` with git commit, dirty flag, manifest SHA-256, feature
columns, seed, schedule, update count, selection rule and full epoch history.
