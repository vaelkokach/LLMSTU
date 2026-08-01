# Branch B — pre-registered test-split protocol

**Registered:** 2026-08-01, **before** any Branch-B model saw the test split.
**Companion to:** `TEST_SPLIT_PROTOCOL.md` (Branch A, closed 2026-08-01).

Until now Branch B had **no test set at all**: `attention/sequence_builder.py`
drew its own 80/20 video shuffle rather than reading
`grounding_data/llmstu_tools/outputs/splits.json`, so 23 of the detector's 27
held-out videos sat in the cue model's training set and no Branch-B number could
legitimately be called a test result (`FINDINGS.md §3.4c`;
`outputs/thesis_audit_report.md` A0-2).

All Branch-B sequences have now been re-partitioned onto the detector's leak-free
video-wise split via `grounding_data/llmstu_seq_split_manifest.json`
(73 / 27 / 27 videos = 4,390 / 1,085 / 1,056 sequences = 185,050 / 42,702 / 43,733
frames, zero video overlap). Every model below was **retrained from scratch** on
that partition. The test videos have never been read by any of them.

---

## 1. What was decided on validation, and is now frozen

Decided from `work_dirs/thesis/tables/table_a_val.{json,md}` and
`table_b_gold_dedup.md`, over 3 seeds each, with paired video-level bootstrap
tests. Nothing below may change after the test run.

### 1.1 Architecture

| finding | evidence |
|---|---|
| MS-TCN beats the temporal transformer by **+0.1007** macro-F1 (3/3 seed pairs significant) | `cmp_mstcn556_vs_transformer556.json` |
| ASRF beats the temporal transformer by **+0.1043** macro-F1 (3/3 significant) | `cmp_asrf556_vs_transformer556.json` |
| ASRF vs MS-TCN is a **tie**: +0.0036 macro-F1, 0/3 significant | `cmp_asrf556_vs_mstcn556.json` |

**Selected primary frame model: ASRF at 556 dims.** ASRF is chosen over the tied
MS-TCN because it additionally produces an explicit boundary probability, which
the event layer consumes; the tie means this is a design choice, not a
performance claim, and it is recorded as such.

### 1.2 Feature block

| contrast | Δ macro-F1 | significant | verdict |
|---|---|---|---|
| +head pose (556 − 552), transformer | **+0.0294** | **3/3** | real |
| +expression isolated (563expr − 556), transformer | +0.0009 | 0/3 | none |
| +dynamics isolated (563dyn − 556), transformer | −0.0105 | 0/3 | none |
| +expression+dynamics (570 − 556), transformer | −0.0031 | 0/3 | none |
| +expression+dynamics (570 − 556), MS-TCN | +0.0012 | 0/3 | none (balanced acc **−0.0327, 3/3**) |
| +expression+dynamics (570 − 556), ASRF | +0.0008 | 0/3 | none |

**Selected feature block: 556 (base + head pose).** The extra 14 dims earn
nothing under three different architectures, and cost balanced accuracy under
MS-TCN. 570 is still carried onto test so the null is reported on test as well
rather than only asserted from validation.

### 1.3 Checkpoint selection rule

Highest mean validation macro-F1 over a trailing 5-epoch window (`--select-window 5`),
fixed before training. Chosen because single-epoch validation macro-F1 in this
project swings ±0.05 between adjacent epochs, so `argmax` over 90 epochs selects
partly on noise.

### 1.4 Calibration

Temperature scaling, temperature fitted **on validation predictions only** and
applied unchanged to test. Temperature scaling cannot alter any argmax, so it
cannot change accuracy, macro-F1 or the confusion matrix; only ECE / Brier / NLL
and the coverage–risk curve move. Asserted in code.

### 1.5 Alert threshold

The abstention threshold for the dashboard is read off the **validation**
coverage–risk curve and held fixed. It is not tuned on test and not tuned on the
human-gold set.

---

## 2. Exactly what will be run on the test split

One evaluation pass per checkpoint, `attention/thesis_eval/run_eval.py`,
`--split test`, batch size 1, video-level cluster bootstrap with 2,000 resamples.

| # | system | dims | seeds | role |
|---|---|---|---|---|
| 1 | transformer | 552 | 42, 43, 44 | base-feature baseline |
| 2 | transformer | 556 | 42, 43, 44 | project's historic architecture |
| 3 | mstcn | 556 | 42, 43, 44 | convolutional segmentation baseline |
| 4 | asrf | 556 | 42, 43, 44 | **selected primary model** |
| 5 | asrf | 570 | 42, 43, 44 | feature-null confirmation |

Plus one majority-class control computed from the test labels.

## 3. What will be reported

Per system, mean ± sd over the three seeds, with the seed-42 video-level
bootstrap 95% CI alongside:

accuracy · balanced accuracy · macro precision / recall / F1 · weighted F1 ·
per-class precision / recall / F1 / AUPRC / AUROC · macro-AUPRC · macro-AUROC ·
confusion matrix · ECE · classwise ECE · Brier · NLL · reliability bins.

Plus, from the stored archives: temperature-scaled calibration metrics and the
coverage–risk curve.

## 4. What will NOT happen after the run

- No architecture, feature block, hyperparameter, seed count, checkpoint-selection
  rule, calibration method or alert threshold will be changed.
- No model will be retrained.
- No additional test evaluation will be run.
- If a test number disappoints, it is reported as measured. The validation table
  stays exactly as it is.

## 5. Standing caveats that apply to the test numbers

- Labels on the test split are **pseudo-labels** from the same Qwen3-VL-family
  teacher as training. The test split therefore measures generalisation to unseen
  *videos*, not agreement with human ground truth. The human-gold diagnostic set
  (16 distinct episodes, 10 tracks) is the only human-referenced measurement and
  is small and deliberately selected.
- The test split shares its video partition with Branch A, so a system-level claim
  is available; but the two branches are still measured with different metric
  families (R@k vs macro-F1) and must not be combined into one number.

## 6. Reproduction

```bash
cd LLMDet
python -m attention.thesis_eval.eval_all --root work_dirs/thesis/ladder --split test \
    --device cuda:0 --n-boot 2000
python -m attention.thesis_eval.eval_all --root work_dirs/thesis/arch   --split test \
    --device cuda:0 --n-boot 2000 --refine-asrf
python -m attention.thesis_eval.aggregate --root work_dirs/thesis/ladder --split test \
    --out work_dirs/thesis/tables
python -m attention.thesis_eval.aggregate --root work_dirs/thesis/arch   --split test \
    --out work_dirs/thesis/tables
```
