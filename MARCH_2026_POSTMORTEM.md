# Post-mortem: why the March 2026 experiments are not usable

**Written:** 2026-07-30
**Scope:** the detector fine-tuning runs (Mar 25–29 2026), the E0/E1/E2 ablation, the
temporal attention model, and the real-time inference path.
**Verdict:** none of the March headline numbers can be cited in the thesis. The
detector runs measured the wrong thing, the ablation compared a checkpoint against
itself, the temporal model was a 2-class classifier reporting 4-class accuracy, and
the inference path could not execute at all on the current library stack.

Every claim below is tagged **[verified]** if reproduced from an on-disk artifact
during this write-up, or **[audit]** if it comes from the 2026-07-29 audit and the
raw artifact was not re-checked. Evidence paths are collected in the appendix.

---

## 0. Summary

| # | Failure | Effect on results | Severity |
|---|---|---|---|
| 1 | Train/val split leaked temporally | Headline R@1=0.613 measures memorisation | Fatal |
| 2 | Best checkpoints deleted by pruning | Headline number is unreproducible | Fatal |
| 3 | "E2" is the same file as E0 | A third of the ablation never existed | Fatal |
| 4 | E1 fine-tune degraded the model | Reported as a contribution; it is a regression | Major |
| 5 | LVIS AP silently returned 0.0000 | Fake metric row instead of an error | Major |
| 6 | 2 of 4 behaviour classes had zero samples | Model is 2-class; F1 = 0.0 for two classes | Fatal |
| 7 | Safety guard for #6 was switched off | The bug was detectable and was suppressed | Major |
| 8 | Accuracy computed as mean-of-batch-means | 0.6442 reported vs 0.6186 true | Minor |
| 9 | BERT 3D mask vs SDPA | Inference crashed before producing a frame | Fatal |
| 10 | Unreachable empty-region guard | Latent crash on legitimate inputs | Major |
| 11 | Caption→box binding was ordinal + a stub | Labels attached to the wrong students | Fatal |

Failures 1, 3, 5, 6, 7 and 10 share one property: **they made the system look like it
was working.** None of them produced an error message. That is the real lesson of
this post-mortem, and it is discussed in §6.

---

## 1. The detector numbers measured memorisation, not generalisation

### 1.1 The split leaked

The dataset is video frames sampled at 1 fps. The March split was made by shuffling
**frames**, not videos. Consecutive frames of the same student in the same seat under
the same lighting therefore landed on both sides of the split — the audit found every
validation frame had a training frame within ±2.5 seconds. **[audit]**

A model does not have to generalise to score well on that. It has to recognise a
near-duplicate of an image it was trained on. The reported **R@1 = 0.613** is
therefore an upper bound on memorisation, not a measure of grounding quality, and it
cannot appear in the thesis as a performance figure. **[audit]**

This single fact invalidates the March detector results more thoroughly than any
other item in this document. Everything downstream that was tuned against that
number was tuned against noise.

### 1.2 The checkpoint that produced it no longer exists

`max_keep_ckpts` pruning plus manual cleanup deleted iterations 37500, 40000 and
50000 — including the best one. Only `iter_22500.pth` and `iter_25000.pth` survive
in `work_dirs/grounding_dino_swin_t_student_only/`. **[verified]**

So even setting leakage aside, the headline number could not be reproduced or
re-evaluated on a clean split. There is no artifact to go back to.

**Fixed for the current runs:** `student_llmstu_exact.py` keeps all 16 checkpoints,
and the runbook requires copying the best one to `work_dirs/thesis_bundle/checkpoints/`
before any cleanup.

---

## 2. The E0/E1/E2 ablation compared a checkpoint against itself

`work_dirs/thesis_bundle/manifest.json` reads: **[verified]**

```json
{
  "e0_ckpt": "work_dirs/grounding_dino_swin_t/iter_15000.pth",
  "e1_best": "work_dirs/grounding_dino_swin_t_student_e1/iter_12000.pth",
  "e2_best": "work_dirs/grounding_dino_swin_t/iter_15000.pth",
  "ablation_root": "work_dirs/ablation_e0_e1_e2"
}
```

`e2_best` and `e0_ckpt` are **the same path**. E2 was never trained. Any table
reporting three experimental conditions is reporting two, with one duplicated —
and because it is literally the same file, "E2" would score identically to E0 and
look like a stable, reproducible result rather than an error.

Note also `ablation_root` points at `work_dirs/ablation_e0_e1_e2`, which does not
exist; the real directory is `work_dirs/ablation_e0_e1` — and its name already
records that only two arms were ever run. **[verified]**

### 2.1 E1 was a regression, not an improvement

From the two evaluation logs: **[verified]**

| arm | R@1 | R@5 | R@10 |
|---|---|---|---|
| E0 (pre-trained) | 0.1614 | **0.5038** | 0.6064 |
| E1 (fine-tuned) | 0.1614 | **0.4078** | 0.5848 |

Fine-tuning cost **9.6 points of R@5** and 2.2 of R@10, and moved R@1 by 0.0002.
E1 made the model worse. It was carried forward in the thesis bundle as a
contribution.

The flat R@1 alongside a large R@5 drop is itself diagnostic: the model's top-1
choice was unaffected while its ranking of the remaining candidates degraded —
consistent with training on labels whose caption→box binding is wrong (§5.3).

---

## 3. Why metrics came out as literal zeros

Two independent zero-valued metrics, two different causes.

### 3.1 LVIS AP = 0.0000 — a silent fallback

March runs report: **[verified]**

```
WARNING - LVISFixedAPMetric got empty `self.results`. Please ensure that the
          processed results are properly added into `self.results` in `process`
INFO    - No valid LVIS detections found for this evaluation. Returning zeroed AP metrics.
Iter(test) [100/100]  lvis_fixed_ap/AP: 0.0000  AP50: 0.0000  AP75: 0.0000 ...
```

The metric accumulated **no results at all**, and `mmdet/evaluation/metrics/lvis_metric.py:499-511`
returns a dictionary of zeros instead of raising: **[verified]**

```python
if len(new_results) == 0:
    logger.info('No valid LVIS detections found for this evaluation. '
                'Returning zeroed AP metrics.')
    return {'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0, ...}
```

This is a local modification to the metric, presumably added to stop a crash. It
converted a total evaluation failure into a well-formed metric row that flows into
logs and tables looking like a measurement. A crash would have been strictly better.

That the harness — not the model — was at fault is provable from the timeline: **[verified]**

| date | run | LVIS AP |
|---|---|---|
| 2025-06-16/19 | `grounding_dino_swin_t` | 0.4153 – 0.4198 |
| 2026-02-05/06 | `grounding_dino_swin_t` | 0.4160 |
| 2026-03-26/27 | `..._student_classroom` (5 runs) | **0.0000** |

The same evaluation scored ~0.42 in June 2025 and February 2026, then 0.0000 across
five consecutive March runs. AP does not fall to exactly zero by degradation; it
falls to exactly zero when nothing reaches the evaluator.

### 3.2 Two behaviour classes had zero training samples

`work_dirs/attention_temporal/train_summary.json`: **[verified]**

```json
{"best_val_acc": 0.6442,
 "train_class_hist": [342, 0, 0, 359],
 "val_class_hist":   [162, 0, 0, 226],
 "class_weights": [0.00583, 1.99431, 1.99431, 0.00556]}
```

Classes 1 and 2 (`distracted`, `sleeping`) have **zero samples in both train and
val**. The model is a 2-class classifier reporting a 4-class task. Per-class metrics
confirm it: `precision = [0.625, 0.0, 0.0, 0.621]`, `f1 = [0.321, 0.0, 0.0, 0.737]`.
The confusion matrix has two entirely empty rows and columns. **[verified]**

Those two classes are the *off-task behaviours the thesis exists to detect*. The
system was reporting 64% accuracy at detecting attention loss while being
structurally incapable of predicting inattention.

**Two independent root causes, both in `sequence_builder.py`:** **[verified]**

*(a) Label contamination.* Per-region labels were derived from the region phrase
concatenated with the whole-image caption and all image-level tags:

```python
def _region_to_label(region_phrase, tags, caption):
    text = " ".join([region_phrase, caption] + tags).lower()
```

Every student in a frame therefore sees the same caption and tags. The global text
dominates the keyword vote, so all students in a frame tend to collapse to whichever
class the image-level description mentions. Per-student label diversity is destroyed
before training begins.

*(b) Majority collapse.* Each temporal track was then reduced to a single label:

```python
y = int(np.bincount(np.array(labels, dtype=np.int64)).argmax())
```

A student who looks at their screen for 25 frames and at their phone for 5 becomes
"screen_oriented" — the phone episode is erased. This deletes exactly the transient
events the system is supposed to detect, and it biases hardest against the rarest
classes.

*(c) Weighting made it worse.* The class weights `[0.0058, 1.994, 1.994, 0.0056]`
assign ~340× more weight to the two classes with **no samples** than to the two that
exist. The absent classes cannot contribute gradient, so the effective loss was
scaled down by ~170× for everything that mattered.

### 3.3 The guard that would have caught this was switched off

The codebase already had `enforce_full_class_coverage`, a check that aborts training
when a class has no samples. It was disabled in an uncommitted config edit before the
March run. **[audit]**

The bug was not merely undetected — it was detectable, and the detector was turned
off. It is now re-enabled (`configs/attention_temporal.yaml: enforce_full_class_coverage: true`).

---

## 4. The accuracy figure was wrong even for the classes that did exist

`train_temporal_ddp.py` accumulated per-batch accuracies and averaged them: **[verified]**

```python
accs.append(float((pred[valid] == y[valid]).float().mean().item()))
...
return ..., float(np.mean(accs))
```

`np.mean(accs)` is a mean of batch means. It weights a final partial batch of 3
samples the same as a full batch of 32, and batches differ further because `valid`
masks out unlabelled samples. Recomputing from the saved confusion matrix:

```
correct = 240, total = 388  ->  true micro accuracy = 0.6186
reported best_val_acc                              = 0.6442
```

**[verified]** — a 2.6-point inflation, on top of the fact that the number describes
a 2-class problem.

---

## 5. Why inference did not work at all

The detector could train but could not run. Four separate blockers:

### 5.1 BERT attention mask vs SDPA (fatal)

Grounding-DINO feeds the text encoder a **3D** special-token relation mask. Newer
`transformers` releases route BERT through SDPA by default, and
`_prepare_4d_attention_mask_for_sdpa` accepts only 2D masks:

```
ValueError: too many values to unpack (expected 2)
```

Inference crashed before the first frame. This is an environment-drift failure: the
code was written against a `transformers` version whose default attention path
tolerated the 3D mask, and the installed version changed underneath it. Nothing in
the repo pinned it.

**Fixed** at `mmdet/models/language_models/bert.py:203` by forcing the eager path:

```python
self.model = HFBertModel.from_pretrained(
    name, add_pooling_layer=add_pooling_layer, config=config,
    attn_implementation='eager')
```

**[verified]** — the fix is present, and today's ARM A run exercises it successfully.

### 5.2 An unreachable empty-region guard (latent crash)

`grounding_dino.py` builds `selected_queries` per batch element, skipping any sample
with no region conversations, then guards against the empty case before concatenating.
A `torch.cat` had been hoisted **above** the guard, making it unreachable — so a batch
where every sample legitimately had zero regions crashed inside `torch.cat` instead of
taking the guarded path.

**Fixed** — the guard at `grounding_dino.py:983` now precedes the `torch.cat` at :987. **[verified]**

### 5.3 The caption→box binding was ordinal, and its "smart" alternative was a stub

This is the thesis's central technical problem, and in March it was unimplemented.

`grounding_data/QWEN3-VL/jsonl_formatter.py` parses `"Student 3: looking at phone"`
and assigns it by **ordinal position**: **[verified]**

```python
if student_num <= len(detections):
    det = detections[student_num - 1]      # Student N -> Nth detection
```

The Nth detection is in **detector-confidence order**, which has no relationship to
whatever order the captioning model implicitly used. One missed or extra detection
shifts every subsequent student by one and silently mislabels the entire frame.

A function named `_find_nearest_bbox` exists and looks like it solves this. It does not: **[verified]**

```python
def _find_nearest_bbox(self, char_pos, detections, caption):
    ...
    # Simple heuristic: use first detection or average
    # In practice, you might want more sophisticated matching
    if len(detections) == 1:
        return detections[0]["bbox"]
    # Return first detection as default
    return detections[0]["bbox"]
```

Both branches return `detections[0]`. The `char_pos` and `caption` arguments are
never used. A reader auditing the pipeline would see a plausibly-named function and
reasonably assume matching was handled.

Measured on LLMSTU's exact correspondences (72,398 frames, 261,397 assignments),
ordinal binding is **26.1% correct**; Hungarian and Sinkhorn reach **92.3%**. **[audit]**
Roughly three quarters of every training label in the March runs described the
wrong student. This is the most likely explanation for the E1 regression in §2.1.

### 5.4 Supporting breakage

- `tokens_positive` held **word indices where the loader expects character offsets**,
  so text spans did not align with the phrases they were meant to mark. Only
  `annotations/*_fixed_*.jsonl` have valid spans. **[audit]**
- 39.8% of frames in the old ODVG had **no grounded regions at all**. **[audit]**
- `grounding_dino_swin_t.py:281` pointed at a nonexistent `ann_file` — a half-finished
  edit left in the config. **[audit]**
- A stray debug `print` sat in `grounding_dino.py`'s inference path. Removed. **[audit]**

---

## 6. The common thread

The individual bugs are ordinary. The pattern is not, and it is worth stating plainly
because it is the part that generalises:

**Every one of the fatal failures produced plausible output rather than an error.**

- A leaked split yields a *higher* score, so it reads as success.
- A metric that returns zeros on total failure yields a formatted row, so it reads as a measurement.
- Zero-sample classes yield 64% accuracy on the remaining two, so they read as a working model.
- A stub named `_find_nearest_bbox` reads as implemented matching.
- E2 pointing at E0's file yields a reproducible-looking number rather than a missing-file error.

The system never said it was broken. It said `best_val_acc: 0.6442`. Consequently the
March effort optimised a pipeline whose measurements were disconnected from what it
was supposed to measure — the metrics were not merely inaccurate, they were
insensitive to the thing being studied.

Three practices follow, and they are now in place:

1. **Fail loudly.** No metric may return a zeroed result on empty input; no
   experiment packager may accept a missing or duplicated checkpoint. The
   `package_thesis_artifacts.py` script now fails hard and byte-compares E2 against
   E0 specifically to prevent §2 from recurring.
2. **Split by the unit that carries the correlation.** Video-wise, never frame-wise
   or shard-wise, for 1 fps data.
3. **Verify label semantics, not just label presence.** Class histograms are checked
   before training, and pseudo-labels are audited against a human gold set
   (93.9% field accuracy / 85.9% cue accuracy, measured 2026-07-29).

### 6.1 The same class of bug, caught today

While launching ARM A this morning, all 4 ranks died immediately:

```
IndexError: list index out of range   (text_transformers.py:320, conv = source[0])
```

Cause: `ODVGDataset` substitutes `conversations = []` for a missing key
(`mmdet/datasets/odvg.py:157`) rather than failing, and the regenerated ODVG files
lacked that field. **[verified]**

This is §6's pattern again — a silent default papering over missing data — but with
the opposite outcome: because the consumer crashed instead of coping, it surfaced in
two minutes instead of contaminating a run. **A loud failure is a feature.** Fixed by
`grounding_data/llmstu_tools/add_conversations.py`.

---

## 7. Status of each item

| # | Failure | Status |
|---|---|---|
| 1 | Split leakage | **Fixed** — video-wise 73/27/27 split, `make_splits.py` |
| 2 | Checkpoint pruning | **Fixed** — all ckpts kept; runbook mandates copy-out |
| 3 | E2 == E0 | **Fixed** — packager byte-compares and fails loudly |
| 4 | E1 regression | **Superseded** — retraining on corrected labels |
| 5 | LVIS zeroed AP | **Known** — fallback still present; do not cite LVIS numbers |
| 6 | Zero-sample classes | **Fixed** — 6-class taxonomy, per-frame labels, no majority collapse |
| 7 | Disabled guard | **Fixed** — `enforce_full_class_coverage: true` |
| 8 | Mean-of-batch-means | **Fixed** — micro-accuracy from the confusion matrix |
| 9 | SDPA mask | **Fixed** — `attn_implementation='eager'` |
| 10 | Unreachable guard | **Fixed** — guard hoisted above `torch.cat` |
| 11 | Ordinal binding | **Fixed** — Hungarian/Sinkhorn; ablation running today |

Item 5 is deliberately left as-is: the zeroing fallback is in vendored `mmdet` code,
LVIS is not a thesis metric, and changing it now would perturb a stack that is
mid-experiment. **No LVIS number from 2026 should be reported.**

### 7.1 What replaces the March numbers

| quantity | March (invalid) | replacement |
|---|---|---|
| Detector grounding | R@1 0.613 (leaked) | `student_llmstu_exact`, clean split — running |
| Matching quality | not measured | ordinal 26.1% vs Hungarian 92.3% |
| Matching → grounding | not measured | ARM A/B/C ablation — running today |
| Temporal model | 0.6442 (2 of 4 classes) | macro-F1 0.384, 6 classes, leak-free |
| Pseudo-label quality | assumed | 93.9% field / 85.9% cue vs human gold |
| Real-time capability | claimed | 7.1 FPS measured; 2.0 FPS at 30 students |

The replacement numbers are lower. They are also the first numbers in this project
that mean what they say.

---

## Appendix: evidence index

| Claim | Source |
|---|---|
| Surviving checkpoints only | `LLMDet/work_dirs/grounding_dino_swin_t_student_only/iter_{22500,25000}.pth` |
| E2 == E0 | `LLMDet/work_dirs/thesis_bundle/manifest.json` |
| E0 metrics | `work_dirs/ablation_e0_e1/e0/20260325_193738/20260325_193738.log` |
| E1 metrics | `work_dirs/ablation_e0_e1/e1/20260325_200039/20260325_200039.log` |
| LVIS 0.0000 | `work_dirs/grounding_dino_swin_t_student_classroom/20260327_123711/...log:2098-2100` (+4 more) |
| LVIS zeroing code | `LLMDet/mmdet/evaluation/metrics/lvis_metric.py:499-511` |
| LVIS 0.4197 (working) | `work_dirs/grounding_dino_swin_t/20250617_111421/...log` |
| LVIS 0.4160 (Feb 2026) | `work_dirs/grounding_dino_swin_t/20260206_224223/...log` |
| Class histogram | `work_dirs/attention_temporal/train_summary.json` |
| Per-class F1 = 0 | `work_dirs/attention_temporal/checkpoints/best_val_class_metrics.json` |
| Confusion matrix | `work_dirs/attention_temporal/checkpoints/best_val_confusion_matrix.npy` |
| Label contamination | `git show 02fd4c9^:LLMDet/attention/sequence_builder.py` (`_region_to_label`) |
| Majority collapse | same file, `y = np.bincount(labels).argmax()` |
| Mean-of-batch-means | `git show 02fd4c9^:LLMDet/attention/train_temporal_ddp.py:159,179` |
| Ordinal binding | `grounding_data/QWEN3-VL/jsonl_formatter.py:67-73` |
| `_find_nearest_bbox` stub | `grounding_data/QWEN3-VL/jsonl_formatter.py:148-170` |
| SDPA fix | `LLMDet/mmdet/models/language_models/bert.py:200-203` |
| Empty-region guard | `LLMDet/mmdet/models/detectors/grounding_dino.py:983-987` |
| `conversations` default | `LLMDet/mmdet/datasets/odvg.py:157` |
| Gold-set audit | `grounding_data/llmstu_tools/outputs/gold_eval_summary.md` |
