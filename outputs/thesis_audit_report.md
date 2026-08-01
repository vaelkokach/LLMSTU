# Phase 0 — Repository and evaluation audit

**Date:** 2026-08-01 · **Repo state at audit:** commit `bb8fe0c`, clean working tree
**Scope:** everything the thesis intends to cite for Branch A (grounding) and Branch B
(visible-cue + event layers), audited against `Thesis_Topic.md`, `FINDINGS.md` and
`Thesis_Research_and_Metrics_Addendum.md`.

Verification commands are given inline. Every claim below was checked against
repository evidence; where a document and the repository disagree, the repository is
reported and the conflict is stated rather than resolved silently.

---

## 0. Headline conclusions

| # | Finding | Severity | Status |
|---|---|---|---|
| A0-1 | The three temporal datasets (552/556/570) are **bit-identical** in their shared columns, labels, timestamps and split. The historic ablation was genuinely controlled. | ✅ good news | verified |
| A0-2 | **Branch B has no test split at all.** The sequence builder draws its own 80/20 shuffle, independent of `splits.json`. 23 of the detector's 27 test videos sit in the cue model's training set. | 🔴 blocking | fixed (re-partition) |
| A0-3 | `attention/events.py` maps **`inactivity` onto the same cue as `head_down`**, so every head-down episode is emitted twice. The "24 human-gold episodes" contain 7 duplicates plus 1 duplicated `return_to_task` marker → **16 distinct episodes**. | 🔴 material | documented + de-duplicating evaluator built |
| A0-4 | The DDP trainer computes validation metrics on **rank 0's shard only** — no all-gather — and selects `best.pth` on that shard metric. | 🟠 known, confirmed | superseded by single-process trainer |
| A0-5 | Checkpoint selection took the **argmax of a metric that swings ±0.05 between adjacent epochs**, over 90 epochs. The reported "best" values carry a selection-on-noise upward bias. | 🟠 new | fixed (smoothed selection rule) |
| A0-6 | The 128-vs-127 video discrepancy is fully explained: `video_0162_…` has 1,358 recovered frames but **zero LLMSTU label records**. It is the unique zero-coverage video. | ✅ resolved | documented |
| A0-7 | The pseudo-label model identity **cannot be resolved from repository evidence**. Label records carry `model_confidence` but no model-name field. | 🟠 unresolvable | documented as uncertainty |
| A0-8 | The script that produced the cited single-process numbers is `attention/eval_baseline_chain.py` (its output is byte-identical to `baseline_chain_eval.json`). It computes only accuracy / macro-F1 / per-class F1 — **no balanced accuracy, AUPRC, calibration, confusion matrix or CIs**. | 🟠 gap | superseded by `attention/thesis_eval` |
| A0-9 | `FINDINGS.md §9 "Open items"` lists four items as blocking that are **complete**, and lists the dashboard as "dropped from scope" while §6d.3 reports it built and verified. | 🟡 stale | patch proposed |

---

## 1. Checkpoints and datasets — the ablation ladder

### 1.1 Authoritative checkpoint locations

| model | dims | checkpoint | selection metric | trained |
|---|---|---|---|---|
| base | 552 | `LLMDet/work_dirs/attention_temporal_v2/checkpoints/best.pth` | rank-0-shard macro-F1 0.40957 | 2026-07-29 |
| +head pose | 556 | `LLMDet/work_dirs/attention_temporal_hp/checkpoints/best.pth` | rank-0-shard macro-F1 0.43640 | 2026-07-31 |
| +expression +dynamic | 570 | `LLMDet/work_dirs/attention_temporal_full/checkpoints/best.pth` | rank-0-shard macro-F1 0.43995 | 2026-08-01 |

Each directory also holds all 90 epoch checkpoints (~14 GB per model, ~42 GB total).
Nothing was deleted during this audit.

Branch A (detector), byte-verified per `FINDINGS.md §3.4b`:
`LLMDet/work_dirs/thesis_bundle/checkpoints/main_llmstu_exact_iter25000_final.pth`
(+ `_iter22500_bestR1.pth`, and the three ablation arms).

### 1.2 Sequence datasets — identity verification (Phase 0 items 2–4)

```
grounding_data/llmstu_sequences        552-dim  built 2026-07-29
grounding_data/llmstu_sequences_hp     556-dim  built 2026-07-31
grounding_data/llmstu_sequences_full   570-dim  built 2026-08-01
```

All three verified as follows.

| check | result |
|---|---|
| number of sequences | 6,531 / 6,531 / 6,531 |
| total frames | 271,485 in all three |
| val frames | 60,811 in all three |
| video split (`meta.json["split"]`) hash | identical (`a1b129bcbc67`) — 102 train / 25 val videos |
| build seed | 42 in all three |
| first 552 feature columns | **max abs difference 0.0** vs the 552-dim build, on every sampled sequence |
| columns 552:556 (head pose) | **max abs difference 0.0** between the 556- and 570-dim builds |
| per-frame labels after the legacy remap | **exactly equal** across all three |
| timestamps `t` | **exactly equal** across all three |
| train class histogram | `[160764, 13416, 10893, 5489, 10557, 9555]` in all three `train_summary.json` |
| val class histogram | `[46502, 4213, 3293, 1323, 2595, 2885]` in all three |

The only `meta.json` difference is `label_majority` on 299 of 6,531 rows: the 552-dim
build predates the 7→6-class taxonomy reduction and stores `uncertain` as id 6, the
later builds store it as id 5. Both loaders apply `LEGACY_7CLASS_REMAP`, and the
identical class histograms above confirm the two vintages are label-equivalent.

**Conclusion:** the 552→556→570 comparison was a genuinely controlled single-variable
ablation. This is a real strength and should be stated as such.

### 1.3 Dense vs deduplicated records (Phase 0 item 4)

Confirmed. `sequence_builder.py` is invoked on `labels_tracked.jsonl` (283,913 dense
records), not `labels_dedup.jsonl` (84,950). Evidence: 6,531 sequences with uniform
1.0 s sampling, matching the `FINDINGS 6.0b` record of a discarded first 556-dim build
that used dedup and produced 3,913 sequences at 4.3 s median gap. Dedup is correct for
the detector (it prevents near-duplicate frames inflating a split) and wrong for a
sequence model (it destroys temporal regularity).

### 1.4 Feature column layout (enables the isolated ablations)

Established by reading `sequence_builder.build_sequences_llmstu` and
`precompute_affect.py`, and confirmed numerically:

```
[  0:552]  base      CLIP(512) + bbox geometry(8) + colour(24) + posture(8)
[552:556]  headpose  yaw, pitch, roll, face_found
[556:563]  express   7 basic-expression probabilities (ViT FER over the MediaPipe face box)
[563:570]  dynamic   motion_now/mean/std, scale_change, gaze_dev_yaw/pitch/mag
```

**Consequence — the addendum's §5 P0.6 ladder needs no feature re-extraction.** The
563-expression and 563-dynamic rungs the project has never run are column slices of the
existing 570-dim arrays. This removes the main cost objection to the isolated ablation
*and* removes any possibility of extraction drift confounding it.

---

## 2. The split defect (A0-2) — Branch B has no test set

`attention/sequence_builder.py:213` defines its own splitter:

```python
def split_videos(video_ids, val_fraction, seed):
    vids = sorted(video_ids); random.Random(seed).shuffle(vids)
    n_val = max(1, int(round(len(vids) * val_fraction)))
    return set(vids[n_val:]), set(vids[:n_val])
```

It never reads `grounding_data/llmstu_tools/outputs/splits.json`. So:

* the detector uses a **73 / 27 / 27** video split with a reserved test set;
* the cue model uses an independent **102 / 25** split with **no test set**;
* 23 of the detector's 27 test videos are in the cue model's *training* set.

`FINDINGS.md §3.4c` states this correctly and estimates a rebuild at ~2.5 h. The audit
finds it is far cheaper than that: the split is a property of `video_id`, and every
sequence's `video_id` is recorded in `meta.json`, so the datasets can be **re-partitioned
by metadata alone** with no feature extraction at all.

Produced: `grounding_data/llmstu_seq_split_manifest.json`

| split | videos | sequences | frames |
|---|---|---|---|
| train | 73 | 4,390 | 185,050 |
| val | 27 | 1,085 | 42,702 |
| test | 27 | 1,056 | 43,733 |

Zero video overlap between splits. Both branches now share one split, so a system-level
claim ("detector → cue model → events, all evaluated on the same held-out videos") is
finally available.

**The three archived checkpoints cannot be evaluated on this test split** — they were
trained on 23 of its 27 videos. Retraining is mandatory, and is cheap (minutes per run).

---

## 3. Event-layer defects

### 3.1 `inactivity` is an alias of `head_down` (A0-3)

`attention/events.py:26-35`:

```python
"head_down":  {CUE_TO_ID["head_down"]},
"inactivity": {CUE_TO_ID["head_down"]},   # identical cue set
```

Both channels segment the *same* boolean series with the same parameters, so they emit
identical episodes. Confirmed in the gold file:

```
$ python - <<< 'Counter(channel for e in gold_events.jsonl)'
head_down 7 · inactivity 7 · phone_use 5 · return_to_task 3 · peer_interaction 2   (= 24)
```

and the 7 `inactivity` episodes have byte-identical `(t_start, t_end)` to the 7
`head_down` ones. Additionally `('video_0127…', seat 4)` carries
`return_to_task (51.9, 51.9)` **twice**.

Distinct gold episodes: **7 head_down + 5 phone_use + 2 return_to_task + 2
peer_interaction = 16**, not 24.

Effects on published numbers:
* `head_down` is double-weighted in every aggregate event count and in false-alerts/hour;
* the duplicated zero-length marker is **unmatchable by construction** — greedy matching
  is one-to-one, so one of the pair is a guaranteed miss for every system;
* recall *ratios* (6/24, 10/24, 8/24, 18/24) are not wildly wrong because the duplication
  inflates numerator and denominator roughly together, but they are not clean.

The new evaluator reports both `gold_raw` (24, historic continuity) and `gold_dedup`
(16, honest denominator).

### 3.2 The `inactivity` channel has no distinct signal

`events.py` documents this: "a dedicated inactivity signal needs motion features
(future work)". The project now *has* motion features (the `dynamic` block, columns
563:570). This is a concrete, cheap improvement, but it changes the event taxonomy and
is therefore listed as remaining work rather than executed inside a controlled ablation.

---

## 4. Evaluator audit (Phase 0 item 5)

Three harnesses have produced numbers that entered `FINDINGS.md`.

| harness | process model | per-frame? | padding | aggregation | verdict |
|---|---|---|---|---|---|
| `train_temporal_ddp.validate()` | DDP, **per-rank shard** | yes | mask, correct | confusion matrix over the shard | ⛔ not citable as an absolute |
| `eval_baseline_chain.py` | single process, batch = 1 | yes (`per_frame=True` passed) | none needed | confusion-free direct comparison over all frames | ✅ correct but thin |
| `eval_events_vs_human.py` | single process, batch = 1 | yes, **raises** if `per_frame` false | none needed | per-track | ✅ correct after its two 2026-07-31 fixes |

Specific checks:

* **Class ordering** — all three import `CUE_CLASSES` from `taxonomy.py`; none infers an
  order from data. ✅
* **Batch-weighted averaging** — none of the three averages per-batch metrics; all
  concatenate first. ✅ (The March defect is genuinely gone.)
* **Distributed reduction** — `validate()` has no `all_gather`/`all_reduce`. Its
  `macro_f1` is rank 0's shard. With `DistributedSampler` over 1,452 val sequences on
  4 ranks that is ~363 sequences. ⛔ Confirms `FINDINGS §6f`.
* **Feature dimensions** — `eval_events_vs_human.py` validates width against
  `input_dim` and exits rather than silently truncating. ✅
* **Sequence padding and masks** — the single-process evaluators use batch size 1, so
  they never exercise the mask. The mask path was therefore **untested** until now;
  `attention/tests/test_thesis_eval.py` adds padding-invariance and batch-size-invariance
  tests for all three architectures.
* **`macro_f1` present-only averaging** — `eval_baseline_chain.macro_f1` skips classes
  with zero support. All 6 classes are present in val, so it is equivalent here, but the
  behaviour differs from the trainer's version on any subset. Both are now reimplemented
  once, in `thesis_eval/metrics.py`, with the behaviour under test.

**Provenance note.** `attention_temporal_v2/single_process_eval.json` is byte-identical
to `baseline_chain_eval.json` (`diff` returns 0), which identifies the generating script
— it was not recorded anywhere. The command is now recorded in this report and the new
evaluator writes its own `command.txt` beside every result.

### 4.1 What the cited single-process numbers do and do not contain

`single_process_eval.json` contains accuracy, macro-F1 and per-class F1 — and an
`event_layer_model_vs_pseudolabel` block that is **model vs teacher**, superseded by the
model-vs-human evaluation and not to be cited. It contains **no** balanced accuracy, no
AUPRC, no AUROC, no confusion matrix, no ECE/Brier/NLL, no confidence intervals and no
prediction archive. Every one of those is required by the addendum and none could be
recomputed without re-running inference.

---

## 5. Selection-on-noise (A0-5)

From `work_dirs/attention_temporal_v2/train_6class_90ep.log`, adjacent epochs:

```
[epoch 85] val_macro_f1=0.3492
[epoch 86] val_macro_f1=0.3826
[epoch 87] val_macro_f1=0.3930
[epoch 88] val_macro_f1=0.3933
[epoch 89] val_macro_f1=0.3433
```

Swings of ±0.05 between adjacent epochs, against a 556-vs-570 single-process gap of
**0.0018**. Taking `argmax` over 90 such draws selects substantially on noise and biases
the reported maximum upward. This applies identically to all three models, so the
*ordering* is not obviously corrupted, but no difference of this size is interpretable
without repeated seeds and interval estimates — which is precisely the addendum's P0.3
and P1.13.

The new trainer selects on the mean macro-F1 over a trailing 5-epoch window and records
both the selected and final checkpoints.

---

## 6. The 128 / 127 video discrepancy (Phase 0 item 6)

Resolved definitively.

| stage | count |
|---|---|
| appearance chains recovered (`video_recovery_report.json`) | 128 |
| frames with a recovered video id (`frame_to_video.json`) | 104,069 |
| distinct source frames present in `LLMSTU/labels_slim.jsonl` | 93,910 |
| frames with a video id but **no** LLMSTU label record | 10,159 |
| videos in `splits.json` and in every sequence dataset | 127 |

**Excluded video: `video_0162_0_10_20251026030716_20251026032847`** — 1,358 recovered
frames, **0** label records. It is the unique video with zero label coverage (next
lowest is `video_0131…` at 27.4%; median coverage 99.7%). With no label records it never
enters `labels_tracked.jsonl`, hence never `labels_dedup.jsonl`, `splits.json` or any
sequence build. The exclusion is consistent across the whole pipeline and is not a split
bug.

Why LLMSTU produced no labels for it is not recoverable from the repository (the
captioner's own logs are not retained). State it as: *one of 128 recovered videos
carries no pseudo-labels and is excluded upstream of all splits; 127 videos are used.*

---

## 7. Pseudo-label model provenance (Phase 0 item 7)

**Unresolvable from repository evidence.** Checked:

* `LLMSTU/labels_slim.jsonl` records carry `model_confidence` but **no model-name field**;
* `grounding_data/QWEN3-VL/` and `grounding_data/stu_img/Qwen3-VL_Student_action_emotion.jsonl`
  belong to the *older whole-frame* pipeline, which is a different corpus;
* no generation script, config or log for the LLMSTU captioning run is in the repository.

`FINDINGS.md §2.1` already flags this and attributes the `Qwen3-VL` references to the
older pipeline. The audit confirms the flag and cannot strengthen it.

**Required thesis wording:** the pseudo-labels were generated by a Qwen3-VL-family
vision-language model; the exact point release is not recorded in the pipeline outputs
and is reported as a reproducibility limitation. Do **not** assert "Qwen3.5-VL" as a
verified fact in the methods section without an external record.

---

## 8. Stale sections of `FINDINGS.md` (Phase 0 item 8)

| location | problem | proposed correction |
|---|---|---|
| §9 item 1 | "Re-run the matching full sweep with `spatial_weight = 0.0` … until then the citable matching number is 81.5%" | **Done** (§3.2, 2026-07-31). Citable number is **0.7896**. The "81.5%" here contradicts both 92.3% and 79.0% elsewhere and should be deleted. |
| §9 item 3 | "Main run (40k iters) — the headline grounding number" | **Done** at 25k iters (§3.4b), and superseded by the test-split result (§3.4c). |
| §9 item 4 | "Model-vs-human event metrics on the 984 dense crops" | **Done** (§6.0). |
| §9 "Dropped from scope — Live dashboard" | Contradicts §6d.3, which reports the dashboard built and end-to-end verified on real video. | Delete; the dashboard is restored. |
| §1 status board, "Temporal model" | Labelled without a split. It is a **validation** number on a split that has no test set. | Label explicitly as validation, on the builder's own 102/25 split. |
| §5.3 / §6.0 "the ceiling" | The teacher is called "the ceiling" / "theoretical ceiling". | Use *empirical teacher benchmark*. 18/24 is what one particular pseudo-labeller achieved, not a bound on what any model could achieve. |
| §6.0a note 3 | Correctly warns the 1.6→10.4 s onset change is not like-for-like, but no common-subset number was ever computed. | Now computable; see Phase 4. |
| §7 | "Real-time performance: 7.1 FPS" with no latency percentiles. | Needs p50/p90/p95/p99 and a student-count curve before "real-time" is claimed. |
| §9a.3 | Says mediapipe is "implemented, **not installed**" | mediapipe 1.0.0 **is** installed and is the backend that produced `head_pose_cache.npz`. §6.0b supersedes it. |

Two numbers that must **never** be restored: the contaminated **92.3%** caption–box
matching value (`spatial_weight=0.5` leaked the ground-truth ordering) and any DDP
shard-averaged macro-F1 (0.4096 / 0.4364 / 0.4400) as an absolute.

---

## 9. Conflicts between the source documents

| conflict | resolution |
|---|---|
| Addendum §4.2 says "the current 570-dim configuration has the best reported DDP macro-F1"; `FINDINGS §6f` shows that single-process, **556 ≥ 570**. | The repository (single-process re-measure) wins. The addendum's own P0.1 asks for exactly this correction, so there is no real disagreement — but the addendum's §4.2 framing must not be carried into the thesis. |
| Addendum §5 P0.6 proposes "563 base + head pose + expression" and "563 base + head pose + dynamic". | Matches the verified column layout exactly; implementable as slices. Adopted. |
| Addendum §2.2 recommends benchmarking DirectMHP/6DRepNet/UniGaze. | Feasible in principle, but only **190 GB** of disk remain and the environment is a live thesis training environment. Scoped down; see the limitations section of the final report. |
| Addendum §2.3 recommends downloading CMOSE / DAiSEE / EngageWild. | Disk budget makes a multi-dataset download imprudent, and DIPSER is 780 GB (`FINDINGS` changelog 2026-07-31). Treated as a literature-positioning table (Table D), not new training. |
| `Thesis_Topic.md` names "boredom, perplexity, curiosity". | `FINDINGS §6d.1` measures AUROC 0.544 for boredom from basic expressions. The measurement stands; the thesis must report the limitation, not the aspiration. |

---

## 10. Obsolete or invalid outputs (do not cite)

| artifact | reason |
|---|---|
| `LLMDet/work_dirs/matching/results*sw0.5*` (92.3%) | ground-truth leak via the spatial prior |
| any `best_val_macro_f1` in `attention_temporal*/train_summary.json` | rank-0-shard metric |
| `attention_temporal*/single_process_eval.json` → `event_layer_model_vs_pseudolabel` | model-vs-teacher events; superseded by model-vs-human |
| `attention_temporal_v2/realtime_*.mp4`, `attention_temporal/checkpoints/*` (March) | pre-rebuild pipeline; see `MARCH_2026_POSTMORTEM.md` |
| `attention_temporal_hp/dipser_correlation_smoke.json` (ρ = −0.023) | measures domain shift, not the cue↔engagement link; `FINDINGS §6c` already says so |
| SCB zero-shot precision 0.187 / R@1 0.0322 | uninterpretable: SCB annotates only behaviour-exhibiting students |
| the three archived 552/556/570 checkpoints **on the new test split** | trained on 23 of its 27 videos |

---

## 11. Implementation map

| # | Recommendation (addendum) | Repo status found | Work required | Data | Compute | Scientific value | Confounding risk | Artifact | Priority |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Unified single-process evaluator | partial (`eval_baseline_chain.py`: 3 metrics, no archive) | new package | existing | CPU/1 GPU, minutes | **high** — every Table A number | low | `attention/thesis_eval/` | P0 |
| 2 | Balanced acc, AUPRC, AUROC, ECE, Brier, NLL, CM | absent | new | existing | negligible | high | low | `metrics.json` | P0 |
| 3 | Video/track cluster bootstrap CIs | absent | new | existing | CPU minutes | **high** — the 0.0018 gap is uninterpretable without it | low | `bootstrap` block | P0 |
| 4 | Branch-B test split | **absent entirely** | metadata re-partition + retrain | existing | ~15 min/model | **highest** — no defensible test number exists otherwise | none (video-wise) | `llmstu_seq_split_manifest.json` | P0 |
| 5 | Segmental F1@10/25/50, edit, event P/R/F1@tIoU, onset/offset/duration MAE, delay, FA/h | only match count, onset, duration, FA/h | new module | gold cache | CPU | high | low | `segmentation.py` | P0 |
| 6 | Common matched-subset boundary errors | absent (flagged in FINDINGS) | new | gold cache | CPU | high — resolves the 1.6→10.4 s artefact | low | `common_matched_subset` | P0 |
| 7 | Isolated 563-expr / 563-dyn ablations | never run | column slice + train | **none new** | ~15 min × 6 runs | **high** — the 556→570 claim is confounded today | none | ladder sweep | P0 |
| 8 | ≥3 seeds for small deltas | 1 seed | sweep | none new | 15 runs | high | none | `seed_summary` | P0 |
| 9 | MS-TCN baseline | absent | new model | none new | ~20 min × 3 | high | none | `models.MSTCN` | P1 |
| 10 | ASRF boundary regression | absent | new model | none new | ~20 min × 3 | **highest of the model work** — targets the measured failure | none | `models.ASRF` | P1 |
| 11 | ASFormer | absent | new model | none new | ~30 min × 3 | medium (MS-TCN/ASRF cover the claim) | none | — | P2 |
| 12 | Temperature scaling + abstention | absent | new | prediction archive | CPU seconds | high — dashboard-facing | low | `calibrate.py` | P1 |
| 13 | DirectMHP / 6DRepNet / UniGaze benchmark | MediaPipe only | new envs + weights | new downloads | GPU + **disk risk** | medium | env risk to a live training environment | — | P2, scoped down |
| 14 | CMOSE / DAiSEE / EngageWild training | absent | new pipelines | 10s–100s of GB | high | medium; separate task | **must not merge taxonomies** | Table D | P2, documentation only |
| 15 | Runtime p50/p90/p95/p99 + student-count curve | FPS only | extend profiler | none | GPU minutes | medium–high | low | Table E | P1 |
| 16 | Results register | absent | new | all outputs | none | high | none | `FINAL_RESULTS_REGISTER.{json,md}` | P0 |

---

## 12. Reproduction of this audit

```bash
cd /home/jovyan/Computer_vision

# 1.2 dataset identity
python - <<'PY'
import json, numpy as np
from pathlib import Path
lut = np.arange(7); lut[6] = 5
b, h, f = (Path(f"grounding_data/llmstu_sequences{s}") for s in ("", "_hp", "_full"))
for fn in sorted(p.name for p in (b/"val").glob("*.npz"))[:8]:
    A, B, C = (np.load(d/"val"/fn) for d in (b, h, f))
    assert np.abs(A["x"] - C["x"][:, :552]).max() == 0
    assert np.abs(B["x"][:, 552:556] - C["x"][:, 552:556]).max() == 0
    assert np.array_equal(lut[np.clip(A["y_frames"], 0, 6)], lut[np.clip(C["y_frames"], 0, 6)])
    assert np.array_equal(A["t"], C["t"])
print("identical")
PY

# 6 excluded video
python - <<'PY'
import json
from collections import Counter
ftv = json.load(open("grounding_data/llmstu_tools/outputs/frame_to_video.json"))
labelled = {json.loads(l)["src_frame"] for l in open("grounding_data/LLMSTU/labels_slim.jsonl")}
tot, lab = Counter(ftv.values()), Counter(ftv[s] for s in labelled if s in ftv)
print(sorted((lab.get(v, 0) / tot[v], v) for v in tot)[:2])
PY

# 3.1 duplicate gold channels
python - <<'PY'
import json
from collections import Counter
g = [json.loads(l) for l in open("grounding_data/llmstu_tools/outputs/gold_events.jsonl")]
print(Counter(e["channel"] for e in g))
print(len({(e["channel"], e["t_start"], e["t_end"], e["video_id"], e["seat_id"]) for e in g}), "distinct")
PY

# 4 harness provenance
diff LLMDet/work_dirs/attention_temporal_v2/{baseline_chain_eval,single_process_eval}.json && echo identical

# new split manifest + evaluator tests
cd LLMDet && python -m pytest attention/tests/test_thesis_eval.py -q
```
