# Student-unit caption↔box matching (Branch A novelty)

Standalone package (numpy/scipy/torch/transformers; no mmdet dependency)
implementing and evaluating strategies for assigning per-student caption
units to detected person boxes:

| module | role |
|---|---|
| `student_unit_matcher.py` | unit parsing + `ordinal`, `ordinal_lr`, `hungarian`, `sinkhorn` (log-domain, dustbin-padded for unbalanced N≠M) |
| `compatibility.py` | cost components: CLIP ViT-B/32 crop↔text similarity, reading-order spatial prior, detector confidence, format validity; per-assignment confidence components for calibration |
| `evaluate_matching.py` | exact-GT evaluation harness (assignment accuracy, wrong-assignment rate, crowding breakdown, IoU@0.5 region correctness) |
| `calibration.py` | logistic/isotonic calibration of assignment confidence; AUROC/AUPRC/ECE/Brier |
| `run_matching_experiment.py` | CLI comparing all four strategies |

## Ground truth

`evaluate_matching.load_gt` reads `grounding_data/llmstu_tools/outputs/correspondence_gt.jsonl`
(one record per src_frame: `{"src_frame", "students": [{bbox_person, det_conf,
person_idx, crop_file, caption, labels}]}`), produced by the W1 data tooling.
Until that file exists, an equivalent stand-in is derived directly from
`grounding_data/LLMSTU/labels/shard_000.jsonl` — the correspondence is exact
either way because each LLMSTU row *is* one student crop. Pass `--gt` to swap
in the real file; no code change needed.

## Experiment design

Boxes are presented in detector-confidence order (what the real pipeline
emits — the audit showed there is **no** spatial sort). The VLM's "Student
1..N" numbering order is unknown, so `--unit-order {lr,conf}` simulates the
two hypotheses. Each hypothesis makes its matching ordinal baseline perfect
*by construction* (lr → `ordinal_lr` = 1.0, conf → `ordinal` = 1.0); those
cells are upper bounds, not results. The meaningful number is the
**order-invariant accuracy of content-based matching** (hungarian/sinkhorn),
which needs no assumption about how the VLM numbers students.

## Preliminary results (500 frames, shard_000 stand-in GT, CPU CLIP)

Assignment accuracy (1,846 unit↔box decisions, frames with ≥2 students):

| strategy | overall | n=2 | n=3 | n≥4 | wrong-rate |
|---|---|---|---|---|---|
| ordinal (pipeline baseline) | **0.265** | 0.426 | 0.383 | 0.210 | 0.735 |
| hungarian (CLIP only) | 0.815 | 0.948 | 0.921 | 0.767 | 0.185 |
| sinkhorn (CLIP only) | 0.817 | 0.948 | 0.921 | 0.771 | 0.183 |
| hungarian (CLIP + 0.5·spatial) | **0.940** | 1.000 | 0.980 | 0.920 | 0.060 |

Read: if the VLM numbers students in reading order while the pipeline binds
"Student N" to the Nth confidence-ordered detection (today's behavior), ~73%
of pseudo-label regions land on the wrong student. Content-based Hungarian
matching alone recovers 81.5%; adding a mild reading-order prior reaches 94%.
Sinkhorn ≈ Hungarian here (assignment-level agreement), as expected when the
plan is near-integral; its value is the soft plan for confidence estimation.

**Caveats:** (1) stand-in GT uses LLMSTU captions as unit texts — richer and
more discriminative than the old "action, emotion" units, so 0.815 is an
optimistic ceiling for matching *old* captions; rerun against real old-caption
units on the gold set. (2) In the synthetic-lr protocol the `spatial`
calibration feature is an *oracle* (correct ⇔ rank agreement) — calibration
AUROC 1.0 with it is leakage; without it (clip_sim_z, margin, n_boxes):
AUROC 0.76, AUPRC 0.93, ECE 0.02 after isotonic. Real calibration must be fit
on the gold calibration split.

## Scaling up (needs approval — GPU)

CPU throughput is ~25 crops/s (~45 s / 500 frames). Full 93,910-frame sweep:
pass `--device cuda --frames 999999`, ~15 min on one A100 (CLIP ViT-B/32,
batch 256). Refinement of the actual pseudo-label JSONL (re-assigning regions
in the old ODVG file via hungarian) should be added as a small driver reusing
`compatibility.build_cost_matrix` once W1's video-wise splits exist.
