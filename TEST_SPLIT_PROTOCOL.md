# Test-split evaluation — pre-registration and result

Binding protocol from `THESIS_PLAN.md` P0.6: the test split is used **exactly
once**, after all decisions are frozen. This file was written **before** the run.

## Freeze state

| item | value |
|---|---|
| git commit | `7c563f9` |
| date | 2026-08-01 |
| detector checkpoint | `work_dirs/thesis_bundle/checkpoints/main_llmstu_exact_iter25000_final.pth` |
| detector config | `configs/student_llmstu_exact.py` |
| test annotations | `outputs/odvg_test.jsonl` — 9,405 frames, 27 videos, **never previously read** |

## Pre-registered — what will be reported

**Detector grounding on the held-out test split**: R@1, R@5, R@10, R@-1 via
`Flickr30kMetric` at IoU 0.5, identical protocol to the validation runs.
Whatever comes out is reported, including if it is worse than validation.

## Pre-registered — what will NOT be claimed, and why

**No temporal-model test number.** The temporal sequences were split
independently of the detector's train/val/test division:

| | overlap with the 27 detector-test videos |
|---|---|
| temporal **train** videos (102) | **23** |
| temporal val videos (25) | 4 |

Evaluating the cue model on those videos would score it on data it trained on.
The temporal model's own 102/25 split IS video-wise and leak-free within itself,
so **macro-F1 0.4098 remains a valid held-out VALIDATION number** — it is simply
not a test number, and must be labelled as validation in the write-up.

Closing this properly would mean rebuilding the sequences under the detector's
video split and retraining (~2.5 h). Recorded as a known limitation rather than
silently reporting a validation figure as a test figure.

## Result — single run, 2026-08-01 07:58 UTC

`Flickr30kMetric`, IoU 0.5, 9,405 frames / 27 held-out videos:

| metric | validation | **TEST (held-out)** | Δ |
|---|---|---|---|
| **R@1** | 0.6343 | **0.6462** | **+0.0119** |
| R@5 | 0.9808 | **0.9891** | +0.0083 |
| R@10 | 0.9963 | **0.9982** | +0.0019 |
| R@-1 | 0.9994 | 0.9997 | +0.0003 |

→ `LLMDet/work_dirs/logs/FINAL_test_split.log`, config
`LLMDet/configs/eval_test_split.py`

### Reading this

**Test slightly EXCEEDS validation.** That is the healthiest outcome available:

1. **No validation overfitting.** Every threshold, prompt and configuration
   decision in this project was made against the validation split. If those
   choices had been fitted to validation noise, test would sit below it. It does
   not.
2. **Independent confirmation the split is leak-free.** Two disjoint video sets
   agree to within ~1 point. Compare March, where the leaked split produced
   0.6103 that no clean set could reproduce.
3. The +1.2 point gap is ordinary between-split variance over 27 videos, not
   evidence the test set is easier. It should not be presented as an improvement.

**CITE 0.6462 as the final test result** and 0.6343 as validation. The protocol
is now closed: the test split has been used, and any further tuning would
invalidate this number. Re-using it would make it a second validation set.

### Not claimed

No temporal-model test figure, for the reason pre-registered above — 23 of these
27 videos are in the cue model's training set. Its macro-F1 **0.4098 remains a
validation number** and is labelled as such throughout.
