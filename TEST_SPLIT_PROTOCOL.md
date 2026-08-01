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

## Result

*(filled in immediately after the single run — see below)*
