# Comparing every trained model on a single image

How to run all detector variants on one frame and read the result. Written
2026-07-31; every number below was measured, not estimated.

---

## 0. Setup

```bash
cd /home/jovyan/Computer_vision/LLMDet
export PYTHONPATH=/home/jovyan/Computer_vision/LLMDet
```

`PYTHONPATH` is mandatory — `mmdet` is a *local* package here, not an installed
one. Without it: `ModuleNotFoundError: No module named 'mmdet'`.

Use `--device cpu` if a training job holds the GPUs (~5–15 s per model per
prompt); `cuda:0` is ~20× faster.

---

## 1. The models

One best checkpoint per variant, all in
`work_dirs/thesis_bundle/checkpoints/` (4.0–4.3 GB each). All share the
Grounding-DINO Swin-T architecture, so one config
(`configs/student_llmstu_exact.py`) constructs every one — only weights differ.

| # | variant | checkpoint | val R@1 | what it is |
|---|---|---|---|---|
| 1 | **E0 base** | `e0_iter15000.pth` | — | original MM-Grounding-DINO, never fine-tuned by us |
| 2 | **E1** | `e1_best.pth` | — | ⚠️ a **regression** — cost 9.6 pts R@5 vs its own baseline |
| 3 | ~~E2~~ | `e2_best.pth` | — | ❌ **byte-identical to E0** (MD5 verified). Never trained. Excluded by default |
| 4 | **ARM A** ordinal | `arm_a_ordinal_iter10000.pth` | 0.3230 | ~72% of training labels bound to the wrong student |
| 5 | **ARM B** hungarian | `arm_b_hungarian_iter10000.pth` | 0.4954 | **+17.2 R@1 over ARM A** — the ablation's headline |
| 6 | **ARM C** hungarian+p90 | `arm_c_hungarian_p90_iter10000.pth` | 0.4787 | 40% fewer, higher-confidence pairs — a negative result |
| 7 | **MAIN** ⭐ | `main_llmstu_exact_iter25000_final.pth` | **0.6343** | exact correspondences — **the citable model** |

Two **temporal** models also exist. They are *not* image detectors — they consume
556-dim feature sequences, not pixels, and cannot be run with these commands:

| model | checkpoint | macro-F1 |
|---|---|---|
| 552-dim baseline | `work_dirs/attention_temporal_v2/checkpoints/best.pth` | 0.4096 |
| 556-dim + head pose | `work_dirs/attention_temporal_hp/checkpoints/best.pth` | 0.4364 |

---

## 2. Run all models on one image

```bash
IMG=../Computer_vision/LLMDet/test2.jpg

python tools/compare_all_models.py \
    --image "$IMG" \
    --prompts "reading" "smiling" "crying" \
    --score-thr 0.15 \
    --device cuda:0 \
    --gt-jsonl ../grounding_data/llmstu_tools/outputs/odvg_val.jsonl \
    --out work_dirs/model_comparison
```

`--gt-jsonl` is optional; supply it and the script also reports **recall of the
annotated students**, which is the only number here that means anything on its
own (see §4). Annotated images land in `work_dirs/model_comparison/`, with
**green = ground truth, orange = prediction**. Add `--include-e2` to include the
E0 duplicate if you want to see for yourself that it is identical.

### Pick your own frame

```bash
python -c "
import json,random
rows=[json.loads(l) for l in open('../grounding_data/llmstu_tools/outputs/odvg_val.jsonl')]
random.seed(1)
for r in random.sample(rows,5):
    print('../grounding_data/stu_img/frames/'+r['filename'])
    print('   GT:', r['grounding']['caption'])"
```

---

## 3. Measured output on one frame

Frame `..._video_0031_...`, ground truth
`"using laptop. listening attentively."` (5 annotated students), `--score-thr 0.15`:

| model | prompt | n | max score | recall |
|---|---|---|---|---|
| E0 base | a student sitting | 11 | 0.395 | **0.80** |
| E0 base | using phone | 2 | 0.230 | 0.00 |
| E1 (regression) | a student sitting | 4 | 0.268 | 0.40 |
| E1 (regression) | using phone | 0 | 0.017 | 0.00 |
| ARM A ordinal | a student sitting | 2 | 0.495 | 0.40 |
| ARM A ordinal | using phone | 2 | 0.545 | 0.40 |
| ARM B hungarian | a student sitting | 8 | 0.312 | **0.80** |
| ARM B hungarian | using phone | 8 | 0.325 | **0.80** |
| ARM C p90 | a student sitting | 4 | 0.235 | 0.60 |
| ARM C p90 | using phone | 5 | 0.376 | 0.60 |
| **MAIN** | a student sitting | 3 | 0.399 | 0.40 |
| **MAIN** | using phone | 3 | 0.444 | 0.60 |

---

## 4. How to read this — three traps

**Trap 1: a single image cannot rank the models.** MAIN scores 0.6343 R@1 over
9,352 validation frames but shows recall 0.40 on *this* frame, while E0 shows
0.80. That is sampling noise, not evidence. Use these commands to *see behaviour*
— where boxes land, how scores shift with the prompt — and use `FINDINGS.md` for
ranking.

**Trap 2: detection count is not evidence of understanding.** Every model returns
boxes for essentially any prompt, because grounding detectors localise the
best-matching region; they are not trained to answer "is this phrase present?".
The control that proves this is not a fine-tuning defect:

```bash
python tools/infer_image.py --ckpt work_dirs/thesis_bundle/checkpoints/e0_iter15000.pth \
    --image "$IMG" --prompt "empty chair" --score-thr 0.15 --device cuda:0
```

The **never-fine-tuned** E0 returns **23 detections for "empty chair"**; MAIN
returns 5. Fine-tuning made the model *more* conservative, not less.

**Trap 3: absolute scores are not calibrated across models.** ARM A hits max
0.545 while MAIN hits 0.444 on the same frame — ARM A is not "more confident and
therefore better". Compare scores *across prompts within one model*; never across
models.

---

## 5. What each comparison is actually good for

**ARM A vs ARM B** — the clearest thing to look at. Identical frames, captions,
boxes, schedule and seed; only the caption→box binding differs. This is what
+17.2 R@1 looks like in pixels, and it is the ablation the thesis rests on.

```bash
for C in arm_a_ordinal_iter10000 arm_b_hungarian_iter10000; do
    python tools/infer_image.py --ckpt work_dirs/thesis_bundle/checkpoints/$C.pth \
        --image "$IMG" --prompt "using phone" --score-thr 0.15 --device cuda:0
done
```

**E0 vs MAIN** — what 25k iterations on exact correspondences bought.

**E1** — kept as a cautionary artifact. It is worse than the baseline it started
from (−9.6 R@5); see `MARCH_2026_POSTMORTEM.md` §2.1.

**E2** — do not treat as a third condition. It is E0's file under another name;
the March "three-arm ablation" was two arms with one duplicated.

---

## 6. Thresholds

Use **`--score-thr 0.15`**, not the 0.45 that was shipped in
`attention_temporal.yaml` until 2026-07-31. That value was inherited from the
pre-fine-tuning model. Measured on MAIN over 120 validation frames:

| prompt @ threshold | precision | recall | F1 |
|---|---|---|---|
| `"student"` @ 0.45 (old default) | 0.937 | **0.256** | 0.402 |
| `"student"` @ 0.15 | 0.801 | 0.776 | 0.788 |
| **`"a student sitting"` @ 0.10** | 0.774 | 0.868 | **0.818** |

→ `work_dirs/probe/threshold_sweep_iter25000.json`

⚠️ Do **not** use a behaviour phrase as a person-detector prompt. `"using laptop"`
scores the highest aggregate F1 of any prompt, but only because it names the
majority activity — it finds on-task students well and systematically misses the
phone-users and head-down students the system exists to detect.

---

## 7. Quantitative check on one model

Pictures show behaviour; for numbers use the probe, which scores localization and
grounding separately against the validation annotations:

```bash
python tools/probe_main_checkpoint.py \
    --ckpt work_dirs/thesis_bundle/checkpoints/main_llmstu_exact_iter25000_final.pth \
    --n 15 --device cuda:0 --out work_dirs/probe
```

---

## 8. Troubleshooting

| symptom | cause |
|---|---|
| `ModuleNotFoundError: mmdet` | `PYTHONPATH` unset — §0 |
| `can't open file '.../checkpoints/tools/...'` | you are not in `LLMDet/`; `cd` there first |
| 0 detections | threshold too high; try `--score-thr 0.05` |
| `CUDA out of memory` | a training job holds the GPUs — use `--device cpu` |
| `ValueError: too many values to unpack` | the SDPA/BERT bug, fixed at `mmdet/models/language_models/bert.py:203` |
