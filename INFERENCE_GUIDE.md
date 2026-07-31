# Running inference on the detector checkpoints

How to point the fine-tuned grounding detector at your own images and read the
output. Written 2026-07-31 against `iter_5000.pth`; the same commands work for
any checkpoint (swap the `--ckpt` path).

---

## 0. Two things that will bite you

**1. `PYTHONPATH` is mandatory.** `mmdet` is a *local* package inside `LLMDet/`,
not an installed one. Without this you get `ModuleNotFoundError: No module named
'mmdet'`:

```bash
export PYTHONPATH=/home/jovyan/Computer_vision/LLMDet
```

**2. Do NOT use `score_thr: 0.45`.** That value is in
`configs/attention_temporal.yaml`, inherited from the *pre*-fine-tuning model.
The word `"student"` appears in **0 of 36,339** fine-tuning captions (training
captions are cue phrases like `"using laptop. listening attentively."`), so that
prompt is out of distribution for the fine-tuned head and its scores collapse.

Measured on `iter_5000`, 15 val frames:

| prompt @ threshold | precision | recall | F1 |
|---|---|---|---|
| `"student"` @ 0.45 (shipped) | 1.000 | **0.429** | **0.600** |
| `"student"` @ 0.15 | 0.857 | 0.857 | 0.857 |
| `"person"` @ 0.25 | 0.867 | 0.929 | 0.897 |

**Use `--score-thr 0.15` as the starting point.** A threshold sweep on the final
`iter_25000` checkpoint runs automatically when training finishes and writes
`work_dirs/probe/threshold_sweep_iter25000.json` — use its recommendation once
that exists, because confidence calibration sharpens during training.

---

## 1. Setup (run once per shell)

```bash
cd /home/jovyan/Computer_vision/LLMDet
export PYTHONPATH=/home/jovyan/Computer_vision/LLMDet
```

**Device.** While a training run is using the GPUs, pass `--device cpu`
(default). CPU is ~5–15 s per image for Swin-T — slow but it will not contend
with training. Once GPUs are free, `--device cuda:0` is roughly 20× faster.

Check whether training is running:

```bash
nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader
```

---

## 2. Single image, any prompt

```bash
python tools/infer_image.py \
    --ckpt work_dirs/student_llmstu_exact/iter_5000.pth \
    --image /path/to/your/frame.jpg \
    --prompt "student" \
    --score-thr 0.15 \
    --device cpu
```

Prints every detection with its score and box, and writes an annotated copy to
`work_dirs/infer_out/`. Real output:

```
prompt='student'  3 detections >= 0.15  (max score 0.483)
    score 0.483  box (1027,1)-(1439,705)
    score 0.443  box (532,133)-(665,462)
    score 0.206  box (143,259)-(284,587)
```

### A whole folder

```bash
python tools/infer_image.py \
    --ckpt work_dirs/student_llmstu_exact/iter_5000.pth \
    --image ../grounding_data/stu_img/frames \
    --prompt "student" --score-thr 0.15 --max-images 10
```

### Grab some real validation frames to try

```bash
python -c "
import json,random
rows=[json.loads(l) for l in open('../grounding_data/llmstu_tools/outputs/odvg_val.jsonl')]
random.seed(1)
for r in random.sample(rows,5):
    print('../grounding_data/stu_img/frames/'+r['filename'])
    print('   GT:', r['grounding']['caption'])
"
```

---

## 3. What you can prompt it with

The model is open-vocabulary, so any phrase is *accepted*. But it was fine-tuned
on exactly these 11 phrases, so these are what it knows well:

```
using laptop          listening attentively    sleeping head down
looking away          reading                  using phone
talking to peer       writing notes            raising hand
eating or drinking    sitting at desk
```

Two different jobs, and they behave differently:

| goal | prompt | what you should see |
|---|---|---|
| **find the students** | `"student"` / `"person"` | a box per student |
| **find a behaviour** | `"using phone"` | boxes only on students doing it |

Try the same image with several prompts — this is the clearest demonstration
that grounding works rather than just detection:

```bash
for P in "student" "using phone" "sleeping head down" "talking to peer"; do
    python tools/infer_image.py \
        --ckpt work_dirs/student_llmstu_exact/iter_5000.pth \
        --image /path/to/frame.jpg --prompt "$P" --score-thr 0.15
done
```

> ⚠️ **Do not use a behaviour phrase as your person-detector prompt.**
> `"using laptop"` scores the *highest* F1 of any prompt (0.966 @ thr 0.08), but
> only because `using_laptop` is the majority activity — it finds most students
> by naming what most of them are doing, while systematically missing phone
> users, head-down and talking students. Those minorities are the entire point
> of attention-loss detection. Use a generic anchor (`student`, `person`) for
> localization.

---

## 4. Quantitative check against ground truth

`infer_image.py` shows you pictures. To get numbers, use the probe — it scores
localization and grounding separately against the val annotations:

```bash
python tools/probe_main_checkpoint.py \
    --ckpt work_dirs/student_llmstu_exact/iter_5000.pth \
    --n 15 --device cpu --out work_dirs/probe
```

Output:

```
--- LOCALIZATION (prompt 'student', IoU>=0.5) ---
  precision 0.889   recall 0.432   F1 0.582      <- at default thr 0.35
--- GROUNDING (top box for each cue phrase) ---
  19/32 correct = 0.594
```

Annotated images land in `work_dirs/probe/` with **green = ground truth,
orange = prediction**.

Add `--score-thr 0.05` to see the same checkpoint reach recall 0.946 — proof
that the students *are* being found and the issue is score calibration, not
detection.

---

## 5. Threshold sweep (choosing a threshold yourself)

```bash
python tools/sweep_detector_threshold.py \
    --ckpt work_dirs/student_llmstu_exact/iter_5000.pth \
    --n 30 --device cpu \
    --out work_dirs/probe/my_sweep.json
```

Tries 6 prompts × 13 thresholds and prints a precision/recall/F1 table per
prompt, then recommends the best **generic** prompt and threshold. Inference
runs once per (frame, prompt) and thresholds are evaluated offline, so raising
`--n` costs much less than it looks.

---

## 6. Available checkpoints

```bash
ls -1 work_dirs/student_llmstu_exact/*.pth
```

The main run saves every 2,500 iterations up to 25,000. Validation scores so
far (grounding R@1 on the held-out video-wise split):

| checkpoint | R@1 | R@5 | R@10 |
|---|---|---|---|
| `iter_2500.pth` | 0.5630 | 0.9619 | 0.9932 |
| `iter_5000.pth` | 0.5946 | 0.9722 | 0.9950 |

For reference, the ablation arms finished at R@1 0.3230 (ordinal), **0.4954**
(Hungarian), 0.4787 (Hungarian+p90) — the main run is already well past all of
them by iteration 5,000, which is the expected benefit of exact-correspondence
labels over inferred ones.

Other checkpoints worth pointing the same commands at, in
`work_dirs/thesis_bundle/checkpoints/`:

```
arm_a_ordinal_iter10000.pth        arm_b_hungarian_iter10000.pth
arm_c_hungarian_p90_iter10000.pth  e0_iter15000.pth   (pre-fine-tuning baseline)
```

Comparing `arm_a` against `arm_b` on the same image is the most direct way to
*see* what the +17.2 R@1 gap looks like.

---

## 7. Troubleshooting

| symptom | cause |
|---|---|
| `ModuleNotFoundError: mmdet` | `PYTHONPATH` not set — see §1 |
| `0 detections` | threshold too high; try `--score-thr 0.05` |
| `CUDA out of memory` | training is using the GPUs — use `--device cpu` |
| very slow (~15 s/image) | expected on CPU; use `cuda:N` when free |
| `ValueError: too many values to unpack` | the SDPA/BERT bug — fixed in `mmdet/models/language_models/bert.py:203`; make sure you are not on an older copy |
