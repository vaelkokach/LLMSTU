# Orientation

Classroom student-attention analysis for a master's thesis. Two halves:

* **Detector** — a vendored LLMDet/GroundingDINO fork (`LLMDet/mmdet/`, `LLMDet/llava/`,
  `LLMDet/ram/`) fine-tuned to localise students. Test R@1 **0.6462** under a closed
  pre-registered protocol (`TEST_SPLIT_PROTOCOL.md`). Do not tune against test.
* **Temporal cue model** — MS-TCN / ASRF / Transformer over per-student feature
  sequences, predicting a visible-cue class per frame.

`FINDINGS.md` is the long-form experimental record and the source of truth for any
number. `THESIS_DEFENSIBILITY_REVIEW.md` lists the open threats.
`docs/EXPERIMENT_STATUS.md` is the current state and what is in flight.

## Layout

| path | what |
|---|---|
| `LLMDet/attention/` | the cue pipeline: features, tracking, taxonomy, temporal models |
| `LLMDet/attention/thesis_eval/` | training (`train.py`), evaluation (`run_eval.py`), metrics |
| `LLMDet/configs/` | mmdet detector configs; `attention_runtime.yaml` is the deployment config |
| `LLMDet/work_dirs/thesis/` | checkpoints + eval outputs, one dir per run |
| `tools/dashboard/` | the served UI, model registry, session replay, live pipeline |
| `deploy/hf_space_live/` | the Hugging Face Space (Docker) |
| `grounding_data/` | sequences, manifests, label jsonl — **gitignored, HPC only** |

## Conventions that are load-bearing

**Never spend the test split.** `TEST_SPLIT_PROTOCOL.md` is closed for the detector.
Evaluate on `--split val` unless deliberately spending test once, under protocol.

**Feature columns are absolute.** `thesis_eval/data.py` defines named `LAYOUTS`
(`v570`, `v1074_head`); each npz records which one it is. A width or layout mismatch
is fatal by design — reading the wrong columns would train happily on nonsense.

**Abstention is real.** `_reliable` taxonomies map unsupported classes to
`IGNORE_INDEX`; they leave the loss and the metrics together. Every such number must
be quoted with its **coverage** (`metrics.json` carries `coverage`).

**Calibration travels with the checkpoint.** `session_replay.load_model` refuses to
run a model whose calibration is missing rather than defaulting to threshold 0.
Registry ids may be seed-pinned as `variant@sNN`.

**`.gitignore` has bitten this repo twice.** An unanchored `datasets/` silently
excluded two vendored source directories from every clone.
`attention/tests/test_vendored_tree_complete.py` guards the class — keep it green.

## Running things

```bash
cd LLMDet
python -m attention.thesis_eval.train --experiment-id NAME --model mstcn \
  --feature-config 556_hp --taxonomy cue6 --seed 42 --epochs 90 \
  --output-dir work_dirs/thesis/... --device cuda:0

python -m attention.thesis_eval.run_eval --ckpt .../best.pth --split val \
  --out .../eval_val --device cuda:0     # sequence root comes from the checkpoint

python -m pytest attention/tests/ -q     # run from LLMDet/, not the repo root
```

Tests import `attention.*`, so **run pytest from `LLMDet/`**. Two test modules need
`mmcv` and will not collect without it.

## Environment

* **HPC** — 8×A100, no outbound network. Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
  for anything touching CLIP or a HF model; the weights are in `~/.cache/huggingface`.
  Cannot reach GitHub: changes arrive as a tarball extracted at the repo root.
  Parallel runs use `CUDA_VISIBLE_DEVICES=N` with `--device cuda:0`.
* **HF Space** — `WaelK/classroom-attention-cues-live`, t4-medium, private.
  Artifacts fetched at runtime from `WaelK/llmstu-dashboard-artifacts`; nothing
  student-derived is baked into the image. `deploy/hf_space/README_DEPLOY.md` has the
  full deployment story including what phase 2 actually needed.
* **Data is restricted.** `artifacts.lock.json` marks the detector checkpoint and the
  LLMSTU corpus `restricted` pending the ethics question in
  `docs/branch_c/RELEASE_RISK_REGISTER.md`.
