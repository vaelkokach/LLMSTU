# Environment Audit — Branch C (Reproducibility/Release)

Audit only. No packages installed, nothing downloaded, no GPU workload run.
Every line below is backed by a command actually executed on the live box on
2026-08-09. Working directory: `/home/jovyan/Computer_vision`, branch
`branch-c/overt-cue`.

---

## 1. Interpreter and key package versions

Two conda environments exist on this box:

```
$ conda env list
# conda environments:
#
base                  *  /opt/conda
video-pipeline           /opt/conda/envs/video-pipeline
```

`base` is the **active** environment (the one every command in this audit ran
in unless noted) and is the one the dashboard and `attention/` package are
actually exercised from — nothing in the repo activates `video-pipeline`
(`grep -rniE 'video-pipeline|conda activate|source activate'
tools/dashboard` returns nothing).

```
$ python3 -c "import sys; print(sys.prefix); print(sys.version)"
/opt/conda
3.11.9 | packaged by conda-forge | (main, Apr 19 2024, 18:36:13) [GCC 12.3.0]
```

Key packages, `base` (prefix `/opt/conda`):

```
$ pip list | grep -iE '^(torch|torchvision|torchaudio|numpy|opencv|mmcv|mmengine|mmdet|transformers|mediapipe|ultralytics|fastapi|flask|uvicorn|streamlit|gradio|scikit-learn|pandas|pillow|scipy|PyYAML)\b'
mediapipe                         1.0.0
mmcv                              2.2.0
mmengine                          0.10.7
numpy                             1.26.4
opencv-python                     4.11.0.86
pandas                            2.2.2
pillow                            10.3.0
scikit-learn                      1.5.0
scipy                             1.13.1
torch                             2.2.2+cu121
torchaudio                        2.2.2+cu121
torchvision                       0.17.2+cu121
transformers                      4.44.2
```
`PyYAML 6.0.1` confirmed separately (`pip list | grep -iE '^PyYAML'`).

**`mmdet` is NOT pip-installed in either environment** — confirmed by
`python3 -c "import mmdet"` failing with `ModuleNotFoundError` in both `base`
and `video-pipeline`. It resolves only because every dashboard entry point
does `sys.path.insert(0, <repo>/LLMDet)` before importing, which puts the
**vendored copy at `LLMDet/mmdet/`** (version `3.3.0`, see §5) on `sys.path`
ahead of anything pip-installed. This is a real, working setup, but it is
undocumented — nothing in the repo says "you must run from a working
directory where `LLMDet/` is importable" outside the dashboard scripts'
`sys.path` hacks themselves.

`ultralytics`, `fastapi`, `flask`, `uvicorn`, `streamlit`, `gradio`: **not
installed, and not needed** — the dashboard (`tools/dashboard/server.py`) is
built on stdlib `http.server.BaseHTTPRequestHandler` /
`ThreadingHTTPServer`, not a web framework. There is no dependency on any of
these five packages anywhere in `tools/dashboard/*.py` or `LLMDet/attention/*.py`.

Second environment, `video-pipeline` (prefix `/opt/conda/envs/video-pipeline`),
for reference — versions differ non-trivially from `base` and from what the
runtime code was actually exercised against:

```
$ /opt/conda/envs/video-pipeline/bin/pip list | grep -iE '^(torch|torchvision|numpy|opencv|mmcv|mmengine|pandas|pillow|scipy)\b'
mmcv                   2.1.0
mmengine               0.10.7
numpy                  2.2.6
opencv-contrib-python  4.13.0.92
opencv-python          4.13.0.92
pandas                 2.3.3
Pillow                 9.4.0
scipy                  1.15.3
torch                  2.5.1
torchvision            0.20.1
```
`mediapipe`/`transformers`/`mmdet` are absent from this env's grep output
entirely. **Finding:** two live environments exist with materially different
numpy (1.26.4 vs 2.2.6) and mmcv (2.2.0 vs 2.1.0) versions; nothing in the
repo states which one is authoritative, and `video-pipeline` looks unused by
any tracked code path.

`LLMDet/README.md` (upstream LLMDet project's own stated dev environment,
§"3 Our Experiment Environment") recommends:
```
pytorch==2.2.1+cu121
transformers==4.37.2
numpy==1.22.2 (numpy should be lower than 1.24, recommend for numpy==1.23 or 1.22)
mmcv==2.2.0, mmengine==0.10.5
```
Compared against the live `base` env: torch is close (2.2.2 vs 2.2.1),
mmcv matches exactly (2.2.0), mmengine is close (0.10.7 vs 0.10.5), but
**`numpy 1.26.4` is running against upstream's own explicit "must be below
1.24" warning**, and `transformers 4.44.2` is 7 minor versions ahead of the
`4.37.2` upstream pinned. Neither mismatch has caused an observed failure in
this audit (no code was run), but it is an unverified compatibility gap
inherited silently.

---

## 2. CUDA / cuDNN / driver / FFmpeg

```
$ nvidia-smi
Driver Version: 535.54.03    CUDA Version: 12.2
8x NVIDIA A100-SXM4-40GB, 40960 MiB each
```

```
$ python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.cuda.is_available())"
2.2.2+cu121 12.1 8902 True
```

So: driver supports CUDA 12.2, torch was built against CUDA 12.1 (compatible,
forward-compatible driver), bundled cuDNN 8.9.2. `torch.cuda.is_available()`
is `True` on this box (not exercised further — no GPU workload run per this
task's read-only/no-GPU constraint).

```
$ ffmpeg -version
ffmpeg version 4.4.2-0ubuntu0.22.04.1
libavutil      56. 70.100
libavcodec     58.134.100
```
System FFmpeg is the Ubuntu 22.04 distro build (gpl-enabled, libx264/libx265
present). Nothing in the repo pins an FFmpeg version or checks it at runtime.

---

## 3. Dependency declaration files

```
$ git ls-files | grep -E 'requirements|environment|pyproject|setup\.py|Pipfile|poetry'
(no output)
```

**There is no `requirements.txt`, `environment.yml`, `pyproject.toml`,
`setup.py`, `Pipfile`, or Poetry lockfile anywhere in the tracked repository.**
There is also no `Dockerfile`, `docker-compose.yml`, or `Makefile`
(`git ls-files | grep -iE 'dockerfile|docker-compose|Makefile'` — no output).
The only environment description that exists at all is the prose paragraph in
`LLMDet/README.md` §3 quoted above, which is the **upstream LLMDet project's**
README, not something written for this fork, and it does not cover
`mediapipe`, `opencv`, `scikit-learn`, or anything in `attention/` or
`tools/dashboard/`.

**This is a major finding.** Nothing currently lets a third party
reconstruct this environment from the repo alone; every version in §1 was
recovered by introspecting the live box, not by reading a manifest.

---

## 4. Minimal true dependency set for the Branch-B runtime path

Derived by grepping every `import`/`from` line in `LLMDet/attention/**/*.py`
and `tools/dashboard/*.py` (commands run and full output captured during this
audit) and classifying each name.

**Stdlib** (no action needed): `argparse`, `base64`, `collections`
(`defaultdict`, `deque`, `Counter`), `csv`, `dataclasses`, `datetime`,
`hashlib`, `http.server`, `itertools`, `json`, `multiprocessing`, `os`,
`pathlib`, `random`, `re`, `shutil`, `subprocess`, `sys`, `tempfile`,
`threading`, `time`, `traceback`, `typing`, `urllib`, `warnings`, `zipfile`.

**Third-party** (the actual install surface):
| Package | Where it's used |
|---|---|
| `numpy` | almost every module in `attention/` and `attention/thesis_eval/` |
| `torch` (+ `torch.nn`, `torch.nn.functional`, `torch.utils.data`, `torch.distributed`, `torch.utils.tensorboard` in training-only scripts) | `temporal_model.py`, `thesis_eval/models.py`, `runtime.py`, `train_temporal_ddp.py`, `pipeline_bridge.py` |
| `opencv-python` (`cv2`) | `features.py`, `head_pose.py`, `precompute_session.py`, `pipeline_bridge.py`, `sequence_builder.py`, etc. |
| `PyYAML` (`yaml`) | `realtime_infer.py`, `eval_baseline_chain.py`, `eval_events_vs_human.py`, `pipeline_bridge.py`, `precompute_session.py` |
| `Pillow` (`PIL.Image`) | `features.py`, `precompute_affect.py` |
| `transformers` (`CLIPModel`/`CLIPProcessor`, `AutoImageProcessor`/`AutoModelForImageClassification`) | `features.py` (guarded import), `precompute_affect.py` (guarded import) |
| `mediapipe` (`mediapipe.tasks.python.vision`, `BaseOptions`) | `head_pose.py`, `bench_face_backends.py`, `precompute_head_pose*.py`, `precompute_affect.py` — all guarded/deferred imports, i.e. only needed for the MediaPipe-backed head-pose/face path, not the whole runtime |
| `mmdet` (`mmdet.apis.inference.inference_detector`, `init_detector`) | `detector_adapter.py` only, via the **vendored** copy at `LLMDet/mmdet/` (see §5) — not pip-installed |
| `scikit-learn` (`sklearn.metrics.average_precision_score`) | `train_temporal_ddp.py` only (training script, guarded import, not the inference/dashboard path) |

**Local** (this repo, not third-party): everything under `attention.*` and
`attention.thesis_eval.*` importing each other (e.g. `attention.taxonomy`,
`attention.events`, `attention.temporal_model`, `attention.thesis_eval.data`,
`attention.thesis_eval.calibrate`), plus the dashboard's own siblings
(`sources`, `model_registry`, `session_replay`, `pipeline_bridge`,
`precompute_session`).

**Minimal true third-party set to run the dashboard against a precomputed
session** (the cheapest, GPU-free path — `session_replay.py`'s imports):
`numpy`, `opencv-python`, `torch` (CPU build is enough for this path; the
model forward pass is small). No `mediapipe`, no `mmdet`, no `transformers`
are on this path — those only enter through `pipeline_bridge.py`/
`precompute_session.py` (the "run a fresh video through the whole pipeline"
path), which additionally needs `PyYAML`, `mmdet` (vendored), `mediapipe`,
`transformers`, `Pillow`.

---

## 5. Vendored code inside `LLMDet/`

`LLMDet/mmdet/` is a vendored copy of **mmdetection**, version string taken
directly from the vendored source:

```
$ grep -n '__version__' LLMDet/mmdet/version.py
__version__ = '3.3.0'
```

It is tracked in git (326 files: `git ls-files LLMDet/mmdet | wc -l` → 326),
not a submodule (`.gitmodules` does not exist), added whole in one commit and
touched by exactly one more:

```
$ git log --oneline -- LLMDet/mmdet
02fd4c9 Rebuild pipeline around LLMSTU per-student dataset
de0c883 Initial commit (code only)
```

**It has been locally modified** — the second commit changed four files
inside the vendored tree:

```
$ git show --stat 02fd4c9 -- LLMDet/mmdet
 LLMDet/mmdet/evaluation/metrics/flickr30k_metric.py    | 16 +++++++-
 LLMDet/mmdet/models/dense_heads/grounding_dino_head.py | 27 ++++++++++---
 LLMDet/mmdet/models/detectors/grounding_dino.py        | 82 ++++++++++++++++++++++++++++----------
 LLMDet/mmdet/models/language_models/bert.py            |  5 ++-
```
The commit message for `02fd4c9` documents the intent of one of these edits
("SDPA->eager BERT attention" — i.e. `language_models/bert.py`); the other
three (flickr30k metric, grounding-dino head, grounding-dino detector) are
undocumented diffs against upstream mmdet 3.3.0. Anyone trying to reproduce
results against a pip-installed `mmdet==3.3.0` instead of this vendored copy
would silently get different behaviour in exactly these four files.

`LLMDet/` also vendors **LLaVA** (`LLMDet/llava/`) — not investigated in
depth for this audit (out of scope: the runtime path in §4 does not import
it), but present and tracked.

No `VERSION` file or equivalent exists at the `LLMDet/` top level for the
LLMDet project itself; version provenance for the outer LLMDet fork is only
inferable from `LLMDet/README.md`'s CVPR2025 changelog, not a machine-readable
field.

---

## Summary of major findings for this doc

1. No dependency-declaration file of any kind exists in the repo (requirements/environment.yml/pyproject/setup.py/Pipfile/Dockerfile/Makefile all absent).
2. Two conda environments exist with diverging numpy/mmcv/opencv versions (`base` vs `video-pipeline`); only `base` is actually used by tracked code, and nothing states this.
3. `mmdet` is not pip-installable as used — it only works via `sys.path` insertion of the vendored `LLMDet/mmdet/` (v3.3.0, locally patched in 4 files, 2 of them undocumented in the commit message).
4. Live `numpy` version (1.26.4) violates the upstream LLMDet project's own documented constraint ("numpy should be lower than 1.24").
5. The actual dashboard/replay path needs far fewer third-party packages (numpy, opencv, torch) than the full live-pipeline path (+ PyYAML, mmdet, mediapipe, transformers, Pillow) — this distinction is not documented anywhere for a future reproducer.
