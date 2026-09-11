# Deploying the dashboard as a Hugging Face Space

Two phases. Phase 1 gets a working Space with model switching and no GPU. Phase 2
adds the detector so the Space can analyse new video. Do them in order — if you
install mmcv against a CUDA build before anything works, a version mismatch and
an application bug look identical.

---

## Before you start: decide the Space's visibility

**Make the Space private.**

`artifacts.lock.json` marks two artifacts `restricted`, gated on RISK 1 in
`docs/branch_c/RELEASE_RISK_REGISTER.md` — an unresolved ethics/consent question:
the detector checkpoint, and the LLMSTU corpus. The precomputed session cache is
derived from student video, and `THESIS_DEFENSIBILITY_REVIEW.md` §A records that
no ethics approval, consent form, or data-management plan was found in the repo.

Pushing either to Hugging Face — even to a private repo — puts institutionally
governed data on a third-party host. That is a decision for you and your
supervisor, not one to make by default because a deploy step needed a file. Two
things follow:

1. **Nothing student-derived is baked into the image.** `app.py` fetches
   artifacts at runtime from a private repo, so access stays revocable in one
   place instead of being copied into every image layer and build cache.
2. **The Space boots without them.** With no `ARTIFACT_REPO`, it serves the
   tracked cue log: the UI renders and model switching is disabled. You can
   demonstrate the interface without uploading anything restricted.

If the ethics question is still open, phase 1 with no artifact repo is a
defensible thing to show. Phase 2 is not, until it is resolved.

---

## Phase 1 — replay + model switching

### 1. Create the artifact repo (private)

```bash
huggingface-cli repo create llmstu-dashboard-artifacts --type model --private
```

Upload the dev bundle contents, preserving the repo-relative layout
`tools/dashboard/sessions/...` and `LLMDet/work_dirs/thesis/...` — `app.py`
links those two trees into place by exactly those paths:

```bash
huggingface-cli upload WaelK/llmstu-dashboard-artifacts ./dash_dev_extracted . --repo-type model
```

### 2. Create the Space

Docker SDK, hardware **CPU basic** for phase 1 — the temporal models are small
and this path never touches the detector, so a GPU costs money and buys nothing.

Copy into the Space repo root:

```
Dockerfile          <- deploy/hf_space/Dockerfile
app.py              <- deploy/hf_space/app.py
requirements.txt    <- deploy/hf_space/requirements.txt
README.md           <- deploy/hf_space/README.md      (the Space card)
tools/              <- tools/
LLMDet/attention/   <- LLMDet/attention/
LLMDet/configs/     <- LLMDet/configs/
```

### 3. Set the Space secrets and variables

| name | kind | value |
|---|---|---|
| `HF_TOKEN` | **secret** | a read token for the artifact repo |
| `ARTIFACT_REPO` | variable | `WaelK/llmstu-dashboard-artifacts` |
| `DASHBOARD_MODEL` | variable | `arch/mstcn_556_hp` (phase 1 / CPU only — see below) |
| `SESSION` | variable | `0325` |

`arch/mstcn_556_hp` rather than the registry default: it has the best test
macro-F1 of any deployable variant (0.5011 vs ASRF's 0.4940) *and* replays about
4.6x faster on CPU (0.88x real time vs 0.19x). On CPU hardware the default is
the wrong choice on both axes.

**This argument is phase-1-only, and does not transfer.** The GPU Space
(`deploy/hf_space_live/`) has a T4, so the replay-speed half is moot, and the
thesis cites `ff_det/mstcn_553_ff_s42` as the deployed system. That app therefore
defaults to `ff_det/mstcn_553_facefound@s42`. It previously inherited this
CPU default, so a live Space with `DASHBOARD_MODEL` unset served a model trained
on the FaceLandmarker mesh rather than the BlazeFace detector — giving up the
+30% FPS of FINDINGS 11.16 — while presenting itself as the deployed system
(FINDINGS §14.6).

**Keep the seed pin.** The unpinned id `ff_det/mstcn_553_facefound` resolves to
the best *validation* seed, s43, which is a different checkpoint with its own
fitted calibration. Without `@s42` the Space cannot serve the checkpoint the
thesis names.

### 4. Verify

The build takes a few minutes. When it is up, check in this order:

1. the page renders and the header pill leaves `connecting…`;
2. **Model** is populated and switching one re-decides the session;
3. alert coverage in the model card changes between variants — it ranges 78.7%
   down to 30.5% across the registry at the same 85% alert precision.

### 5. Check the artifact repo is current

The registry the Space builds is derived from whatever landed in the artifact
repo, so a checkpoint that was never uploaded is simply not offered and nothing
says it was expected. From a machine with network:

```bash
# on the HPC
python tools/dashboard/artifact_manifest.py --emit artifacts_manifest.json
# wherever the repo is checked out
python tools/dashboard/artifact_manifest.py --verify artifacts_manifest.json \
    --root /path/to/snapshot_of_llmstu-dashboard-artifacts
```

It lists every file the Space needs — checkpoints, `run_record.json`,
`eval_val/metrics.json` and the fitted calibration for each of the 24
live-capable variants, 98 files and ~565 MB — with sizes and sha256, and flags
any model whose calibration is missing. A model without one is refused at
selection time rather than at boot, so it is exactly the kind of gap that
survives a smoke test.

If the page loads but every panel stays empty, the API calls are 404ing. That was
fixed by resolving them against `document.baseURI`; confirm the Space is running
the current `index.html`.

---

## Phase 2 — live and uploaded video

Needs the detector, which means mmcv/mmengine against a CUDA build, and the
`research` extra from `pyproject.toml`. Changes required:

* `python:3.11-slim` with a cu121 torch wheel, **not** a CUDA base image: the
  torch wheels bundle the CUDA runtime and the driver comes from the host, so a
  CUDA base image adds ~2 GB of duplicated runtime. Python 3.11 is forced by the
  only prebuilt mmcv wheel that exists for this stack (cp311);
* the version triple `python 3.11 + torch 2.2.x/cu121 + mmcv==2.2.0` — this is
  the step that fails most often, and it fails at import time with a message
  about a missing symbol rather than anything about versions;
* Space hardware with a GPU;
* **both** vendored trees copied in, `LLMDet/mmdet/` *and* `LLMDet/llava/`.
  Copying only mmdet builds fine and then dies with `No module named 'llava'`:
  `mmdet/models/detectors/grounding_dino.py:26` imports `llava.constants` at
  module level, so the detector cannot even be imported without it. Only llava's
  `.py` files are needed — the eval tables and webpage assets are ~9 MB of
  nothing;
* its pip prerequisites: addict, matplotlib, pycocotools, shapely, six,
  terminaltables, yapf, **scipy** (hungarian_assigner) and **fairscale**
  (grounding_dino), plus tqdm and rich, which arrive transitively today but are
  imported directly by the tree;
* the detector checkpoint in the artifact repo — see the visibility note above;
* a longer startup timeout: the first request loads GroundingDINO.

Put an import check in the Dockerfile:

```dockerfile
RUN python -c "import sys; sys.path.insert(0, 'LLMDet'); from mmdet.utils import register_all_modules; register_all_modules(); import mmdet.models.detectors.grounding_dino as g; print('ok', g.__name__)"
```

`register_all_modules()` is what the detector calls on first use and it walks
the whole datasets and models registry. Without this, every missing package
costs a full rebuild *plus* a video upload to discover, one package at a time,
because the failure surfaces inside a background analysis job rather than at
startup. scipy, fairscale and llava were all found this way.

**Watch the storage shape.** If durable storage is a mounted bucket, do not put
the session cache on it: `sessions/*/frames/` is ~900 JPEGs averaging 78 KB —
77% of the artifact files and 1.4% of the bytes — and each costs a cache
metadata sidecar write. That combination stalled a boot at 25% with repeated
`[Errno 5] Input/output error`. Weights (273 files, 4.94 GB, ~18 MB each) belong
on the bucket; the session cache belongs on ephemeral disk, where it re-fetches
in seconds. Probe the mount with a timeout, too — a wedged FUSE mount makes an
unbounded write hang the Space before it logs its first line.

Expect ~5.77 FPS at 6 students at the deployed 3:2 striding. Surface that in the
UI rather than implying real time — at 3:2 the striding also costs cue agreement
(0.947 against a 1:1 reference) and finds 13 of 15 episodes.
