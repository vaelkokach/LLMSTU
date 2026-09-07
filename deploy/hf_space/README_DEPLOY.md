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
| `DASHBOARD_MODEL` | variable | `arch/mstcn_556_hp` |
| `SESSION` | variable | `0325` |

`arch/mstcn_556_hp` rather than the registry default: it has the best test
macro-F1 of any deployable variant (0.5011 vs ASRF's 0.4940) *and* replays about
4.6x faster on CPU (0.88x real time vs 0.19x). On CPU hardware the default is
the wrong choice on both axes.

### 4. Verify

The build takes a few minutes. When it is up, check in this order:

1. the page renders and the header pill leaves `connecting…`;
2. **Model** is populated and switching one re-decides the session;
3. alert coverage in the model card changes between variants — it ranges 78.7%
   down to 30.5% across the registry at the same 85% alert precision.

If the page loads but every panel stays empty, the API calls are 404ing. That was
fixed by resolving them against `document.baseURI`; confirm the Space is running
the current `index.html`.

---

## Phase 2 — live and uploaded video

Needs the detector, which means mmcv/mmengine against a CUDA build, and the
`research` extra from `pyproject.toml`. Changes required:

* base image `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` rather than
  `python:3.11-slim`;
* a CUDA torch wheel matching that CUDA version, then `mmcv==2.2.0` built or
  installed against *that exact* torch — this is the step that fails most often,
  and it fails at import time with a message about a missing symbol rather than
  anything about versions;
* Space hardware with a GPU;
* `LLMDet/mmdet/` copied in (it is vendored, not pip-installed) plus its
  prerequisites: addict, matplotlib, pycocotools, shapely, six, terminaltables,
  yapf;
* the detector checkpoint in the artifact repo — see the visibility note above;
* a longer startup timeout: the first request loads GroundingDINO.

Expect ~5.77 FPS at 6 students at the deployed 3:2 striding. Surface that in the
UI rather than implying real time — at 3:2 the striding also costs cue agreement
(0.947 against a 1:1 reference) and finds 13 of 15 episodes.
