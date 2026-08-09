# Reproducibility

Status: **this repository is not, and cannot honestly be described as,
100% publicly reproducible.** The underlying corpus is identifiable video of
real students in a real classroom, and no documented ethics/consent basis for
that data collection exists yet (`docs/branch_c/RELEASE_RISK_REGISTER.md`
RISK 1). Every claim below is scoped around that fact, not around it.

This document is the map: what `python bootstrap.py --profile <X>` actually
gets you for each of the four profiles, what a fresh clone can and cannot do
*today* (2026-08-09), and exactly which results require the restricted
dataset and are therefore never publicly reproducible from this repo alone.

It builds on, and does not repeat, three earlier audit documents:
`docs/branch_c/ENVIRONMENT_AUDIT.md` (what is actually installed and
imported), `docs/branch_c/RELEASE_RISK_REGISTER.md` (why this repo could not
be responsibly published today), and `docs/branch_c/LICENSE_AUDIT.md` (which
third-party components are safe to depend on and which are traps).

---

## 0. The one-command entry point

```
python bootstrap.py --profile {demo,research,full-data,hpc}
```

Flags: `--dry-run` (print what would happen, fetch nothing),
`--offline` / `--no-download` (never touch the network; report gaps
instead), `--lock <path>` (use a different lock file, mostly for testing).

`bootstrap.py` and the resolver it calls (`tools/artifacts/download.py`)
never invent a substitute for a missing/corrupted file, never proceed past a
failed checksum, and always print size + source + licence before fetching
anything. `artifacts.lock.json` is the single source of truth for every
artifact this can fetch; see its own `"notes"` and `"status_values"` fields
for the exact contract (three-state `status`: `public` / `restricted` /
`manual`, and a three-state `redistribution.permitted`: `yes` / `no` /
`pending_ethics_review`).

`bootstrap.py` never runs `pip install`. Python dependencies come from
`pyproject.toml` (`pip install -e ".[demo]"` etc.) as a separate, explicit
step — see §4.

---

## 1. The four profiles

### `demo`

**Guarantees:** the dashboard server starts, serves the UI, and replays an
already-precomputed session end to end (`tools/dashboard/demo_session.jsonl`,
`live_session*.jsonl` — all already tracked in git, containing per-student
cue/confidence/dwell records with no images or names). The deployed temporal
checkpoint (MS-TCN, 553-dim features, seed 42) and its calibration thresholds
are available and loadable on CPU.

**Does NOT let you:** point the dashboard at a new video. That needs the
detector + CLIP + MediaPipe (the `research` profile), and there is currently
no publicly shippable sample video to try it on — the only footage this
project has is the restricted LLMSTU corpus.

**Artifact count:** 2 (`checkpoint.mstcn_553_ff_s42`,
`config.mstcn_553_ff_thresholds`). No GPU. No third-party model download —
Python deps only (`pip install -e ".[demo]"`: numpy, torch (CPU is enough),
opencv-python, PyYAML, Pillow).

**⚠ The honest gap, found while building this:** those two files are
currently **gitignored** by the blanket `LLMDet/work_dirs/` rule in
`.gitignore`, and no public release asset has ever been created for them
(`artifacts.lock.json` records both as `status: "manual"`,
`source.kind: "unpublished"`). That means:

* On **this development box**, in **this working tree**,
  `python bootstrap.py --profile demo` reports both artifacts
  verified-present immediately (they are sitting on disk, just not tracked
  by git) and exits 0.
* On a **genuine `git clone` of this repository from a fresh machine**, the
  same command reports both as missing, with status `manual`, and exits
  non-zero — there is nowhere for it to fetch them from yet.

Closing this gap is a repository-policy decision this agent did not make
unilaterally, for two reasons: (a) `.gitignore` is outside this agent's
owned paths, and (b) both checkpoints are trained on features derived from
the restricted LLMSTU corpus, so publishing them at all inherits RISK 1's
open ethics question (`artifacts.lock.json` marks their
`redistribution.permitted` as `"pending_ethics_review"`, not `"yes"`).
**Recommendation**, not action taken: once that determination is made, either
(1) force-add the two files to git — precedent already exists for
`mstcn_553_ff_thresholds.json`, which *is* force-added despite the same
`work_dirs/` ignore rule — or (2) run
`python -m tools.artifacts.upload publish --id checkpoint.mstcn_553_ff_s42 --backend <github|hf|zenodo>`
and fill in the resulting URL in `artifacts.lock.json`'s `source.url`. Until
one of those happens, "clone and run the demo" is true **on this box**, not
yet true **from a bare `git clone`**.

### `research`

**Guarantees:** everything in `demo`, plus the full live pipeline —
detector → tracker → CLIP appearance features → head-pose/face backend —
so a researcher can run **their own** video through the pipeline end to end,
on CPU or GPU, without needing the restricted LLMSTU dataset at all.

**Does NOT let you:** reproduce this thesis's own reported numbers (that
needs the actual LLMSTU corpus and its splits — `full-data`).

**Artifacts fetched (all `status: "public"`, real URLs, verified against
the live host without downloading the body — see §3):** 6DRepNet360 weights
(named as the head-pose escalation path, not yet wired into the deployed
config), the CLIP backbone (`openai/clip-vit-base-patch32`, via
`huggingface_hub`, not a direct download), three MediaPipe model assets, and
the three MM-Grounding-DINO Swin-T/B/L checkpoints used as detector-training
init weights. `pip install -e ".[demo,research]"` additionally installs
`transformers`, `mediapipe`, and the vendored `LLMDet/mmdet/` tree's own
prerequisites (`mmcv`, `mmengine`, `addict`, `matplotlib`, `pycocotools`,
`shapely`, `six`, `terminaltables`, `yapf` — **not** `mmdet` itself, see §4).

You will also need the **detector checkpoint**
(`checkpoint.detector_main_llmstu_exact_iter25000`, 4.29 GB) to run detection
on your own video with the exact model this thesis used — that checkpoint is
`status: "restricted"` (fine-tuned directly on the restricted footage; see
§2) and is never auto-fetched. Bring your own detector checkpoint, or a
generic Grounding-DINO/LLMDet checkpoint, if you don't have access to it.

### `full-data`

**Guarantees:** everything in `research`, plus the restricted LLMSTU
dataset itself, for whoever already holds institutional access to it. This
is the only profile that can reproduce the thesis's reported evaluation
numbers (`attention/thesis_eval`) and re-run the six-sweep checkpoint
comparison the dashboard's model-comparison dropdown shows.

**`bootstrap.py --profile full-data` will never fetch these
artifacts automatically, with or without network access.** Two entries carry
`status: "restricted"`:

* `checkpoint.detector_main_llmstu_exact_iter25000` — fine-tuned directly on
  frames of the restricted video; the closest derivative of identifiable
  footage in this whole catalogue.
* `data.llmstu_restricted_corpus` — a pointer entry to the full dataset tree
  (raw crops, sequence feature caches, splits, gold annotations); the actual
  per-file inventory (487 artifacts) is `docs/branch_c/ARTIFACT_MANIFEST.json`
  from the audit phase, not duplicated here.

If you already have access, place the files at the paths
`python -m tools.artifacts.download list --profile full-data` prints, then
re-run `verify`.

**Not currently publicly reproducible, and not expected to become so without
an institutional ethics/consent determination** (RISK 1). This profile exists
so that *someone who already has legitimate access* has a documented,
automatable path — not as a promise that access can be obtained by running a
command.

### `hpc`

**Guarantees:** everything in `full-data`, plus the pretrained init weights
and configs to retrain the detector or the temporal-model sweeps from
scratch on a multi-GPU cluster
(`LLMDet/tools/train.py`, `attention/thesis_eval/launch_sweep.py`).

**Not bit-reproducible.** GPU training is not deterministic across
hardware/CUDA/driver versions; re-running matches the recorded macro-F1
within the reported seed variance (see each checkpoint's `run_record.json`),
not byte-for-byte. `bootstrap.py` never launches training itself — it only
stages the artifacts training needs, then stops.

**Operating constraint this profile does not enforce for you:** this
project's own working rule is at most 4 of the box's 8 GPUs occupied at
once, and explicit approval before any GPU/training job. `bootstrap.py`
has no opinion about your cluster's policies — that discipline is on you.

---

## 2. What is, and is not, publicly reproducible — the honest line

**Publicly reproducible from a bootstrap-resolved environment alone, no
restricted data needed:**
- The dashboard UI and its replay of the already-tracked session logs
  (`demo` — modulo the gitignore gap in §1).
- Running the full detect→track→feature→classify pipeline on **your own**
  video (`research`), using the public detector-init weights, CLIP, and
  MediaPipe — but not with *this thesis's own* fine-tuned detector.
- The unit test suite (234 tests as of this change: the original 188 plus 46
  new in `LLMDet/attention/tests/test_artifacts.py`), all CPU-only, all
  network-mocked where relevant.

**NOT publicly reproducible, and not expected to become so without an
external decision:**
- This thesis's own reported macro-F1/balanced-accuracy numbers
  (`attention/thesis_eval`), the six-sweep architecture/feature-ladder
  comparison, and anything keyed to the specific 127-video LLMSTU corpus —
  all require `data.llmstu_restricted_corpus`.
- Detection with *this thesis's* actual fine-tuned detector weights
  (`checkpoint.detector_main_llmstu_exact_iter25000`) — trained directly on
  the restricted footage; `status: "restricted"` regardless of network
  access.
- Anything gated by RISK 1: the underlying data collection's own ethics/
  consent basis is undocumented. This is an institutional determination,
  not a repo-engineering fix, and it gates the two items above independent
  of whether someone technically has a copy of the files.

**No top-level `LICENSE` file exists yet either** (`docs/branch_c/
LICENSE_AUDIT.md` §2) — flagged there, not fixed here (outside this agent's
owned paths); a public release needs one before anything else in this
document matters.

---

## 3. What "verified" means in this reproducibility story

Per this task's own hard rule, no checksum in `artifacts.lock.json` was
invented. Every `verified: true` entry was established one of these ways,
all on 2026-08-09, all without downloading a full artifact body unless that
exact file was already sitting on disk from prior work:

- **Already-local files** (6DRepNet360, the three MediaPipe assets, the
  three MM-Grounding-DINO checkpoints, the two thesis checkpoints, the
  calibration JSON, the detector checkpoint): sha256 computed directly from
  the file on disk — a local read, not a network download, including for
  the 4.29 GB detector checkpoint (21 seconds, `sha256sum`).
- **Cross-checked against the live host without downloading the body:** for
  the three MediaPipe assets, a `HEAD` request's `Content-Length` **and**
  the GCS `x-goog-hash` MD5 header were both compared against the local
  file and matched exactly. For the three MM-Grounding-DINO checkpoints, a
  `HEAD` request's `Content-Length` matched the local file size exactly
  (and the filename's trailing 8 hex characters match the head of the
  locally-computed sha256, consistent with mmdetection's own naming
  convention — an internal consistency check, not treated as a security
  guarantee on its own).
- **CLIP** (`openai/clip-vit-base-patch32`): sha256 computed from the local
  `huggingface_hub` cache blob (whose filename is itself the sha256, per
  `huggingface_hub`'s own LFS storage convention — self-consistent). The
  pinned revision was independently confirmed live (`HEAD` to
  `/resolve/main/config.json` → 307 redirect to the exact same revision
  hash). The weight blob itself was not re-fetched over the network this
  session (a second `HEAD` attempt was blocked by this session's own
  tooling permissions) — see `artifacts.lock.json`'s `verification_note` on
  that entry for the precise, undramatized caveat.

Nothing in `artifacts.lock.json` records `verified: true` with `sha256:
null`; the schema validator (`tools/artifacts/validate_lock_schema`, and
`test_every_verified_artifact_has_a_real_looking_sha256` in
`test_artifacts.py`) enforces that pairing mechanically, not just by
convention.

---

## 4. Environment and container story

There is **no Dockerfile, no `environment.yml`, and no CI-published
container image** for this project, before or after this change (that
remains true — building one was not in this agent's owned paths).
`pyproject.toml` is the first machine-readable dependency declaration this
repository has ever had; before it, every version in
`docs/branch_c/ENVIRONMENT_AUDIT.md` §1 had to be recovered by introspecting
the live box, not by reading a manifest.

**What `pyproject.toml` gives you:**
- Core deps (`numpy==1.26.4`, `torch==2.2.2`) — the cheapest GPU-free path
  (dashboard + session replay + temporal-model inference).
- `[demo]` extra: `opencv-python`, `PyYAML`, `Pillow`.
- `[research]` extra: `transformers`, `mediapipe`, plus `mmcv`/`mmengine`
  and the vendored `LLMDet/mmdet/` tree's other runtime prerequisites
  (`addict`, `matplotlib`, `pycocotools`, `shapely`, `six`,
  `terminaltables`, `yapf`).
- `[train]` extra: `scikit-learn` (training scripts only).

**Why `mmdet` itself is never declared, anywhere:** it is not pip-installed
on the reference box. `LLMDet/attention/detector_adapter.py` resolves it via
`sys.path.insert(0, LLMDet/)` onto the **vendored, locally-patched** copy at
`LLMDet/mmdet/` (v3.3.0, patched in 4 files —
`docs/branch_c/ENVIRONMENT_AUDIT.md` §5). Declaring `mmdet==3.3.0` as a
dependency would silently fetch the unpatched upstream package instead of
using the patched vendored tree the code was actually written against —
strictly worse than the current undeclared-but-documented state.

**A known, unresolved gap this change surfaced rather than fixed:** live
`numpy` (1.26.4) violates `LLMDet/README.md`'s own upstream recommendation
("numpy should be lower than 1.24"). No failure has ever been observed from
this (the 234-test suite passes under it), but "no observed failure" is not
"verified compatible" — `pyproject.toml` pins to what is *actually running*,
with this caveat stated, rather than silently claiming compliance with
upstream's stricter ask.

**Two live conda environments exist on the reference box** (`base` and
`video-pipeline`) with materially different numpy/mmcv/opencv versions;
`pyproject.toml` pins to `base`, the one every tracked code path actually
uses (`docs/branch_c/ENVIRONMENT_AUDIT.md` §1). `video-pipeline` looks
unused by any tracked code and is not represented here at all.

**GPU builds:** `torch==2.2.2` in `pyproject.toml` is the plain PyPI (CPU)
build; the reference box actually runs `2.2.2+cu121` against driver-level
CUDA 12.2 / cuDNN 8.9.2. The `+cu121` local version segment is not resolvable
from plain PyPI — GPU users should install a matching CUDA wheel from
PyTorch's own index after `pip install -e .`, then treat the pin here as a
floor, not a full recipe. This is a real, currently-uncontainerized gap;
building an image or lockfile that pins the CUDA build exactly is future
work, not something this change claims to have closed.

---

## 5. CI: what runs on a hosted runner, what does not, and why

See `.github/workflows/ci.yml`'s own header comment for the authoritative,
maintained version of this; summarized:

**Runs on every push/PR, hosted, no GPU:** install from `pyproject.toml`
(`[demo,research,train]`), the CPU test suite
(`LLMDet/attention/tests`, 234 tests), `bootstrap.py --profile demo
--dry-run`, `artifacts.lock.json` schema validation, and a scan for
hard-coded `/home/jovyan` paths in committed runtime code/configs
(`tools/artifacts/ci_path_scan.py`).

**Cannot run on hosted CI, and how it's covered instead:**
- *GPU inference/training correctness* — no GPU on hosted runners. Covered
  by manual verification on the project's own GPU box before each release
  tag; the CUDA/cuDNN/driver versions actually exercised are recorded in
  `docs/branch_c/ENVIRONMENT_AUDIT.md` §2.
- *The thesis's reported metrics* — need the restricted corpus
  (`status: "restricted"`), never fetched in CI. Covered by the frozen
  `run_record.json` / `eval_val/`, `eval_test/` artifacts already checked
  into `LLMDet/work_dirs/thesis/**`, which CI does not regenerate.
- *Real artifact downloads* — CI only dry-runs the resolver
  (`--dry-run` touches no network). A real, non-dry-run
  `bootstrap.py --profile research` pulls several GB; run it by hand when
  validating a release, not on every push.

**Left failing on purpose, not a CI bug:** the hard-coded-path scan is
expected to currently report 8 hits against `main`
(`docs/branch_c/RELEASE_RISK_REGISTER.md` RISK 4's 6 blocking files plus 2
non-blocking docstring examples) — those files are outside this agent's
owned paths. The CI job exists to make the count visible and stop it from
growing, not to claim it is already zero; `test_artifacts.py`'s own
hardcoded-path tests are scoped to this task's *own* new files (which are
clean) plus the scanning mechanism itself, for exactly this reason — see
that file's module docstring.

---

## 6. Quick reference: commands

```bash
# What would the demo profile need, without touching the network?
python bootstrap.py --profile demo --dry-run

# Actually resolve it (fetches only what --offline/status don't block)
python bootstrap.py --profile demo

# What's missing for full-data with no network at all?
python -m tools.artifacts.download offline --profile full-data

# Verify everything already on disk against the lock file
python -m tools.artifacts.download verify

# Validate the lock file's own schema
python -m tools.artifacts.download validate-schema

# Install Python deps for a given tier (separate from bootstrap.py)
pip install -e ".[demo]"          # dashboard + replay
pip install -e ".[demo,research]" # + full live pipeline
pip install -e ".[demo,train]"    # + training scripts

# Run the CPU test suite (from LLMDet/)
cd LLMDet && python -m pytest attention/tests -q
```
