# Instructor dashboard

Real-time analytics and alerts over the attention-cue pipeline, plus a control
for **which trained model is producing the cues**.

```
  video ──► detector ──► tracker ──► CLIP + head pose ──► temporal model ──► cues
            └──────────── model-independent ───────────┘   └── the only ──┘
                          cached once per video               part that
                                                              varies
```

Everything left of the temporal model is identical for every checkpoint in the
registry, so it runs **once per video** and is cached. Switching models then
re-decides every cue in the session without touching the detector — which is
what makes the selector usable at all on a machine with no GPU, where a single
detector pass is the expensive part of the whole system.

---

## 0. Start here

**The one-off setup is already done.** Thresholds are fitted for all 13
deployable models, and `0325.mp4` is already analysed. From the repo root:

```bash
python tools/dashboard/server.py --session tools/dashboard/sessions/0325 \
    --model arch/mstcn_556_hp --host 127.0.0.1
```

Open <http://localhost:8080>. It starts playing immediately — no GPU, no
detector, nothing to wait for.

Why `--model arch/mstcn_556_hp` and not the registry default: on CPU the default
(ASRF) replays at **0.19× real time** while MS-TCN manages 0.88×. Drop the flag
to get the best validation score instead; the reasoning is in §1.

Three things to try once it is up:

1. **Switch models.** Pick another entry in *Model*. The session restarts and
   every cue is re-decided — the detector never runs. Watch `alert coverage` in
   the card: it ranges 78.7% down to 30.5% across the registry at the same 85%
   alert precision.
2. **Add your own recording.** Drop a video on *Recording*, press **Analyse**,
   watch the progress bar. Budget ~15 min per 900 frames on CPU (1.04 fps). It
   plays automatically when done.
3. **Point it at a camera.** Restart with `--video 0` (webcam) or
   `--video rtsp://user:pass@host/stream`. See §2 for what changes when the
   source is live.

### Reading the page

| panel | what it shows |
|---|---|
| **Live view** | the frame with a box per tracked student, coloured and labelled by cue |
| **Class overview** | students tracked, how many are off-task now, and the off-task fraction over time |
| **Per-student cues** | one row per seat: current cue, how long it has held, and the model's confidence |
| **Alerts** | sustained episodes only — never a single frame |
| **Model** | the running checkpoint and what it scores; switch here |
| **Recording** | upload, analyse, and choose what is playing |

Cues are **visible behaviours**, not mental states: `screen_oriented`,
`looking_away`, `head_down`, `turned_to_peer`, `phone_use`, and `uncertain`. The
system reports "head down for 45 s" and never "not paying attention" — it cannot
see attention, and saying otherwise would be a claim the data does not support.

`uncertain` is not a sixth behaviour, it is an **abstention**: the model's
confidence fell below its display threshold, so the chip declines to name a cue.
The raw prediction is still recorded underneath — abstention withholds a claim,
not the evidence.

An alert needs **three** things at once, which is why the alert count stays far
below the number of off-task students: a cue in the alertable set, held for its
minimum duration (phone 15 s, looking away 20 s, head down and turned-to-peer
30 s), *and* model confidence above that model's own alert threshold. A student
who glances away cannot generate one.

Seats are track numbers assigned in detection order. They are not identities,
they do not persist across runs, and no demographic attribute is read anywhere
in the pipeline.

---

## 1. What the selector offers

`model_registry.py` reduces the 54 checkpoints under `work_dirs/thesis/` to one
entry per **variant** (sweep × architecture × feature config), choosing the seed
with the best macro-F1 on the **validation** split.

```
python tools/dashboard/model_registry.py
```

| variant | best seed | val macro-F1 | val mean ± sd | test macro-F1 | live |
|---|---|---|---|---|---|
| `arch/asrf_556_hp` | s43 | **0.5258** | 0.5063 ± 0.0202 | 0.4940 | yes ← **default** |
| `ff_det/mstcn_553_facefound` | s43 | 0.5121 | 0.4913 ± 0.0208 | — | yes |
| `arch/mstcn_556_hp` | s43 | 0.5067 | 0.5027 ± 0.0050 | 0.5011 | yes |
| `posefix/asrf_556_hp` | s42 | 0.5041 | 0.4980 ± 0.0103 | — | yes |
| `ff_bp/mstcn_553_facefound` | s42 | 0.4937 | 0.4819 ± 0.0138 | — | yes |
| `posefix/mstcn_556_hp` | s42 | 0.4829 | 0.4695 ± 0.0124 | — | yes |
| `ff_det/transformer_553_facefound` | s44 | 0.4251 | 0.4149 ± 0.0106 | — | yes |
| `ff_bp/transformer_553_facefound` | s43 | 0.4168 | 0.4045 ± 0.0117 | — | yes |
| `ladder/transformer_556_hp` | s44 | 0.4092 | 0.4020 ± 0.0069 | 0.4025 | yes |
| `posefix/transformer_556_hp` | s42 | 0.4073 | 0.4008 ± 0.0107 | — | yes |
| `headpose/transformer_553_facefound` | s44 | 0.4034 | 0.3962 ± 0.0062 | — | yes |
| `headpose/transformer_555_angles` | s44 | 0.3914 | 0.3779 ± 0.0155 | — | yes |
| `ladder/transformer_552_base` | s43 | 0.3777 | 0.3726 ± 0.0049 | 0.3844 | yes |
| `arch/mstcn_570_full` | s43 | 0.5178 | 0.5038 ± 0.0121 | 0.4924 | **no** |
| `arch/asrf_570_full` | s44 | 0.5172 | 0.5071 ± 0.0096 | 0.5011 | **no** |
| `ladder/transformer_563_expr` | s42 | 0.4069 | 0.4029 ± 0.0036 | 0.4242 | **no** |
| `ladder/transformer_570_full` | s44 | 0.4058 | 0.3989 ± 0.0072 | 0.3959 | **no** |
| `ladder/transformer_563_dyn` | s42 | 0.3944 | 0.3914 ± 0.0030 | 0.4058 | **no** |

Three decisions in that table are worth stating explicitly, because each could
have been made the flattering way instead.

**Ranking and the default use validation only.** `TEST_SPLIT_PROTOCOL.md` spent
the test split once, under a pre-registration. A dropdown sorted by test
macro-F1 would spend it again on every page load. Test numbers are displayed —
they are the honest generalisation estimate — but they rank nothing and choose
nothing. (Only the `arch` and `ladder` sweeps were ever evaluated on test, which
is why most rows show `—`.)

**Two macro-F1 figures per model, not one.** The dashboard has to run *one*
checkpoint, so it runs the best of the variant's three seeds — but the thesis
tables report the seed mean, and best-of-3 reads higher by construction. The
default's 0.5258 is a selection; its 0.5063 ± 0.0202 is the measurement. The UI
shows both, labelled.

**The non-deployable models are listed, disabled, with the reason.** `563_expr`,
`563_dyn` and `570_full` cannot run in a streaming path: the expression block
needs a second per-crop GPU model, and the dynamics block is a whole-track
statistic (fidget variance, a personalised median gaze baseline) that no
streaming path can produce honestly. Dropping them from the list would have made
the ablation look smaller than it was; the previous bridge instead
**zero-padded** up to their width, which produced a running system whose extra
14 dims were all zeros.

### Measured cost of each choice (CPU, `0325.mp4`, 900 frames)

| stage | rate | |
|---|---|---|
| **Analyse** — detector + tracker + CLIP + both head-pose backends | **1.04 fps** | 865 s for 900 frames; one-off per video |
| Replay, MS-TCN head | **26 fps** | 0.88× real time |
| Replay, ASRF head | **5.8 fps** | **0.19× real time** |

ASRF is 4.5× more expensive at inference than MS-TCN for +0.019 validation
macro-F1 — and on test the ordering reverses. On CPU an ASRF session plays at
about a fifth of real time, and `--speed` will not help: the pacing loop is
already waiting on compute rather than on the clock. **For a live demonstration,
switch to `arch/mstcn_556_hp`.** It is one dropdown entry away.

### The default: `arch/asrf_556_hp` seed 43

ASRF over 556-dim features (CLIP + geometry + colour + posture + head pose),
val macro-F1 0.5258 for this seed, 0.5063 ± 0.0202 over three.

This is **not** the model in `attention_runtime.yaml`, which deploys
`ff_det/mstcn_553_facefound` seed 42. That config chose speed: §11.16 replaced
the FaceLandmarker mesh with a BlazeFace detector for +30% FPS at a statistical
tie, and MS-TCN is cheaper than ASRF. Neither consideration applies here —
nothing about a cached session is real-time-constrained — so the dashboard
defaults to the best validation score and leaves the deployment config alone.
Both are one dropdown entry apart.

---

## 2. Running it

### From the page: upload a recording

```bash
python tools/dashboard/server.py --host 127.0.0.1
#  -> http://localhost:8080, then drop a video on the Recording panel
```

Drop a file (or click to browse), press **Analyse**, watch the progress bar,
and it starts playing when the pass finishes. After that it is a *session*: it
replays instantly and the model dropdown switches freely.

The two buttons are a real choice, not a preference:

| | what runs | switching models | when |
|---|---|---|---|
| **Analyse** | detector + tracker + features, once, into a cache | instant afterwards | any recording you will look at more than once — i.e. almost always |
| **Play** on an un-analysed file | the whole pipeline, live | restarts the recording | a quick look, or when you want the current config end to end |

Uploads land in `uploads/`, sessions in `sessions/`; both are just directories,
so anything dropped in by hand appears in the picker too. A session is paired to
its video by name (`lecture.mp4` → `sessions/lecture`), and a half-built session
— a directory with no `meta.json` — reads as absent rather than as broken.

Cancelling an analysis deletes the partial cache rather than keeping it. A cache
covering the first 40 seconds of a lecture would otherwise replay as though that
were the whole lecture.

**Exposure.** The page can write files, and there is no authentication — this is
a thesis demo, not a service. The server binds `0.0.0.0` by default and warns at
startup when uploads are enabled on that address. Use `--host 127.0.0.1` when
the browser is on the same machine (as above), or `--no-upload` to serve
read-only. Filenames from the browser are treated as hostile: basename only,
character allowlist, extension allowlist, and a containment check under
`uploads/`. Uploads are capped at 8 GB and streamed to disk, never buffered.

### The same thing from the command line

Both setup steps are **already done** in this repo. You need them only after
retraining (step 0) or for a new video you would rather not upload (step 1).

```bash
# 0. thresholds for every model, fitted on ITS OWN validation predictions.
#    Seconds. Rerun after adding or retraining a checkpoint; without it the
#    dashboard refuses to load that model rather than running it uncalibrated.
python tools/dashboard/calibrate_registry.py

# 1. per video: the expensive pass — exactly what the Analyse button runs.
#    Detector, tracker, CLIP and BOTH head-pose backends. ~1.04 fps on CPU.
python tools/dashboard/precompute_session.py \
    --config LLMDet/configs/attention_runtime.yaml \
    --video LLMDet/0325.mp4 --frames 900 --device cpu \
    --out tools/dashboard/sessions/0325

# 2. the dashboard. Instant model switching, no detector, no GPU.
python tools/dashboard/server.py --session tools/dashboard/sessions/0325
#    -> http://localhost:8080
```

Server flags worth knowing:

| flag | effect |
|---|---|
| `--model VARIANT_ID` | start on a specific checkpoint instead of the registry default |
| `--host 127.0.0.1` | do not expose the port (and its upload endpoint) to the network |
| `--device cuda:0` | use a GPU if one is free; CPU is the default everywhere |
| `--blur-faces` | blur the top 35% of each box — for a classroom or a screenshot |
| `--speed 4` | replay faster than real time. Only helps when the model can keep up: ASRF on CPU is already compute-bound at 0.19×, so this does nothing for it |
| `--no-upload` | serve read-only |
| `--analyse-frames N` | frame cap for the Analyse button (default 900) |
| `--port` | default 8080 |

### Checking it still works

```bash
python tools/dashboard/verify_dashboard.py --cache tools/dashboard/sessions/0325
```

Replays several models over one cache and asserts they actually disagree,
re-reads every `eval_val/metrics.json` to confirm the model card is not showing
hand-copied numbers, and exercises the HTTP surface including that a
non-deployable model is refused. Exits non-zero on any failure. Last run: all
8 HTTP checks and both data checks passed (FINDINGS 11.22).

### Headless — diff two models over the same session

```bash
python tools/dashboard/session_replay.py --cache tools/dashboard/sessions/0325 \
    --model arch/asrf_556_hp   --out /tmp/asrf.jsonl
python tools/dashboard/session_replay.py --cache tools/dashboard/sessions/0325 \
    --model ladder/transformer_552_base --out /tmp/base.jsonl
```

Same frames, same boxes, same features — the only difference is the temporal
head, so any disagreement is attributable to the model.

### Live: the viewer's own camera, from the page

The **Live camera** panel captures with `getUserMedia` in the browser and POSTs
JPEG frames to `/api/camera/frame`. The server runs the detector, the tracker and
the selected model on each one and returns the frame with boxes and cue labels
drawn on it, through the same `/api/state` the rest of the page already polls.

```bash
python tools/dashboard/server.py --config LLMDet/configs/attention_runtime.yaml \
    --session tools/dashboard/sessions/0325 --device cuda:0
#  -> open the page, press "Start camera"
```

The panel only appears when the server was started with `--config`; without a
detector there is nothing to run on the frames.

**Why the capture is in the browser and not `--video 0`.** `--video 0` opens a
capture device *on the machine running the server*. On a Hugging Face Space no
camera is attached to that machine, and on a shared GPU box device 0 belongs to
whoever plugged it in. Neither is what a viewer means by "my camera". The two
modes are both kept, and they are different things: `--video 0` is for running
the dashboard locally, the panel is for everyone else.

Three consequences worth stating plainly:

* **Video leaves the viewer's machine.** Frames are sent to the server, held
  only long enough to analyse, and not written to disk — but they are sent. The
  server binds `0.0.0.0` and has no authentication (see the exposure note in
  `server.py`), so use `--host 127.0.0.1`, or put it behind something, before
  pointing a real classroom at it. `--blur-faces` applies here as everywhere.
* **`getUserMedia` needs a secure context**: `https`, or `http` on `localhost`.
  On plain `http` to a remote host the browser exposes no camera API at all, and
  the panel says so rather than failing silently.
* **Most frames are dropped, and that is the mechanism.** The browser pushes at
  up to 10/s; the pipeline manages ~1 fps on CPU and a few on a GPU. Measured on
  a T4-class CPU run: 539 pushed, 97 analysed, 442 skipped (82%). Skipping is
  what keeps the overlay current — see the next section — and the panel reports
  all three counts rather than showing a frame rate that looks healthy.

The page pushes one frame at a time and schedules the next only after the
previous POST returns, so it self-paces to whatever the server can absorb
instead of queueing frames the pipeline would drop anyway.

### Live: a webcam, an IP camera, or a video file

```bash
# webcam / capture card — the device index
python tools/dashboard/server.py --config LLMDet/configs/attention_runtime.yaml \
    --video 0 --device cuda:0

# IP camera
python tools/dashboard/server.py --config LLMDet/configs/attention_runtime.yaml \
    --video rtsp://user:pass@10.0.0.5/stream1 --device cuda:0

# a file, played through the full pipeline rather than from a cache
python tools/dashboard/server.py --config LLMDet/configs/attention_runtime.yaml \
    --video LLMDet/0325.mp4 --device cpu
```

`--video` takes a device index, a URL with a scheme OpenCV can open
(`rtsp/rtsps/http/https/udp/tcp/rtmp`), or a path. Model switching works here
too, but each switch restarts the source and re-runs the whole pipeline.

**A camera is not a file, in two ways that matter.**

*Frames keep arriving whether or not anything reads them.* The pipeline runs at
a few frames a second, so reading sequentially would mean consuming a queue: the
overlay falls further behind the room the longer it runs, unbounded, with
nothing on screen saying so. A reader thread drains the source and hands the
pipeline the newest frame; everything in between is dropped and **counted**. The
header shows `2.1 fps processed · 92% of 25 fps camera dropped`, amber past 50%
and red past 90%. `cv2.CAP_PROP_BUFFERSIZE` is not used for this — the FFMPEG
backend that handles RTSP ignores it.

*Timestamps have to be wall-clock.* Alert thresholds are durations — "head down
for 30 s". For a file, frame index ÷ fps is real time. For a live source with
dropped frames it is not: at 2 fps processed against a 25 fps camera, `n/fps`
runs about 12× slow, and a 30-second episode would be announced roughly six
minutes late. Live sources timestamp from the wall clock instead.

**Throughput.** Measured 5.77 FPS at detector-stride 3:2 on a GPU (§11.17),
p50 158 ms. That is the ceiling for live use; a real classroom camera at 25 fps
will drop most frames, which is fine for sustained-episode alerts (the shortest
per-channel minimum is 2 s) and not fine for anything needing every frame. On
CPU it will be far slower — how much slower is unmeasured, and is the first
thing worth measuring when the machine frees up.

**One caveat not yet addressed.** `max_age: 45` and `min_hits: 8` are counted in
*processed* frames and were tuned at ~25 fps processed. At 2 fps a track
survives 45 processed frames ≈ 22 s of occlusion instead of 1.8 s. The stride
controller already rescales `min_hits`; `max_age` is not rescaled by capture
rate. Worth checking against a real camera before deployment.

### From a recorded cue log (no model)

```bash
python tools/dashboard/server.py --replay tools/dashboard/live_session.jsonl
```

Stdlib only — no torch, no mmdet. A cue log stores decisions rather than
features, so the selector is disabled rather than pretending to work.

---

## 3. Things that are refused rather than papered over

Each of these was a real failure mode in this project before it became a check.

| situation | what happens |
|---|---|
| a model has no fitted thresholds | load fails, pointing at `calibrate_registry.py`. An uncalibrated model never abstains, so it would look *more* decisive than a calibrated one — precisely backwards |
| a model cannot reach 85% selective accuracy at any threshold | it may display cues but raises **no alerts**, and the UI says so. The bar is not lowered to let a weak model page an instructor |
| a checkpoint's feature width disagrees with the extractor | `predict_window` raises. The old path zero-padded, making an absent measurement indistinguishable from a real one |
| a checkpoint's architecture disagrees with the config | `load_runtime_model` builds from the checkpoint's own `spec`. The old path caught the exception and ran on **randomly initialised weights** |
| a model trained on the BlazeFace flag | is fed the BlazeFace block, not the landmarker one, chosen from its own `run_record.json` |
| a session cache predates `source_width` | replay refuses rather than drawing every box at the wrong scale |
| a live source outruns the pipeline | frames are dropped to stay current and the drop rate is shown, instead of the overlay silently falling behind the room |
| an IP camera URL is passed | opened as a URL. It used to be run through `Path(...).resolve()`, which turned `rtsp://cam/s` into `<cwd>/rtsp:/cam/s` and failed naming a path the user never typed |
| an analysis is cancelled | the partial cache is deleted. Keeping it would let the first 40 s of a lecture replay as though it were the whole lecture |
| an upload names `../../etc/passwd.mp4` | reduced to `passwd.mp4` inside `uploads/`, with a containment check behind that |
| an upload would overwrite an existing name | written as `lecture-2.mp4`. Overwriting would silently invalidate any session already built from the old file |
| a model switch arrives mid-run | the previous producer thread is stopped and joined first; if it will not stop, the switch returns 409 rather than running two pipelines into one state |

---

## 4. Files

| file | role |
|---|---|
| `sources.py` | uploads and session caches on disk; filename sanitising, upload streaming |
| `model_registry.py` | sweeps → one entry per variant, best validation seed, deployability |
| `calibrate_registry.py` | per-model temperature and abstention thresholds, validation only |
| `precompute_session.py` | the model-independent front end, cached once per video |
| `session_replay.py` | one model over one cache; also a headless CLI |
| `pipeline_bridge.py` | the full live pipeline, with the same push interface |
| `server.py` | stdlib HTTP server, state, alert logic, `/api/model` switching |
| `index.html` | the page |
