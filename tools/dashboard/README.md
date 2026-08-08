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

```bash
# 0. one-off: thresholds for every model, fitted on ITS OWN validation
#    predictions. Seconds. Without this the dashboard refuses to load a model.
python tools/dashboard/calibrate_registry.py

# 1. one-off per video: the expensive pass — what the Analyse button runs.
#    Detector, tracker, CLIP and BOTH head-pose backends. CPU-bound.
python tools/dashboard/precompute_session.py \
    --config LLMDet/configs/attention_runtime.yaml \
    --video LLMDet/0325.mp4 --frames 900 --device cpu \
    --out tools/dashboard/sessions/0325

# 2. the dashboard. Instant model switching, no detector, no GPU.
python tools/dashboard/server.py --session tools/dashboard/sessions/0325
#    -> http://localhost:8080
```

Useful flags: `--speed 4` to replay faster than real time, `--blur-faces` for a
classroom deployment or a screenshot, `--model VARIANT_ID` to start on something
other than the default, `--port`.

### Headless — diff two models over the same session

```bash
python tools/dashboard/session_replay.py --cache tools/dashboard/sessions/0325 \
    --model arch/asrf_556_hp   --out /tmp/asrf.jsonl
python tools/dashboard/session_replay.py --cache tools/dashboard/sessions/0325 \
    --model ladder/transformer_552_base --out /tmp/base.jsonl
```

Same frames, same boxes, same features — the only difference is the temporal
head, so any disagreement is attributable to the model.

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
