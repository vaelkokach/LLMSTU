# Dashboard integration — design and inventory for a Branch-C mode

**Status:** design + inventory document. No code in `tools/dashboard/*`,
`LLMDet/attention/*`, or any config was modified to produce this file.
**Written:** 2026-08-09, at repo HEAD `6d4cea0` on `branch-c/overt-cue`.
**Scope:** how a future Branch-C ("OVERT-Cue") mode could be added to the
working dashboard without touching its current behaviour, and exactly what
would have to be tested to prove that.

**Provenance note on the frozen fixtures this document builds on.** A prior
agent froze two regression fixtures at `outputs/branch_c/dashboard/legacy_fixture/`
before being killed by a session limit:
- `MANIFEST.json` — full narrative manifest for `arch/asrf_556_hp`
  (`tools/dashboard/model_registry.py`'s current `default_entry()`, i.e. the
  legacy dashboard's default model), recorded at git commit `3f9b760`.
- `replay_arch_asrf_556_hp.jsonl` (900 lines, sha256
  `3129821955dbe4cc612d7abf3b5da97297b43215cc6f0070220d0000b8f1aa5b`) —
  matches the manifest exactly.
- `replay_ff_det_mstcn_553_facefound.jsonl` (900 lines, sha256
  `8fccc33965d4ba624c2c7feb2c981f1b157c6e1d6d5aca8d233ac5a6c5a10643`, on disk
  at 2026-08-09 13:47 UTC, five minutes before this task started) — this is
  the fixture for the **other** "legacy" system, the one
  `LLMDet/configs/attention_runtime.yaml:86` loads
  (`work_dirs/thesis/ff_det/mstcn_553_ff_s42/checkpoints/best.pth`). Its cue
  distribution (89.8% screen_oriented / 7.7% uncertain / 1.2% phone_use / 0.6%
  looking_away / 0.5% turned_to_peer / 0.2% head_down, over 5,073 student-frame
  observations, verified directly by parsing the file for this document) matches
  `verify_dashboard.log:8` ("ff_det/mstcn_553_facefound … screen_oriented 90%
  uncertain 8% phone_use 1%") within rounding, and its checkpoint hash
  (`714485556fcfe54cbaa264eee16e7f0b4365374b4010574c86ba8647e2b3296b`) was
  confirmed against the file on disk. **`MANIFEST.json` does not yet describe
  this second file** — no command, no per-field summary, no checkpoint/
  calibration hashes recorded for it. That gap is not fixed here: this
  document's owner is `docs/branch_c/DASHBOARD_INTEGRATION.md` only, and
  extending `MANIFEST.json` is the fixture-freezing agent's file, not mine.
  It is flagged in §8 as a prerequisite for using this second fixture as a
  release gate.
- Separately, `git merge-base --is-ancestor 3f9b760… HEAD` returns false, and
  `git branch -a --contains 3f9b760…` lists no branch — the commit object
  exists locally but is not reachable from any ref, presumably from an
  isolated worktree/session that was never merged. This is the same class of
  provenance quirk `MANIFEST.json`'s own `recorded_git_commit_caveat` already
  flags for `meta.json`'s recorded commit, not a new problem, and it does not
  affect the file-hash claims above, which were checked directly against the
  files on disk rather than by checking out that commit.

---

## 1. Dependency map

### 1.1 HTTP surface (`tools/dashboard/server.py`)

All eight routes verified live in `outputs/branch_c/dashboard/legacy_fixture/verify_dashboard.log:20-28`
("== 4. HTTP surface ==", 8/8 PASS) and independently confirmed here by
grepping `tools/dashboard/index.html` for every `fetch(`/`xhr.open(` call
(`index.html:213,276,289,297,314,357,411,468`) — the browser client calls
exactly these eight and no others.

| method | path | request body | response shape | handler |
|---|---|---|---|---|
| GET | `/`, `/index.html` | — | `index.html` bytes | `server.py:202-204` |
| GET | `/api/state` | — | `{t, running, source, source_id, frame (b64 jpeg or null), students, alerts[<=25], history[[t,frac]], class_summary, model, notice, capture, job}` | `server.py:205-221` |
| GET | `/api/models` | — | `{models: [ModelEntry.to_json()], active, switchable, switch_cost}` | `server.py:222-228` |
| GET | `/api/sources` | — | `{sources, active, upload_enabled, max_upload_mb, accepts, can_analyse}` | `server.py:229-238` |
| POST | `/api/model` | `{variant_id}` | `{active: variant_id}` | `server.py:245-247` → `switch_model` (`server.py:492-504`) |
| POST | `/api/source` | `{source_id, mode?}` | `{active: source_id}` | `server.py:248-251` → `select_source` (`server.py:507-526`) |
| POST | `/api/upload?name=` | raw bytes | `{id, name, size_mb}` | `server.py:252-253,286-296` |
| POST | `/api/analyse` | `{source_id, max_frames?}` | `{status, name}` (job) | `server.py:254-258` → `start_analyse` (`server.py:533-560`) |
| POST | `/api/cancel` | — | `{cancelling: true}` | `server.py:259-265` |

Error convention (`server.py:266-277`): `ValueError`→400, `FileNotFoundError`→404,
`SystemExit`→400, `RuntimeError`→409, anything else→500 with
`{type(e).__name__}: {e}`. Any Branch-C failure path (§6) must raise one of
these four exception types to get a sane HTTP status instead of a bare 500.

### 1.2 Call graph, HTTP route → model

```
POST /api/source or /api/model
  -> select_source / switch_model            (server.py:507-526 / 492-504)
    -> start(source, entry)                  (server.py:454-489)
      -> RUNNER.start(target=...)            (server.py:152-160, Runner class 112-161)
        -> run_session(entry, cache_dir)      (server.py:363-382)   [source.kind == "session"]
             SR.SessionCache(cache_dir)                              session_replay.py:59-99
             SR.load_model(entry, device)                            session_replay.py:102-116
               -> runtime.load_runtime_model(ckpt, device, calibration)  runtime.py:100-143
             SR.replay(cache, entry, bundle, push_frame, ...)          session_replay.py:119-203
               -> cache.live_vector(rows, entry.head_pose_backend)     session_replay.py:95-99
               -> runtime.predict_window(bundle, window)               runtime.py:146-183
               -> push_frame(t, jpg, students, cue_names)               server.py:299-340
        -> run_live_source(entry, video)      (server.py:385-416)   [source.kind == "video"]
             pipeline_bridge.run_live(...)                            pipeline_bridge.py:108-363
               -> FrozenLLMDetAdapter.detect()                        pipeline_bridge.py:158-169
               -> IoUTracker.update()/.coast()                        pipeline_bridge.py:172-178,261-267
               -> HeadPoseEstimator(backend=entry.head_pose_backend)  pipeline_bridge.py:183-190
               -> StudentFeatureExtractor.extract_batch()             pipeline_bridge.py:191-194,270
               -> runtime.load_runtime_model / predict_window          pipeline_bridge.py:204,282
               -> push_frame(...)                                      server.py:299-340 (via push_fn)
        -> run_replay(path)                   (server.py:347-360)   [source.kind == "cuelog"]
             reads JSONL rows -> push_frame(r["t"], None, r["students"], r.get("cue_names", []))
```

### 1.3 Ownership, by stage

| stage | owner (file:line) | live path | session-replay path |
|---|---|---|---|
| detection | `attention/detector_adapter.py` (`FrozenLLMDetAdapter`) | `pipeline_bridge.py:158-169` | none — boxes come pre-baked from `features.npz` (`session_replay.py:73` `self.bbox`), produced once offline by `precompute_session.py` using the same detector |
| tracking | `attention/tracking.py` (`IoUTracker`) | `pipeline_bridge.py:172-178, 261-267` | none — `track_id` is a cached column (`session_replay.py:71`), fixed at precompute time |
| head pose | `attention/head_pose.py` (`HeadPoseEstimator`) | `pipeline_bridge.py:183-190` (backend from `entry.head_pose_backend`, `pipeline_bridge.py:183-184`) | selects between two cached blocks (`session_replay.py:75-76`, `95-99`), same `entry.head_pose_backend` |
| feature assembly | `attention/features.py` (`StudentFeatureExtractor`) | `pipeline_bridge.py:191-194`; width asserted against `bundle.live_input_width`, never padded (`pipeline_bridge.py:205-214`) | `SessionCache.live_vector()` (`session_replay.py:95-99`): `concat(base, head[backend])` |
| temporal model + calibration + abstention | `attention/thesis_eval/runtime.py` (`RuntimeBundle`, `load_runtime_model`, `predict_window`) | `pipeline_bridge.py:204,282` | `session_replay.py:114,169` |
| events / alerts | **inlined in `server.py`**, not `attention/events.py` | `ALERT_AFTER_S` dict (`server.py:104-105`), fired inside `push_frame` (`server.py:299-340`) | same `push_frame`, shared by both paths |
| rendering | `session_replay.py:_draw` (`206-239`, cached-frame overlay) / `pipeline_bridge.py:309-328` (live cv2 draw) | both encode JPEG q=70, base64'd in `server.py:303-304`; browser reads only `st.cue`, `st.dwell`, `st.conf` (`index.html:490-495`) | — |

The events/alerts row is a real, previously-established fact worth restating
here because it changes what "the event layer" means for a Branch-C mode:
`server.py`'s docstring (`server.py:50`) and a code comment (`server.py:102`)
both claim alignment with `attention/events.py`'s `EventConfig`, but
`attention/events.py` (`EVENT_CHANNELS`, `EventConfig`, `segment_events`,
defined at `attention/events.py:42,67-82,106-136`) is never imported anywhere
under `tools/dashboard/` — confirmed by grep, zero hits for
`attention.events`/`EventConfig`/`segment_events` in `tools/dashboard/*.py`.
The dashboard's actual alert rule is a single-threshold-per-cue dwell check
(`server.py:104-105,321-340`), independently reimplemented and *simpler* than
`attention/events.py`'s hysteresis (`min_duration_s` + `max_gap_s`,
`attention/events.py:85-103`). A Branch-C mode that wants the more rigorous
event semantics BRANCH_C_PROTOCOL.md §3.1/§6 describes (six-cue macro-F1,
causal-only features, proper episode segmentation) needs its own alert path —
see §4.

---

## 2. Current JSONL schema, from real records

Four fields recur everywhere but are not universal, so this is written per
field, not per file.

### 2.1 Top level

Every JSONL line is one video frame:

```
{"t": <float, seconds>, "students": {<seat_id str>: <student record>}, ...}
```

- `t` — always present, always a float.
- `students` — always present; a dict keyed by `str(track_id)` ("seat"); may
  be `{}` on a frame with no confirmed track.
- `dt` — present in the four `tools/dashboard/*.jsonl` cue-log files
  (`demo_session.jsonl`, `live_session.jsonl`, `live_session_mstcn.jsonl`,
  `live_session_facedet.jsonl`), written by `pipeline_bridge.py:330-333`
  (`rec.write(json.dumps({"t":..., "students":..., "dt":..., "cue_names":...}))`).
  **Absent** in both frozen fixture files
  (`outputs/branch_c/dashboard/legacy_fixture/replay_*.jsonl`), because those
  were written by `session_replay.py`'s headless `main()`/`push()`
  (`session_replay.py:270-274`, `fh.write(json.dumps({"t": t, "students": students}))`),
  which does not carry `dt` or `cue_names` at all. `server.py`'s consumer,
  `run_replay` (`server.py:347-360`), tolerates the absence:
  `r.get("dt", 1.0)` (`server.py:358`) defaults to 1 second/frame. So the
  fixture files *would* replay through `--replay` today, just at a different
  assumed frame spacing than they were actually recorded at (0.5 s at
  30 fps/frame-index-spacing-2, not 1.0 s) — a real, if minor, schema gap
  worth fixing if the fixture is ever driven through the live server rather
  than through `session_replay.py --out` directly.
- `cue_names` — present in the same four cue-log files, absent in both
  fixtures, for the same reason. `push_frame` never reads this parameter
  (`server.py:299` signature includes it, the body `server.py:301-340` never
  touches it) — it is accepted and silently dropped. **Confirmed dead
  parameter** as of this commit.

### 2.2 Per-student record — union of fields observed, with which files carry which

| field | type | `demo_session.jsonl` | `live_session.jsonl` | `live_session_{mstcn,facedet}.jsonl` | frozen fixtures |
|---|---|---|---|---|---|
| `cue` | str, one of the 6 taxonomy cues + `uncertain` | yes | yes | yes | yes |
| `conf` | float, rounded to 2dp | yes | yes | yes | yes |
| `dwell` | float, seconds on the current displayed cue | yes | yes | yes | yes |
| `alerted` | bool | yes | yes | yes | yes |
| `bbox` | `[x1,y1,x2,y2]` float list, rounded to 1dp | **no** | **no** | **no** | **yes** |
| `raw_cue` | str, pre-abstention argmax | **no** | **no** | yes | yes |
| `abstained` | bool | **no** | **no** | yes | yes |
| `alert_allowed` | bool | **no** | **no** | yes | yes |

Read directly off real lines for this document:
- `demo_session.jsonl:1` — `{"cue":"screen_oriented","conf":0.85,"dwell":0.0,"alerted":false}`, no `bbox`/`raw_cue`/`abstained`/`alert_allowed`. This is the oldest format, predating abstention — exactly the case `server.py:326-327`'s comment describes ("Replay logs recorded before abstention existed carry no `alert_allowed` key, so the default admits them").
- `live_session.jsonl` (first non-empty frame, `t=0.533`) — same reduced shape as `demo_session.jsonl`, still no `bbox`/`raw_cue`/`abstained`/`alert_allowed`, despite `cue_names` already including `"uncertain"`.
- `live_session_mstcn.jsonl` / `live_session_facedet.jsonl` (first non-empty frame, `t=0.533`) — full abstention triplet present (`raw_cue`, `abstained`, `alert_allowed`), but still **no `bbox`**.
- `outputs/branch_c/dashboard/legacy_fixture/replay_arch_asrf_556_hp.jsonl` (first non-empty frame, `t=0.5`) — the only files with `bbox` present, alongside the full abstention triplet.

Why the difference: `pipeline_bridge.run_live`'s `push_fn` call
(`pipeline_bridge.py:296-307`) has *always* written `bbox`, `raw_cue`,
`abstained`, `alert_allowed` for as long as that abstention logic has
existed, and `session_replay.replay` (`session_replay.py:181-190`) matches it
field-for-field. The three checked-in `.jsonl` files without `bbox` are
therefore recordings made either before that field was added to the writer,
or via some other code path than the one currently in the repo — this is
schema drift across time, not a live divergence between two current code
paths. `push_frame`'s own logic (`server.py:299-340`) never reads `bbox`; it
is consumed only by the browser's future use (it is not currently drawn
client-side either — `index.html:490-495` reads only `cue`/`dwell`/`conf`),
so the gap is inert today but would matter to any client that starts reading
`bbox` from an old file. `.get()` with defaults protects every currently-read
field (`dwell` defaults via `s.get("dwell", 0)` at `server.py:330`,
`alert_allowed` defaults via `s.get("alert_allowed", True)` at
`server.py:328`); `cue` and `conf` are read unconditionally
(`s["cue"]`/direct dict access in the same block) and would `KeyError` on a
record missing them — none of the six files omit either.

---

## 3. Proposed Branch-C schema version

**Rule: additive only.** Every existing key keeps its name, type and meaning.
A consumer that only reads the §2 fields must see byte-identical values.

### 3.1 New top-level field

```json
{"schema_version": "1.0.0", ...}
```

Absent → `"1.0.0"` (today's implicit schema, matching `dashboard_session/1.0.0`
already used as `meta.json`'s `cache_version` per `MANIFEST.json`'s
`session_cache.meta_json.cache_version`). A Branch-C-carrying record is
`"2.0.0-branch_c"` or similar; consumers that don't understand the string
still see every field they already parse.

### 3.2 New per-student namespace: `branch_c`

Present only when the active mode is OVERT-Cue Demo or Research Replay
(§4); absent (not `null`, not `{}` — genuinely absent) under Legacy Branch B,
so an old consumer's `for k, v in student.items()` loop sees no new keys to
misinterpret.

```json
"branch_c": {
  "reliability": {
    "modalities": ["appearance", "seat_pose", "body"],
    "alpha": [0.61, 0.33, 0.06],
    "r": [1.42, 0.51, -0.88],
    "tau": 1.0
  },
  "pose": {
    "source": "causal_seat",
    "ref_confidence": 0.83,
    "geodesic_rel_rad": 0.27,
    "omega_rad_s": [0.01, -0.02, 0.00],
    "R_ref_source": "causal_seat"
  },
  "provenance": {
    "experiment_id": "overt_full_fold2_s43",
    "checkpoint": "outputs/branch_c/checkpoints/fold2/overt_full_s43/best.pth",
    "feature_schema_version": "branch_c_pose/1.0.0",
    "fold": 2
  },
  "abstention_reason": "display_threshold"
}
```

Field sourcing, so this is not invented:
- `reliability.alpha` / `reliability.r` — exactly the `diagnostics` dict
  `ReliabilityFusion.forward` returns, keys `"alpha"` and `"r"`
  (`LLMDet/attention/branch_c/fusion.py:139-141,171-172`). The module's own
  docstring is explicit that these are "routing/reliability diagnostics …
  never … explanations … never … causal" (`fusion.py:21-27`); the schema
  reuses that exact framing so a future UI string cannot drift from it.
- `pose.source`, `pose.ref_confidence` — `ReferenceEstimate.source` and
  `.confidence` (`LLMDet/attention/branch_c/canonical.py:309-320`); the three
  concrete estimator sources are the literal strings `"scene"`, `"torso"`,
  `"causal_seat"` / `"causal_seat:insufficient"` / `"scene:missing"`
  (`canonical.py:348,351,369,410,416`).
- `pose.geodesic_rel_rad`, `pose.omega_rad_s` — two of the six named blocks in
  `POSE_FEATURE_LAYOUT` (`canonical.py:442-450`: `rot6d_rel(6)`, `log_rel(3)`,
  `geodesic_rel(1)`, `omega(3)`, `alpha(3)`, `ref_confidence(1)` = 17 dims
  total). Not all 17 dims need to round-trip into JSON per frame; the two
  picked here (`geodesic_rel` and `omega`) are the ones with a direct,
  human-legible unit (an angle, an angular rate) that a dashboard could show
  next to the cue chip without retraining an instructor to read a 6D vector.
- `provenance.*` — mirrors the fields `outputs/branch_c/RUNS.jsonl` is already
  required to carry per run (BRANCH_C_PROTOCOL.md §9: "git commit, dirty-diff
  hash, environment lock hash, dataset/split hash, feature-schema hash,
  checkpoint hash, command, seed"), narrowed to what an instructor-facing
  record needs to answer "which model said this" — `fold` matters uniquely to
  Branch C because BRANCH_C_PROTOCOL.md §2.1 registers **nested grouped
  cross-validation**, unlike Legacy Branch B's single train/val/test split;
  there is no single "the Branch-C model," only "fold N's model."
- `abstention_reason` — new, not present in any existing record. Today's
  abstention (`runtime.py:174-181`) is binary (`abstained: bool`) and gives no
  reason beyond "below `display_threshold`." Branch C's reliability gate adds
  a second possible reason — every modality masked out
  (`fusion.py:66-69,168-171`, "returns exactly zero … which is what makes
  `z_fused == z_base`") — so an instructor-facing string needs to distinguish
  "the model doesn't know" from "nothing useful was observed this frame." The
  field is a closed string enum: `"display_threshold"` |
  `"all_modalities_missing"` | `"reference_uninitialised"` (the last for
  `canonical.py:405-410`'s `min_observations` gate).

### 3.3 Worked example — before / after

**Before** (today, `arch/asrf_556_hp`, taken verbatim from
`replay_arch_asrf_556_hp.jsonl`'s first non-empty frame):

```json
{"t": 0.5, "students": {"6": {
  "cue": "looking_away", "conf": 0.5,
  "bbox": [784.1, 484.2, 896.6, 684.1],
  "dwell": 0.0, "alerted": false,
  "raw_cue": "looking_away", "abstained": false, "alert_allowed": false
}}}
```

**After** (same frame, hypothetical OVERT-Cue Demo mode; every "before" key
unchanged, `branch_c` added, `schema_version` added at top level):

```json
{"schema_version": "2.0.0-branch_c", "t": 0.5, "students": {"6": {
  "cue": "looking_away", "conf": 0.5,
  "bbox": [784.1, 484.2, 896.6, 684.1],
  "dwell": 0.0, "alerted": false,
  "raw_cue": "looking_away", "abstained": false, "alert_allowed": false,
  "branch_c": {
    "reliability": {"modalities": ["appearance", "seat_pose"], "alpha": [0.71, 0.29], "r": [0.9, -0.4], "tau": 1.0},
    "pose": {"source": "causal_seat", "ref_confidence": 0.62, "geodesic_rel_rad": 1.31, "omega_rad_s": [0.04, 0.01, -0.02]},
    "provenance": {"experiment_id": "overt_full_fold2_s43", "checkpoint": "outputs/branch_c/checkpoints/fold2/overt_full_s43/best.pth", "feature_schema_version": "branch_c_pose/1.0.0", "fold": 2},
    "abstention_reason": null
  }
}}}
```

### 3.4 Which consumers read which fields — evidence, not assertion

Grepped directly (repeated from the citations above, collected here as the
compatibility argument):

| consumer | fields read | citation |
|---|---|---|
| `server.py` `push_frame` | `s["cue"]`, `s.get("dwell", 0)`, `s.get("alert_allowed", True)`, `s.get("alerted")` | `server.py:307,313,321,328,330` |
| `server.py` `/api/state` | passes the whole `STATE["students"]` dict through unmodified (no per-field access) | `server.py:213` |
| `index.html` (browser) | `st.cue`, `st.dwell`, `st.conf` | `index.html:490,494,495` |
| `session_replay.py main()`'s `push()` | `s["cue"]` only (for the cue histogram) | `session_replay.py:271-272` |
| `verify_dashboard.py`'s `agreement()`/`replay_cues()` | `s["cue"]` (per-frame agreement between two models) | inferred from `verify_dashboard.log`'s "do the models actually differ?" section; not re-derived line-by-line here since `tools/dashboard/verify_dashboard.py` is out of this document's file scope to edit, only to read for citation |

No existing consumer reads a field that this schema removes, renames, or
retypes, and no existing consumer iterates `student.keys()` expecting a
closed set (`push_frame` and `index.html` both name-address specific keys) —
so an unrecognised `branch_c` key added under a student record is inert to
every consumer above by construction, not by convention.

---

## 4. The integration seam

### 4.1 Recommended seam

Three modes, one new CLI flag, zero changes to the Legacy Branch B code path
when the flag is absent or `legacy`:

```
python tools/dashboard/server.py --session ... [--dashboard-mode {legacy,demo,research_replay}]
```

- **Default `legacy`** (or the flag omitted): today's `server.py:609-700`
  `main()`, byte-for-byte. This is what makes the rollback in §7 a single
  flag flip rather than a code revert.
- **`demo`**: one frozen, gate-passed Branch-C checkpoint (mirroring
  `model_registry.default_entry()`, `model_registry.py:220-224`) with the
  full `branch_c` namespace populated per §3.
- **`research_replay`**: pick any (fold, seed) combination from the nested-CV
  grid BRANCH_C_PROTOCOL.md §2.1 registers, for reproducing a specific
  reported number interactively rather than only from a table.

This mirrors the project's only existing config-selection precedent — a CLI
flag with an explicit default, not an environment variable (grepped: no
`os.environ`/`os.getenv` reference anywhere in
`tools/dashboard/*.py`/`LLMDet/attention/thesis_eval/runtime.py`/
`LLMDet/configs/*.yaml`). It is the same pattern `--model`
(`server.py:624-626`) already uses to pick among `Legacy Branch B` variants
without a second binary or a second server process.

### 4.2 Exact functions/files a future change would touch

| new artefact | mirrors | role |
|---|---|---|
| `attention/branch_c/runtime.py` (new file) | `attention/thesis_eval/runtime.py`'s `RuntimeBundle`/`load_runtime_model`/`predict_window` (`runtime.py:72-183`) | loads a Branch-C checkpoint by its own `spec` (same strict-loading contract, §4.3), runs `ReliabilityFusion.forward` (`fusion.py:151-172`), returns a result dict shaped like `predict_window`'s **plus** the `branch_c` sub-dict of §3.2 |
| `tools/dashboard/branch_c_registry.py` (new file) | `tools/dashboard/model_registry.py`'s `scan()`/`find()`/`default_entry()` (`model_registry.py:146-228`) | enumerates Branch-C checkpoints across folds/seeds instead of Legacy Branch B's flat variant list; kept **separate from** `model_registry.entries` so a Branch-C `variant_id` can never satisfy `MR.find(CONTEXT.get("entries", []), variant_id)` (`server.py:494`) by accident (§5) |
| `server.py:main()` (`server.py:609-700`) | itself | add `--dashboard-mode`; branch to a new `start()` variant only when `mode != "legacy"` |
| `server.py:start()` / `run_session()` / `run_live_source()` (`server.py:363-489`) | themselves | add a `demo`/`research_replay` branch that calls the Branch-C loader instead of `session_replay.load_model`/`SR.replay`, and merges the `branch_c` dict into each student record **after** building the existing dict — never in place of it |

### 4.3 Why this is the smallest seam

1. **No existing file's current behaviour changes when the flag is absent.**
   Every touched function (`main`, `start`, `run_session`, `run_live_source`)
   gets a new conditional branch, not a rewritten one; the `legacy` branch is
   the exact code that runs today.
2. **The strict-checkpoint-loading contract already generalises.**
   `attention_runtime.yaml:39-42`'s note — "model architecture and feature
   width are read from the CHECKPOINT's own spec, not from this file" — and
   `runtime.py:100-143`'s implementation of that (`ck.get("spec")`,
   `strict=True` state-dict loading, `runtime.py:107-111,124-127`) is
   architecture-agnostic: nothing in it assumes MS-TCN/ASRF/transformer over
   `attention.thesis_eval.models`. A Branch-C checkpoint that saves the same
   `spec` shape (confirmed real: `LLMDet/work_dirs/thesis/arch/asrf_556_hp_s43/run_record.json`
   has `spec = {experiment_id, model, feature_config, seed, ...}`) can reuse
   the identical strict-load discipline in a sibling loader without
   duplicating the "never guess, never pad" logic — the lead's note that this
   behaviour "already exists" (`attention_runtime.yaml:39-41`) is confirmed
   correct and nothing here reinvents it.
3. **A second, separate registry is smaller than a shared one with a type
   tag.** Keeping Branch-C entries out of `CONTEXT["entries"]` means
   `switch_model`'s existing lookup (`server.py:494`,
   `MR.find(CONTEXT.get("entries", []), variant_id)`) needs **zero changes**
   to remain incapable of loading a Branch-C variant through the Legacy
   Branch-B code path — the isolation in §5 falls out of the seam's shape
   rather than needing a runtime guard.
4. **The schema is additive**, so `push_frame` (`server.py:299-340`) needs no
   change to keep working when it receives a student dict carrying an extra
   `branch_c` key — every read in it is a named-key `.get()`/`[...]`, never an
   iteration.

---

## 5. Mode isolation

Concretely, per state category, where it lives today and what would leak if
a Branch-C mode were bolted on carelessly:

| state | lives at | leak risk if careless | how isolation should be enforced |
|---|---|---|---|
| **Live per-frame state** (`frame_jpeg_b64`, `students`, `alerts`, `history`, `class_summary`, `capture`) | module-level `STATE` dict (`server.py:85-99`) | none by default: `reset_state()` (`server.py:163-172`) is called via `Runner.start()`'s `on_start` callback (`server.py:156-157,175`) on **every** switch, session or model, clearing all of the above. Already safe for a mode switch as long as the new mode also goes through `RUNNER.start(...)`. | keep using `Runner.start()` for every mode; do not add a second entry point that writes `STATE` directly |
| **Tracker identity** | local `tracker = IoUTracker(...)` inside `pipeline_bridge.run_live` (`pipeline_bridge.py:173-178`); session replay has no tracker object at all — `track_id` is a cache column (`session_replay.py:71`) | none today: a fresh `IoUTracker` is constructed per call, not reused. Risk appears **only** if a Branch-C live path is added that shares one `IoUTracker` instance across mode switches for efficiency — track IDs from a Legacy-Branch-B run would then leak into a Branch-C run's dwell/alert bookkeeping (`dwell` dict keyed by `track_id`, `pipeline_bridge.py:224,289-295`). | keep constructing a fresh tracker (and a fresh `hist`/`dwell`/`StrideController`, all already function-local: `pipeline_bridge.py:171-176,223-224`; `session_replay.py:137-140`) per `run_*` call, as today |
| **Feature buffers** (`hist` deques) | function-local `defaultdict(lambda: deque(maxlen=win))` (`pipeline_bridge.py:223`, `session_replay.py:139`) | none — fresh per call, discarded when the function returns/thread stops | no change needed; a Branch-C loader should follow the same "fresh local buffer per replay/run call" convention |
| **Calibration temperature / thresholds** | fields on a freshly-returned `RuntimeBundle` (`runtime.py:72-97`), never cached at module level | none today for Legacy Branch B — a stale `bundle` only survives inside the stopped thread's closure, and `Runner.stop()` joins that thread (default `join_timeout=60.0`, `server.py:141-150`) before the new one starts | a Branch-C `RuntimeBundle`-equivalent must be constructed the same way (fresh return value, no module-level cache); if it is cached for performance, the cache key must include the mode, not just the checkpoint path |
| **CONTEXT dict** (`entry`, `source`, `switch_cost`, `config`, `device`, `entries`) | module-level `CONTEXT = {}` (`server.py:177`), mutated by `start()`/`switch_model()`/`select_source()`/`main()` | **highest structural risk**: `CONTEXT` is one flat namespace with no mode key today. If a Branch-C code path writes `CONTEXT["entry"]` using the same key a Legacy lookup later reads without checking which registry produced it, a Legacy calibration file could get paired with a Branch-C checkpoint's logits, or vice versa — silently, because nothing currently type-tags what is in `CONTEXT["entry"]`. | add `CONTEXT["mode"]` and `CONTEXT["branch_c_entries"]` (kept separate from `CONTEXT["entries"]` per §4.2); every function that reads `CONTEXT["entry"]` must first check `CONTEXT["mode"]` agrees with the entry's origin; a unit test should construct a `legacy` `ModelEntry` and a `demo`-mode Branch-C entry with the *same* `variant_id` string and assert `switch_model`/its Branch-C equivalent never cross-loads one for the other |
| **Session cache identity** | `sessions/<video-stem>/` keyed by filename only (`sources.py:126-127` `session_dir_for`) | a session built for Branch-C (extended `features.npz` carrying quality/context inputs `ReliabilityHead` needs, `fusion.py:92-101`) would collide on-disk with a same-named Legacy session, and `start_analyse` already refuses to rebuild over an existing one (`server.py:546-547`, "already has a session; delete it first") — which prevents silent overwrite but also means the two systems cannot both have a session for `0325.mp4` under today's naming. | version `meta.json` (it already has an implicit `cache_version`, e.g. `dashboard_session/1.0.0` per `MANIFEST.json`'s `session_cache.meta_json.cache_version`) and have `SessionCache.__init__` (`session_replay.py:62-82`) or its Branch-C successor refuse to load a cache whose recorded schema version doesn't support the requested mode, with the exact message shape specified in §6 — never silently reinterpret an old cache's columns as new ones |
| **Alert thresholds** (`ALERT_AFTER_S`) | module-level constant (`server.py:104-105`) | none today, and it should stay a shared constant — Branch C keeps the same six-cue taxonomy (BRANCH_C_PROTOCOL.md §3.1), so a `phone_use` episode means the same thing in either mode. Risk is only that Branch C's richer event semantics (§1.3's note that `attention/events.py` is unused today) might tempt an implementer to give Branch C a *different* dwell threshold for the same cue name, which would make the two modes' alerts incomparable for no principled reason. | if Branch C's event layer differs from `ALERT_AFTER_S`, it must be a distinctly-named constant, not a silent override of the same name |

**How this gets tested**: a single pytest that (a) starts the server
programmatically in `legacy` mode, replays a few frames, records
`STATE`/`CONTEXT` snapshots; (b) switches to `demo` mode; (c) switches back to
`legacy`; (d) asserts every Legacy-mode field in the final `STATE`/`CONTEXT`
snapshot matches the first snapshot's *shape and value provenance* — same
`entry.checkpoint`, same calibration file path, same tracker/dwell state
reset to empty. This is the direct executable form of the "mode switching
leaks no state" compatibility gate already registered in
BRANCH_C_PROTOCOL.md §7.3.

---

## 6. Failure and fallback

**Non-negotiable:** the legacy launch path (`python tools/dashboard/server.py
--session ...`, no `--dashboard-mode` flag) must succeed on a machine that
has never heard of Branch C. This has one direct consequence for how the new
code is written: every Branch-C-only import must be deferred inside the
mode-specific function, exactly like `pipeline_bridge.run_live` already
defers `cv2`/`torch`/`yaml`/`attention.*` imports to inside the function
body (`pipeline_bridge.py:126-137`) rather than at module top, and exactly
like `session_replay.replay` defers `cv2`/`attention.taxonomy`/
`attention.thesis_eval.runtime` the same way (`session_replay.py:128-130`).
`import attention.branch_c.fusion` (or `canonical`, or a future full-range
head-pose package) must not appear at `server.py`'s module level.

**Required error content**, matching the existing message style at
`runtime.py:107-111` (missing spec), `runtime.py:114-120` (column overflow),
`pipeline_bridge.py:186-190` (head-pose backend unavailable),
`pipeline_bridge.py:210-214` (feature-width mismatch, "Refusing to pad — a
zero block is indistinguishable from a real measurement"), and
`session_replay.py:65-68,108-113` (missing cache / missing calibration):

A Branch-C failure must name, in one message:
1. **which model** — the mode (`demo`/`research_replay`) and, for
   `research_replay`, the (fold, seed) pair;
2. **which checkpoint** — the resolved path it tried to load;
3. **which feature schema** — the expected `feature_schema_version` /
   column count versus what the session cache or live extractor actually
   produced (mirroring `pipeline_bridge.py:211-213`'s exact phrasing
   pattern: "live features are `{got}`-dim but `{experiment_id}` needs …");
4. **the recovery action** — concretely, one of: "select `--dashboard-mode
   legacy`", "rebuild the session cache with the Branch-C precompute path",
   or "checkpoint not found at `{path}`; retrain or point at a different
   fold/seed."

None of these may be swallowed into a bare `except Exception: pass` — the
March post-mortem's whole reason for existing
(`attention/thesis_eval/runtime.py:14-25`'s docstring narrates exactly this
failure class: `strict=False` inside a bare `try/except` silently running on
random weights) is precisely the failure mode a Branch-C loader must not
reintroduce. Raise `SystemExit` for a hard configuration error (matching
`runtime.py`'s convention) or `RuntimeError` for a recoverable one that
`server.py`'s existing `do_POST` handler already turns into HTTP 409
(`server.py:272-273`) — never a bare `except`.

---

## 7. Rollback

Because the seam in §4 gates every Branch-C behaviour behind
`--dashboard-mode`, defaulting to `legacy`, the operational rollback is not a
code change at all:

```
python tools/dashboard/server.py --session tools/dashboard/sessions/0325
```

— today's exact invocation, unchanged, with the flag simply omitted (or
passed explicitly as `--dashboard-mode legacy`). No redeploy, no config
edit, no restart-with-a-different-checkpoint dance: this is the same command
already recorded as reproducing `MANIFEST.json`'s frozen fixture.

If the seam's *code* itself needs undoing (e.g. the new conditional branches
in `main()`/`start()` introduced a bug reachable even in `legacy` mode — the
one way the "additive only" promise could be broken), the documented
one-command rollback is:

```
git revert <sha-of-the-branch-c-integration-commit>
```

This is exact and singular *because* §4.3 keeps the integration to a small,
identifiable set of new files plus new conditional branches in four existing
functions — if it is landed as one commit (as the repository's convention of
one focused commit per change already shows throughout this branch's log,
e.g. `582df32`, `29afa5f`, `6229db2`), reverting that one commit restores
every touched file to its pre-integration byte content, which is strictly
stronger than the flag-based rollback above (it also removes the new files
and the new argparse option, not just avoids exercising them).

---

## 8. Regression test plan

**Inputs already in hand**, not to be regenerated by this document:
- `outputs/branch_c/dashboard/legacy_fixture/MANIFEST.json` — full command,
  hashes, and `per_field_summary` for `arch/asrf_556_hp` over
  `tools/dashboard/sessions/0325`.
- `outputs/branch_c/dashboard/legacy_fixture/replay_arch_asrf_556_hp.jsonl` —
  the frozen output, sha256 `3129821955dbe4cc612d7abf3b5da97297b43215cc6f0070220d0000b8f1aa5b`.
- `outputs/branch_c/dashboard/legacy_fixture/replay_ff_det_mstcn_553_facefound.jsonl` —
  the second system's frozen output, sha256
  `8fccc33965d4ba624c2c7feb2c981f1b157c6e1d6d5aca8d233ac5a6c5a10643` (see the
  provenance note at the top of this document: its `MANIFEST.json` entry does
  not exist yet — **prerequisite before this file can gate anything**: the
  fixture-freezing agent needs to add its command/hash block to
  `MANIFEST.json`, the same way the ASRF one is documented, or a "regression"
  against it cannot be told apart from "we never recorded what produced it").
- `outputs/branch_c/dashboard/legacy_fixture/verify_dashboard.log` — 8/8 HTTP
  checks PASS, 18/18 model-card-vs-evaluator matches, 4-model cross-comparison
  showing genuine disagreement (85.0–91.5%), all against the same session.

**Procedure, before any Branch-C code is merged:**
1. Confirm the environment still reproduces the baseline: re-run
   `MANIFEST.json`'s `command` (`outputs/branch_c/dashboard/legacy_fixture/MANIFEST.json`
   → `command_str`) and diff the resulting sha256 against
   `3129821955dbe4cc612d7abf3b5da97297b43215cc6f0070220d0000b8f1aa5b`. This
   step exists to separate "the integration changed something" from "the
   machine/BLAS build changed something," per the manifest's own
   `how_a_future_test_should_use_this` steps 3-4.
2. Re-run `python tools/dashboard/verify_dashboard.py --cache
   tools/dashboard/sessions/0325 --device cpu --port <free port>` and diff its
   output against `verify_dashboard.log`'s baseline (8/8 HTTP checks, 18/18
   card matches, same four-model pairwise-agreement numbers to within the
   noise of re-running on the same machine — these are deterministic per
   `MANIFEST.json`'s `why_this_command_is_deterministic`, so an exact match is
   the expectation, not a tolerance).

**Procedure, after Branch-C code lands (with the new mode present but
exercised only via `--dashboard-mode demo`/`research_replay`, never
implicitly):**
3. Re-run step 1's exact command **unmodified** (no new flags). Byte-identical
   sha256 is the target and is achievable on CPU — the lead independently
   reproduced it once already (`MANIFEST.json`'s narrative: "confirmed it
   reproduces BYTE-IDENTICALLY … 2m34s on CPU"). Any deviation here means the
   `legacy` code path changed, which §4.3 says it must not.
4. Re-run step 2 unmodified; same 8/8 and 18/18 expectation.
5. Only once 3–4 pass, add new tests exercising `--dashboard-mode demo` and
   `research_replay` against a Branch-C session cache, asserting the §3
   schema (`schema_version` present, `branch_c` sub-dict present with the
   exact keys of §3.2, every §2 field still present with its original type)
   and the §5 isolation test (switch `legacy` → `demo` → `legacy`, assert
   `STATE`/`CONTEXT` return to a Legacy-equivalent shape).

**Tolerance.** Byte-identical sha256 for steps 1/3, because it is proven
achievable on CPU for this exact command
(`predict_window` is `@torch.no_grad()`, `runtime.py:146`; no autocast; no
wall-clock pacing in headless replay, `session_replay.py:277`
`realtime=False`). If a *deliberate, reviewed* change to floating-point
behaviour is later made to shared code (e.g. a numerically different but
equivalent reformulation inside `predict_window`), fall back to
`MANIFEST.json`'s own documented tolerance: same `n_frames_written`, same
`track_ids`, `cue_distribution` within ±1 frame per class, same
`n_abstained_observations`.

**Release blocker.** Any sha256 mismatch in steps 1/3, or any
`per_field_summary` deviation beyond the stated tolerance, **that is not
traceable to a deliberately changed, hash-recorded input** (checkpoint file,
calibration file — both hash-checked in `MANIFEST.json`) is a release
blocker per BRANCH_C_PROTOCOL.md §7.3 ("Legacy Branch-B dashboard replay
reproduces its frozen fixture within predefined tolerances … Legacy Branch B
remains the dashboard default until all three gates pass"). Any drop below
8/8 HTTP checks or 18/18 card matches in step 4 is equally a blocker — those
numbers are currently 8/8 and 18/18, not "mostly passing."

---

## 9. Risk register — integration-specific

Ranked by how much of the legacy path each risk can silently damage if
mis-implemented, each with the smallest mitigation.

**R1 (highest) — `CONTEXT` has no mode tag, so a Branch-C entry and a Legacy
entry can share ambient state by construction.**
Detailed in §5's `CONTEXT` row: `CONTEXT["entry"]` is one flat key read by
multiple functions (`start`, `switch_model`, `run_session`,
`run_live_source`) with no check on which registry produced the value.
*Smallest mitigation:* add `CONTEXT["mode"]` as the single new key gating
every other Branch-C read/write of `CONTEXT`; one assertion at the top of
each touched function (`assert CONTEXT.get("mode", "legacy") == expected`)
turns a silent cross-load into an immediate, loud `AssertionError` instead of
a wrong number quietly shown to an instructor.

**R2 — module-level import of a Branch-C dependency breaks the legacy launch
path on a machine without it.**
`server.py` currently imports nothing beyond stdlib and `sources` at module
level (`server.py:69-83`); `torch`/`cv2`/`attention.*` are all deferred into
function bodies in both `pipeline_bridge.py:126-137` and
`session_replay.py:128-130`. A careless `from attention.branch_c import
fusion` at the top of `server.py` or a new module would make `--session
tools/dashboard/sessions/0325` (no Branch-C anything) fail at import time on
a machine that has, say, never installed a full-range head-pose package
Branch C might add later. *Smallest mitigation:* a one-line lint/test —
`python -c "import ast; ...` or simpler, `grep -n "^import\|^from" server.py`
— asserting no `attention.branch_c` import appears outside a function body,
run in CI before merge.

**R3 — the second frozen fixture
(`replay_ff_det_mstcn_553_facefound.jsonl`) has no manifest entry, so a
regression against it cannot be distinguished from "we never recorded what
produced it."**
Detailed at the top of this document and in §8. *Smallest mitigation:* one
manifest entry, symmetrical to the existing `arch/asrf_556_hp` block, added
by the fixture-freezing agent (not this document) before this fixture is
used as a release gate — a five-minute follow-up, not a redesign.

**R4 — session cache schema has no version field consumers check, so a
Branch-C-extended `features.npz` could be loaded by the Legacy path (or vice
versa) and silently produce a wrong-shape vector rather than an obvious
error.**
`SessionCache.__init__` (`session_replay.py:62-82`) trusts `meta.json` and
`features.npz` to have the columns it expects; there is no explicit
"`cache_version` must be >= X" assertion anywhere in the read path today
(only an implicit one — `session_replay.py:95-99`'s `if backend not in
self.head: raise SystemExit`, which catches a *missing block*, not a
*wrong-but-present* one). *Smallest mitigation:* one new assertion at the top
of `SessionCache.__init__` checking `meta.json`'s `cache_version` against a
minimum the calling mode requires, with the exact §6 error shape — this is a
single `if` statement, not a cache-format migration.

**R5 — `Runner.stop()`'s 60 s join timeout could turn a slow Branch-C
inference frame into a spurious "previous run did not stop" `RuntimeError`
on a mode switch.**
`server.py:141-150`: `stop(self, join_timeout=60.0)` raises if the previous
thread is still alive after 60 s. ASRF already costs 4.5x MS-TCN per the
lead's prior finding (per-frame ~170 ms at 5.9 fps measured in
`replay_arch_asrf_556_hp.log:3`, "900 frames in 152.2s (5.9 fps)"); a Branch-C
model with an added `ReliabilityFusion` forward pass and pose-canonicalisation
math (`canonical.py`'s SO(3) log/exp calls, none of which are asymptotically
expensive but all of which add wall-clock on CPU) is unlikely to individually
exceed the 60 s budget for *one frame*, but a mode switch mid-window (waiting
for `should_stop()` to be checked once per frame,
`pipeline_bridge.py:248-249`/`session_replay.py:146`) could still be slow
enough to be visible to a user as a stuck "switching…" state, without
actually breaking anything. *Smallest mitigation:* none required for
correctness; if it becomes a UX complaint, lower `should_stop()`'s check
granularity (e.g. check inside the per-track loop too, not only per frame) —
an optimisation, not a regression fix.
