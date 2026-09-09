#!/usr/bin/env python3
"""Instructor dashboard — real-time analytics, alerts, and a model selector.

Implements the `Thesis_Topic.md` deliverable:

    "The project will give teachers/professors a dynamic dashboard showing
     real-time analytics and alerts so enabling quick interventions..."

Serves a live view over the pipeline: annotated video frame, per-student cue
state, a rolling alert log for sustained off-task episodes, and class-level
analytics over time — plus two controls the thesis needs and a product would
not: **which recording is being analysed**, and **which trained model is
producing the cues**.

Sources
-------
The page can be pointed at three kinds of thing, and the difference between them
is cost, not features:

``session``   a cache built by ``precompute_session`` from one video. Everything
              except the temporal head is already computed, so replay needs no
              detector, no GPU, and switching models is instant. This is what an
              uploaded recording becomes once analysed.
``video``     a file, webcam index or camera URL run through the whole pipeline
              live. Expensive, and a model switch restarts it.
``cue log``   a recorded JSONL of decisions. Stdlib only — no torch, no mmdet —
              and no model to switch, because a cue log stores conclusions
              rather than features.

A recording uploaded through the page lands as a ``video``, is analysed once
into a ``session``, and is used from the session thereafter.

Why a model selector is part of the deliverable
-----------------------------------------------
The thesis is a comparison of architectures and feature blocks, and the tables
in `work_dirs/thesis/tables/` are frame-level macro-F1 on held-out splits. A
number like 0.50 does not tell a reader what the difference between two models
*looks like* to an instructor. Running the same classroom minute through each
checkpoint does.

Selection is on **validation** only, in the dropdown ordering and in the
default. Test numbers are shown but never rank anything: `TEST_SPLIT_PROTOCOL.md`
spent the test split once, and a UI that sorted by it would spend it again on
every page load.

DESIGN CONSTRAINTS honoured from the rest of the project:
  * Language is VISIBLE-CUE only. The UI says "head down 45 s", never
    "not paying attention" — the taxonomy's founding principle
    (attention/taxonomy.py) and the supervisor's reframing.
  * Alerts fire on sustained EPISODES (attention/events.py), not single frames,
    and only when the model's own validation-fitted alert threshold allows it.
    A model that cannot reach 85% selective accuracy raises no alerts at all,
    and the UI says which.
  * Privacy: no identity, no demographics. Students are seat numbers. Faces can
    be blurred with --blur-faces, and the served frame is downscaled.

    python server.py --session tools/dashboard/sessions/0325   # cheapest
    python server.py --config LLMDet/configs/attention_runtime.yaml --video 0
    python server.py --replay live_session.jsonl
    python server.py --config LLMDet/configs/attention_runtime.yaml
        # no source: upload one from the page

UPLOADS AND EXPOSURE. The page can write video files to ``uploads/``. The server
binds 0.0.0.0 by default, which on a shared machine means anyone who can reach
the port can upload. There is no authentication — this is a thesis demo, not a
deployed service. Use ``--host 127.0.0.1`` when the browser is on the same
machine, or ``--no-upload`` to serve read-only.
"""
import argparse
import base64
import json
import sys
import threading
import time
import traceback
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sources as SRC                                           # noqa: E402

STATE = {
    "frame_jpeg_b64": None,
    "t": 0.0,
    "students": {},        # seat -> {cue, since, conf}
    "alerts": deque(maxlen=100),
    "history": deque(maxlen=600),   # (t, off_task_fraction)
    "class_summary": {},
    "running": False,
    "source": "",
    "source_id": None,
    "model": {},           # the active entry, as shown in the header
    "notice": "",          # e.g. "alerts disabled for this model"
    "capture": {},         # live-source throughput and drop rate
    "job": {},             # background analyse job
}
LOCK = threading.Lock()

# Alert only after a cue has persisted this long — matches EventConfig
# min_duration_s. Single-frame flicker must never page an instructor.
ALERT_AFTER_S = {"phone_use": 15.0, "head_down": 30.0,
                 "turned_to_peer": 30.0, "looking_away": 20.0}


# ---------------------------------------------------------------------------
# runner: owns the one thread that is currently producing frames
# ---------------------------------------------------------------------------

class Runner:
    """Starts, stops and replaces one background thread.

    Switching model or source must not leave the old thread running: two
    pipelines pushing into one STATE would interleave output from two
    configurations under one configuration's name, which is the sort of
    plausible-looking result this project keeps having to hunt down.
    """

    def __init__(self, on_start=None):
        self.thread = None
        self._stop = threading.Event()
        self.lock = threading.Lock()
        self.on_start = on_start

    def should_stop(self):
        return self._stop.is_set()

    def alive(self):
        return self.thread is not None and self.thread.is_alive()

    def request_stop(self):
        """Ask, without waiting. A cancel button should return immediately;
        the worker notices at its next frame, which on CPU can be seconds."""
        self._stop.set()

    # Generous, because the live source checks the stop flag once per frame and
    # a single detector frame on CPU is seconds. A tight timeout would turn a
    # slow switch into a spurious 409.
    def stop(self, join_timeout=60.0):
        self._stop.set()
        t = self.thread
        if t and t.is_alive():
            t.join(timeout=join_timeout)
            if t.is_alive():
                raise RuntimeError(
                    "the previous run did not stop within "
                    f"{join_timeout:.0f}s; refusing to start another")
        self.thread = None

    def start(self, target, *args, **kwargs):
        with self.lock:
            self.stop()
            self._stop = threading.Event()
            if self.on_start:
                self.on_start()
            self.thread = threading.Thread(target=target, args=args,
                                           kwargs=kwargs, daemon=True)
            self.thread.start()


def reset_state():
    with LOCK:
        STATE["frame_jpeg_b64"] = None
        STATE["t"] = 0.0
        STATE["students"] = {}
        STATE["alerts"].clear()
        STATE["history"].clear()
        STATE["class_summary"] = {}
        STATE["capture"] = {}
        STATE["running"] = False


RUNNER = Runner(on_start=reset_state)    # produces frames
JOBS = Runner()                          # builds session caches
CONTEXT = {}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _err(self, code, msg):
        self._send(code, json.dumps({"error": str(msg)}))

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            p = Path(__file__).with_name("index.html")
            return self._send(200, p.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/state":
            with LOCK:
                return self._send(200, json.dumps({
                    "t": STATE["t"],
                    "running": STATE["running"],
                    "source": STATE["source"],
                    "source_id": STATE["source_id"],
                    "frame": STATE["frame_jpeg_b64"],
                    "students": STATE["students"],
                    "alerts": list(STATE["alerts"])[-25:],
                    "history": list(STATE["history"]),
                    "class_summary": STATE["class_summary"],
                    "model": STATE["model"],
                    "notice": STATE["notice"],
                    "capture": STATE["capture"],
                    "job": STATE["job"],
                }))
        if path == "/api/models":
            return self._send(200, json.dumps({
                "models": [e.to_json() for e in CONTEXT.get("entries", [])],
                "active": STATE["model"].get("variant_id"),
                "switchable": bool(CONTEXT.get("entries")),
                "switch_cost": CONTEXT.get("switch_cost", ""),
            }))
        if path == "/api/sources":
            return self._send(200, json.dumps({
                "sources": SRC.list_sources(CONTEXT.get("cli_video"),
                                            CONTEXT.get("cli_session")),
                "active": STATE["source_id"],
                "upload_enabled": CONTEXT.get("upload_enabled", False),
                "max_upload_mb": round(SRC.MAX_UPLOAD_BYTES / 1e6),
                "accepts": sorted(SRC.VIDEO_EXTS),
                "can_analyse": bool(CONTEXT.get("config")),
                "stream_enabled": bool(CONTEXT.get("config")),
                "stream_schemes": [x.rstrip(":/") for x in SRC.STREAM_SCHEMES],
            }))
        self._err(404, "not found")

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/api/model":
                e = switch_model(self._json().get("variant_id", ""))
                return self._send(200, json.dumps({"active": e.variant_id}))
            if u.path == "/api/source":
                b = self._json()
                sid = select_source(b.get("source_id", ""), b.get("mode"))
                return self._send(200, json.dumps({"active": sid}))
            if u.path == "/api/upload":
                return self._upload(q.get("name", [""])[0])
            if u.path == "/api/analyse":
                b = self._json()
                return self._send(200, json.dumps(
                    start_analyse(b.get("source_id", ""),
                                  b.get("max_frames"))))
            if u.path == "/api/cancel":
                JOBS.request_stop()
                with LOCK:
                    if STATE["job"].get("status") == "running":
                        STATE["job"] = {**STATE["job"],
                                        "message": "cancelling…"}
                return self._send(200, json.dumps({"cancelling": True}))
        except ValueError as e:
            return self._err(400, e)
        except FileNotFoundError as e:
            return self._err(404, e)
        except SystemExit as e:            # raised by the model loaders
            return self._err(400, e)
        except RuntimeError as e:
            return self._err(409, e)
        except Exception as e:             # noqa: BLE001 - report, don't hang
            traceback.print_exc()
            return self._err(500, f"{type(e).__name__}: {e}")
        self._err(404, "not found")

    def _json(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            raise ValueError("malformed JSON")

    def _upload(self, name):
        if not CONTEXT.get("upload_enabled"):
            raise RuntimeError("uploads are disabled (--no-upload)")
        if not name:
            raise ValueError("missing ?name=")
        n = int(self.headers.get("Content-Length") or 0)
        dest = SRC.save_upload(self.rfile, n, name)
        print(f"[dashboard] uploaded {dest.name} ({n / 1e6:.1f} MB)")
        return self._send(200, json.dumps({
            "id": f"video:{dest}", "name": dest.name,
            "size_mb": round(n / 1e6, 1)}))


def push_frame(t, jpeg_bytes, students, cue_names):
    """Update dashboard state from one processed frame."""
    with LOCK:
        STATE["t"] = float(t)
        if jpeg_bytes is not None:
            STATE["frame_jpeg_b64"] = base64.b64encode(jpeg_bytes).decode()
        STATE["students"] = students

        off = [s for s in students.values() if s["cue"] in ALERT_AFTER_S]
        frac = len(off) / max(len(students), 1)
        STATE["history"].append([round(float(t), 1), round(frac, 3)])

        counts = defaultdict(int)
        for s in students.values():
            counts[s["cue"]] += 1
        STATE["class_summary"] = {
            "n_students": len(students),
            "off_task": len(off),
            "off_task_pct": round(100 * frac),
            "by_cue": dict(counts),
        }

        for seat, s in students.items():
            thr = ALERT_AFTER_S.get(s["cue"])
            # Selective prediction: an alert also needs the model to be confident
            # enough, at the threshold fitted on ITS OWN validation predictions
            # (tools/dashboard/calibrate_registry.py). Replay logs recorded
            # before abstention existed carry no `alert_allowed` key, so the
            # default admits them and old sessions still replay.
            if not s.get("alert_allowed", True):
                continue
            if thr and s.get("dwell", 0) >= thr and not s.get("alerted"):
                STATE["alerts"].append({
                    "t": round(float(t), 1),
                    "seat": seat,
                    "cue": s["cue"],
                    "dwell": round(s["dwell"]),
                    # Visible-cue phrasing only. Never "inattentive".
                    "text": f"Seat {seat}: {s['cue'].replace('_', ' ')} "
                            f"for {round(s['dwell'])}s",
                })
                s["alerted"] = True


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------

def run_replay(path):
    """Replay a recorded cue log — no model, no torch, no GPU."""
    with LOCK:
        STATE["running"] = True
        STATE["source"] = f"cue log: {Path(path).name}"
        STATE["model"] = {}
    rows = [json.loads(l) for l in open(path)]
    for r in rows:
        if RUNNER.should_stop():
            break
        push_frame(r["t"], None, r["students"], r.get("cue_names", []))
        time.sleep(max(0.0, min(1.0, r.get("dt", 1.0))))
    with LOCK:
        STATE["running"] = False


def run_session(entry, cache_dir):
    """Replay a precomputed session cache through one model."""
    import session_replay as SR
    cache = SR.SessionCache(cache_dir)
    bundle, cal = SR.load_model(entry, CONTEXT["device"])
    with LOCK:
        STATE["running"] = True
        STATE["source"] = (f"session: {Path(cache_dir).name} "
                           f"({cache.meta['n_frames']} frames @ "
                           f"{cache.fps:.0f} fps)")
        STATE["model"] = model_card(entry, cal)
        STATE["notice"] = "" if cal.get("alerts_enabled", True) else \
            cal.get("alerts_disabled_reason", "")
    print(f"[dashboard] {entry.variant_id} — {bundle.describe()}")
    SR.replay(cache, entry, bundle, push_frame,
              should_stop=RUNNER.should_stop,
              blur_faces=CONTEXT["blur_faces"], realtime=True,
              speed=CONTEXT["speed"])
    with LOCK:
        STATE["running"] = False


def run_live_source(entry, video):
    """Full pipeline over a file, a webcam or an IP camera.

    Expensive: the detector dominates, and on CPU it dominates completely. For a
    live source that shows up as a drop rate rather than as a slowdown — the
    reader thread keeps the newest frame and the pipeline skips the rest — so
    the capture stats are surfaced to the page instead of only the log.
    """
    import session_replay as SR
    from pipeline_bridge import classify_source, run_live
    cal_path = SR.calibration_path(entry.variant_id)
    cal = json.loads(cal_path.read_text()) if cal_path.exists() else {}
    kind, _ = classify_source(video)
    with LOCK:
        STATE["running"] = True
        STATE["source"] = (
            f"{kind}: {Path(video).name if kind == 'file' else video} "
            f"on {CONTEXT['device']}")
        STATE["model"] = model_card(entry, cal)
        STATE["notice"] = "" if cal.get("alerts_enabled", True) else \
            cal.get("alerts_disabled_reason", "")

    def on_stats(s):
        with LOCK:
            STATE["capture"] = s

    run_live(CONTEXT["config"], video, push_frame,
             blur_faces=CONTEXT["blur_faces"], max_frames=CONTEXT["max_frames"],
             device=CONTEXT["device"], entry=entry,
             should_stop=RUNNER.should_stop, stats_fn=on_stats)
    with LOCK:
        STATE["running"] = False


def model_card(entry, cal):
    """What the header shows about the running model.

    Both the deployed single-seed figure and the seed mean are carried, because
    they are different claims: the dashboard runs one checkpoint chosen as the
    best of three on validation, while the thesis tables report the mean over
    those three. Showing only the first would quietly overstate the model.
    """
    return {
        "variant_id": entry.variant_id,
        "label": entry.label,
        "experiment_id": entry.experiment_id,
        "seed": entry.seed,
        "n_seeds": entry.n_seeds,
        "sweep_label": entry.sweep_label,
        "head_pose_backend": entry.head_pose_backend,
        "val_macro_f1": entry.val.get("macro_f1"),
        "val_seed_mean": entry.val_seed_mean,
        "test_macro_f1": entry.test.get("macro_f1"),
        "test_seed_mean": entry.test_seed_mean,
        "display_threshold": cal.get("display_threshold"),
        "alert_threshold": cal.get("alert_threshold"),
        "alerts_enabled": cal.get("alerts_enabled", True),
        # Alert coverage is the instructor-facing consequence of model quality
        # and is NOT ordered like macro-F1. Every model is held to the same 85%
        # alert precision, so what a weaker one gives up is the *share of the
        # class it can say anything about*: 78.7% at the top of the registry,
        # 30.5% at the bottom. Two students in three, or two in seven.
        "alert_coverage": cal.get("alert_coverage"),
        "alert_selective_accuracy": cal.get("alert_selective_accuracy"),
        "display_coverage": cal.get("display_coverage"),
        "checkpoint": entry.checkpoint,
    }


def start(source, entry=None):
    """Point the dashboard at ``source`` (a dict from ``sources.list_sources``).

    ``source["kind"]`` decides the cost: a session replays from the cache, a
    video runs the whole pipeline, a cue log runs neither.
    """
    entry = entry or CONTEXT.get("entry")
    CONTEXT["source"] = source
    with LOCK:
        STATE["source_id"] = source.get("id")

    if source["kind"] == "cuelog":
        CONTEXT["switch_cost"] = ""
        return RUNNER.start(run_replay, source["path"])

    if entry is None:
        raise RuntimeError("no model selected")
    CONTEXT["entry"] = entry

    if source["kind"] == "session":
        CONTEXT["switch_cost"] = (
            "instant — the detector, tracker and features are cached, so only "
            "the temporal head re-runs")
        return RUNNER.start(run_session, entry, source["path"])

    if source["kind"] in ("video", "stream"):
        if not CONTEXT.get("config"):
            raise RuntimeError(
                "no detector config; start the server with --config to run "
                "the full pipeline")
        CONTEXT["switch_cost"] = (
            "reconnects to the camera and re-runs the whole pipeline; a live "
            "stream has no cache to fall back on"
            if source["kind"] == "stream" else
            "restarts the source and re-runs the whole pipeline; analyse it "
            "into a session to make switching instant")
        return RUNNER.start(run_live_source, entry, source["path"])

    raise ValueError(f"unknown source kind {source['kind']!r}")


def switch_model(variant_id):
    import model_registry as MR
    entry = MR.find(CONTEXT.get("entries", []), variant_id)
    if entry is None:
        raise ValueError(f"unknown model {variant_id!r}")
    if not entry.deployable:
        raise ValueError(f"{variant_id} cannot run live: {entry.blocked_reason}")
    src = CONTEXT.get("source")
    if src is None or src["kind"] == "cuelog":
        raise ValueError("this source has no model to switch "
                         "(a recorded cue log holds cues, not features)")
    start(src, entry)
    return entry


def select_source(source_id, mode=None):
    """Point the dashboard at a different recording.

    ``mode`` picks what to do with a video that already has a session:
    ``"session"`` (default when one exists) replays the cache, ``"video"``
    forces the full pipeline. It is not a preference — running a video live when
    a cache exists is a genuine choice between fidelity to the current config
    and speed.
    """
    kind, path = SRC.parse_id(source_id)
    if kind == "stream":
        # Typed by the user rather than discovered on disk, so there is nothing
        # in list_sources to look up. parse_id has already validated the URL.
        src = {"id": f"stream:{path}", "kind": "stream",
               "name": path, "path": path, "ready": True}
        start(src)
        return src["id"]

    listed = {s["id"]: s for s in SRC.list_sources(CONTEXT.get("cli_video"),
                                                   CONTEXT.get("cli_session"))}
    src = listed.get(source_id)
    if src is None:
        raise FileNotFoundError(f"no such source {source_id!r}")

    if kind == "video" and src.get("session") and mode != "video":
        src = listed.get(f"session:{Path(src['session']).resolve()}") or src
    start(src)
    return src["id"]


# ---------------------------------------------------------------------------
# analyse: build a session cache in the background
# ---------------------------------------------------------------------------

def start_analyse(source_id, max_frames=None):
    """Kick off a precompute pass over an uploaded/recorded video."""
    kind, video = SRC.parse_id(source_id)
    if kind == "stream":
        raise ValueError(
            "a live stream cannot be analysed into a session: precompute needs "
            "a finite source, and a camera has no end. Select it as a source "
            "to run the pipeline live instead.")
    if kind != "video":
        raise ValueError("only a video can be analysed; a session already is")
    if not video.exists():
        raise FileNotFoundError(f"{video} is gone")
    if not CONTEXT.get("config"):
        raise RuntimeError("no detector config; start the server with --config")
    if JOBS.alive():
        raise RuntimeError("an analysis is already running")

    out = SRC.session_dir_for(video)
    if SRC.session_meta(out) is not None:
        raise RuntimeError(f"{out.name} already has a session; delete it first")

    # Analysis is the detector-bound part of the system. Leaving a live pipeline
    # running alongside it would have both fighting for the same cores and make
    # the progress estimate meaningless.
    RUNNER.stop()
    with LOCK:
        STATE["running"] = False
        STATE["job"] = {"status": "running", "name": video.name,
                        "source_id": source_id, "pct": 0,
                        "message": "starting…"}
    JOBS.start(_analyse_thread, video, out,
               int(max_frames or CONTEXT["analyse_frames"]))
    return {"status": "running", "name": video.name}


def _analyse_thread(video, out, max_frames):
    from precompute_session import precompute

    def on_progress(p):
        with LOCK:
            STATE["job"] = {
                "status": "running", "name": video.name, **p,
                "message": (f"{p['frames_done']}/{p['frames_target']} frames · "
                            f"{p['fps']:.2f} fps · ~{p['eta_s']}s left"),
            }

    t0 = time.time()
    try:
        precompute(CONTEXT["config"], str(video), str(out),
                   device=CONTEXT["device"], max_frames=max_frames,
                   progress_fn=on_progress, should_stop=JOBS.should_stop)
    except KeyboardInterrupt as e:                    # cancelled
        with LOCK:
            STATE["job"] = {"status": "cancelled", "name": video.name,
                            "message": str(e)}
        return
    except BaseException as e:                        # noqa: BLE001
        traceback.print_exc()
        with LOCK:
            STATE["job"] = {"status": "failed", "name": video.name,
                            "message": f"{type(e).__name__}: {e}"}
        return

    meta = SRC.session_meta(out) or {}
    with LOCK:
        STATE["job"] = {
            "status": "done", "name": video.name, "pct": 100,
            "session_id": f"session:{out.resolve()}",
            "message": (f"{meta.get('n_frames', '?')} frames, "
                        f"{meta.get('n_tracks', '?')} students, "
                        f"{round(time.time() - t0)}s"),
        }
    # Show the result rather than leaving the operator to find the new entry.
    try:
        select_source(f"session:{out.resolve()}")
    except Exception:                                  # noqa: BLE001
        traceback.print_exc()


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0",
                    help="bind address. Use 127.0.0.1 when the browser is on "
                         "this machine — uploads are unauthenticated.")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--session", default=None,
                    help="a session cache from precompute_session.py")
    ap.add_argument("--replay", default=None, help="a recorded cue log (JSONL)")
    ap.add_argument("--config", default="LLMDet/configs/attention_runtime.yaml",
                    help="detector config, needed to analyse or stream a video")
    ap.add_argument("--video", default=None,
                    help="a file, a webcam index (0), or a camera URL")
    ap.add_argument("--model", default=None,
                    help="variant id, e.g. arch/asrf_556_hp. "
                         "Default: best deployable variant on validation.")
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda:N")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="session replay speed multiplier (1.0 = source fps)")
    ap.add_argument("--analyse-frames", type=int, default=900,
                    help="frame cap when analysing an uploaded video")
    ap.add_argument("--blur-faces", action="store_true")
    ap.add_argument("--max-frames", type=int, default=100000)
    ap.add_argument("--no-upload", action="store_true",
                    help="serve read-only: no file uploads accepted")
    args = ap.parse_args()

    cfg = Path(args.config)
    CONTEXT.update({
        "config": str(cfg) if cfg.exists() else None,
        "device": args.device,
        "blur_faces": args.blur_faces,
        "speed": args.speed,
        "max_frames": args.max_frames,
        "analyse_frames": args.analyse_frames,
        "cli_video": args.video,
        "cli_session": args.session,
        "upload_enabled": not args.no_upload,
        "entries": [],
    })
    if cfg.exists():
        SRC.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        SRC.SESSION_DIR.mkdir(parents=True, exist_ok=True)
    else:
        print(f"[dashboard] no detector config at {args.config}: sessions and "
              f"cue logs will work, analysing and streaming a video will not")

    entry = None
    if not args.replay:
        import model_registry as MR
        entries = MR.scan()
        entry = (MR.find(entries, args.model) if args.model
                 else MR.default_entry(entries))
        if entry is None:
            raise SystemExit(f"unknown model {args.model!r}. Known: "
                             + ", ".join(e.variant_id for e in entries))
        if not entry.deployable:
            raise SystemExit(f"{entry.variant_id}: {entry.blocked_reason}")
        # A seed-pinned id (variant@sNN) is not one of the ranked rows, which
        # carry each variant's best validation seed. Show it anyway, first, or
        # the dropdown reports no active model while one is plainly running.
        if all(e.variant_id != entry.variant_id for e in entries):
            entries = [entry] + entries
        CONTEXT["entries"] = entries
        CONTEXT["entry"] = entry

    if args.replay:
        start({"kind": "cuelog", "id": f"cuelog:{args.replay}",
               "path": args.replay})
    elif args.session:
        start({"kind": "session", "id": f"session:{Path(args.session).resolve()}",
               "path": str(Path(args.session).resolve())}, entry)
    elif args.video:
        start({"kind": "video", "id": f"video:{args.video}",
               "path": args.video}, entry)
    else:
        print("[dashboard] no source yet — upload a recording from the page, "
              "or pick one already in tools/dashboard/{uploads,sessions}/")

    if CONTEXT["upload_enabled"] and args.host == "0.0.0.0":
        print("[dashboard] WARNING: bound to 0.0.0.0 with uploads enabled and "
              "no authentication. Use --host 127.0.0.1 or --no-upload on a "
              "shared machine.")

    # Threading, not the plain HTTPServer: a model switch stops the producer
    # thread and can take seconds, and on a single-threaded server that would
    # freeze the polling GETs too — the page would look crashed while working.
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"dashboard: http://{'localhost' if args.host == '0.0.0.0' else args.host}"
          f":{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
