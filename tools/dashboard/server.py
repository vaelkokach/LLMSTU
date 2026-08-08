#!/usr/bin/env python3
"""Instructor dashboard — real-time analytics, alerts, and a model selector.

Implements the `Thesis_Topic.md` deliverable:

    "The project will give teachers/professors a dynamic dashboard showing
     real-time analytics and alerts so enabling quick interventions..."

Serves a live view over the pipeline: annotated video frame, per-student cue
state, a rolling alert log for sustained off-task episodes, and class-level
analytics over time — plus a control the thesis needs and a product would not:
**which trained model is producing the cues**.

Why a model selector is part of the deliverable
-----------------------------------------------
The thesis is a comparison of architectures and feature blocks, and the tables
in `work_dirs/thesis/tables/` are frame-level macro-F1 on held-out splits. A
number like 0.50 does not tell a reader what the difference between two models
*looks like* to an instructor. Running the same classroom minute through each
checkpoint does. The dropdown offers one entry per variant — best validation
seed, from `model_registry` — and switching it re-decides every cue in the
session without re-running the detector.

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

Three sources, in order of what they cost:

    # cached session — no GPU, no detector, model switching is instant
    python server.py --session sessions/0325

    # live from a video (expensive; --device cpu runs with no GPU at all)
    python server.py --config ../../LLMDet/configs/attention_runtime.yaml \
        --video ../../LLMDet/0325.mp4 --device cpu

    # a recorded cue log, cues only, no model switching
    python server.py --replay live_session.jsonl

Stdlib only for the server itself — torch is imported lazily and only by the
two sources that need it, so `--replay` still runs anywhere.
"""
import argparse
import base64
import json
import sys
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

STATE = {
    "frame_jpeg_b64": None,
    "t": 0.0,
    "students": {},        # seat -> {cue, since, conf}
    "alerts": deque(maxlen=100),
    "history": deque(maxlen=600),   # (t, off_task_fraction)
    "class_summary": {},
    "running": False,
    "source": "",
    "model": {},           # the active entry, as shown in the header
    "notice": "",          # e.g. "alerts disabled for this model"
    "capture": {},         # live-source throughput and drop rate
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
    """Starts, stops and replaces the thread that feeds the dashboard.

    Switching models must not leave the old thread running: two pipelines
    pushing into one STATE would interleave cues from two models under one
    model's name, which is the sort of plausible-looking output this project
    keeps having to hunt down.
    """

    def __init__(self):
        self.thread = None
        self._stop = threading.Event()
        self.lock = threading.Lock()

    def should_stop(self):
        return self._stop.is_set()

    # Generous, because the live source checks the stop flag once per frame and
    # a single detector frame on CPU is seconds. A tight timeout would turn a
    # slow switch into a spurious 409.
    def stop(self, join_timeout=60.0):
        self._stop.set()
        t = self.thread
        if t and t.is_alive():
            t.join(timeout=join_timeout)
            if t.is_alive():
                # Do not start a second producer on top of a live one.
                raise RuntimeError(
                    "the previous run did not stop within "
                    f"{join_timeout:.0f}s; refusing to start another")
        self.thread = None

    def start(self, target, *args, **kwargs):
        with self.lock:
            self.stop()
            self._stop = threading.Event()
            reset_state()
            self.thread = threading.Thread(target=target, args=args,
                                           kwargs=kwargs, daemon=True)
            self.thread.start()


RUNNER = Runner()
CONTEXT = {}     # how to restart the current source with a different model


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

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            p = Path(__file__).with_name("index.html")
            return self._send(200, p.read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/state":
            with LOCK:
                s = {
                    "t": STATE["t"],
                    "running": STATE["running"],
                    "source": STATE["source"],
                    "frame": STATE["frame_jpeg_b64"],
                    "students": STATE["students"],
                    "alerts": list(STATE["alerts"])[-25:],
                    "history": list(STATE["history"]),
                    "class_summary": STATE["class_summary"],
                    "model": STATE["model"],
                    "notice": STATE["notice"],
                    "capture": STATE["capture"],
                }
            return self._send(200, json.dumps(s))
        if self.path == "/api/models":
            return self._send(200, json.dumps({
                "models": [e.to_json() for e in CONTEXT.get("entries", [])],
                "active": STATE["model"].get("variant_id"),
                "switchable": CONTEXT.get("switchable", False),
                "switch_cost": CONTEXT.get("switch_cost", ""),
            }))
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/api/model":
            return self._send(404, json.dumps({"error": "not found"}))
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, json.dumps({"error": "malformed JSON"}))
        try:
            entry = switch_model(body.get("variant_id", ""))
        except SystemExit as e:            # raised by the loaders on bad state
            return self._send(400, json.dumps({"error": str(e)}))
        except (RuntimeError, ValueError) as e:
            return self._send(409, json.dumps({"error": str(e)}))
        return self._send(200, json.dumps({"active": entry.variant_id}))


def push_frame(t, jpeg_bytes, students, cue_names):
    """Update dashboard state from one processed frame."""
    with LOCK:
        STATE["t"] = float(t)
        if jpeg_bytes is not None:
            STATE["frame_jpeg_b64"] = base64.b64encode(jpeg_bytes).decode()
        STATE["students"] = students

        off = [s for s in students.values()
               if s["cue"] in ALERT_AFTER_S]
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
    rows = [json.loads(l) for l in open(path)]
    for r in rows:
        if RUNNER.should_stop():
            break
        push_frame(r["t"], None, r["students"], r.get("cue_names", []))
        time.sleep(max(0.0, min(1.0, r.get("dt", 1.0))))
    with LOCK:
        STATE["running"] = False


def run_session(entry, cache, device, blur_faces, speed):
    """Replay a precomputed session cache through one model."""
    import session_replay as SR
    bundle, cal = SR.load_model(entry, device)
    with LOCK:
        STATE["running"] = True
        STATE["source"] = (f"session: {cache.dir.name} "
                           f"({cache.meta['n_frames']} frames @ "
                           f"{cache.fps:.0f} fps)")
        STATE["model"] = model_card(entry, cal)
        STATE["notice"] = "" if cal.get("alerts_enabled", True) else \
            cal.get("alerts_disabled_reason", "")
    print(f"[dashboard] {entry.variant_id} — {bundle.describe()}")
    SR.replay(cache, entry, bundle, push_frame,
              should_stop=RUNNER.should_stop, blur_faces=blur_faces,
              realtime=True, speed=speed)
    with LOCK:
        STATE["running"] = False


def run_live_source(entry, config, video, device, blur_faces, max_frames):
    """Full pipeline over a video, a webcam or an IP camera.

    Expensive: the detector dominates, and on CPU it dominates completely. For a
    live source that shows up as a drop rate rather than as a slowdown — the
    reader thread keeps the newest frame and the pipeline skips the rest — so
    the capture stats are surfaced to the page instead of only the log.
    """
    import session_replay as SR
    from pipeline_bridge import classify_source, run_live
    cal_path = SR.calibration_path(entry.variant_id)
    cal = json.loads(cal_path.read_text()) if cal_path.exists() else {}
    kind, source = classify_source(video)
    with LOCK:
        STATE["running"] = True
        STATE["source"] = (f"{kind}: "
                           f"{Path(video).name if kind == 'file' else video}"
                           f" on {device}")
        STATE["model"] = model_card(entry, cal)
        STATE["notice"] = "" if cal.get("alerts_enabled", True) else \
            cal.get("alerts_disabled_reason", "")

    def on_stats(s):
        with LOCK:
            STATE["capture"] = s

    run_live(config, video, push_frame, blur_faces=blur_faces,
             max_frames=max_frames, device=device, entry=entry,
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
        "checkpoint": entry.checkpoint,
    }


def switch_model(variant_id):
    """Restart the current source under a different model."""
    import model_registry as MR
    entries = CONTEXT.get("entries", [])
    entry = MR.find(entries, variant_id)
    if entry is None:
        raise ValueError(f"unknown model {variant_id!r}")
    if not entry.deployable:
        raise ValueError(f"{variant_id} cannot run live: {entry.blocked_reason}")
    if not CONTEXT.get("switchable"):
        raise ValueError("this source has no model to switch "
                         "(a recorded cue log holds cues, not features)")
    CONTEXT["start"](entry)
    return entry


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--session", default=None,
                    help="a session cache from precompute_session.py")
    ap.add_argument("--replay", default=None, help="a recorded cue log (JSONL)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--model", default=None,
                    help="variant id, e.g. arch/asrf_556_hp. "
                         "Default: best deployable variant on validation.")
    ap.add_argument("--device", default="cpu",
                    help="cpu (default) or cuda:N")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="session replay speed multiplier (1.0 = source fps)")
    ap.add_argument("--blur-faces", action="store_true")
    ap.add_argument("--max-frames", type=int, default=100000)
    args = ap.parse_args()

    if args.replay:
        CONTEXT["switchable"] = False
        CONTEXT["entries"] = []
        RUNNER.start(run_replay, args.replay)

    elif args.session or (args.config and args.video):
        import model_registry as MR
        entries = MR.scan()
        entry = (MR.find(entries, args.model) if args.model
                 else MR.default_entry(entries))
        if entry is None:
            raise SystemExit(
                f"unknown model {args.model!r}. Known: "
                + ", ".join(e.variant_id for e in entries))
        if not entry.deployable:
            raise SystemExit(f"{entry.variant_id}: {entry.blocked_reason}")
        CONTEXT["entries"] = entries
        CONTEXT["switchable"] = True

        if args.session:
            import session_replay as SR
            cache = SR.SessionCache(args.session)
            CONTEXT["switch_cost"] = (
                "instant — the detector, tracker and features are cached, so "
                "only the temporal head re-runs")
            CONTEXT["start"] = lambda e: RUNNER.start(
                run_session, e, cache, args.device, args.blur_faces, args.speed)
        else:
            CONTEXT["switch_cost"] = (
                "restarts the video from the beginning and re-runs the whole "
                "pipeline; precompute a session cache to make switching instant")
            CONTEXT["start"] = lambda e: RUNNER.start(
                run_live_source, e, args.config, args.video, args.device,
                args.blur_faces, args.max_frames)
        CONTEXT["start"](entry)

    else:
        raise SystemExit(
            "no source. Pass --session DIR (cheapest), --config CFG --video VID, "
            "or --replay FILE.")

    # Threading, not the plain HTTPServer: a model switch stops the producer
    # thread and can take seconds, and on a single-threaded server that would
    # freeze the polling GETs too — the page would look crashed while working.
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"dashboard: http://localhost:{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
