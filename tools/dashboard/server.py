#!/usr/bin/env python3
"""Instructor dashboard — real-time analytics and alerts.

Implements the `Thesis_Topic.md` deliverable:

    "The project will give teachers/professors a dynamic dashboard showing
     real-time analytics and alerts so enabling quick interventions..."

Serves a live view over the running pipeline: annotated video frame, per-student
cue state, a rolling alert log for sustained off-task episodes, and class-level
attention analytics over time.

DESIGN CONSTRAINTS honoured from the rest of the project:
  * Language is VISIBLE-CUE only. The UI says "head down 45 s", never
    "not paying attention" — the taxonomy's founding principle
    (attention/taxonomy.py) and the supervisor's reframing.
  * Alerts fire on sustained EPISODES (attention/events.py), not single frames,
    so a student glancing away does not generate a notification.
  * Privacy: no identity, no demographics. Students are seat numbers. Faces can
    be blurred with --blur-faces for classroom deployment, and the served frame
    is downscaled.

Stdlib only (http.server) — same choice as tools/gold_annotator, so it runs
anywhere the pipeline runs with no extra dependencies.

Usage:
    # live from the pipeline
    python server.py --config ../../LLMDet/configs/attention_temporal_hp.yaml \
        --video 0325.mp4 --port 8080

    # replay a precomputed session (no GPU needed)
    python server.py --replay session.jsonl --port 8080
"""
import argparse
import base64
import json
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

STATE = {
    "frame_jpeg_b64": None,
    "t": 0.0,
    "students": {},        # seat -> {cue, since, conf}
    "alerts": deque(maxlen=100),
    "history": deque(maxlen=600),   # (t, off_task_fraction)
    "class_summary": {},
    "running": False,
    "source": "",
}
LOCK = threading.Lock()

# Alert only after a cue has persisted this long — matches EventConfig
# min_duration_s. Single-frame flicker must never page an instructor.
ALERT_AFTER_S = {"phone_use": 15.0, "head_down": 30.0,
                 "turned_to_peer": 30.0, "looking_away": 20.0}


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
                }
            return self._send(200, json.dumps(s))
        self._send(404, json.dumps({"error": "not found"}))


def push_frame(t, jpeg_bytes, students, cue_names):
    """Update dashboard state from one processed frame."""
    import numpy as np
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
            # enough. The threshold is fitted on the validation split and frozen
            # (attention/thesis_eval/runtime.py); at it, retained frames are
            # correct 85.4% of the time versus 75.6% at full coverage. Replay
            # logs recorded before abstention existed carry no `alert_allowed`
            # key, so the default admits them and old sessions still replay.
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


def run_replay(path):
    """Replay a recorded session — lets the dashboard be demoed without a GPU."""
    STATE["running"] = True
    STATE["source"] = f"replay: {Path(path).name}"
    rows = [json.loads(l) for l in open(path)]
    t0 = time.time()
    for r in rows:
        push_frame(r["t"], None, r["students"], r.get("cue_names", []))
        time.sleep(max(0.0, min(1.0, r.get("dt", 1.0))))
    STATE["running"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--replay", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--blur-faces", action="store_true")
    ap.add_argument("--max-frames", type=int, default=100000)
    args = ap.parse_args()

    if args.replay:
        threading.Thread(target=run_replay, args=(args.replay,),
                         daemon=True).start()
    elif args.config and args.video:
        from pipeline_bridge import run_live
        threading.Thread(target=run_live,
                         args=(args.config, args.video, push_frame,
                               args.blur_faces, args.max_frames),
                         daemon=True).start()
    else:
        print("no source: pass --replay FILE, or --config CFG --video VID")

    srv = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"dashboard: http://localhost:{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
