"""Replay a cached session through any one temporal model.

``precompute_session`` cached everything that does not depend on the model. This
runs the part that does: assemble the live feature vector, slide a window per
track, and ask one checkpoint what each student's visible cue is.

Two things this must get right, because both are silent when wrong.

**Which head-pose block.** The cache holds the FaceLandmarker block and the
BlazeFace block side by side. Each model gets the one its training features were
built from (``ModelEntry.head_pose_backend``, derived from the checkpoint's own
``run_record.json``). Handing a model the other block would still produce a
556-wide vector, still run, and still print cues — just worse ones, for a reason
nothing in the output would reveal.

**Which thresholds.** Confidence is comparable across models only after each has
its own temperature. ``calibrate_registry`` fits one per model on validation;
this refuses to invent a fallback if that file is missing, because a model
running at ``threshold = 0`` never abstains and would look *more* decisive than
a well-calibrated one — the wrong way round.

Everything else mirrors ``pipeline_bridge.run_live`` frame for frame, so a
cached replay and a live run of the same model produce the same cue stream.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable, Dict, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "LLMDet"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np                                              # noqa: E402

CALIBRATION_DIR = (REPO / "LLMDet" / "work_dirs" / "thesis" / "runtime"
                   / "dashboard")

#: BGR overlay colours, matching the chip colours in index.html.
#:
#: Keyed by class NAME rather than by id, so a model predicting a regrouped
#: taxonomy draws in the right colour without a second table: `on_task` is the
#: green `screen_oriented` is, `off_task` and `down_or_hidden` are the red the
#: cues they merge are. Any name not listed falls back to grey, which is the
#: honest default — an unrecognised class is one this overlay cannot interpret.
CUE_COLOUR = {
    # cue6
    "screen_oriented": (61, 220, 132),
    "head_down": (86, 95, 255),
    "phone_use": (86, 95, 255),
    "looking_away": (84, 180, 255),
    "turned_to_peer": (84, 180, 255),
    "uncertain": (160, 160, 160),
    # onoff / onoff_reliable
    "on_task": (61, 220, 132),
    "off_task": (86, 95, 255),
    # coarse3_reliable
    "down_or_hidden": (86, 95, 255),
}

#: Grey. Also what an abstention draws as, since `displayed_cue` is the
#: ABSTAIN_LABEL ("uncertain") whenever the model was not confident enough.
UNKNOWN_COLOUR = (160, 160, 160)


def colour_for(cue: str):
    """BGR for one class name, grey for anything this table does not know."""
    return CUE_COLOUR.get(cue, UNKNOWN_COLOUR)


def calibration_path(variant_id: str) -> Path:
    return CALIBRATION_DIR / f"{variant_id.replace('/', '__')}.json"


class SessionCache:
    """The precomputed front end for one video."""

    def __init__(self, cache_dir: str):
        self.dir = Path(cache_dir).resolve()
        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            raise SystemExit(
                f"{self.dir} is not a session cache (no meta.json). Build one "
                f"with tools/dashboard/precompute_session.py.")
        self.meta = json.loads(meta_path.read_text())
        z = np.load(self.dir / "features.npz")
        self.frame_idx = z["frame_idx"]
        self.track_id = z["track_id"]
        self.bbox = z["bbox"]
        self.base = z["base"]
        self.head = {"mediapipe": z["head_landmarker"],
                     "mediapipe_detector": z["head_detector"]}
        self.frame_index = z["frame_index"]

        # How many JPEGs the cache actually has. Replay works without them --
        # it just pushes cue data and no image -- so a cache whose frame writes
        # failed looks exactly like a cue log at the UI: an empty video panel
        # and no error. Counting them here lets the caller say so.
        fdir = self.dir / "frames"
        self.n_cached_frames = (sum(1 for _ in fdir.glob("*.jpg"))
                                if fdir.is_dir() else 0)

        # rows grouped by frame, in frame order
        self._by_frame: Dict[int, list] = defaultdict(list)
        for r, f in enumerate(self.frame_idx):
            self._by_frame[int(f)].append(r)

    @property
    def fps(self) -> float:
        return float(self.meta.get("fps", 25.0))

    def rows_for(self, frame: int):
        return self._by_frame.get(int(frame), [])

    def jpeg(self, frame: int) -> Optional[bytes]:
        p = self.dir / "frames" / f"{int(frame):06d}.jpg"
        return p.read_bytes() if p.exists() else None

    def live_vector(self, rows, backend: str) -> np.ndarray:
        """[n, 556] = base | the head-pose block this model was trained on."""
        if backend not in self.head:
            raise SystemExit(f"cache has no {backend!r} head-pose block")
        return np.concatenate([self.base[rows], self.head[backend][rows]], axis=1)


def load_model(entry, device: str = "cpu"):
    """Build the runtime bundle for a registry entry, with its own thresholds."""
    from attention.thesis_eval.runtime import load_runtime_model

    cal = calibration_path(entry.variant_id)
    if not cal.exists():
        raise SystemExit(
            f"no calibration for {entry.variant_id} at {cal}. Run "
            f"tools/dashboard/calibrate_registry.py. Refusing to fall back to "
            f"threshold 0: an uncalibrated model never abstains, so it would "
            f"appear more confident than a calibrated one purely because "
            f"nobody fitted it.")
    bundle = load_runtime_model(str(REPO / entry.checkpoint), device=device,
                                calibration=str(cal))
    return bundle, json.loads(cal.read_text())


def replay(cache: SessionCache, entry, bundle, push_fn: Callable,
           should_stop: Callable[[], bool] = lambda: False,
           blur_faces: bool = False, realtime: bool = True,
           speed: float = 1.0, overlay: bool = True) -> int:
    """Stream one model's cues over the cached session.

    ``push_fn(t, jpeg_bytes, students, cue_names)`` matches
    ``pipeline_bridge.run_live`` so the server does not care which produced it.
    """
    import cv2
    from attention.thesis_eval.runtime import StrideController, predict_window

    # The names the MODEL predicts, from its own checkpoint spec. Pushing
    # CUE_CLASSES here would have told the UI to render six cue chips for a
    # two-class model, so the legend and the model would disagree.
    class_names = list(bundle.class_names)

    inf = cache.meta.get("inference", {})
    win = int(inf.get("window_size", 32))
    minf = int(inf.get("min_frames_for_pred", 4))
    # The detector stride is already baked into the cache (frames it skipped
    # have coasted boxes). Only the temporal stride is still ours to apply.
    stride = StrideController(1, int(inf.get("temporal_stride", 1)))

    hist = defaultdict(lambda: deque(maxlen=win))
    dwell: Dict[int, Dict] = {}
    fps = cache.fps
    n_pushed = 0
    wall0 = time.time()

    for frame in cache.frame_index:
        if should_stop():
            break
        frame = int(frame)
        t = frame / fps
        rows = cache.rows_for(frame)
        students: Dict[str, Dict] = {}

        # Forget tracks the tracker dropped, on empty frames too — otherwise a
        # reused track id would inherit a stale cached prediction.
        stride.drop_missing(int(cache.track_id[r]) for r in rows)

        if rows:
            vec = cache.live_vector(rows, entry.head_pose_backend)
            for k, r in enumerate(rows):
                hist[int(cache.track_id[r])].append(vec[k])

            for k, r in enumerate(rows):
                tid = int(cache.track_id[r])
                h = hist[tid]
                if len(h) < minf:
                    continue
                if stride.should_predict(tid, frame):
                    res = stride.store(tid, frame,
                                       predict_window(bundle, np.stack(list(h))))
                else:
                    res = stride.cached(tid)
                cue = res["displayed_cue"]

                prev = dwell.get(tid)
                if prev and prev["cue"] == cue:
                    prev["dwell"] = t - prev["since"]
                else:
                    dwell[tid] = {"cue": cue, "since": t, "dwell": 0.0,
                                  "alerted": False}
                st = dwell[tid]
                students[str(tid)] = {
                    "cue": cue,
                    "conf": round(float(res["confidence"]), 2),
                    "bbox": [round(float(v), 1) for v in cache.bbox[r]],
                    "dwell": st["dwell"],
                    "alerted": st["alerted"],
                    "raw_cue": res["cue"],
                    "abstained": bool(res["abstained"]),
                    "alert_allowed": bool(res["alert_allowed"]),
                }

        jpg = cache.jpeg(frame)
        if jpg is not None and overlay and students:
            jpg = _draw(cv2, jpg, students, cache.meta, blur_faces)

        push_fn(t, jpg, students, class_names)
        n_pushed += 1

        if realtime and speed > 0:
            target = wall0 + (n_pushed / (fps * speed))
            time.sleep(max(0.0, target - time.time()))

    return n_pushed


def _draw(cv2, jpeg_bytes: bytes, students: Dict[str, Dict], meta: Dict,
          blur_faces: bool) -> bytes:
    """Draw boxes and cue labels onto the cached (clean, downscaled) frame.

    Boxes are cached in full-resolution coordinates and the frame is stored
    downscaled, so every coordinate is scaled by the same factor the JPEG was.
    """
    import numpy as np
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jpeg_bytes
    src_w = meta.get("source_width")
    if not src_w:
        raise SystemExit(
            "this session cache predates `source_width` in meta.json, so the "
            "cached boxes cannot be scaled onto the cached frames. Rebuild it "
            "with precompute_session.py — drawing at the wrong scale would put "
            "every box in the wrong place while still looking like a result.")
    scale = img.shape[1] / float(src_w)
    for seat, s in students.items():
        x1, y1, x2, y2 = [int(v * scale) for v in s["bbox"]]
        col = colour_for(s["cue"])
        if blur_faces:
            hh = max(1, int(0.35 * (y2 - y1)))
            roi = img[max(0, y1):max(0, y1) + hh, max(0, x1):max(0, x2)]
            if roi.size:
                img[max(0, y1):max(0, y1) + hh, max(0, x1):max(0, x2)] = \
                    cv2.GaussianBlur(roi, (31, 31), 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img, f"{seat} {s['cue'].replace('_', ' ')}",
                    (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1,
                    cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes() if ok else jpeg_bytes


def main():
    """Headless replay: useful to diff two models over the same cache."""
    import argparse
    import model_registry as MR

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--model", default=None,
                    help="variant id, e.g. arch/asrf_556_hp (default: registry default)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="write per-frame cues as JSONL")
    args = ap.parse_args()

    entries = MR.scan()
    entry = MR.find(entries, args.model) if args.model else MR.default_entry(entries)
    if entry is None:
        raise SystemExit(f"unknown model {args.model!r}")
    if not entry.deployable:
        raise SystemExit(f"{entry.variant_id} is not deployable: {entry.blocked_reason}")
    # A cache holds base + both head-pose blocks and nothing else. Without this
    # the run reaches predict_window's width assert, which correctly reports
    # 556 against 1074 but reads as a broken extractor rather than as a model
    # this cache cannot serve.
    if not entry.replay_capable:
        raise SystemExit(
            f"{entry.variant_id} cannot replay a session cache: "
            f"{entry.blocked_reason}")

    cache = SessionCache(args.cache)
    bundle, cal = load_model(entry, args.device)
    print(f"[replay] {entry.variant_id} seed {entry.seed} — {bundle.describe()}")
    print(f"[replay] head-pose block: {entry.head_pose_backend}")

    fh = open(args.out, "w") if args.out else None
    counts: Dict[str, int] = defaultdict(int)

    def push(t, jpg, students, cue_names):
        for s in students.values():
            counts[s["cue"]] += 1
        if fh:
            fh.write(json.dumps({"t": t, "students": students}) + "\n")

    t0 = time.time()
    n = replay(cache, entry, bundle, push, realtime=False, overlay=False)
    el = time.time() - t0
    print(f"[replay] {n} frames in {el:.1f}s ({n / max(el, 1e-9):.1f} fps, "
          f"temporal head only)")
    total = sum(counts.values()) or 1
    for cue, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cue:16} {c:6d}  {100 * c / total:5.1f}%")
    if fh:
        fh.close()
        print(f"written: {args.out}")


if __name__ == "__main__":
    main()
