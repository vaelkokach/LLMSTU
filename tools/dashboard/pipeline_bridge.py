"""Bridge the real-time pipeline into the dashboard.

Runs detector -> tracker -> features -> temporal model over a video and pushes
per-frame state to the dashboard. Also records a replay log so the dashboard can
be demonstrated later without a GPU (useful for a thesis defence on a laptop).

Kept separate from server.py so the dashboard has no torch/mmdet dependency in
replay mode — `python server.py --replay session.jsonl` needs stdlib only.
"""
import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

LLMDET_ROOT = Path(__file__).resolve().parents[2] / "LLMDet"
sys.path.insert(0, str(LLMDET_ROOT))

#: URL schemes OpenCV can open directly (FFMPEG/GStreamer backends).
_STREAM_SCHEMES = ("rtsp://", "rtsps://", "http://", "https://", "udp://",
                   "tcp://", "rtmp://")


def classify_source(video):
    """-> ("camera" | "stream" | "file", opencv_argument)

    Three source kinds need three different treatments and only the file case
    is a path. ``Path(url).resolve()`` silently turns ``rtsp://cam/stream`` into
    ``<cwd>/rtsp:/cam/stream``, so an IP camera used to fail with "cannot open"
    naming a path the user never typed.

    Camera and stream are *live*: they produce frames whether or not anything is
    consuming them, which changes both how frames must be read and what a
    timestamp means. A file does neither.
    """
    s = str(video)
    if s.isdigit():
        return "camera", int(s)
    if s.lower().startswith(_STREAM_SCHEMES):
        return "stream", s
    return "file", str(Path(s).resolve())


class LatestFrame:
    """Reader thread that keeps only the newest frame from a live source.

    A live camera produces frames on its own clock. The pipeline consumes them
    at a few per second, so reading sequentially means consuming a *queue*: the
    displayed frame falls further behind the room the longer the dashboard runs,
    with no upper bound and nothing on screen to say so. An instructor would be
    alerted about a student who put their phone away two minutes ago.

    ``cv2.CAP_PROP_BUFFERSIZE`` is honoured by some backends and ignored by the
    FFMPEG one that handles RTSP, so it cannot be relied on. This drains the
    source in its own thread instead and hands out the most recent frame,
    turning the lag into dropped frames — which is the right trade for a
    dashboard, and is counted so the drop rate can be reported rather than
    guessed at.
    """

    def __init__(self, cap):
        import threading
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0            # frames read from the source
        self.taken = 0          # frames the pipeline actually processed
        self.alive = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.alive:
            ok, f = self.cap.read()
            if not ok:
                self.alive = False
                break
            with self.lock:
                self.frame = f
                self.seq += 1

    def read(self, timeout=5.0):
        """Newest unseen frame, or (False, None) if the source stopped."""
        deadline = time.time() + timeout
        last = -1
        while time.time() < deadline:
            with self.lock:
                if self.frame is not None and self.seq != last:
                    self.taken += 1
                    return True, self.frame
                last = self.seq
            if not self.alive:
                return False, None
            time.sleep(0.005)
        return self.alive, None

    @property
    def dropped(self):
        with self.lock:
            return max(0, self.seq - self.taken)

    def release(self):
        self.alive = False
        self.thread.join(timeout=2.0)
        self.cap.release()


def run_live(config_path, video, push_fn, blur_faces=False, max_frames=100000,
             record=None, device=None, entry=None, should_stop=None,
             stats_fn=None):
    """Detector -> tracker -> features -> temporal model over a live video.

    ``entry`` is an optional ``model_registry.ModelEntry``. When given, its
    checkpoint, calibration and head-pose backend override the config's, which
    is what lets the dashboard's model selector drive a live run as well as a
    cached one. The head-pose backend has to come from the entry rather than the
    config because it decides what the extractor *produces*: a model trained on
    the BlazeFace flag and fed the FaceLandmarker block still runs and still
    prints cues.

    ``stats_fn(dict)`` is called once a second for a live source with the
    processed frame rate and the drop rate. A live dashboard that cannot keep up
    with the camera is still useful, but only if it says so — an overlay that
    looks current and is thirty seconds stale is worse than one labelled stale.
    """
    import cv2
    import numpy as np
    import torch
    import yaml

    from attention.detector_adapter import FrozenLLMDetAdapter
    from attention.features import StudentFeatureExtractor
    from attention.head_pose import HeadPoseEstimator
    from attention.thesis_eval.runtime import load_runtime_model, predict_window
    from attention.tracking import IoUTracker
    from attention.taxonomy import CUE_CLASSES
    from attention.realtime_infer import _det_appearance_feature

    cfg = yaml.safe_load(open(config_path))
    # Config paths (detector config/checkpoint, CLIP dirs) are written relative
    # to LLMDet/. Resolve from there so the dashboard can be launched from
    # anywhere rather than only from that directory.
    import os
    cwd0 = os.getcwd()
    # Resolve caller-relative paths BEFORE chdir, or outputs land in LLMDet/
    # instead of where the caller asked for them.
    if record:
        record = str(Path(record).resolve())
    kind, source = classify_source(video)
    os.chdir(LLMDET_ROOT)
    # The caller decides, then the config, then CPU. Grabbing cuda:0 whenever a
    # GPU exists is wrong on a shared box: the GPUs may belong to someone else's
    # job, and the dashboard is expected to be demonstrable without one.
    dev = device or cfg.get("detector", {}).get("device") or "cpu"
    if str(dev).startswith("cuda") and not torch.cuda.is_available():
        dev = "cpu"

    det = FrozenLLMDetAdapter(
        config_path=cfg["detector"]["config_path"],
        checkpoint_path=cfg["detector"]["checkpoint_path"],
        text_prompt=cfg["detector"].get("text_prompt", "a student sitting"),
        score_thr=float(cfg["detector"].get("score_thr", 0.10)),
        device=dev,
        max_det=int(cfg["detector"].get("max_det", 80)),
        min_rel_area=float(cfg["detector"].get("min_rel_area", 0.01)),
        max_rel_area=float(cfg["detector"].get("max_rel_area", 0.60)),
        min_aspect_ratio=float(cfg["detector"].get("min_aspect_ratio", 0.22)),
        max_aspect_ratio=float(cfg["detector"].get("max_aspect_ratio", 1.25)),
        nms_iou_thr=float(cfg["detector"].get("nms_iou_thr", 0.5)))
    from attention.thesis_eval.runtime import StrideController as _SC
    _stride_cfg = _SC(int(cfg.get("inference", {}).get("detector_stride", 1)),
                      int(cfg.get("inference", {}).get("temporal_stride", 1)))
    tracker = IoUTracker(
        iou_match_thr=float(cfg["tracking"].get("iou_match_thr", 0.35)),
        max_age=int(cfg["tracking"].get("max_age", 30)),
        min_hits=_stride_cfg.adjusted_min_hits(int(cfg["tracking"].get("min_hits", 3))),
        appearance_weight=float(cfg["tracking"].get("appearance_weight", 0.35)),
        min_match_score=float(cfg["tracking"].get("min_match_score", 0.25)))

    # Head pose is REQUIRED, not best-effort: the deployed model is 556-dim and
    # 4 of those dims are head pose. Swallowing a failure here used to leave the
    # extractor at 552 dims, which the old zero-padding path then hid.
    backend = (entry.head_pose_backend if entry is not None
               else cfg.get("features", {}).get("head_pose_backend", "mediapipe"))
    hp = HeadPoseEstimator(backend=backend) if backend else None
    if hp is not None and not hp.available():
        raise SystemExit(
            f"head-pose backend {backend!r} is unavailable, but the deployed "
            "model needs its 4 dims. Install it or point --config at a "
            "checkpoint trained without head pose.")
    feat = StudentFeatureExtractor(
        clip_model_name=cfg["features"].get("clip_model_name",
                                            "openai/clip-vit-base-patch32"),
        device=dev, head_pose=hp)

    # Build from the CHECKPOINT's own spec. A YAML/checkpoint disagreement used
    # to raise inside a bare except and run the dashboard on random weights.
    if entry is not None:
        from session_replay import calibration_path
        ckpt = str(Path(__file__).resolve().parents[2] / entry.checkpoint)
        cal = str(calibration_path(entry.variant_id))
    else:
        ckpt, cal = cfg["temporal_checkpoint"], cfg.get("calibration")
    bundle = load_runtime_model(ckpt, device=dev, calibration=cal)
    # The extractor always emits base + the 4 head-pose columns; a checkpoint
    # trained on a subset (e.g. 553_facefound) selects its columns inside
    # predict_window. Assert the EXTRACTOR width, not the model width.
    want = bundle.live_input_width
    if feat.output_dim() != want:
        raise SystemExit(
            f"live features are {feat.output_dim()}-dim but "
            f"{bundle.experiment_id} needs an extractor producing {want} "
            f"({bundle.feature_config}). Refusing to pad — a zero block is "
            "indistinguishable from a real measurement.")
    print(f"[dashboard] temporal model: {bundle.describe()}")

    from attention.thesis_eval.runtime import StrideController
    stride = StrideController(
        detector_stride=int(cfg["inference"].get("detector_stride", 1)),
        temporal_stride=int(cfg["inference"].get("temporal_stride", 1)))
    win = int(cfg["inference"]["window_size"])
    minf = int(cfg["inference"].get("min_frames_for_pred", 4))
    hist = defaultdict(lambda: deque(maxlen=win))
    dwell = {}
    rec = open(record, "w") if record else None

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(
            f"cannot open {kind} source {source!r}. For a webcam pass the "
            f"device index (--video 0); for an IP camera pass the full URL "
            f"(--video rtsp://user:pass@host/stream).")
    live = kind in ("camera", "stream")
    # A live source is drained by a reader thread so the pipeline always gets
    # the newest frame; a file is read sequentially, since every frame matters
    # and none of them are going stale.
    reader = LatestFrame(cap) if live else cap
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = 0
    t_prev = time.time()
    t_start = time.time()
    t_stats = t_start
    print(f"[dashboard] source: {kind} {source!r}"
          + (" — frames are dropped to stay current" if live else ""))
    while n < max_frames:
        # Lets the dashboard abandon a run mid-video when the user picks a
        # different model, instead of leaving two pipelines pushing frames.
        if should_stop is not None and should_stop():
            break
        ok, frame = reader.read()
        if not ok:
            break
        if frame is None:       # live source stalled; keep waiting
            continue
        # Timestamps drive the dwell thresholds that raise alerts ("head down
        # for 30 s"), so they have to be in the same seconds the instructor is
        # living in. For a file, frame index over fps IS that clock. For a live
        # source it is not: frames are dropped, so n/fps would run slow and a
        # 30 s episode would be announced minutes late.
        t = (time.time() - t_start) if live else (n / fps)
        if stride.should_detect(n):
            dets = det.detect(frame)
            tracks = tracker.update(dets, [_det_appearance_feature(frame, d.bbox_xyxy)
                                           for d in dets])
        else:
            tracks = tracker.coast()
        stride.drop_missing(tr.track_id for tr in tracks)
        students = {}
        if tracks:
            fv = feat.extract_batch(frame, [tr.bbox_xyxy for tr in tracks])
            for tr, f in zip(tracks, fv):
                hist[tr.track_id].append(f)

            for tr in tracks:
                h = hist[tr.track_id]
                if len(h) < minf:
                    continue
                # predict_window applies the validation-fitted temperature and
                # both abstention thresholds, and asserts the feature width.
                if stride.should_predict(tr.track_id, n):
                    r = stride.store(tr.track_id, n,
                                     predict_window(bundle, np.stack(list(h))))
                else:
                    r = stride.cached(tr.track_id)
                cue = r["displayed_cue"]
                conf = r["confidence"]
                # Dwell accumulates on the DISPLAYED cue, so an abstention
                # interrupts an episode rather than silently extending it.
                prev = dwell.get(tr.track_id)
                if prev and prev["cue"] == cue:
                    prev["dwell"] = t - prev["since"]
                else:
                    dwell[tr.track_id] = {"cue": cue, "since": t, "dwell": 0.0,
                                          "alerted": False}
                st = dwell[tr.track_id]
                students[str(tr.track_id)] = {
                    "cue": cue, "conf": round(conf, 2),
                    # Track ids are assigned in detection order and are NOT
                    # comparable between runs; the box is, and it is what lets
                    # tools/verify_stride_equivalence.py match students across
                    # configurations the way a human comparing two overlays would.
                    "bbox": [round(float(v), 1) for v in tr.bbox_xyxy],
                    "dwell": st["dwell"], "alerted": st["alerted"],
                    # raw prediction preserved even when abstaining: the point
                    # of abstention is to withhold an alert, not evidence
                    "raw_cue": r["cue"], "abstained": r["abstained"],
                    "alert_allowed": r["alert_allowed"]}

        vis = frame.copy()
        for tr in tracks:
            x1, y1, x2, y2 = [int(v) for v in tr.bbox_xyxy]
            s = students.get(str(tr.track_id))
            cue = s["cue"] if s else "…"
            col = (61, 220, 132) if cue == "screen_oriented" else \
                  (86, 95, 255) if cue in ("head_down", "phone_use") else \
                  (84, 180, 255) if cue in ("looking_away", "turned_to_peer") else \
                  (160, 160, 160)
            if blur_faces:
                hh = max(1, int(0.35 * (y2 - y1)))
                roi = vis[y1:y1 + hh, x1:x2]
                if roi.size:
                    vis[y1:y1 + hh, x1:x2] = cv2.GaussianBlur(roi, (31, 31), 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            cv2.putText(vis, f"{tr.track_id} {cue.replace('_',' ')}",
                        (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        small = cv2.resize(vis, (960, int(960 * vis.shape[0] / vis.shape[1])))
        ok2, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        push_fn(t, buf.tobytes() if ok2 else None, students, CUE_CLASSES)
        if rec:
            rec.write(json.dumps({"t": t, "students": students,
                                  "dt": time.time() - t_prev,
                                  "cue_names": CUE_CLASSES}) + "\n")
        t_prev = time.time()
        n += 1

        if live and stats_fn is not None and time.time() - t_stats >= 1.0:
            t_stats = time.time()
            dropped = reader.dropped
            stats_fn({
                "live": True,
                "processed_fps": round(n / max(t_stats - t_start, 1e-9), 2),
                "frames_processed": n,
                "frames_dropped": dropped,
                "drop_pct": round(100 * dropped / max(dropped + n, 1)),
                "source_fps": round(fps, 1),
            })

    if live:
        # The drop rate is the honest measure of how far behind real time the
        # pipeline is running: 9 dropped for every 1 processed means the model
        # sees the room at a tenth of its actual frame rate.
        dropped = reader.dropped
        total = dropped + n
        print(f"[dashboard] live source: processed {n}, dropped {dropped} "
              f"({100 * dropped / max(total, 1):.0f}% of {total}), "
              f"{n / max(time.time() - t_start, 1e-9):.2f} processed fps")
    reader.release() if live else cap.release()
    os.chdir(cwd0)
    if rec:
        rec.close()
    print(f"[dashboard] finished after {n} frames")
