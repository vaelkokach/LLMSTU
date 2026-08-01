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


def run_live(config_path, video, push_fn, blur_faces=False, max_frames=100000,
             record=None):
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
    video = str(Path(video).resolve()) if not str(video).isdigit() else video
    os.chdir(LLMDET_ROOT)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

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
    tracker = IoUTracker(
        iou_match_thr=float(cfg["tracking"].get("iou_match_thr", 0.35)),
        max_age=int(cfg["tracking"].get("max_age", 30)),
        min_hits=int(cfg["tracking"].get("min_hits", 3)),
        appearance_weight=float(cfg["tracking"].get("appearance_weight", 0.35)),
        min_match_score=float(cfg["tracking"].get("min_match_score", 0.25)))

    # Head pose is REQUIRED, not best-effort: the deployed model is 556-dim and
    # 4 of those dims are head pose. Swallowing a failure here used to leave the
    # extractor at 552 dims, which the old zero-padding path then hid.
    backend = cfg.get("features", {}).get("head_pose_backend", "mediapipe")
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
    bundle = load_runtime_model(cfg["temporal_checkpoint"], device=dev,
                               calibration=cfg.get("calibration"))
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

    win = int(cfg["inference"]["window_size"])
    minf = int(cfg["inference"].get("min_frames_for_pred", 4))
    hist = defaultdict(lambda: deque(maxlen=win))
    dwell = {}
    rec = open(record, "w") if record else None

    cap = cv2.VideoCapture(int(video) if str(video).isdigit() else video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = 0
    t_prev = time.time()
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t = n / fps
        dets = det.detect(frame)
        tracks = tracker.update(dets, [_det_appearance_feature(frame, d.bbox_xyxy)
                                       for d in dets])
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
                r = predict_window(bundle, np.stack(list(h)))
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

    cap.release()
    os.chdir(cwd0)
    if rec:
        rec.close()
    print(f"[dashboard] finished after {n} frames")
