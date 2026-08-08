"""Run the model-independent front end once, so switching models is instant.

Every deployable checkpoint in the registry consumes some subset of the *same*
556-column live vector. The parts that produce that vector — the LLMDet
detector, the tracker, CLIP, and the two head-pose backends — are identical for
all of them; only the temporal head differs. Re-running the front end on every
model switch would therefore recompute identical numbers and, on CPU, would make
each switch a minutes-long wait for a result that cannot differ.

So the front end runs **once per video** and its output is cached:

    frame_idx, track_id, bbox      what the tracker saw
    base            [N, 552]       CLIP + geometry + colour + posture
    head_landmarker [N, 4]         FaceLandmarker mesh: yaw, pitch, roll, found
    head_detector   [N, 4]         BlazeFace: 0, 0, 0, found
    frames/%06d.jpg                the clean downscaled frame

Two head-pose blocks, not one, because the sweeps disagree about which produced
their training features: models under ``ff_det`` were trained on the BlazeFace
flag and everything else on the landmarker mesh (FINDINGS 11.16). Caching both
lets ``session_replay`` hand each model the block it was actually trained on. A
single block would have forced either a silent train/deploy mismatch for half
the registry, or a second full pass per switch.

Frames are cached **clean**, without boxes or labels. The overlay depends on the
model's predictions, so it is drawn at replay time — a cached overlay would show
the previous model's cues under the new model's name.

The cache is a measurement of one video by one detector, so ``meta.json``
records the detector checkpoint, the prompt, the thresholds, the strides and the
git commit. A cache whose detector settings no longer match the config it is
replayed against is refused rather than silently reused.

    python tools/dashboard/precompute_session.py \
        --config LLMDet/configs/attention_runtime.yaml \
        --video LLMDet/0325.mp4 --frames 900 --device cpu \
        --out tools/dashboard/sessions/0325

COST. This is the expensive step and it is entirely CPU-bound when ``--device
cpu``: the detector dominates, then CLIP, then the two face models per student.
It is also the only step that needs the video at all — once the cache exists,
the dashboard and every model switch run from it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LLMDET_ROOT = REPO / "LLMDet"
sys.path.insert(0, str(LLMDET_ROOT))

CACHE_VERSION = "dashboard_session/1.0.0"


def _rel(p: Path) -> str:
    """Repo-relative when possible; a config may legitimately live elsewhere."""
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def precompute(config_path: str, video: str, out_dir: str, device: str = "cpu",
               max_frames: int = 900, jpeg_width: int = 960,
               progress_every: int = 25) -> Path:
    import cv2
    import numpy as np
    import yaml

    from attention.detector_adapter import FrozenLLMDetAdapter
    from attention.features import StudentFeatureExtractor, crop_boxes
    from attention.head_pose import HeadPoseEstimator
    from attention.realtime_infer import _det_appearance_feature
    from attention.thesis_eval.runtime import StrideController
    from attention.tracking import IoUTracker

    cfg = yaml.safe_load(open(config_path))
    out = Path(out_dir).resolve()
    (out / "frames").mkdir(parents=True, exist_ok=True)
    video_abs = str(Path(video).resolve())

    # Detector config/checkpoint and the CLIP cache dir are written relative to
    # LLMDet/, so resolve everything caller-relative first and then chdir.
    import os
    cwd0 = os.getcwd()
    os.chdir(LLMDET_ROOT)
    try:
        d = cfg["detector"]
        det = FrozenLLMDetAdapter(
            config_path=d["config_path"], checkpoint_path=d["checkpoint_path"],
            text_prompt=d.get("text_prompt", "a student sitting"),
            score_thr=float(d.get("score_thr", 0.10)), device=device,
            max_det=int(d.get("max_det", 80)),
            min_rel_area=float(d.get("min_rel_area", 0.01)),
            max_rel_area=float(d.get("max_rel_area", 0.60)),
            min_aspect_ratio=float(d.get("min_aspect_ratio", 0.22)),
            max_aspect_ratio=float(d.get("max_aspect_ratio", 1.25)),
            nms_iou_thr=float(d.get("nms_iou_thr", 0.5)))

        inf = cfg.get("inference", {})
        stride = StrideController(int(inf.get("detector_stride", 1)),
                                  int(inf.get("temporal_stride", 1)))
        t = cfg["tracking"]
        tracker = IoUTracker(
            iou_match_thr=float(t.get("iou_match_thr", 0.35)),
            max_age=int(t.get("max_age", 30)),
            min_hits=stride.adjusted_min_hits(int(t.get("min_hits", 3))),
            appearance_weight=float(t.get("appearance_weight", 0.35)),
            min_match_score=float(t.get("min_match_score", 0.25)))

        # head_pose=None: the extractor emits the 552-dim base only. Both
        # head-pose blocks are computed here instead, over the SAME crops, so
        # one CLIP pass serves every model in the registry.
        feat = StudentFeatureExtractor(
            clip_model_name=cfg["features"].get(
                "clip_model_name", "openai/clip-vit-base-patch32"),
            device=device, head_pose=None)
        if feat.output_dim() != 552:
            raise SystemExit(
                f"expected a 552-dim base from the extractor, got "
                f"{feat.output_dim()}. The cache layout assumes "
                f"base(552) + 4 + 4; fix the layout, do not pad.")

        backends = {}
        for name, backend in (("landmarker", "mediapipe"),
                              ("detector", "mediapipe_detector")):
            hp = HeadPoseEstimator(backend=backend)
            if not hp.available():
                raise SystemExit(
                    f"head-pose backend {backend!r} is unavailable. Both are "
                    f"required: the registry contains models trained on each, "
                    f"and feeding one the other's block is a silent "
                    f"train/deploy mismatch.")
            backends[name] = hp

        cap = cv2.VideoCapture(video_abs)
        if not cap.isOpened():
            raise SystemExit(f"cannot open {video_abs}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        rows_frame, rows_track, rows_bbox = [], [], []
        rows_base, rows_lm, rows_det = [], [], []
        frame_index = []          # frames actually written, in order
        n = 0
        t0 = time.time()
        while n < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            if stride.should_detect(n):
                dets = det.detect(frame)
                tracks = tracker.update(
                    dets, [_det_appearance_feature(frame, x.bbox_xyxy)
                           for x in dets])
            else:
                tracks = tracker.coast()

            if tracks:
                boxes = [tr.bbox_xyxy for tr in tracks]
                base = feat.extract_batch(frame, boxes)
                crops, _clipped, valid_idx = crop_boxes(frame, boxes)
                # Degenerate boxes produce no crop, so head pose defaults to the
                # all-zero block — the same thing extract_batch writes for them.
                lm = np.zeros((len(boxes), 4), dtype=np.float32)
                dd = np.zeros((len(boxes), 4), dtype=np.float32)
                for crop, i in zip(crops, valid_idx):
                    lm[i] = backends["landmarker"].estimate(crop)
                    dd[i] = backends["detector"].estimate(crop)
                for j, tr in enumerate(tracks):
                    rows_frame.append(n)
                    rows_track.append(int(tr.track_id))
                    rows_bbox.append([float(v) for v in tr.bbox_xyxy])
                    rows_base.append(base[j])
                    rows_lm.append(lm[j])
                    rows_det.append(dd[j])

            small = cv2.resize(
                frame, (jpeg_width, int(jpeg_width * frame.shape[0] / frame.shape[1])))
            cv2.imwrite(str(out / "frames" / f"{n:06d}.jpg"), small,
                        [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_index.append(n)

            n += 1
            if progress_every and n % progress_every == 0:
                el = time.time() - t0
                print(f"[precompute] {n} frames, {len(rows_frame)} student-frames, "
                      f"{el:.0f}s elapsed, {n / max(el, 1e-9):.2f} fps", flush=True)
        cap.release()
    finally:
        os.chdir(cwd0)

    if not rows_frame:
        raise SystemExit("no tracked students in this clip — nothing to cache")

    np.savez_compressed(
        out / "features.npz",
        frame_idx=np.asarray(rows_frame, dtype=np.int32),
        track_id=np.asarray(rows_track, dtype=np.int32),
        bbox=np.asarray(rows_bbox, dtype=np.float32),
        base=np.stack(rows_base).astype(np.float32),
        head_landmarker=np.stack(rows_lm).astype(np.float32),
        head_detector=np.stack(rows_det).astype(np.float32),
        frame_index=np.asarray(frame_index, dtype=np.int32))

    meta = {
        "cache_version": CACHE_VERSION,
        "git_commit": _git_commit(),
        "config": _rel(Path(config_path).resolve()),
        "video": video_abs,
        "fps": float(fps),
        "n_frames": n,
        "n_student_frames": len(rows_frame),
        "n_tracks": int(len(set(rows_track))),
        # Boxes are cached in SOURCE pixels while frames are cached downscaled,
        # so the replay overlay needs both widths to scale correctly.
        "source_width": int(cap_w),
        "source_height": int(cap_h),
        "jpeg_width": jpeg_width,
        "device": device,
        "wall_clock_s": round(time.time() - t0, 1),
        # Enough of the detector's identity to refuse a stale cache.
        "detector": {k: cfg["detector"].get(k) for k in
                     ("config_path", "checkpoint_path", "text_prompt",
                      "score_thr", "max_det", "nms_iou_thr")},
        "tracking": cfg["tracking"],
        "inference": cfg.get("inference", {}),
        "layout": {"base": 552, "head_landmarker": 4, "head_detector": 4},
        "note": ("Front end only. Contains no temporal-model output, so it is "
                 "shared by every model in the registry and does not need "
                 "rebuilding when the selected model changes."),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[precompute] {n} frames, {len(rows_frame)} student-frames, "
          f"{meta['n_tracks']} tracks in {meta['wall_clock_s']}s")
    print(f"[precompute] written: {out}")
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="LLMDet/configs/attention_runtime.yaml")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu",
                    help="cpu (default) or cuda:N. The dashboard is built to "
                         "run with no GPU at all.")
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--jpeg-width", type=int, default=960)
    args = ap.parse_args()
    precompute(args.config, args.video, args.out, args.device, args.frames,
               args.jpeg_width)


if __name__ == "__main__":
    main()
