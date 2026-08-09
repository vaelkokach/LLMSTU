"""Build the multi-expert Branch-C sequences for the fusion arms.

One array per sequence carrying every expert's evidence side by side, plus an
explicit per-modality quality vector, so the fusion arms (protocol arms 7-12, 17)
can be trained without re-deriving features per run.

Column layout — recorded here and asserted by the loader, because a silently
shifted block is the failure mode this project has already paid for once:

    [  0:552]  appearance   the proven 552-dim CLIP/geometry/colour base
    [552:561]  head         6DRepNet360 rotation matrix, flattened (9)
    [561:569]  quality      8 availability/quality signals (see below)
    [569:575]  motion       6 CAUSAL box-trajectory features

Quality block, columns 561-568:
    0 face_found · 1 det_conf · 2 occluded · 3 head_kpts/5 · 4 face_kpts/5
    5 head_span_px/300 · 6 box_w/2812 · 7 box_h/1050

Motion block, columns 569-574 — **backward differences only**:
    0 d_cx · 1 d_cy · 2 d_w · 3 d_h · 4 speed · 5 d_speed

The existing ``dynamic`` block (570-dim builds, columns 563:570) is deliberately
NOT reused. ``compute_dynamic`` derives its statistics over the *whole track*,
which is acausal, and BRANCH_C_PROTOCOL.md §6 forbids feeding an acausal feature
to an online model. Frame 0 gets zero motion rather than a wrapped or forward
difference.

Row alignment reuses ``patch_pose_columns.replay_chunks`` and inherits its proof:
every recovered chunk's timestamps must equal the stored ``t`` array exactly.

    python tools/branch_c/build_branch_c_sequences.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "LLMDet"))

from attention.thesis_eval.patch_pose_columns import replay_chunks  # noqa: E402

LAYOUT = {
    "appearance": (0, 552),
    "head": (552, 561),
    "quality": (561, 569),
    "motion": (569, 575),
}
TOTAL_DIM = 575
FRAME_W, FRAME_H = 2812.0, 1050.0


def causal_motion(boxes: list) -> np.ndarray:
    """Backward-difference box trajectory features. Frame 0 is zero by definition."""
    b = np.asarray(boxes, dtype=np.float64)
    n = len(b)
    out = np.zeros((n, 6), dtype=np.float32)
    if n < 2:
        return out
    cx = (b[:, 0] + b[:, 2]) * 0.5 / FRAME_W
    cy = (b[:, 1] + b[:, 3]) * 0.5 / FRAME_H
    w = (b[:, 2] - b[:, 0]) / FRAME_W
    h = (b[:, 3] - b[:, 1]) / FRAME_H
    out[1:, 0] = np.diff(cx)
    out[1:, 1] = np.diff(cy)
    out[1:, 2] = np.diff(w)
    out[1:, 3] = np.diff(h)
    speed = np.hypot(out[:, 0], out[:, 1])
    out[:, 4] = speed
    out[1:, 5] = np.diff(speed)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=REPO / "grounding_data/llmstu_sequences_full")
    ap.add_argument("--dst", type=Path, default=REPO / "grounding_data/llmstu_sequences_branch_c")
    ap.add_argument("--labels", type=Path,
                    default=REPO / "grounding_data/llmstu_tools/outputs/labels_tracked.jsonl")
    ap.add_argument("--frame-to-video", type=Path,
                    default=REPO / "grounding_data/llmstu_tools/outputs/frame_to_video.json")
    ap.add_argument("--rotation-cache", type=Path,
                    default=REPO / "outputs/branch_c/cache/head_pose_fullrange/merged.npz")
    ap.add_argument("--face-cache", type=Path,
                    default=REPO / "grounding_data/llmstu_tools/outputs/head_pose_cache.npz")
    args = ap.parse_args()

    rot = np.load(args.rotation_cache, allow_pickle=True)
    rot_idx = {str(n): i for i, n in enumerate(rot["names"])}
    rot_R, rot_ok = rot["R"], rot["ok"]

    fc = np.load(args.face_cache, allow_pickle=False)
    face_found = {str(n): float(v) for n, v in zip(fc["names"], fc["vecs"][:, 3])}

    labels = {}
    with open(args.labels) as f:
        for line in f:
            r = json.loads(line)
            labels[r["file_name"]] = r
    print(f"rotations {len(rot_idx):,}   labels {len(labels):,}")

    for sp in ("train", "val"):
        (args.dst / sp).mkdir(parents=True, exist_ok=True)

    n_seq = n_frames = n_missing_rot = n_missing_lab = 0
    for rec in replay_chunks(args.labels, args.frame_to_video):
        name = f"{rec['split']}/sample_{rec['sample_idx']:06d}.npz"
        fp = args.src / name
        if not fp.exists():
            raise SystemExit(f"replay produced {name}, absent from {args.src}; abort")
        z = np.load(fp)
        x_src, t = z["x"], z["t"].astype(np.float64)
        if x_src.shape[0] != len(rec["times"]) or not np.array_equal(t, rec["times"]):
            raise SystemExit(f"{name}: replay/stored timestamp mismatch; abort")

        T = x_src.shape[0]
        x = np.zeros((T, TOTAL_DIM), dtype=np.float32)
        x[:, 0:552] = x_src[:, 0:552]

        for i, fn in enumerate(rec["file_names"]):
            j = rot_idx.get(fn)
            if j is not None and rot_ok[j]:
                x[i, 552:561] = rot_R[j]
            else:
                n_missing_rot += 1
            r = labels.get(fn)
            if r is None:
                n_missing_lab += 1
                continue
            bx = r.get("bbox_person") or [0, 0, 0, 0]
            x[i, 561] = face_found.get(fn, 0.0)
            x[i, 562] = float(r.get("det_conf", 0.0))
            x[i, 563] = 1.0 if r.get("occluded") else 0.0
            x[i, 564] = float(r.get("head_kpts", 0)) / 5.0
            x[i, 565] = float(r.get("face_kpts", 0)) / 5.0
            x[i, 566] = float(r.get("head_span_px", 0.0)) / 300.0
            x[i, 567] = (float(bx[2]) - float(bx[0])) / FRAME_W
            x[i, 568] = (float(bx[3]) - float(bx[1])) / FRAME_H

        x[:, 569:575] = causal_motion(rec["boxes"])

        np.savez_compressed(args.dst / name, x=x, y_frames=z["y_frames"],
                            y=z["y"], t=z["t"])
        n_seq += 1
        n_frames += T

    meta = json.loads((args.src / "meta.json").read_text())
    meta["branch_c_layout"] = {k: list(v) for k, v in LAYOUT.items()}
    meta["branch_c_total_dim"] = TOTAL_DIM
    meta["branch_c_note"] = (
        "Motion is causal (backward differences). The acausal 570-dim dynamic block "
        "is deliberately excluded, per BRANCH_C_PROTOCOL.md section 6."
    )
    (args.dst / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"wrote {n_seq:,} sequences / {n_frames:,} frames -> {args.dst}")
    print(f"missing rotations {n_missing_rot}   missing labels {n_missing_lab}")
    for k, (a, b) in LAYOUT.items():
        print(f"  {k:<11} [{a}:{b}]  {b-a} dims")


if __name__ == "__main__":
    main()
