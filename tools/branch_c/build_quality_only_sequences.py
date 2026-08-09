"""Build quality/missingness-only sequences for the registered shortcut audit.

BRANCH_C_PROTOCOL.md §5.1 requires a classifier trained on **availability and
quality signals alone** — no appearance, no orientation. Its macro-F1 upper-bounds
how much of any model's performance is obtainable from *whether evidence exists*
rather than from what the evidence says. FINDINGS §11.10 and §12.13a make this
non-optional for this corpus: `face_found` alone was ~80% of the historic
head-pose gain, and it tracks physical face visibility (head_down 15.1%,
uncertain 3.9%) rather than orientation.

Eight signals, none of which describe what the student looks like or which way
they are facing:

    0  face_found          did the face detector fire at all
    1  det_conf            person-detector confidence
    2  occluded            annotation occlusion flag
    3  head_kpts / 5       how many head keypoints were recoverable
    4  face_kpts / 5       how many face keypoints were recoverable
    5  head_span_px / 300  apparent head size
    6  box width / 2812    apparent person width
    7  box height / 1050   apparent person height

Layout note: the array is written 570 columns wide with the eight signals in
columns 0-7 and **every other column structurally zero**, so it trains under the
existing ``552_base`` feature config with no change to
``attention/thesis_eval/data.py``. Modifying a shared Branch-A/B module to add a
config for one diagnostic is not worth the regression risk; 544 dead inputs cost
a dilated TCN essentially nothing and the information content is exactly the
eight signals.

Row alignment reuses ``patch_pose_columns.replay_chunks``, which reproduces the
builder's emission order and is verified by requiring every recovered chunk's
timestamps to equal the stored ``t`` array exactly.

    python tools/branch_c/build_quality_only_sequences.py
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

N_QUALITY = 8
FRAME_W, FRAME_H = 2812.0, 1050.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=REPO / "grounding_data/llmstu_sequences_full")
    ap.add_argument("--dst", type=Path, default=REPO / "grounding_data/llmstu_sequences_quality")
    ap.add_argument("--labels", type=Path,
                    default=REPO / "grounding_data/llmstu_tools/outputs/labels_tracked.jsonl")
    ap.add_argument("--frame-to-video", type=Path,
                    default=REPO / "grounding_data/llmstu_tools/outputs/frame_to_video.json")
    ap.add_argument("--pose-cache", type=Path,
                    default=REPO / "grounding_data/llmstu_tools/outputs/head_pose_cache.npz")
    ap.add_argument("--pipeline-only", action="store_true",
                    help="Drop occluded/head_kpts/face_kpts. Those three come from the "
                         "same annotation record the cue label is derived from, so they "
                         "are teacher-side rather than measurements available at "
                         "inference. This flag isolates the five genuinely "
                         "pipeline-measured signals and is what bounds the shortcut "
                         "claim (FINDINGS 12.19).")
    args = ap.parse_args()

    d = np.load(args.pose_cache, allow_pickle=False)
    face_found = {str(n): float(v) for n, v in zip(d["names"], d["vecs"][:, 3])}

    rec_by_name = {}
    with open(args.labels) as f:
        for line in f:
            r = json.loads(line)
            rec_by_name[r["file_name"]] = r
    print(f"labels {len(rec_by_name):,}   face_found cache {len(face_found):,}")

    for sp in ("train", "val"):
        (args.dst / sp).mkdir(parents=True, exist_ok=True)

    n_seq = n_frames = n_missing = 0
    for rec in replay_chunks(args.labels, args.frame_to_video):
        name = f"{rec['split']}/sample_{rec['sample_idx']:06d}.npz"
        fp = args.src / name
        if not fp.exists():
            raise SystemExit(f"replay produced {name}, absent from {args.src}; abort")
        z = np.load(fp)
        x, t = z["x"], z["t"].astype(np.float64)
        if x.shape[0] != len(rec["times"]) or not np.array_equal(t, rec["times"]):
            raise SystemExit(f"{name}: replay/stored timestamp mismatch; abort")

        q = np.zeros((x.shape[0], x.shape[1]), dtype=np.float32)
        for i, fn in enumerate(rec["file_names"]):
            r = rec_by_name.get(fn)
            if r is None:
                n_missing += 1
                continue
            bx = r.get("bbox_person") or [0, 0, 0, 0]
            if args.pipeline_only:
                q[i, 0] = face_found.get(fn, 0.0)
                q[i, 1] = float(r.get("det_conf", 0.0))
                q[i, 2] = float(r.get("head_span_px", 0.0)) / 300.0
                q[i, 3] = (float(bx[2]) - float(bx[0])) / FRAME_W
                q[i, 4] = (float(bx[3]) - float(bx[1])) / FRAME_H
            else:
                q[i, 0] = face_found.get(fn, 0.0)
                q[i, 1] = float(r.get("det_conf", 0.0))
                q[i, 2] = 1.0 if r.get("occluded") else 0.0
                q[i, 3] = float(r.get("head_kpts", 0)) / 5.0
                q[i, 4] = float(r.get("face_kpts", 0)) / 5.0
                q[i, 5] = float(r.get("head_span_px", 0.0)) / 300.0
                q[i, 6] = (float(bx[2]) - float(bx[0])) / FRAME_W
                q[i, 7] = (float(bx[3]) - float(bx[1])) / FRAME_H

        np.savez_compressed(args.dst / name, x=q, y_frames=z["y_frames"], y=z["y"], t=z["t"])
        n_seq += 1
        n_frames += x.shape[0]

    for f in ("meta.json",):
        if (args.src / f).exists():
            (args.dst / f).write_text((args.src / f).read_text())
    print(f"wrote {n_seq:,} sequences / {n_frames:,} frames -> {args.dst}")
    print(f"crops missing from labels: {n_missing}")
    n = 5 if args.pipeline_only else N_QUALITY
    print(f"informative columns: 0..{n-1}; all others structurally zero")
    if args.pipeline_only:
        print("pipeline-measured signals only; occluded/head_kpts/face_kpts excluded")


if __name__ == "__main__":
    main()
