"""Calibrate and validate the geometric head crop for the full-range estimator.

The corpus stores person crops. 6DRepNet360 wants head crops. ``head_span_px``
cannot bridge the gap - it is floored at exactly 120 px and sits at a median 0.43x
the person-box height, so it is a loose bound, not a head size. This script fixes
the one free parameter of the geometric alternative and, more importantly, checks
that the result is not garbage.

The check is the point. 6DRepNet360 emits a rotation for *every* input, including
a crop containing no head at all, so "it produced angles" is no evidence that it
worked. The only cheap ground truth available is MediaPipe's own estimate on the
frames where MediaPipe found a face. If the two disagree there, the crop is wrong
and any downstream conclusion drawn from these angles would be an artifact.

    python tools/branch_c/calibrate_head_crop.py --device cuda:1 --n 3000

Reports, per candidate fraction: circular correlation and median absolute
difference of yaw against MediaPipe, on the overlap subset only. Writes nothing
except its report - choosing the constant is a human decision recorded in
``head_pose_fullrange.DEFAULT_HEAD_FRACTION``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "LLMDet"))

CROPS = REPO / "grounding_data/LLMSTU/crops"
MP_CACHE = REPO / "grounding_data/llmstu_tools/outputs/head_pose_cache_bbox_person.npz"
FOLDS = REPO / "outputs/branch_c/splits/branch_c_folds.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--n", type=int, default=3000, help="crops sampled from the overlap")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--fractions", default="0.30,0.35,0.42,0.50,0.60,1.00")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import cv2
    import torch
    from attention.branch_c.head_pose_fullrange import FullRangeHeadPose, crop_head

    dev_videos = set(json.loads(FOLDS.read_text())["folds"][0]["inner_train_videos"])

    cache = np.load(MP_CACHE, allow_pickle=True)
    names, vecs = cache["names"], cache["vecs"]
    face_found = vecs[:, 3] > 0.5
    print(f"MediaPipe cache: {len(names):,} crops, face_found on {face_found.sum():,}")

    # Development videos only. The crop filenames in this cache do not carry the
    # video id, so map through the sibling cache that does.
    named = np.load(REPO / "grounding_data/llmstu_tools/outputs/head_pose_cache.npz",
                    allow_pickle=True)["names"]
    in_dev = np.array([any(v in n for v in dev_videos) for n in named]) if len(named) == len(names) else np.ones(len(names), bool)

    idx = np.where(face_found & in_dev)[0]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(idx, size=min(args.n, len(idx)), replace=False)
    print(f"sampled {len(idx):,} face-visible development crops\n")

    images, mp_yaw = [], []
    for i in idx:
        p = CROPS / str(named[i] if len(named) == len(names) else names[i])
        img = cv2.imread(str(p))
        if img is None:
            continue
        images.append(img)
        mp_yaw.append(float(vecs[i, 0]) * 90.0)
    mp_yaw = np.asarray(mp_yaw)
    print(f"loaded {len(images):,} images\n")

    est = FullRangeHeadPose(device=args.device).load()

    def circ_corr(a_deg, b_deg):
        a, b = np.deg2rad(a_deg), np.deg2rad(b_deg)
        a -= np.arctan2(np.sin(a).mean(), np.cos(a).mean())
        b -= np.arctan2(np.sin(b).mean(), np.cos(b).mean())
        return float((np.sin(a) * np.sin(b)).sum() /
                     np.sqrt((np.sin(a) ** 2).sum() * (np.sin(b) ** 2).sum()))

    print(f"{'fraction':>9}{'n':>7}{'circ corr':>12}{'|corr|':>9}"
          f"{'median |dYaw|':>15}{'6DRep yaw sd':>14}")
    print("-" * 66)
    for frac in [float(x) for x in args.fractions.split(",")]:
        crops = []
        keep = []
        for j, img in enumerate(images):
            c = crop_head(img, None, frac)
            if c is not None:
                crops.append(c)
                keep.append(j)
        yaws = []
        for s in range(0, len(crops), args.batch):
            R = est.estimate_batch(crops[s:s + args.batch])
            yaws.append(est.to_euler_degrees(R)[:, 0].numpy())
        yaw = np.concatenate(yaws) if yaws else np.array([])
        ref = mp_yaw[keep]
        cc = circ_corr(yaw, ref)
        # sign/offset-free comparison: align by the circular mean before differencing
        d = np.abs(((yaw - yaw.mean()) - (ref - ref.mean()) + 180) % 360 - 180)
        print(f"{frac:>9.2f}{len(yaw):>7}{cc:>12.3f}{abs(cc):>9.3f}"
              f"{np.median(d):>15.1f}{yaw.std():>14.1f}")
    print("-" * 66)
    print("A crop that contains the head should show |circ corr| clearly above the")
    print("fraction=1.00 row, which feeds the whole person box and is the control.")


if __name__ == "__main__":
    main()
