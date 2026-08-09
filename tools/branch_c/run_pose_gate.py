"""The registered full-range pose gate (BRANCH_C_PROTOCOL.md amendment A1).

Re-runs exactly the pre-test of FINDINGS 12.13 — same development videos, same
contrasts, same AUC statistic, same two causal reference estimators — but on
full-range 6DRepNet360 rotations instead of the cached MediaPipe angles.

The gate criterion was registered before this cache existed:

    If seat-relative canonicalisation of full-range rotation still yields ~null
    AUC improvement over raw rotation on the four face-visible classes, the pose
    arms (5, 6, 13) are not trained and the result is reported as a negative
    result.

Development data only: fold 0's inner-train videos. No outer fold, no legacy
Branch-B test split.

    python tools/branch_c/run_pose_gate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "LLMDet"))

from attention.branch_c.canonical import (  # noqa: E402
    geodesic_distance, matrix_to_euler, so3_exp, so3_log,
)
from attention.taxonomy import CUE_CLASSES, map_record  # noqa: E402

CACHE = REPO / "outputs/branch_c/cache/head_pose_fullrange/merged.npz"
LABELS = REPO / "grounding_data/llmstu_tools/outputs/labels_tracked.jsonl"
FOLDS = REPO / "outputs/branch_c/splits/branch_c_folds.json"


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUC, folded so 0.5 means no separation in either direction."""
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    a = (ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return max(a, 1 - a)


def causal_running_reference(R: torch.Tensor, momentum: float = 0.05) -> torch.Tensor:
    """Per-track reference from strictly past frames, tangent-space averaged.

    Returns the reference to use *at* each frame, i.e. index t reflects frames
    < t only. Frame 0 gets identity.
    """
    T = R.shape[0]
    out = torch.empty_like(R)
    cur = torch.eye(3, dtype=R.dtype)
    started = False
    for t in range(T):
        out[t] = cur
        if not started:
            cur = R[t].clone()
            started = True
        else:
            cur = cur @ so3_exp(so3_log(cur.transpose(-1, -2) @ R[t]).unsqueeze(0) * momentum)[0]
    return out


def tangent_mean(R: torch.Tensor, iters: int = 6) -> torch.Tensor:
    """Karcher mean on SO(3). A componentwise mean of rotations is not a rotation."""
    m = R[0].clone()
    for _ in range(iters):
        delta = so3_log(m.transpose(-1, -2).unsqueeze(0) @ R).mean(0)
        m = m @ so3_exp(delta.unsqueeze(0))[0]
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    dev = set(json.loads(FOLDS.read_text())["folds"][args.fold]["inner_train_videos"])
    d = np.load(CACHE, allow_pickle=True)
    names, Rflat, ok = d["names"], d["R"], d["ok"]
    idx_of = {str(n): i for i, n in enumerate(names)}
    print(f"cache: {len(names):,} crops, valid {ok.sum():,} ({ok.mean()*100:.1f}%)")

    tracks = defaultdict(list)
    n_lab = 0
    with open(LABELS) as f:
        for line in f:
            r = json.loads(line)
            if r["video_id"] not in dev:
                continue
            i = idx_of.get(r["file_name"])
            if i is None or not ok[i]:
                continue
            cue = map_record(r)  # returns a cue class id (int), taxonomy.py:81
            if cue is None:
                continue
            tracks[(r["video_id"], str(r["seat_id"]))].append(
                (r["src_frame"], i, int(cue))
            )
            n_lab += 1
    print(f"development crops with a cue label and a rotation: {n_lab:,} "
          f"across {len(tracks):,} (video, seat) tracks\n")

    # ---- assemble per-track causal features -------------------------------
    feats = {"raw_yaw": [], "raw_geo": [], "causal_rel": [], "scene_rel": []}
    labs = []
    track_R, track_lab, track_key = {}, {}, []
    for key, items in tracks.items():
        items.sort(key=lambda z: z[0])          # temporal order within the track
        R = torch.from_numpy(Rflat[[i for _, i, _ in items]].reshape(-1, 3, 3)).double()
        track_R[key] = R
        track_lab[key] = np.array([c for _, _, c in items])
        track_key.append(key)

    # scene reference: Karcher mean per seat_id over OTHER videos (leave-one-video-out)
    by_seat = defaultdict(list)
    for (vid, seat) in track_key:
        by_seat[seat].append(vid)
    seat_pool = defaultdict(list)
    for (vid, seat), R in track_R.items():
        seat_pool[seat].append((vid, R))

    global_mean = tangent_mean(torch.cat([track_R[k][::37] for k in track_key])[:4000])

    for key in track_key:
        vid, seat = key
        R = track_R[key]
        labs.append(track_lab[key])

        yaw, _, _ = matrix_to_euler(R)
        feats["raw_yaw"].append(np.rad2deg(yaw.numpy()))
        feats["raw_geo"].append(
            np.rad2deg(geodesic_distance(global_mean.expand_as(R), R).numpy())
        )

        ref_c = causal_running_reference(R)
        feats["causal_rel"].append(
            np.rad2deg(geodesic_distance(ref_c, R).numpy())
        )

        others = [r for v, r in seat_pool[seat] if v != vid]
        if others:
            pool = torch.cat(others)
            ref_s = tangent_mean(pool[:: max(1, len(pool) // 800)][:800])
        else:
            ref_s = global_mean
        feats["scene_rel"].append(
            np.rad2deg(geodesic_distance(ref_s.expand_as(R), R).numpy())
        )

    lab = np.concatenate(labs)
    for k in feats:
        feats[k] = np.concatenate(feats[k])
    print(f"frames: {len(lab):,}\n")

    # ---- H1: coverage and class-conditional spread ------------------------
    print("H1 — coverage by cue (full-range estimator has no detection stage):")
    print(f"{'cue':<17}{'frames':>9}{'coverage':>11}{'mean geo(deg)':>15}{'sd':>8}")
    for c, name in enumerate(CUE_CLASSES):
        m = lab == c
        if m.sum() == 0:
            continue
        print(f"{name:<17}{m.sum():>9,}{100.0:>10.1f}%"
              f"{feats['raw_geo'][m].mean():>15.1f}{feats['raw_geo'][m].std():>8.1f}")

    # ---- the gate ---------------------------------------------------------
    print("\nGATE — AUC separating screen_oriented from each cue")
    print(f"{'contrast':<34}{'raw yaw':>10}{'raw geo':>10}"
          f"{'causal rel':>13}{'scene rel':>12}{'best delta':>12}")
    print("-" * 91)
    face_visible = [1, 3, 4]        # looking_away, turned_to_peer, phone_use
    deltas = []
    for c in [1, 2, 3, 4, 5]:
        m0, mc = lab == 0, lab == c
        if mc.sum() < 100:
            continue
        a = {k: auc(feats[k][mc], feats[k][m0]) for k in feats}
        base = max(a["raw_yaw"], a["raw_geo"])
        best_rel = max(a["causal_rel"], a["scene_rel"])
        delta = best_rel - base
        if c in face_visible:
            deltas.append(delta)
        print(f"{'screen_oriented vs ' + CUE_CLASSES[c]:<34}"
              f"{a['raw_yaw']:>10.3f}{a['raw_geo']:>10.3f}"
              f"{a['causal_rel']:>13.3f}{a['scene_rel']:>12.3f}{delta:>+12.3f}")
    print("-" * 91)
    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    print(f"\nmean AUC delta over the face-visible contrasts: {mean_delta:+.4f}")
    print("Registered criterion (protocol A1): a ~null delta means the pose arms")
    print("(5, 6, 13) are NOT trained and the negative result is reported.")


if __name__ == "__main__":
    main()
