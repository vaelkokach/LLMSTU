"""Generate the full-range head-pose cache over the LLMSTU crop corpus.

Writes rotation matrices for **every** crop - 6DRepNet360 is a regressor with no
detection stage, so coverage is 100% by construction. That is the point: it is
what makes hypothesis H1 (BRANCH_C_PROTOCOL.md 5) testable, by removing the
missingness signal the existing 556-dim block leans on.

    python tools/branch_c/precompute_head_pose_fullrange.py --shards 3

Sharded across GPUs chosen by measured free memory (never more than the standing
cap of four, never onto a card another user is on). Each shard writes its own
atomic .npz plus a completion sentinel; a shard that dies leaves no file that
could be mistaken for finished output. The merge step refuses to run until every
shard's sentinel is present.

Provenance for every shard goes to outputs/branch_c/RUNS.jsonl via
tools/branch_c/provenance.py, per protocol section 9.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "LLMDet"))

CROPS = REPO / "grounding_data/LLMSTU/crops"
NAME_CACHE = REPO / "grounding_data/llmstu_tools/outputs/head_pose_cache.npz"
OUT_DIR = REPO / "outputs/branch_c/cache/head_pose_fullrange"

_spec = importlib.util.spec_from_file_location("prov", REPO / "tools/branch_c/provenance.py")
prov = importlib.util.module_from_spec(_spec)
# Must be registered before exec: @dataclass resolves field types through
# sys.modules[cls.__module__], which raises AttributeError on None if the module
# was created by importlib but never registered.
sys.modules["prov"] = prov
_spec.loader.exec_module(prov)


def run_shard(shard: int, n_shards: int, device: str, batch: int) -> None:
    import cv2
    from attention.branch_c.head_pose_fullrange import (
        DEFAULT_HEAD_FRACTION, FullRangeHeadPose, WEIGHTS_SHA256, crop_head,
    )

    names = np.load(NAME_CACHE, allow_pickle=True)["names"]
    mine = np.arange(len(names))[shard::n_shards]
    out_path = OUT_DIR / f"shard_{shard:02d}.npz"
    shard_dir = OUT_DIR / f"shard_{shard:02d}"
    if prov.is_complete(shard_dir):
        print(f"[shard {shard}] already complete, skipping")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    est = FullRangeHeadPose(device=device).load()
    R_out = np.zeros((len(mine), 9), dtype=np.float32)
    ok = np.zeros(len(mine), dtype=bool)

    t0 = time.time()
    buf, pos = [], []
    def flush():
        if not buf:
            return
        R = est.estimate_batch(buf).reshape(len(buf), 9).numpy()
        R_out[pos] = R
        ok[pos] = True
        buf.clear()
        pos.clear()

    for j, i in enumerate(mine):
        img = cv2.imread(str(CROPS / str(names[i])))
        c = crop_head(img, None, DEFAULT_HEAD_FRACTION) if img is not None else None
        if c is None:
            continue
        buf.append(c)
        pos.append(j)
        if len(buf) >= batch:
            flush()
        if j % 20000 == 0 and j:
            el = time.time() - t0
            print(f"[shard {shard}] {j:,}/{len(mine):,}  {j/el:.0f} crops/s  "
                  f"eta {(len(mine)-j)/max(j/el,1e-6)/60:.1f} min", flush=True)
    flush()

    # np.savez_compressed appends '.npz' to any path that does not already end in
    # it, so a '.partial' temp name comes back as '.partial.npz'. Name the temp
    # with the suffix already present and the rename target is what was written.
    tmp = out_path.with_name(out_path.name + ".partial.npz")
    np.savez_compressed(tmp, index=mine.astype(np.int64), R=R_out, ok=ok)
    os.replace(tmp, out_path)
    prov.write_sentinel(shard_dir, {
        "shard": shard, "n_shards": n_shards, "n": int(len(mine)),
        "n_ok": int(ok.sum()), "device": device,
        "head_fraction": DEFAULT_HEAD_FRACTION, "weights_sha256": WEIGHTS_SHA256,
    })
    wall = time.time() - t0
    print(f"[shard {shard}] done {ok.sum():,}/{len(mine):,} in {wall/60:.1f} min")

    prov.append_record(prov.RunRecord(
        run_id=f"head_pose_fullrange_shard{shard:02d}",
        arm="cache/head_pose_fullrange", stage="cache",
        command=[sys.executable, __file__, "--shard", str(shard)],
        device=device, wall_clock_s=wall,
        checkpoint_sha256=WEIGHTS_SHA256,
        split_manifest_sha256=prov.sha256_file(
            REPO / "outputs/branch_c/splits/branch_c_folds.json"),
        metrics={"n": int(len(mine)), "n_ok": int(ok.sum()),
                 "crops_per_s": float(len(mine) / max(wall, 1e-9))},
        notes=f"6DRepNet360 full-range head pose, head_fraction={DEFAULT_HEAD_FRACTION}",
    ))


def merge(n_shards: int) -> None:
    names = np.load(NAME_CACHE, allow_pickle=True)["names"]
    missing = [s for s in range(n_shards)
               if not prov.is_complete(OUT_DIR / f"shard_{s:02d}")]
    if missing:
        raise SystemExit(f"refusing to merge: shards {missing} have no completion sentinel")
    R = np.zeros((len(names), 9), dtype=np.float32)
    ok = np.zeros(len(names), dtype=bool)
    for s in range(n_shards):
        d = np.load(OUT_DIR / f"shard_{s:02d}.npz")
        R[d["index"]] = d["R"]
        ok[d["index"]] = d["ok"]
    tmp = OUT_DIR / "merged.npz.partial.npz"
    np.savez_compressed(tmp, names=names, R=R, ok=ok)
    os.replace(tmp, OUT_DIR / "merged.npz")
    prov.write_sentinel(OUT_DIR, {"n": int(len(names)), "n_ok": int(ok.sum())})
    print(f"merged {ok.sum():,}/{len(names):,} crops -> {OUT_DIR/'merged.npz'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--shard", type=int, default=None, help="run one shard (internal)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    if args.merge_only:
        merge(args.shards)
        return
    if args.shard is not None:
        run_shard(args.shard, args.shards, args.device or "cuda:0", args.batch)
        return

    gpus = prov.select_free_gpus(n=min(args.shards, prov.MAX_CONCURRENT_GPUS))
    print(f"dispatching {args.shards} shards onto GPUs {gpus}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    procs = []
    for s in range(args.shards):
        dev = f"cuda:{gpus[s % len(gpus)]}"
        log = open(OUT_DIR / f"shard_{s:02d}.log", "w")
        procs.append((s, subprocess.Popen(
            [sys.executable, __file__, "--shard", str(s), "--shards", str(args.shards),
             "--device", dev, "--batch", str(args.batch)],
            stdout=log, stderr=subprocess.STDOUT, cwd=str(REPO)), log))
    fail = False
    for s, p, log in procs:
        rc = p.wait()
        log.close()
        print(f"shard {s}: rc={rc}")
        fail |= rc != 0
    if fail:
        raise SystemExit("a shard failed; evidence left in place, not merging")
    merge(args.shards)


if __name__ == "__main__":
    main()
