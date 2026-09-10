"""Recompute the cue labels of an existing sequence build under a rule version.

Why a sidecar instead of rebuilding the sequences: the repair in
``taxonomy.RULESET_V2_RATIONALE`` changes only which cue class a *record* maps
to. Not one feature column depends on it. Re-running the builder would spend
~40 GPU-minutes re-extracting CLIP over 284k crops to write byte-identical
features next to different labels, and would re-run tracking and chunking --
any drift in which would silently make the new runs incomparable to the
published ones.

So the features stay exactly where they are, in exactly the file the published
numbers were measured on, and only ``y``/``y_cand`` are overridden at load
time. A v1-vs-v2 comparison is then guaranteed to differ in the target and in
nothing else, which is a stronger guarantee than a rebuild could give.

Output NPZ (flat, so a key can contain '/'):

    keys      [N]      str, manifest-relative npz path -- the stable sequence id
    offsets   [N + 1]  int64, slice bounds into the frame arrays
    y         [F]      int64, cue class per frame, 6-class space
    y_cand    [F, 6]   uint8, multi-hot candidate set per frame
    ruleset   scalar   str
    ...plus provenance (label file, its sha256, counts)

Correctness is PROVED, not assumed, in two steps, both of which abort:

  1. The replay of the builder's grouping must reproduce every sequence's
     stored timestamps exactly (inherited from ``patch_pose_columns``).
  2. The v1 labels recomputed here must equal the labels ALREADY STORED in the
     npz, frame for frame, on every sequence -- regardless of which ruleset was
     asked for. This is the check that matters: it proves the replay lines up
     AND that this file's reading of the rules matches the builder's, before
     anything downstream trusts a v2 label.

    python -m attention.thesis_eval.build_cue_labels \\
        --sequence-root ../grounding_data/llmstu_sequences_full_det \\
        --manifest ../grounding_data/llmstu_seq_split_manifest.json \\
        --ruleset v2 --out ../grounding_data/cue_labels_v2.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from attention.taxonomy import (CUE_CLASSES, RULESETS, candidate_set,
                                map_record)
from attention.thesis_eval.patch_pose_columns import replay_chunks

#: Same fold data.py applies on load: builds predating 2026-07-29 carry a
#: 7-class taxonomy where idle_other=5 and uncertain=6, and both become 5.
#: Applied before comparing against stored labels so the two vintages compare.
LEGACY_REMAP_LUT = np.array([0, 1, 2, 3, 4, 5, 5], dtype=np.int64)


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def build(sequence_root: Path, labels: Path, frame_to_video: Path,
          ruleset: str, manifest: Path | None = None) -> dict:
    """Recompute labels for every sequence the replay emits.

    Returns the arrays and provenance that :func:`main` writes. Raises on any
    misalignment rather than writing a label set nobody can trust.
    """
    if ruleset not in RULESETS:
        raise SystemExit(f"unknown ruleset {ruleset!r}; known: {RULESETS}")

    keys: list[str] = []
    y_parts: list[np.ndarray] = []
    cand_parts: list[np.ndarray] = []
    n_frames = 0
    n_v1_checked = 0
    changed = Counter()
    missing_fields = 0

    for rec in replay_chunks(labels, frame_to_video):
        key = f"{rec['split']}/sample_{rec['sample_idx']:06d}.npz"
        fp = sequence_root / key
        if not fp.exists():
            raise SystemExit(
                f"replay produced {key}, which does not exist under "
                f"{sequence_root}. The builder's grouping has drifted -- abort "
                f"rather than write labels against the wrong rows.")
        z = np.load(fp)
        t = z["t"].astype(np.float64)
        # PROOF 1: the builder writes `t` straight from the chunk, so exact
        # equality is the strongest check available that this replay recovered
        # the same frames in the same order.
        if len(t) != len(rec["times"]) or not np.array_equal(t, rec["times"]):
            raise SystemExit(
                f"{key}: replay/stored timestamp mismatch "
                f"({len(rec['times'])} replayed vs {len(t)} stored). Abort.")

        metas = rec["metas"]
        if any(m.get("activity") is None for m in metas):
            missing_fields += 1

        want = np.array([map_record(m, ruleset) for m in metas], dtype=np.int64)
        cand = np.zeros((len(metas), len(CUE_CLASSES)), dtype=np.uint8)
        for i, m in enumerate(metas):
            cand[i, candidate_set(m, ruleset)] = 1

        # PROOF 2: recomputing v1 must reproduce what the builder actually
        # wrote. Run it whatever ruleset was asked for -- it validates the
        # replay and this file's reading of the rules, and it is the only
        # check that can catch a v2 label set that is wrong for a reason
        # unrelated to v2.
        v1 = np.array([map_record(m, "v1") for m in metas], dtype=np.int64)
        stored = LEGACY_REMAP_LUT[np.clip(z["y_frames"].astype(np.int64), 0, 6)]
        if not np.array_equal(v1, stored):
            n_bad = int((v1 != stored).sum())
            raise SystemExit(
                f"{key}: recomputed v1 labels differ from the stored labels on "
                f"{n_bad}/{len(v1)} frames. Either the replay is misaligned or "
                f"the cue rules have changed since this build. Abort -- a v2 "
                f"label set built on top of this would be wrong for a reason "
                f"that has nothing to do with v2.")
        n_v1_checked += len(v1)

        for a, b in zip(stored, want):
            if a != b:
                changed[f"{CUE_CLASSES[a]} -> {CUE_CLASSES[b]}"] += 1

        keys.append(key)
        y_parts.append(want)
        cand_parts.append(cand)
        n_frames += len(want)
        if len(keys) % 1000 == 0:
            print(f"  {len(keys)} sequences, {n_frames} frames", flush=True)

    if not keys:
        raise SystemExit("the replay emitted no sequences -- check --labels "
                         "and --frame-to-video.")

    if manifest is not None:
        want_keys = {r["file"] for r in json.load(open(manifest))["samples"]}
        missing = sorted(want_keys - set(keys))
        if missing:
            raise SystemExit(
                f"{len(missing)} sequences in {manifest} have no recomputed "
                f"label (e.g. {missing[:3]}). Training would read a stale "
                f"label for them. Abort.")

    y = np.concatenate(y_parts)
    cand = np.concatenate(cand_parts, axis=0)
    offsets = np.zeros(len(keys) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(p) for p in y_parts])

    n_changed = int(sum(changed.values()))
    print(f"\nrecomputed {len(keys)} sequences / {n_frames} frames under "
          f"ruleset {ruleset!r}")
    print(f"  v1 self-check passed on {n_v1_checked} frames")
    if missing_fields:
        print(f"  WARNING: {missing_fields} sequences had frames with no cue "
              f"fields in meta (older parse?) -- labels for those default to "
              f"the rule's own unknown handling")
    print(f"  frames whose label CHANGED vs stored: {n_changed} "
          f"({n_changed / max(n_frames, 1):.2%})")
    for k, v in changed.most_common(10):
        print(f"    {k:<44}{v:>9,}")
    hist = Counter(int(v) for v in y)
    print("  new class distribution:")
    for i, c in enumerate(CUE_CLASSES):
        print(f"    {c:<20}{hist.get(i, 0):>9,}  {hist.get(i, 0) / n_frames:6.2%}")

    return {
        "keys": np.array(keys, dtype=object),
        "offsets": offsets,
        "y": y,
        "y_cand": cand,
        "ruleset": ruleset,
        "n_sequences": len(keys),
        "n_frames": n_frames,
        "n_frames_changed": n_changed,
        "transitions": dict(changed),
        "class_counts": {c: hist.get(i, 0) for i, c in enumerate(CUE_CLASSES)},
        "labels_path": str(labels),
        "labels_sha256": file_sha256(labels) if labels.is_file() else "",
        "sequence_root": str(sequence_root),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence-root", type=Path, required=True)
    ap.add_argument("--ruleset", default="v2", choices=list(RULESETS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="split manifest; every sequence it lists must get a "
                         "recomputed label, else abort")
    ap.add_argument("--labels", type=Path,
                    default=Path("../grounding_data/llmstu_tools/outputs/labels_tracked.jsonl"))
    ap.add_argument("--frame-to-video", type=Path,
                    default=Path("../grounding_data/llmstu_tools/outputs/frame_to_video.json"))
    args = ap.parse_args()

    out = build(args.sequence_root, args.labels, args.frame_to_video,
                args.ruleset, args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        keys=out["keys"].astype(str), offsets=out["offsets"],
        y=out["y"], y_cand=out["y_cand"], ruleset=out["ruleset"],
        provenance=json.dumps({k: out[k] for k in (
            "n_sequences", "n_frames", "n_frames_changed", "transitions",
            "class_counts", "labels_path", "labels_sha256", "sequence_root")}))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
