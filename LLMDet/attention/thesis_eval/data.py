"""Sequence loading for the controlled feature ablation ladder.

One physical dataset, five logical feature configurations
------------------------------------------------------------------
``grounding_data/llmstu_sequences_full`` stores 570-dim features whose column
layout is fixed by ``sequence_builder.build_sequences_llmstu``:

    [  0:552 ]  base      CLIP(512) + bbox geometry(8) + colour(24) + posture(8)
    [552:556 ]  headpose  yaw, pitch, roll, face_found        (MediaPipe cache)
    [556:563 ]  express   7 basic-expression probabilities    (ViT FER)
    [563:570 ]  dynamic   fidget/lean motion stats + personalised gaze deviation

Verified against the two older builds (2026-08-01 audit): the first 552 columns
of ``llmstu_sequences_full`` are **bit-identical** to ``llmstu_sequences``, and
columns 552:556 are bit-identical to ``llmstu_sequences_hp``, on every sampled
sequence; labels and timestamps match exactly and all three carry the same
6,531 sequences over the same 127 videos.

Consequence: every rung of the ladder is a *column slice* of one array. No
feature is re-extracted, so the ablation cannot be confounded by extraction
drift, cache staleness or a rebuilt split — the failure mode that already cost
this project one discarded 556-dim build (FINDINGS 6.0b).

Split policy
------------
Sequences are re-partitioned onto the detector's leak-free video-wise 73/27/27
split (``llmstu_tools/outputs/splits.json``) via
``grounding_data/llmstu_seq_split_manifest.json``. The builder's own
``split_videos()`` drew an *independent* 80/20 shuffle, which is why 23 of the
27 detector test videos sat in the cue model's training set and no Branch-B
test number could legitimately exist (FINDINGS 3.4c).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Column layout of the 570-dim build. Slices are half-open, in column order.
# These four tile [0, 570) exactly.
FEATURE_BLOCKS: Dict[str, Tuple[int, int]] = {
    "base": (0, 552),
    "headpose": (552, 556),
    "express": (556, 563),
    "dynamic": (563, 570),
}

#: Sub-blocks that split ``headpose`` into its two very different signals.
#: FINDINGS 6.0b argued that ``face_found`` — whether MediaPipe found a face at
#: all — is the strongest single cue signal in the project (92% detection on
#: `screen_oriented` vs 8% on `head_down`), and that the metric angles were the
#: weaker part. That was never tested in isolation, and it decides whether a
#: stronger head-pose estimator (DirectMHP, 6DRepNet) is worth integrating: if
#: the gain is all ``face_found``, better angles cannot help much.
SUB_BLOCKS: Dict[str, Tuple[int, int]] = {
    "hp_angles": (552, 555),      # yaw, pitch, roll
    "hp_facefound": (555, 556),   # the detection flag alone
}

ALL_BLOCKS: Dict[str, Tuple[int, int]] = {**FEATURE_BLOCKS, **SUB_BLOCKS}

#: The controlled ladder. Each entry lists the blocks kept, in column order.
#: ``563_expr`` and ``563_dyn`` isolate the two families that the historic
#: 556->570 comparison added *together* and therefore could not separate.
FEATURE_CONFIGS: Dict[str, List[str]] = {
    "552_base": ["base"],
    "556_hp": ["base", "headpose"],
    "563_expr": ["base", "headpose", "express"],
    "563_dyn": ["base", "headpose", "dynamic"],
    "570_full": ["base", "headpose", "express", "dynamic"],
    # head-pose decomposition (see SUB_BLOCKS)
    "553_facefound": ["base", "hp_facefound"],
    "555_angles": ["base", "hp_angles"],
}

IGNORE_INDEX = -100

# Sequences built before 2026-07-29 carry the 7-class taxonomy (idle_other=5,
# uncertain=6); both fold onto uncertain=5. Applied to every build so the two
# vintages are label-identical.
LEGACY_REMAP_LUT = np.array([0, 1, 2, 3, 4, 5, 5], dtype=np.int64)


def config_dim(name: str) -> int:
    return sum(ALL_BLOCKS[b][1] - ALL_BLOCKS[b][0] for b in FEATURE_CONFIGS[name])


def column_index(name: str) -> np.ndarray:
    """Column indices selected by a feature config, in ascending order."""
    idx: List[int] = []
    for block in FEATURE_CONFIGS[name]:
        lo, hi = ALL_BLOCKS[block]
        idx.extend(range(lo, hi))
    return np.asarray(sorted(idx), dtype=np.int64)


@dataclass
class Sequence_:
    """One (video, seat) track."""
    key: str            # manifest-relative npz path, the stable sequence id
    video_id: str
    seat_id: int
    split: str
    x: np.ndarray       # [T, D] float32, already sliced to the feature config
    y: np.ndarray       # [T]    int64, 6-class ids
    t: np.ndarray       # [T]    float64 seconds within the source video

    @property
    def track_id(self) -> str:
        return f"{self.video_id}#seat{self.seat_id}"


def load_manifest(path: Path) -> List[dict]:
    return json.load(open(path))["samples"]


def load_split(
    manifest_path: Path,
    sequence_root: Path,
    split: str,
    feature_config: str = "570_full",
    limit: Optional[int] = None,
) -> List[Sequence_]:
    """Load one split, slicing features to ``feature_config``.

    Sequences are returned in deterministic manifest order so that every
    evaluator run over the same split produces byte-identical prediction
    archives.
    """
    if feature_config not in FEATURE_CONFIGS:
        raise KeyError(f"unknown feature config {feature_config!r}; "
                       f"known: {sorted(FEATURE_CONFIGS)}")
    cols = column_index(feature_config)
    rows = [r for r in load_manifest(manifest_path) if r["split"] == split]
    rows.sort(key=lambda r: r["file"])
    if limit is not None:
        rows = rows[:limit]
    out: List[Sequence_] = []
    for r in rows:
        d = np.load(sequence_root / r["file"])
        x = d["x"].astype(np.float32)
        if x.shape[1] != 570:
            raise RuntimeError(
                f"{r['file']} has {x.shape[1]} feature columns; the ablation "
                "ladder requires the 570-dim build (llmstu_sequences_full)")
        y = LEGACY_REMAP_LUT[np.clip(d["y_frames"].astype(np.int64), 0, 6)]
        # Force C-contiguity once here. Column-sliced views are strided, and
        # copying them per batch inside collate cost 183 ms/batch — 80% of an
        # epoch, with the GPU at 3% utilisation.
        out.append(Sequence_(
            key=r["file"], video_id=r["video_id"], seat_id=int(r["seat_id"]),
            split=split, x=np.ascontiguousarray(x[:, cols]), y=y,
            t=d["t"].astype(np.float64)))
    return out


def class_histogram(seqs: Sequence[Sequence_], num_classes: int = 6) -> np.ndarray:
    hist = np.zeros(num_classes, dtype=np.int64)
    for s in seqs:
        hist += np.bincount(s.y, minlength=num_classes)[:num_classes]
    return hist


def sqrt_inverse_frequency_weights(hist: np.ndarray) -> np.ndarray:
    """Matches the historic trainer exactly so retrained models stay comparable
    with the archived 552/556/570 checkpoints."""
    inv = 1.0 / np.sqrt(np.maximum(hist.astype(np.float32), 1.0))
    return (inv / inv.sum() * len(hist)).astype(np.float32)


def to_padded_tensors(seqs: List[Sequence_], device=None, max_len: Optional[int] = None):
    """Materialise a whole split as one padded tensor triple.

    Returns ``(X[N, T, D], Y[N, T], pad_mask[N, T])`` with ``Y`` set to
    ``IGNORE_INDEX`` and ``pad_mask`` True at padding.

    Padding once and keeping the result resident (optionally on the GPU) turns
    every training step into a slice of an existing tensor. Re-padding per batch
    inside a collate function cost ~180 ms/batch here — 80% of the epoch — with
    the GPU sitting at 3% utilisation. The whole 570-dim training split is
    1.3 GB padded, which is trivial next to 40 GB of A100 memory.
    """
    import torch
    n = len(seqs)
    T = max_len or max(s.x.shape[0] for s in seqs)
    d = seqs[0].x.shape[1]
    X = torch.zeros((n, T, d), dtype=torch.float32)
    Y = torch.full((n, T), IGNORE_INDEX, dtype=torch.long)
    mask = torch.ones((n, T), dtype=torch.bool)
    for i, s in enumerate(seqs):
        t = s.x.shape[0]
        X[i, :t] = torch.from_numpy(s.x)
        Y[i, :t] = torch.from_numpy(s.y)
        mask[i, :t] = False
    if device is not None:
        X, Y, mask = X.to(device), Y.to(device), mask.to(device)
    return X, Y, mask


def collate(batch: List[Tuple[np.ndarray, np.ndarray]]):
    """Zero-pad features, IGNORE-pad labels, return a True-at-padding mask."""
    import torch
    xs, ys = zip(*batch)
    max_t = max(x.shape[0] for x in xs)
    d = xs[0].shape[1]
    x_out = torch.zeros((len(xs), max_t, d), dtype=torch.float32)
    y_out = torch.full((len(xs), max_t), IGNORE_INDEX, dtype=torch.long)
    mask = torch.ones((len(xs), max_t), dtype=torch.bool)
    for i, (x, y) in enumerate(zip(xs, ys)):
        x_out[i, : x.shape[0]] = torch.from_numpy(np.ascontiguousarray(x))
        y_out[i, : y.shape[0]] = torch.from_numpy(np.ascontiguousarray(y))
        mask[i, : x.shape[0]] = False
    return x_out, y_out, mask
