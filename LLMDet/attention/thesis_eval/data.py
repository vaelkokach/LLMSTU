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
# A build's column meaning depends on which streams were extracted, so the
# layout is NAMED and stored in the npz rather than inferred from the width.
# Getting this wrong is silent: a config would slice real numbers from the
# wrong columns and train happily on nonsense.
#
# v570 tiles [0, 570) exactly and is every sequence built before the head
# stream existed.
LAYOUTS: Dict[str, Dict[str, Tuple[int, int]]] = {
    "v570": {
        "base": (0, 552),
        "headpose": (552, 556),
        "hp_angles": (552, 555),        # yaw, pitch, roll
        "hp_facefound": (555, 556),     # the detection flag alone
        "express": (556, 563),
        "dynamic": (563, 570),
    },
    # Head stream instead of express/dynamic: neither of those is deployable
    # (expression needs a per-crop FER model, dynamics is a whole-track
    # statistic), so a head build does not carry them and the head block takes
    # their place. 518 = 512 CLIP over the head crop + 6 head-box geometry.
    "v1074_head": {
        "base": (0, 552),
        "headpose": (552, 556),
        "hp_angles": (552, 555),
        "hp_facefound": (555, 556),
        "head": (556, 1074),
    },
}

#: Backwards-compatible aliases; v570 is what every existing caller means.
FEATURE_BLOCKS: Dict[str, Tuple[int, int]] = {
    k: v for k, v in LAYOUTS["v570"].items()
    if k in ("base", "headpose", "express", "dynamic")}
SUB_BLOCKS: Dict[str, Tuple[int, int]] = {
    k: v for k, v in LAYOUTS["v570"].items() if k.startswith("hp_")}
ALL_BLOCKS: Dict[str, Tuple[int, int]] = {**FEATURE_BLOCKS, **SUB_BLOCKS}

#: The controlled ladder: config -> (layout, blocks kept in column order).
#: ``563_expr`` and ``563_dyn`` isolate the two families that the historic
#: 556->570 comparison added *together* and therefore could not separate.
FEATURE_CONFIGS: Dict[str, List[str]] = {
    "552_base": ["base"],
    "556_hp": ["base", "headpose"],
    "563_expr": ["base", "headpose", "express"],
    "563_dyn": ["base", "headpose", "dynamic"],
    "570_full": ["base", "headpose", "express", "dynamic"],
    # head-pose decomposition (see LAYOUTS)
    "553_facefound": ["base", "hp_facefound"],
    "555_angles": ["base", "hp_angles"],
    # head stream (needs a v1074_head build)
    "1070_head": ["base", "head"],
    "1074_hp_head": ["base", "headpose", "head"],
}

#: Which layout each config must be sliced against.
CONFIG_LAYOUT: Dict[str, str] = {
    name: ("v1074_head" if "head" in blocks else "v570")
    for name, blocks in FEATURE_CONFIGS.items()
}

#: Total width of each layout, for validating an npz before slicing it.
LAYOUT_WIDTH: Dict[str, int] = {
    "v570": 570,
    "v1074_head": 1074,
}


IGNORE_INDEX = -100

# Sequences built before 2026-07-29 carry the 7-class taxonomy (idle_other=5,
# uncertain=6); both fold onto uncertain=5. Applied to every build so the two
# vintages are label-identical.
LEGACY_REMAP_LUT = np.array([0, 1, 2, 3, 4, 5, 5], dtype=np.int64)


def config_layout(name: str) -> str:
    """Which stored column layout this feature config must be sliced against."""
    if name not in FEATURE_CONFIGS:
        raise KeyError(f"unknown feature config {name!r}; "
                       f"known: {sorted(FEATURE_CONFIGS)}")
    return CONFIG_LAYOUT[name]


def layout_is_compatible(want: str, stored: str, blocks: List[str]) -> bool:
    """Can a config written for ``want`` be sliced out of a ``stored`` build?

    Only when every block it actually reads is defined at IDENTICAL columns in
    both layouts. v570 and v1074_head agree exactly on ``base`` and the
    head-pose blocks -- columns [0, 556) -- and diverge only after, so 556_hp
    or 553_facefound read the same numbers from either build. A config touching
    ``express``/``dynamic`` (v570 only) or ``head`` (v1074_head only) is
    refused, because there the same column index means different things.

    Checked block by block rather than by trusting a name: the whole point of
    naming layouts was that a column-meaning mismatch is silent.
    """
    a, b = LAYOUTS.get(want), LAYOUTS.get(stored)
    if a is None or b is None:
        return False
    return all(blk in b and a[blk] == b[blk] for blk in blocks)


def config_dim(name: str) -> int:
    blocks = LAYOUTS[config_layout(name)]
    return sum(blocks[b][1] - blocks[b][0] for b in FEATURE_CONFIGS[name])


def column_index(name: str) -> np.ndarray:
    """Column indices selected by a feature config, in ascending order."""
    blocks = LAYOUTS[config_layout(name)]
    idx: List[int] = []
    for block in FEATURE_CONFIGS[name]:
        lo, hi = blocks[block]
        idx.extend(range(lo, hi))
    return np.asarray(sorted(idx), dtype=np.int64)


@dataclass
class Sequence_:
    """One (video, seat) track.

    ``y_cand`` [T, K] is the multi-hot partial label: every class the
    annotation supports. ``y`` is the precedence winner and is always one of
    them, so a single-label loss can ignore ``y_cand`` entirely.
    """
    key: str            # manifest-relative npz path, the stable sequence id
    video_id: str
    seat_id: int
    split: str
    x: np.ndarray       # [T, D] float32, already sliced to the feature config
    y: np.ndarray       # [T]    int64 class ids (IGNORE_INDEX where abstained)
    t: np.ndarray       # [T]    float64 seconds within the source video
    y_cand: Optional[np.ndarray] = None   # [T, K] bool multi-hot candidates

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
    taxonomy: str = "cue6",
) -> List[Sequence_]:
    """Load one split, slicing features to ``feature_config``.

    Sequences are returned in deterministic manifest order so that every
    evaluator run over the same split produces byte-identical prediction
    archives.

    ``taxonomy`` relabels the 6 stored cue ids onto a coarser set. Frames whose
    class the taxonomy abstains on become ``IGNORE_INDEX``, so they leave the
    loss and the metrics together -- the model is not scored on frames it was
    never asked to predict. Features are untouched: only ``y`` changes, which
    is why a taxonomy change needs no sequence rebuild.
    """
    if feature_config not in FEATURE_CONFIGS:
        raise KeyError(f"unknown feature config {feature_config!r}; "
                       f"known: {sorted(FEATURE_CONFIGS)}")
    cols = column_index(feature_config)
    want_layout = config_layout(feature_config)
    want_blocks = FEATURE_CONFIGS[feature_config]
    from attention.taxonomy import CUE_CLASSES, taxonomy_classes, taxonomy_lut
    tax_lut = np.array(taxonomy_lut(taxonomy), dtype=np.int64)
    n_tax_classes = len(taxonomy_classes(taxonomy))
    rows = [r for r in load_manifest(manifest_path) if r["split"] == split]
    rows.sort(key=lambda r: r["file"])
    if limit is not None:
        rows = rows[:limit]
    out: List[Sequence_] = []
    for r in rows:
        d = np.load(sequence_root / r["file"])
        x = d["x"].astype(np.float32)
        # Validate against the layout the FILE declares. Sequences built
        # before layouts were named carry none and are v570 by definition.
        stored = (str(d["layout"]) if "layout" in getattr(d, "files", [])
                  else "v570")
        if stored != want_layout and not layout_is_compatible(
                want_layout, stored, want_blocks):
            raise RuntimeError(
                f"{r['file']} is a {stored!r} build; config "
                f"{feature_config!r} reads {want_blocks}, which {stored!r} "
                f"does not define at the same columns. Pick a config for this "
                f"build, or rebuild with the matching streams.")
        # A width mismatch would not crash later -- it would quietly read the
        # wrong columns and train on nonsense, so it is fatal.
        if x.shape[1] != LAYOUT_WIDTH[stored]:
            raise RuntimeError(
                f"{r['file']} declares layout {stored!r} but has "
                f"{x.shape[1]} columns, not {LAYOUT_WIDTH[stored]}.")
        y = LEGACY_REMAP_LUT[np.clip(d["y_frames"].astype(np.int64), 0, 6)]

        # Candidate sets, in 6-class space. Sequences built before y_cand
        # existed fall back to one-hot, which makes a partial-label objective
        # degrade exactly to the single-label one rather than break.
        if "y_cand" in getattr(d, "files", []):
            cand6 = d["y_cand"].astype(bool)
        else:
            cand6 = np.zeros((len(y), len(CUE_CLASSES)), dtype=bool)
            cand6[np.arange(len(y)), y] = True

        # Collapse the candidate columns the same way as the labels, so a
        # taxonomy that merges two classes merges their candidacy too.
        cand = np.zeros((len(y), n_tax_classes), dtype=bool)
        for src_id, new_id in enumerate(tax_lut):
            if new_id >= 0:
                cand[:, new_id] |= cand6[:, src_id]

        y = tax_lut[y]                      # identity for the default cue6
        # A frame whose every candidate was dropped by the taxonomy has nothing
        # left to predict; IGNORE it rather than inventing a target.
        y = np.where(cand.any(axis=1), y, IGNORE_INDEX)
        # Force C-contiguity once here. Column-sliced views are strided, and
        # copying them per batch inside collate cost 183 ms/batch — 80% of an
        # epoch, with the GPU at 3% utilisation.
        out.append(Sequence_(
            key=r["file"], video_id=r["video_id"], seat_id=int(r["seat_id"]),
            split=split, x=np.ascontiguousarray(x[:, cols]), y=y,
            y_cand=cand, t=d["t"].astype(np.float64)))
    return out


def class_histogram(seqs: Sequence[Sequence_], num_classes: int = 6) -> np.ndarray:
    """Frame counts per class, ignoring frames the taxonomy abstains on.

    np.bincount rejects negative values, and IGNORE_INDEX is -100, so the
    filter is required rather than defensive. It is also correct: class weights
    derived from this histogram should not count frames the model is never
    asked to predict.
    """
    hist = np.zeros(num_classes, dtype=np.int64)
    for s in seqs:
        y = s.y[s.y >= 0]
        if y.size:
            hist += np.bincount(y, minlength=num_classes)[:num_classes]
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


def padded_candidates(seqs: List[Sequence_], num_classes: int,
                      device=None, max_len: Optional[int] = None):
    """[N, T, K] float mask of candidate labels, aligned with to_padded_tensors.

    Separate from to_padded_tensors rather than a fourth return value: every
    existing caller unpacks exactly three, and silently changing that arity is
    the kind of edit that breaks an evaluator six modules away.

    Padding rows are all-zero. They are never read — the loss masks on
    ``y != IGNORE_INDEX`` first — but zero is the honest value for "no
    candidate here" and makes an accidental read produce an obvious NaN rather
    than a plausible wrong number.
    """
    import torch

    n = len(seqs)
    T = max_len or max(len(s.y) for s in seqs)
    C = torch.zeros((n, T, num_classes), dtype=torch.float32)
    for i, s in enumerate(seqs):
        t = min(len(s.y), T)
        if s.y_cand is None:
            valid = s.y[:t] >= 0
            idx = np.nonzero(valid)[0]
            C[i, idx, s.y[:t][valid]] = 1.0
        else:
            C[i, :t] = torch.from_numpy(s.y_cand[:t].astype(np.float32))
    return C.to(device) if device is not None else C
