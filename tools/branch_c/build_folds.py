"""Build the immutable Branch-C nested grouped cross-validation manifest.

Branch C has no external holdout. The corpus contains 128 videos; 127 carry the
Branch-A/B split and the single remaining video
(``video_0162_0_10_20251026030716_20251026032847``) contributes four frames, which
is not a test set. The Branch-C protocol therefore uses nested grouped CV, and
this script is the one place the folds are defined.

Grouping unit is ``video_id``. Camera grouping is vacuous (the corpus has exactly
one camera pose) and subject grouping is unrecoverable (students are not
identified across recordings); both facts are recorded in BRANCH_C_PROTOCOL.md
rather than silently ignored.

Fold construction uses only the *marginal* per-video cue prevalence taken from
``meta.json``'s ``label_majority`` field. No model, no prediction and no feature
is involved, so this is stratification, not selection. Assignment is a
deterministic greedy balance: videos are visited in descending sequence count and
each is placed in the outer fold whose running prevalence vector is furthest from
the corpus mean in the direction that video corrects. The result is reproducible
from the seed alone and is hashed into the manifest.

    python tools/branch_c/build_folds.py --out outputs/branch_c/splits

Re-running with the same inputs must reproduce the same manifest hash. The
manifest is write-once: the script refuses to overwrite an existing file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

CUE_CLASSES = [
    "screen_oriented",
    "looking_away",
    "head_down",
    "turned_to_peer",
    "phone_use",
    "uncertain",
]
N_CUES = len(CUE_CLASSES)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_META = REPO / "grounding_data/llmstu_sequences_full/meta.json"
DEFAULT_SPLITS = REPO / "grounding_data/llmstu_tools/outputs/splits.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def video_profiles(meta: dict) -> Dict[str, dict]:
    """Per-video sequence count and cue-prevalence vector over label_majority."""
    counts: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(N_CUES, dtype=np.int64))
    seqs: Dict[str, int] = defaultdict(int)
    frames: Dict[str, int] = defaultdict(int)
    seats: Dict[str, set] = defaultdict(set)
    for s in meta["samples"]:
        v = s["video_id"]
        counts[v][int(s["label_majority"])] += 1
        seqs[v] += 1
        frames[v] += int(s["length"])
        seats[v].add(str(s["seat_id"]))
    out = {}
    for v in sorted(seqs):
        c = counts[v].astype(np.float64)
        out[v] = {
            "sequences": seqs[v],
            "frames": frames[v],
            "seats": len(seats[v]),
            "cue_counts": counts[v].tolist(),
            "prevalence": (c / max(c.sum(), 1.0)).tolist(),
        }
    return out


def assign_outer_folds(profiles: Dict[str, dict], k: int, seed: int) -> Dict[str, int]:
    """Greedy prevalence-balanced assignment of whole videos to k outer folds.

    Video count per fold is a *hard* constraint (equal to within one video), not a
    penalty term. An earlier version made it a soft cost and the folds collapsed to
    40/61/26/0/0 videos: this corpus is ~85 percent ``screen_oriented`` in every
    video, so the prevalence cost is nearly flat and a soft size term never bites.
    Support balance is what makes a per-fold bootstrap interval interpretable, so
    it is enforced rather than encouraged.

    Within the folds that still have capacity, the choice minimises L1 distance
    from the corpus prevalence plus a sequence-count imbalance term, which
    balances *support* as well as group count. Largest videos first, so the
    biggest sources of imbalance are placed while there is still freedom to
    correct them. Ties break on a seeded permutation, never on dict order, so the
    result does not depend on filesystem iteration.
    """
    rng = np.random.default_rng(seed)
    vids = list(profiles)
    order = sorted(vids, key=lambda v: (-profiles[v]["sequences"], rng.random(), v))

    target = np.zeros(N_CUES)
    total_seq = 0
    for v in vids:
        target += np.asarray(profiles[v]["cue_counts"], dtype=np.float64)
        total_seq += profiles[v]["sequences"]
    target = target / max(target.sum(), 1.0)

    n = len(vids)
    capacity = [n // k + (1 if f < n % k else 0) for f in range(k)]
    fold_counts = [np.zeros(N_CUES) for _ in range(k)]
    fold_seqs = [0] * k
    fold_n = [0] * k
    assign: Dict[str, int] = {}
    seq_target = total_seq / k

    for v in order:
        c = np.asarray(profiles[v]["cue_counts"], dtype=np.float64)
        best, best_cost = None, None
        for f in range(k):
            if fold_n[f] >= capacity[f]:
                continue
            trial = fold_counts[f] + c
            prev = trial / max(trial.sum(), 1.0)
            cost = np.abs(prev - target).sum()
            cost += abs(fold_seqs[f] + profiles[v]["sequences"] - seq_target) / max(seq_target, 1.0)
            if best_cost is None or cost < best_cost:
                best, best_cost = f, cost
        assert best is not None, "capacity exhausted; this cannot happen if sum(capacity) == n"
        assign[v] = best
        fold_counts[best] += c
        fold_seqs[best] += profiles[v]["sequences"]
        fold_n[best] += 1

    return _refine_by_swaps(assign, profiles, k, target, seq_target)


def _fold_cost(counts: np.ndarray, seqs: float, target: np.ndarray, seq_target: float) -> float:
    """Cost of one fold: rare-class support imbalance dominates, then total support.

    Weighted by 1/target so a class at 1 percent prevalence carries the same
    influence as one at 85 percent. Without that weighting the objective is
    effectively blind to ``turned_to_peer``, which is exactly the class whose
    per-fold support decides whether a per-class F1 is defined at all.
    """
    prev = counts / max(counts.sum(), 1.0)
    w = 1.0 / np.maximum(target, 1e-3)
    return float(np.abs(prev - target).dot(w)) + 2.0 * abs(seqs - seq_target) / max(seq_target, 1.0)


def _refine_by_swaps(
    assign: Dict[str, int],
    profiles: Dict[str, dict],
    k: int,
    target: np.ndarray,
    seq_target: float,
    max_sweeps: int = 200,
) -> Dict[str, int]:
    """Deterministic pairwise-swap local search; preserves the video-count balance.

    The greedy pass fills folds in order, so the last videos placed have no
    freedom left and the tail of the assignment is effectively arbitrary. Swapping
    pairs between folds keeps every fold's video count fixed while letting support
    and rare-class prevalence equalise. Videos are visited in sorted order and the
    first strictly-improving swap is taken, so the result is a deterministic
    function of the greedy input.
    """
    counts = [np.zeros(N_CUES) for _ in range(k)]
    seqs = [0.0] * k
    for v, f in assign.items():
        counts[f] += np.asarray(profiles[v]["cue_counts"], dtype=np.float64)
        seqs[f] += profiles[v]["sequences"]

    vids = sorted(assign)
    for _ in range(max_sweeps):
        improved = False
        for i, a in enumerate(vids):
            fa = assign[a]
            ca = np.asarray(profiles[a]["cue_counts"], dtype=np.float64)
            sa = profiles[a]["sequences"]
            for b in vids[i + 1:]:
                fb = assign[b]
                if fa == fb:
                    continue
                cb = np.asarray(profiles[b]["cue_counts"], dtype=np.float64)
                sb = profiles[b]["sequences"]
                before = (
                    _fold_cost(counts[fa], seqs[fa], target, seq_target)
                    + _fold_cost(counts[fb], seqs[fb], target, seq_target)
                )
                na, nb = counts[fa] - ca + cb, counts[fb] - cb + ca
                qa, qb = seqs[fa] - sa + sb, seqs[fb] - sb + sa
                after = (
                    _fold_cost(na, qa, target, seq_target)
                    + _fold_cost(nb, qb, target, seq_target)
                )
                if after < before - 1e-12:
                    assign[a], assign[b] = fb, fa
                    counts[fa], counts[fb] = na, nb
                    seqs[fa], seqs[fb] = qa, qb
                    improved = True
                    break
        if not improved:
            break
    return assign


def assign_inner(train_videos: List[str], profiles: Dict[str, dict], frac: float, seed: int) -> List[str]:
    """Pick the inner-validation videos of one outer fold, prevalence-balanced.

    A single inner holdout rather than full inner k-fold: the screening stage
    evaluates on the order of a hundred configurations, and a k-fold inner loop
    multiplies that by k for a selection decision that a single grouped holdout
    of ~20 videos already resolves. The cost is recorded in the protocol.
    """
    n_val = max(2, int(round(frac * len(train_videos))))
    sub = {v: profiles[v] for v in train_videos}
    folds = assign_outer_folds(sub, k=max(2, int(round(1.0 / frac))), seed=seed)
    # take the fold whose size is closest to the requested count
    by_fold: Dict[int, List[str]] = defaultdict(list)
    for v, f in folds.items():
        by_fold[f].append(v)
    pick = min(by_fold.values(), key=lambda g: abs(len(g) - n_val))
    return sorted(pick)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=Path, default=DEFAULT_META)
    ap.add_argument("--legacy-splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--out", type=Path, default=REPO / "outputs/branch_c/splits")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--inner-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument("--force", action="store_true", help="allow overwrite (protocol violation outside of a dry run)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "branch_c_folds.json"
    if manifest_path.exists() and not args.force:
        raise SystemExit(
            f"{manifest_path} exists. The fold manifest is write-once by protocol; "
            "delete it deliberately or pass --force if you are certain."
        )

    meta = json.loads(args.meta.read_text())
    legacy = json.loads(args.legacy_splits.read_text())
    profiles = video_profiles(meta)

    outer = assign_outer_folds(profiles, args.folds, args.seed)
    folds = []
    for f in range(args.folds):
        test_v = sorted(v for v, g in outer.items() if g == f)
        train_pool = sorted(v for v, g in outer.items() if g != f)
        inner_val = assign_inner(train_pool, profiles, args.inner_frac, args.seed + 1000 + f)
        inner_train = sorted(set(train_pool) - set(inner_val))
        assert not (set(test_v) & set(train_pool)), "outer leakage"
        assert not (set(inner_val) & set(inner_train)), "inner leakage"
        folds.append(
            {
                "fold": f,
                "outer_test_videos": test_v,
                "inner_train_videos": inner_train,
                "inner_val_videos": inner_val,
                "counts": {
                    "outer_test": {
                        "videos": len(test_v),
                        "sequences": sum(profiles[v]["sequences"] for v in test_v),
                        "frames": sum(profiles[v]["frames"] for v in test_v),
                    },
                    "inner_train": {
                        "videos": len(inner_train),
                        "sequences": sum(profiles[v]["sequences"] for v in inner_train),
                        "frames": sum(profiles[v]["frames"] for v in inner_train),
                    },
                    "inner_val": {
                        "videos": len(inner_val),
                        "sequences": sum(profiles[v]["sequences"] for v in inner_val),
                        "frames": sum(profiles[v]["frames"] for v in inner_val),
                    },
                },
                "outer_test_class_sequences": np.sum(
                    np.array([profiles[v]["cue_counts"] for v in test_v], dtype=np.int64).reshape(-1, N_CUES),
                    axis=0,
                ).tolist(),
                "outer_test_prevalence": (
                    np.sum(
                        np.array([profiles[v]["cue_counts"] for v in test_v], dtype=np.float64).reshape(-1, N_CUES),
                        axis=0,
                    )
                    / max(1, sum(profiles[v]["sequences"] for v in test_v))
                ).tolist(),
            }
        )

    body = {
        "protocol": "branch_c_nested_grouped_cv",
        "protocol_version": "1.0.0",
        "created": "2026-08-09",
        "grouping_unit": "video_id",
        "grouping_note": (
            "Camera grouping is vacuous: the corpus has one fixed camera pose. "
            "Subject grouping is unrecoverable: students are not identified across "
            "recordings. video_id is the only defensible grouping unit."
        ),
        "cue_classes": CUE_CLASSES,
        "n_outer_folds": args.folds,
        "inner_scheme": "single prevalence-balanced grouped holdout per outer fold",
        "inner_frac": args.inner_frac,
        "seed": args.seed,
        "n_videos": len(profiles),
        "n_sequences": sum(p["sequences"] for p in profiles.values()),
        "n_frames": sum(p["frames"] for p in profiles.values()),
        "sources": {
            "meta": {
                "path": str(args.meta.relative_to(REPO)),
                "sha256": sha256_file(args.meta),
            },
            "legacy_splits": {
                "path": str(args.legacy_splits.relative_to(REPO)),
                "sha256": sha256_file(args.legacy_splits),
                "counts": {k: len(v) for k, v in legacy.items()},
            },
        },
        "video_profiles": profiles,
        "folds": folds,
    }
    payload = json.dumps(body, indent=2, sort_keys=True)
    body["manifest_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    manifest_path.write_text(json.dumps(body, indent=2, sort_keys=True))

    print(f"wrote {manifest_path}")
    print(f"manifest_sha256 {body['manifest_sha256']}")
    print(f"videos {body['n_videos']}  sequences {body['n_sequences']}  frames {body['n_frames']}")
    for fd in folds:
        c = fd["counts"]
        sup = fd["outer_test_class_sequences"]
        print(
            f"  fold {fd['fold']}: test {c['outer_test']['videos']}v/"
            f"{c['outer_test']['sequences']}s  inner_train {c['inner_train']['videos']}v  "
            f"inner_val {c['inner_val']['videos']}v  "
            f"class support {sup}  min={min(sup)}"
        )
    worst = min(min(fd["outer_test_class_sequences"]) for fd in folds)
    print(f"minimum per-fold per-class sequence support: {worst}")


if __name__ == "__main__":
    main()
