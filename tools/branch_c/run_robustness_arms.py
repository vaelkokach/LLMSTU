"""Registered inference-time arms: 11 (modality dropout), 12 (corruption), 17 (permutation).

None of these retrain. They take the trained fusion checkpoints and re-score the
inner-validation split under controlled perturbations, which is what makes them a
fair test of *robustness* rather than of a differently-fitted model.

    python tools/branch_c/run_robustness_arms.py --device cuda:1

Conditions
----------
clean              unmodified
drop_head          head expert forced unavailable at inference          (arm 11)
drop_motion        motion expert forced unavailable                     (arm 11)
drop_all           both experts unavailable; must degrade to appearance (arm 11)
corrupt_head       head rotations replaced by noise, availability kept  (arm 12)
degrade_quality    head/box size signals scaled down, simulating tiny crops (arm 12)
track_gaps         30% of frames lose motion evidence in contiguous runs (arm 12)
box_jitter         motion features perturbed by Gaussian noise          (arm 12)
permute_alpha      reliability weights shuffled across modalities       (arm 17)

The protocol's fusion gate (§7.1 item 3) asks whether **learned** reliability
fusion beats **uniform** fusion under at least two predefined missing/corrupted
conditions without materially degrading clean performance. That comparison is the
point of running arm 8 (uniform) and arm 9/10 (learned) through the identical
condition set.

Arm 17 is the falsification test: if `permute_alpha` does not degrade performance,
the gate is decorative and no fusion contribution may be claimed whatever arm 8
shows. It is reported whatever it says.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "LLMDet"))

from attention.branch_c.model import LAYOUT, TOTAL_DIM, MultiExpertCueModel  # noqa: E402
from attention.branch_c.train_fusion import batches, evaluate, load_split  # noqa: E402

RUNS = REPO / "outputs/branch_c/runs"
MANIFESTS = REPO / "outputs/branch_c/splits/fold_manifests"
SEQ = REPO / "grounding_data/llmstu_sequences_branch_c"
OUT = REPO / "outputs/branch_c/robustness.json"

FUSION_ARMS = ("arm8_uniform", "arm9_learned", "arm10_full")


def perturb(x: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    """Apply one controlled corruption to a sequence's feature array."""
    h0, h1 = LAYOUT["head"]
    q0, q1 = LAYOUT["quality"]
    m0, m1 = LAYOUT["motion"]
    y = x.copy()
    if condition == "corrupt_head":
        # Availability bit stays set: the estimator "succeeded" and was wrong,
        # which is the case a reliability head is supposed to catch and a plain
        # missingness mask cannot.
        y[:, h0:h1] = rng.normal(0, 0.5, size=(len(y), h1 - h0)).astype(np.float32)
    elif condition == "degrade_quality":
        y[:, q0 + 5] *= 0.3      # head_span
        y[:, q0 + 6] *= 0.3      # box width
        y[:, q0 + 7] *= 0.3      # box height
        y[:, q0 + 1] *= 0.5      # detector confidence
    elif condition == "track_gaps":
        n = len(y)
        for _ in range(max(1, n // 20)):
            s = rng.integers(0, max(1, n - 3))
            e = min(n, s + rng.integers(2, 8))
            y[s:e, m0:m1] = 0.0
    elif condition == "box_jitter":
        y[:, m0:m1] += rng.normal(0, 0.02, size=(len(y), m1 - m0)).astype(np.float32)
    return y


@torch.no_grad()
def score(model, seqs, device, condition, seed) -> dict:
    """Score one condition. Mask overrides are applied inside forward, not to data."""
    rng = np.random.default_rng(seed)
    fwd = {}
    if condition in ("drop_head", "drop_motion", "drop_all"):
        pass  # handled per-batch below, needs the batch shape
    if condition == "permute_alpha":
        fwd["permute_reliability"] = torch.Generator().manual_seed(seed)

    model.eval()
    P, Y = [], []
    data = seqs
    if condition in ("corrupt_head", "degrade_quality", "track_gaps", "box_jitter"):
        data = [{**s, "x": perturb(s["x"], condition, rng)} for s in seqs]

    for x, y, pad in batches(data, 8, False, None):
        x, pad = x.to(device), pad.to(device)
        kw = dict(fwd)
        if condition in ("drop_head", "drop_motion", "drop_all"):
            mask = model.availability(x)
            if condition in ("drop_head", "drop_all"):
                mask[0] = False
            if condition in ("drop_motion", "drop_all"):
                mask[1] = False
            kw["mask_override"] = mask
        out, _ = model(x, pad_mask=pad, **kw)
        p = out["logits"].argmax(-1).cpu().numpy()
        m = ~pad.cpu().numpy()
        P.append(p[m])
        Y.append(y.numpy()[m])

    from sklearn.metrics import f1_score
    P, Y = np.concatenate(P), np.concatenate(Y)
    keep = Y != -100
    P, Y = P[keep], Y[keep]
    return {"macro_f1": float(f1_score(Y, P, average="macro", labels=list(range(6)),
                                       zero_division=0)),
            "accuracy": float((P == Y).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="auto",
                    help="'auto' selects idle devices by measured free memory, never "
                         "more than the project cap and never onto another user's card")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seeds", default="42,43,44")
    args = ap.parse_args()

    if args.gpus == "auto":
        spec = importlib.util.spec_from_file_location(
            "prov", REPO / "tools/branch_c/provenance.py")
        prov = importlib.util.module_from_spec(spec)
        sys.modules["prov"] = prov
        spec.loader.exec_module(prov)
        gpus = prov.select_free_gpus()
    else:
        gpus = [int(g) for g in args.gpus.split(",")]
    print(f"scoring on GPUs {gpus}")

    conditions = ["clean", "drop_head", "drop_motion", "drop_all", "corrupt_head",
                  "degrade_quality", "track_gaps", "box_jitter", "permute_alpha"]
    results = defaultdict(lambda: defaultdict(list))

    # Fold data is loaded once and shared; the per-fold val split is ~1000 sequences
    # and reloading it per checkpoint dominated the single-GPU version's runtime.
    val_by_fold = {f: load_split(MANIFESTS / f"fold_{f}.json", SEQ, "val")
                   for f in [int(x) for x in args.folds.split(",")]}

    jobs = []
    for fold in val_by_fold:
        for arm in FUSION_ARMS:
            for seed in [int(s) for s in args.seeds.split(",")]:
                ck = RUNS / arm / f"{arm}_f{fold}_s{seed}" / "checkpoints/best.pth"
                if ck.exists():
                    jobs.append((arm, fold, seed, ck))
    print(f"{len(jobs)} checkpoints x {len(conditions)} conditions")

    lock = __import__("threading").Lock()

    def run_one(i_job):
        i, (arm, fold, seed, ck) = i_job
        device = f"cuda:{gpus[i % len(gpus)]}"
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        model = MultiExpertCueModel(fusion=blob["fusion"]).to(device)
        model.load_state_dict(blob["state_dict"], strict=True)
        out = {}
        for cond in conditions:
            if cond == "permute_alpha" and blob["fusion"] == "none":
                continue
            out[cond] = score(model, val_by_fold[fold], device, cond, seed)["macro_f1"]
        with lock:
            for cond, v in out.items():
                results[arm][cond].append(v)
            print(f"  {arm} f{fold} s{seed} done on {device}", flush=True)

    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        list(ex.map(run_one, enumerate(jobs)))

    print(f"\n{'arm':<16}" + "".join(f"{c:>17}" for c in conditions))
    print("-" * (16 + 17 * len(conditions)))
    summary = {}
    for arm in FUSION_ARMS:
        if arm not in results:
            continue
        row = f"{arm:<16}"
        summary[arm] = {}
        for c in conditions:
            v = results[arm].get(c, [])
            summary[arm][c] = {"mean": float(np.mean(v)) if v else None,
                               "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                               "n": len(v)}
            row += f"{np.mean(v):>17.4f}" if v else f"{'-':>17}"
        print(row)
    print("-" * (16 + 17 * len(conditions)))

    print("\nDegradation from clean (negative = worse):")
    for arm in summary:
        base = summary[arm]["clean"]["mean"]
        deltas = {c: (summary[arm][c]["mean"] - base)
                  for c in conditions if summary[arm].get(c, {}).get("mean") is not None}
        print(f"  {arm:<16}" + "  ".join(f"{c}:{d:+.4f}" for c, d in deltas.items() if c != "clean"))

    OUT.write_text(json.dumps({"summary": summary, "conditions": conditions,
                               "note": "inner validation only; outer folds not opened"},
                              indent=2, sort_keys=True))
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
