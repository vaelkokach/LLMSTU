"""Train the Branch-C fusion arms (protocol arms 3, 8, 9, 10).

    python -m attention.branch_c.train_fusion --arm arm10_full --fold 0 --seed 42 \
        --output-dir ../outputs/branch_c/runs/arm10_full/arm10_full_f0_s42

Arms, and what each one isolates:

  arm3_appearance   fusion="none"     appearance expert alone; the floor
  arm8_uniform      fusion="uniform"  experts fused with equal weight
  arm9_learned      fusion="learned"  reliability-weighted, cue+smooth losses only
  arm10_full        fusion="learned"  + view / quality-order / counterfactual / calibration

arm8 vs arm9 is protocol arm 8 (does *learning* the weights beat averaging).
arm9 vs arm10 is protocol arm 9 (are the observability losses worth anything).

Selection is on inner-validation macro-F1, the same rule and the same evaluator as
arms 1-5, so the numbers land in one table. The outer folds are never loaded: the
fold manifest marks them "test" and this script only ever asks for "train"/"val".

Auxiliary views used by the full arm, all synthesised on the fly:

*Rotated view* — a random global rotation Q is applied to the head block's
rotation matrices. This is the physically meaningful perturbation for
L_view: the head really is the same head, seen from a rotated camera frame.
*Corrupted view* — the head block is zeroed and its availability bit cleared,
simulating an estimator failure. Used for both L_quality_order (the corrupted
view's predicted reliability must sit below the clean one's) and
L_counterfactual (the corrupted path is pulled towards the clean posterior, with
stop-gradient on the clean side).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from attention.branch_c import losses as LO
from attention.branch_c.model import LAYOUT, TOTAL_DIM, MultiExpertCueModel
from attention.thesis_eval.data import sqrt_inverse_frequency_weights

IGNORE = -100

ARMS = {
    "arm3_appearance": dict(fusion="none", full_losses=False),
    "arm8_uniform": dict(fusion="uniform", full_losses=False),
    "arm9_learned": dict(fusion="learned", full_losses=False),
    "arm10_full": dict(fusion="learned", full_losses=True),
}

DEFAULT_COEFFS = {
    "cue": 1.0, "smooth": 0.15,
    "view": 0.5, "quality_order": 0.1, "counterfactual": 0.5, "calibration": 0.1,
}


# --------------------------------------------------------------------------- data

def load_split(manifest: Path, root: Path, split: str) -> List[dict]:
    man = json.loads(manifest.read_text())
    out = []
    for r in man["samples"]:
        if r["split"] != split:
            continue
        z = np.load(root / r["file"])
        x = z["x"].astype(np.float32)
        if x.shape[1] != TOTAL_DIM:
            raise SystemExit(
                f"{r['file']}: width {x.shape[1]}, expected {TOTAL_DIM}. Refusing to "
                "pad or truncate — rebuild the sequences instead.")
        out.append({"x": x, "y": z["y_frames"].astype(np.int64),
                    "video_id": r["video_id"], "file": r["file"]})
    if not out:
        raise SystemExit(f"no sequences for split={split} in {manifest}")
    return out


def batches(seqs: List[dict], bs: int, shuffle: bool, rng: Optional[np.random.Generator]):
    idx = np.arange(len(seqs))
    if shuffle:
        rng.shuffle(idx)
    for s in range(0, len(idx), bs):
        chunk = [seqs[i] for i in idx[s:s + bs]]
        T = max(len(c["y"]) for c in chunk)
        x = np.zeros((len(chunk), T, TOTAL_DIM), dtype=np.float32)
        y = np.full((len(chunk), T), IGNORE, dtype=np.int64)
        pad = np.ones((len(chunk), T), dtype=bool)
        for i, c in enumerate(chunk):
            n = len(c["y"])
            x[i, :n], y[i, :n], pad[i, :n] = c["x"], c["y"], False
        yield (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(pad))


# ------------------------------------------------------------------- perturbations

def random_global_rotation(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Left-multiply the head rotation block by one random Q per sequence."""
    a, b = LAYOUT["head"]
    B = x.shape[0]
    # Drawn on CPU and moved, so the augmentation is identical for a given seed
    # regardless of which GPU the run lands on — the sweep dispatches to whichever
    # device frees up first, and a device-dependent augmentation would make two
    # "same seed" runs incomparable.
    A = torch.randn(B, 3, 3, generator=gen, dtype=x.dtype).to(x.device)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R, dim1=-2, dim2=-1)).unsqueeze(-2)
    Q[torch.det(Q) < 0, :, 0] *= -1
    out = x.clone()
    Rh = x[..., a:b].reshape(B, -1, 3, 3)
    out[..., a:b] = (Q.unsqueeze(1) @ Rh).reshape(B, -1, 9)
    return out


def corrupt_head(x: torch.Tensor) -> torch.Tensor:
    """Simulate total head-estimator failure: zero the block, clearing availability."""
    a, b = LAYOUT["head"]
    out = x.clone()
    out[..., a:b] = 0.0
    return out


# ----------------------------------------------------------------------- evaluation

@torch.no_grad()
def evaluate(model, seqs, device, bs=8, **fwd) -> dict:
    model.eval()
    P, Y = [], []
    for x, y, pad in batches(seqs, bs, False, None):
        out, _ = model(x.to(device), pad_mask=pad.to(device), **fwd)
        p = out["logits"].argmax(-1).cpu().numpy()
        m = ~pad.numpy()
        P.append(p[m])
        Y.append(y.numpy()[m])
    P, Y = np.concatenate(P), np.concatenate(Y)
    keep = Y != IGNORE
    P, Y = P[keep], Y[keep]
    return {"macro_f1": float(f1_score(Y, P, average="macro", labels=list(range(6)), zero_division=0)),
            "accuracy": float((P == Y).mean()),
            "per_class_f1": f1_score(Y, P, average=None, labels=list(range(6)), zero_division=0).tolist(),
            "n_frames": int(len(Y))}


# ------------------------------------------------------------------------- training

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--sequence-root", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    cfg = ARMS[args.arm]
    dev = args.device
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tr = load_split(args.manifest, args.sequence_root, "train")
    va = load_split(args.manifest, args.sequence_root, "val")
    print(f"{args.arm} f{args.fold} s{args.seed}: train {len(tr)} val {len(va)} seqs")

    counts = np.zeros(6)
    for s in tr:
        for c in range(6):
            counts[c] += int((s["y"] == c).sum())
    # Same class weighting and optimiser as attention/thesis_eval/train.py, so the
    # fusion arms land in one table with arms 1-5 rather than differing in three
    # places at once.
    weights = torch.from_numpy(sqrt_inverse_frequency_weights(counts)).to(dev)

    model = MultiExpertCueModel(fusion=cfg["fusion"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, best_ep, hist = -1.0, -1, []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y, pad in batches(tr, args.batch_size, True, rng):
            x, y, pad = x.to(dev), y.to(dev), pad.to(dev)
            out, diag = model(x, pad_mask=pad)
            # Deep supervision on EVERY stage, including the fused prediction that
            # precedes refinement. See the note in model.forward.
            cue = sum(F.cross_entropy(s.reshape(-1, 6), y.reshape(-1),
                                      weight=weights, ignore_index=IGNORE)
                      for s in out["aux_logits"])
            smooth = sum(LO.smooth_loss(s, ~pad) for s in out["aux_logits"])
            terms = {"cue": cue, "smooth": smooth}
            if cfg["full_losses"]:
                xr = random_global_rotation(x, gen)
                out_r, _ = model(xr, pad_mask=pad)
                terms["view"] = LO.view_consistency_loss(out["logits"], out_r["logits"])

                xc = corrupt_head(x)
                out_c, diag_c = model(xc, pad_mask=pad)
                terms["quality_order"] = LO.quality_order_loss(
                    diag["r"][0], diag_c["r"][0], margin=0.5)
                conf = out["logits"].softmax(-1).max(-1).values.detach()
                retained = diag["mask"][1].float()      # motion still present
                terms["counterfactual"] = LO.counterfactual_loss(
                    out["logits"], out_c["logits"], conf, retained, valid_mask=~pad)
                terms["calibration"] = LO.calibration_loss(out["logits"], y, mode="brier")

            loss = LO.total_loss(terms, DEFAULT_COEFFS)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss)
        sched.step()

        m = evaluate(model, va, dev, args.batch_size)
        hist.append({"epoch": ep, "train_loss": tot, "val_macro_f1": m["macro_f1"],
                     "val_accuracy": m["accuracy"]})
        if m["macro_f1"] > best:
            best, best_ep = m["macro_f1"], ep
            (args.output_dir / "checkpoints").mkdir(exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "arm": args.arm,
                        "fusion": cfg["fusion"], "epoch": ep, "input_dim": TOTAL_DIM,
                        "layout": {k: list(v) for k, v in LAYOUT.items()},
                        "val_macro_f1": best},
                       args.output_dir / "checkpoints/best.pth")
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep:3d} loss {tot:9.2f} macroF1 {m['macro_f1']:.4f}", flush=True)

    wall = time.time() - t0
    rec = {
        "experiment_id": args.output_dir.name, "arm": args.arm, "fold": args.fold,
        "seed": args.seed, "fusion": cfg["fusion"], "full_losses": cfg["full_losses"],
        "coefficients": DEFAULT_COEFFS, "epochs": args.epochs,
        "selected_epoch": best_ep, "selected_val_macro_f1": best,
        "input_dim": TOTAL_DIM, "n_parameters": sum(p.numel() for p in model.parameters()),
        "wall_clock_s": wall, "history": hist,
        "manifest": str(args.manifest), "sequence_root": str(args.sequence_root),
        "note": "inner validation only; outer fold never loaded",
    }
    (args.output_dir / "run_record.json").write_text(json.dumps(rec, indent=2))
    print(f"done in {wall/60:.1f} min; selected epoch {best_ep} macroF1 {best:.4f}")


if __name__ == "__main__":
    main()
