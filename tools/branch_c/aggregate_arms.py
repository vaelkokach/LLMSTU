"""Aggregate the Branch-C arm sweep into the inner-validation results table.

Reports, per arm, the mean and standard deviation of selected validation macro-F1
across the 5 outer folds x 3 seeds, and — because the arms are trained on
*identical* fold/seed pairs — the **paired** difference between arms over those 15
matched runs, which is a far tighter comparison than differencing two marginal
means.

These are inner-validation numbers. The outer folds have not been opened. Nothing
here may be reported as a generalisation estimate.

    python tools/branch_c/aggregate_arms.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "outputs/branch_c/runs"

LABELS = {
    "arm0_mstcn_quality": "arm 0  quality/missingness only (shortcut audit)",
    "arm0b_mstcn_quality5": "arm 0b pipeline-measured quality only (5 signals)",
    "arm1_mstcn_553_ff": "arm 1  base + face_found (deployment / presence-only control)",
    "arm2_mstcn_556_mp": "arm 2  + MediaPipe angles (legacy pose baseline)",
    "arm5_mstcn_556_fr": "arm 5  + 6DRepNet360 full-range rotation (candidate)",
}


def collect() -> dict:
    out = defaultdict(dict)
    for rec in RUNS.glob("*/*/run_record.json"):
        d = json.loads(rec.read_text())
        eid = d["experiment_id"]
        arm = rec.parent.parent.name
        fold = int(eid.split("_f")[-1].split("_s")[0])
        seed = int(eid.split("_s")[-1])
        out[arm][(fold, seed)] = {
            "macro_f1": d["selected_val_macro_f1"],
            "epoch": d["selected_epoch"],
            "wall_s": d["wall_clock_s"],
            "input_dim": d["input_dim"],
            "manifest_sha256": d["manifest_sha256"],
        }
    return out


def paired_bootstrap(diffs: np.ndarray, n: int = 20000, seed: int = 0) -> tuple:
    """Percentile CI on the mean paired difference. Resamples the matched pairs."""
    rng = np.random.default_rng(seed)
    boot = rng.choice(diffs, size=(n, len(diffs)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return float(lo), float(hi), float(min(p, 1.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=Path,
                    default=REPO / "outputs/branch_c/inner_val_results.json")
    args = ap.parse_args()

    data = collect()
    if not data:
        raise SystemExit("no runs found")

    print("BRANCH-C ARM SWEEP — INNER VALIDATION (outer folds NOT opened)\n")
    print(f"{'arm':<62}{'runs':>6}{'macro-F1':>11}{'sd':>8}{'min':>8}{'max':>8}")
    print("-" * 103)
    summary = {}
    for arm in sorted(data):
        v = np.array([r["macro_f1"] for r in data[arm].values()])
        dims = {r["input_dim"] for r in data[arm].values()}
        summary[arm] = {
            "n": len(v), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "input_dim": sorted(dims),
            "per_run": {f"f{f}_s{s}": r["macro_f1"] for (f, s), r in sorted(data[arm].items())},
        }
        print(f"{LABELS.get(arm, arm):<62}{len(v):>6}{v.mean():>11.4f}"
              f"{summary[arm]['sd']:>8.4f}{v.min():>8.4f}{v.max():>8.4f}")
    print("-" * 103)

    print("\nPAIRED CONTRASTS (same fold and seed, 15 matched pairs)")
    print(f"{'contrast':<44}{'pairs':>7}{'mean d':>10}{'95% CI':>22}{'p':>9}{'wins':>8}")
    print("-" * 100)
    contrasts = [("arm0_mstcn_quality", "arm0b_mstcn_quality5"),
                 ("arm1_mstcn_553_ff", "arm0b_mstcn_quality5"),
                 ("arm1_mstcn_553_ff", "arm0_mstcn_quality"),
                 ("arm2_mstcn_556_mp", "arm1_mstcn_553_ff"),
                 ("arm5_mstcn_556_fr", "arm1_mstcn_553_ff"),
                 ("arm5_mstcn_556_fr", "arm2_mstcn_556_mp")]
    pairs_out = {}
    for a, b in contrasts:
        if a not in data or b not in data:
            continue
        keys = sorted(set(data[a]) & set(data[b]))
        if not keys:
            continue
        d = np.array([data[a][k]["macro_f1"] - data[b][k]["macro_f1"] for k in keys])
        lo, hi, p = paired_bootstrap(d)
        wins = int((d > 0).sum())
        sig = "" if lo <= 0 <= hi else "  *"
        name = f"{a.split('_')[0]} - {b.split('_')[0]}"
        print(f"{name:<44}{len(d):>7}{d.mean():>+10.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{p:>9.4f}{wins:>5}/{len(d)}{sig}")
        pairs_out[f"{a}__minus__{b}"] = {
            "n_pairs": len(d), "mean_diff": float(d.mean()),
            "ci95": [lo, hi], "p": p, "wins": wins,
            "per_pair": {f"f{f}_s{s}": float(x) for (f, s), x in zip(keys, d)},
        }
    print("-" * 100)
    print("*  = 95% CI excludes zero. These are INNER-VALIDATION contrasts and are")
    print("   selection evidence, not a generalisation estimate.")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(
        {"summary": summary, "paired": pairs_out,
         "note": "inner validation only; outer folds not opened"},
        indent=2, sort_keys=True))
    print(f"\nwrote {args.json_out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
