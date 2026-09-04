#!/usr/bin/env python
"""Regenerate Table A from archived predictions and diff it against the published CSV.

THESIS_DEFENSIBILITY_REVIEW.md lists as defence question 11: "Can every table be
regenerated from the submitted artifact bundle?" This answers it mechanically for
Table A, and fails loudly if it stops being true.

    python tools/verify_tables.py --predictions-root LLMDet/work_dirs \\
        --tables LLMDet/work_dirs/thesis/tables

Metrics are recomputed here from `y` and `pred` in each `predictions.npz` rather
than read from any stored summary, so agreement is genuine evidence that the
archived predictions reconstruct the table — not a restatement of it. The
implementation is deliberately independent of the evaluator that produced the
tables; if the two ever disagree, that is the finding.

Balanced accuracy is the mean per-class recall over classes present in y or pred;
macro F1 is the unweighted mean per-class F1 over the same class set. Seeds are
averaged after per-seed metrics, matching how the published table aggregates.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TOL = 2e-3   # published table is 4dp; anything under this is formatting, not drift


def metrics(y, pred) -> Tuple[float, float, float]:
    import numpy as np
    classes = np.unique(np.concatenate([y, pred]))
    acc = float((y == pred).mean())
    recalls, f1s = [], []
    for c in classes:
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        rec = tp / (tp + fn) if (tp + fn) else None
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        if rec is not None:
            recalls.append(rec)
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        else:
            f1s.append(0.0)
    return acc, sum(recalls) / len(recalls), sum(f1s) / len(f1s)


def find_runs(root: Path, sweep: str, model: str, dims: str) -> List[Path]:
    """Experiment dirs for one config, across seeds, within ONE sweep.

    Scoping to the sweep is not cosmetic. The same config name recurs across
    sweeps — arch/mstcn_556_hp_s42, deploy_sim/mstcn_556_hp_s42 and
    posefix/mstcn_556_hp_bp_s42 all match a naive `mstcn_556_*_s*` glob. Because
    posefix carries eval_val but no eval_test, an unscoped search silently
    averages six seeds on val and three on test, and reports a table mismatch
    that is entirely an artefact of the search.
    """
    hits = []
    for sweep_dir in sorted(root.rglob(sweep)):
        if not sweep_dir.is_dir():
            continue
        for d in sorted(sweep_dir.glob(f"{model}_{dims}_*_s*")):
            if d.is_dir():
                hits.append(d)
    return hits


def load_published(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def verify(pred_root: Path, table_csv: Path, split: str, sweep: str) -> Tuple[int, int, List[str]]:
    import numpy as np

    rows = load_published(table_csv)
    ok = bad = 0
    problems: List[str] = []
    print(f"\n=== {table_csv.name} ({split}) ===")
    print(f"{'config':<16} {'recomputed':<24} {'published':<24} {'seeds':<6} verdict")

    for r in rows:
        model, dims = r["model"], r["dims"]
        runs = [d for d in find_runs(pred_root, sweep, model, dims)
                if (d / f"eval_{split}" / "predictions.npz").exists()]
        if not runs:
            problems.append(f"{model}_{dims} ({split}): no predictions.npz found")
            print(f"{model+'_'+dims:<16} {'-- missing --':<24}")
            bad += 1
            continue

        per_seed = []
        for d in runs:
            z = np.load(d / f"eval_{split}" / "predictions.npz", allow_pickle=True)
            per_seed.append(metrics(z["y"], z["pred"]))
        got = tuple(sum(v[i] for v in per_seed) / len(per_seed) for i in range(3))
        pub = (float(r["accuracy"]), float(r["balanced_accuracy"]), float(r["macro_f1"]))

        agree = all(abs(got[i] - pub[i]) <= TOL for i in range(3))
        n_pub = int(r.get("n_seeds") or 0)
        seed_note = f"{len(per_seed)}/{n_pub}" if n_pub else str(len(per_seed))
        if agree and (not n_pub or len(per_seed) == n_pub):
            ok += 1
            verdict = "MATCH"
        else:
            bad += 1
            verdict = "DIFFERS"
            if not agree:
                worst = max(range(3), key=lambda i: abs(got[i] - pub[i]))
                name = ["accuracy", "balanced_accuracy", "macro_f1"][worst]
                problems.append(f"{model}_{dims} ({split}): {name} "
                                f"recomputed {got[worst]:.4f} vs published {pub[worst]:.4f}")
            if n_pub and len(per_seed) != n_pub:
                problems.append(f"{model}_{dims} ({split}): {len(per_seed)} seed(s) "
                                f"archived but table reports {n_pub}")
        print(f"{model+'_'+dims:<16} "
              f"{got[0]:.4f} {got[1]:.4f} {got[2]:.4f}    "
              f"{pub[0]:.4f} {pub[1]:.4f} {pub[2]:.4f}    {seed_note:<6} {verdict}")
    return ok, bad, problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-root", default="LLMDet/work_dirs")
    ap.add_argument("--tables", default="LLMDet/work_dirs/thesis/tables")
    ap.add_argument("--sweep", default="arch",
                    help="sweep directory Table A draws from (default: arch)")
    args = ap.parse_args()

    try:
        import numpy  # noqa: F401
    except ImportError:
        print("numpy is required", file=sys.stderr)
        return 2

    pred_root, tables = Path(args.predictions_root), Path(args.tables)
    total_ok = total_bad = 0
    all_problems: List[str] = []
    for split in ("val", "test"):
        csv_path = tables / f"table_a_{split}.csv"
        if not csv_path.exists():
            all_problems.append(f"{csv_path} missing")
            continue
        ok, bad, probs = verify(pred_root, csv_path, split, args.sweep)
        total_ok += ok
        total_bad += bad
        all_problems += probs

    print(f"\n{total_ok} row(s) reproduce from archived predictions, {total_bad} do not.")
    if all_problems:
        print("\nUnresolved:")
        for p in all_problems:
            print(f"  - {p}")
        return 1
    print("Table A is fully regenerable from the archived predictions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
