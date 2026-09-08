"""Which trained models the dashboard is allowed to offer, and which is default.

The sweeps under ``work_dirs/thesis/`` hold 54 checkpoints: 6 sweeps x
{transformer, mstcn, asrf} x {552_base, 553_facefound, 555_angles, 556_hp,
563_expr, 563_dyn, 570_full} x 3 seeds. The dashboard must not present that as a
flat list of 54 equivalent choices — most of them are seeds of the same variant,
and three of the feature configs cannot run live at all.

This module reduces the sweeps to one entry per **variant** (sweep x
architecture x feature config), choosing the seed with the best macro-F1 on the
**validation** split. Selection never reads the test split: the test numbers are
carried along and displayed, but they never decide anything. That is the same
discipline as ``TEST_SPLIT_PROTOCOL.md`` — the test split was spent once, and a
dropdown that ranked models by it would be spending it again, once per page
load.

Three facts about each variant come from the checkpoint's own ``run_record.json``
rather than from a table maintained by hand here:

``spec.feature_config``
    which columns of the live vector the model consumes. Configs needing a
    column at or beyond 556 (``563_expr``, ``563_dyn``, ``570_full``) are marked
    **not deployable**: the expression block needs a second per-crop GPU model
    and the dynamic block is a whole-track statistic that a streaming path
    cannot produce. They appear in the UI, disabled, with the reason — hiding
    them would make the ablation look smaller than it was.

``spec.sequence_root``
    which head-pose backend produced the 4-column block the model was trained
    on: ``_det`` -> BlazeFace detector, everything else -> FaceLandmarker mesh.
    The session cache stores both blocks, so each model is fed the one it was
    trained with instead of whichever the last run happened to use.

``eval_val/metrics.json``
    the evaluator's own numbers (``thesis_eval/1.0.0``), so what the dropdown
    shows is what the thesis tables show, from the same file.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
THESIS = REPO / "LLMDet" / "work_dirs" / "thesis"

#: The live extractor produces base(552) + head pose(4). A feature config that
#: needs a column beyond this cannot be served by a streaming path.
LIVE_MAX_COL = 556

#: Human labels. Keys are the trainer's ``feature_config`` strings.
FEATURE_LABEL = {
    "552_base": "CLIP + geometry + colour + posture",
    "553_facefound": "+ face-found flag",
    "555_angles": "+ head-pose angles",
    "556_hp": "+ head pose (angles + flag)",
    "563_expr": "+ facial expression",
    "563_dyn": "+ motion dynamics",
    "570_full": "+ expression + dynamics",
}

ARCH_LABEL = {"transformer": "Transformer", "mstcn": "MS-TCN", "asrf": "ASRF"}

#: What each sweep varied, so the dropdown explains why two entries share an
#: architecture and a feature config.
SWEEP_LABEL = {
    "ladder": "feature ladder",
    "arch": "architecture sweep",
    "headpose": "head-pose ablation",
    "posefix": "body-pose train/deploy fix (rejected, ref. FINDINGS 11.15)",
    "ff_bp": "face-found, landmarker on the person box",
    "ff_det": "face-found, BlazeFace detector",
}


def _head_pose_backend(sequence_root: str) -> str:
    """The backend whose 4-column block this model was trained on.

    ``llmstu_sequences_full_det`` was built with the BlazeFace detector
    (FINDINGS 11.16); ``llmstu_sequences_full`` and ``..._bp`` with the
    FaceLandmarker mesh. Feeding a model the other block would be a silent
    train/deploy mismatch of exactly the kind 11.15 was written about.
    """
    return "mediapipe_detector" if sequence_root.rstrip("/").endswith("_det") \
        else "mediapipe"


@dataclass
class ModelEntry:
    variant_id: str
    label: str
    sweep: str
    sweep_label: str
    model: str
    feature_config: str
    input_dim: int
    experiment_id: str
    seed: int
    n_seeds: int
    checkpoint: str
    head_pose_backend: str
    sequence_root: str
    deployable: bool
    blocked_reason: str
    val: Dict[str, float] = field(default_factory=dict)
    test: Dict[str, float] = field(default_factory=dict)
    #: mean +/- sd of macro-F1 over the variant's seeds. The thesis tables
    #: report this; the dashboard must run ONE checkpoint, so `val`/`test` above
    #: are that single seed's numbers and read higher than the mean by
    #: construction — best-of-3 is a selection, not a measurement. Both are
    #: shown so the deployed figure is never mistaken for the reported one.
    val_seed_mean: Dict[str, float] = field(default_factory=dict)
    test_seed_mean: Dict[str, float] = field(default_factory=dict)
    val_predictions: str = ""
    calibration: str = ""
    is_default: bool = False

    def to_json(self) -> Dict:
        return asdict(self)


def _metrics(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    m = json.loads(path.read_text())
    return {k: m[k] for k in
            ("macro_f1", "accuracy", "balanced_accuracy", "macro_auprc", "ece",
             "n_frames", "n_videos")
            if k in m}


def _max_column(feature_config: str) -> int:
    """Highest live column the config reads. Falls back to the dim on import
    failure so the registry still builds without torch present."""
    try:
        import sys
        sys.path.insert(0, str(REPO / "LLMDet"))
        from attention.thesis_eval import data as D
        return int(D.column_index(feature_config).max()) + 1
    except Exception:
        return int(feature_config.split("_")[0])


def _collect(thesis_root: Path) -> Dict[str, List[Dict]]:
    """Every evaluated run that has a checkpoint, grouped by variant id."""
    by_variant: Dict[str, List[Dict]] = {}
    for record in sorted(thesis_root.glob("*/*/run_record.json")):
        exp_dir = record.parent
        sweep = exp_dir.parent.name
        rec = json.loads(record.read_text())
        spec = rec["spec"]
        ckpt = exp_dir / "checkpoints" / "best.pth"
        val = _metrics(exp_dir / "eval_val" / "metrics.json")
        if not ckpt.exists() or "macro_f1" not in val:
            # No checkpoint, or never evaluated -> nothing honest to display.
            continue
        by_variant.setdefault(f"{sweep}/{spec['model']}_{spec['feature_config']}",
                              []).append({
            "sweep": sweep, "spec": spec, "dir": exp_dir, "ckpt": ckpt,
            "val": val, "test": _metrics(exp_dir / "eval_test" / "metrics.json"),
        })
    return by_variant


def _seed_mean(seeds: List[Dict], split: str) -> Dict[str, float]:
    vals = [r[split]["macro_f1"] for r in seeds if r[split]]
    if not vals:
        return {}
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5         if len(vals) > 1 else 0.0
    return {"macro_f1": m, "macro_f1_sd": sd, "n": len(vals)}


def _make_entry(variant_id: str, seeds: List[Dict], chosen: Dict) -> ModelEntry:
    """One registry row: `chosen` supplies the checkpoint, `seeds` the spread."""
    spec = chosen["spec"]
    fc = spec["feature_config"]
    maxcol = _max_column(fc)
    deployable = maxcol <= LIVE_MAX_COL
    reason = "" if deployable else (
        f"needs live column {maxcol} of {LIVE_MAX_COL}: the expression block "
        f"requires a per-crop FER model and the dynamics block is a "
        f"whole-track statistic, so neither can be produced by a streaming "
        f"path")
    seq_root = spec.get("sequence_root", "")
    return ModelEntry(
        variant_id=variant_id,
        label=f"{ARCH_LABEL.get(spec['model'], spec['model'])} · {fc}",
        sweep=chosen["sweep"],
        sweep_label=SWEEP_LABEL.get(chosen["sweep"], chosen["sweep"]),
        model=spec["model"],
        feature_config=fc,
        input_dim=int(fc.split("_")[0]),
        experiment_id=spec["experiment_id"],
        seed=int(spec["seed"]),
        n_seeds=len(seeds),
        checkpoint=str(chosen["ckpt"].relative_to(REPO)),
        head_pose_backend=_head_pose_backend(seq_root),
        sequence_root=seq_root,
        deployable=deployable,
        blocked_reason=reason,
        val=chosen["val"],
        test=chosen["test"],
        val_seed_mean=_seed_mean(seeds, "val"),
        test_seed_mean=_seed_mean(seeds, "test"),
        val_predictions=str((chosen["dir"] / "eval_val" / "predictions.npz")
                            .relative_to(REPO)),
    )


def scan(thesis_root: Path = THESIS) -> List[ModelEntry]:
    """One entry per variant, best validation seed, ordered best-first."""
    entries = [_make_entry(vid, seeds, max(seeds, key=lambda r: r["val"]["macro_f1"]))
               for vid, seeds in _collect(thesis_root).items()]
    entries.sort(key=lambda e: (not e.deployable, -e.val["macro_f1"]))
    for e in entries:
        if e.deployable:
            e.is_default = True     # best deployable variant on validation
            break
    return entries


def default_entry(entries: List[ModelEntry]) -> ModelEntry:
    for e in entries:
        if e.is_default:
            return e
    raise SystemExit("no deployable model in the registry")


def find(entries: List[ModelEntry], variant_id: str,
         thesis_root: Path = THESIS) -> Optional[ModelEntry]:
    """Look up a variant, optionally pinned to one seed as `variant@sNN`.

    Representing a variant by its best VALIDATION seed is the right rule for a
    ranked dropdown and the wrong one for a deployment. `attention_runtime.yaml`
    deploys ff_det/mstcn_553_ff_s42, but the variant id `ff_det/
    mstcn_553_facefound` resolves here to s43 -- a different checkpoint carrying
    its own fitted calibration (T 0.9236, alert 0.66 against T 0.9705, alert
    0.64) and a different alert coverage (64.6% against 69.0%).

    Without a pin the live Space cannot serve the checkpoint the thesis names as
    deployed, so a demo offered as "the deployed system" quietly is not one.
    Pinning changes nothing about how the dropdown ranks or defaults.
    """
    if "@s" not in variant_id:
        return next((e for e in entries if e.variant_id == variant_id), None)

    base, _, seed_txt = variant_id.partition("@s")
    try:
        seed = int(seed_txt)
    except ValueError:
        return None
    seeds = _collect(thesis_root).get(base)
    if not seeds:
        return None
    chosen = next((r for r in seeds if int(r["spec"]["seed"]) == seed), None)
    if chosen is None:
        return None
    entry = _make_entry(base, seeds, chosen)
    # Keep the pin visible: it is what the UI shows, what /api/models reports
    # as active, and what calibration_path() resolves against.
    entry.variant_id = variant_id
    return entry


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="write the registry as JSON")
    args = ap.parse_args()

    entries = scan()
    hdr = (f"{'variant':34} {'best':5} {'val F1':>7} {'val mean+-sd':>16} "
           f"{'test F1':>8} {'live':5} {'head pose':18} sweep")
    print(hdr)
    print("-" * len(hdr))
    for e in entries:
        test = f"{e.test['macro_f1']:.4f}" if e.test else "—"
        vm = e.val_seed_mean
        mean = f"{vm['macro_f1']:.4f}+-{vm['macro_f1_sd']:.4f}" if vm else "—"
        print(f"{e.variant_id:34} s{e.seed:<4} "
              f"{e.val['macro_f1']:7.4f} {mean:>16} {test:>8} "
              f"{'yes' if e.deployable else 'NO':5} {e.head_pose_backend:18} "
              f"{e.sweep_label}"
              + ("   <- DEFAULT" if e.is_default else ""))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            [e.to_json() for e in entries], indent=2))
        print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
