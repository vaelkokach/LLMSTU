"""Train the registered Branch-C arms under the frozen nested-CV folds.

Generates one split manifest per outer fold (inner_train -> "train",
inner_val -> "val", outer_test -> "test" and never loaded), then dispatches
independent single-GPU runs of ``attention.thesis_eval.train`` - reusing the
existing trainer rather than writing a second one, so arms 1/2 are measured by
exactly the code that produced the Branch-B numbers.

Registered arms actually trained (BRANCH_C_PROTOCOL.md 4, amendment A1,
FINDINGS 12.15):

  arm1_mstcn_553_ff   base + face_found                    deployment baseline
  arm2_mstcn_556_mp   + MediaPipe yaw/pitch/roll           legacy pose baseline
  arm5_mstcn_556_fr   + 6DRepNet360 full-range rotation    the candidate

Arms 6 and 13 (seat-relative canonicalisation, reference-frame comparison) are
dropped: the registered gate returned a null AUC delta of +0.0083 (FINDINGS
12.15). Arm 4 (quality/presence only, no angles) is identical to arm 1 on this
feature ladder - ``553_facefound`` *is* the presence-only control - so it is not
run twice; the single run serves both roles and this note records why.

Arms 2 and 5 differ in **exactly** the three angle columns. Their 553 other
columns are byte-identical and ``face_found`` changed on 0.0% of frames
(patch_pose_columns verified 0 changes), so any difference between them is
attributable to how head orientation was measured and to nothing else.

    python tools/branch_c/launch_arm_sweep.py --gpus auto

Outer-test folds are never touched here. Opening them is a separate, deliberate
step under protocol section 2.4.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LLMDET = REPO / "LLMDet"

_spec = importlib.util.spec_from_file_location("prov", REPO / "tools/branch_c/provenance.py")
prov = importlib.util.module_from_spec(_spec)
sys.modules["prov"] = prov
_spec.loader.exec_module(prov)

FOLDS = REPO / "outputs/branch_c/splits/branch_c_folds.json"
SEQ_MANIFEST = REPO / "grounding_data/llmstu_seq_split_manifest.json"
MANIFEST_DIR = REPO / "outputs/branch_c/splits/fold_manifests"
OUT_ROOT = REPO / "outputs/branch_c/runs"

ARMS = {
    # Registered shortcut audit (protocol 5.1): quality/missingness signals only,
    # no appearance and no orientation. Upper-bounds how much cue performance is
    # obtainable from whether evidence exists rather than what it says.
    "arm0_mstcn_quality": dict(model="mstcn", feature_config="552_base",
                               root="grounding_data/llmstu_sequences_quality"),
    "arm0b_mstcn_quality5": dict(model="mstcn", feature_config="552_base",
                                 root="grounding_data/llmstu_sequences_quality5"),
    "arm1_mstcn_553_ff": dict(model="mstcn", feature_config="553_facefound",
                              root="grounding_data/llmstu_sequences_full"),
    "arm2_mstcn_556_mp": dict(model="mstcn", feature_config="556_hp",
                              root="grounding_data/llmstu_sequences_full"),
    "arm5_mstcn_556_fr": dict(model="mstcn", feature_config="556_hp",
                              root="grounding_data/llmstu_sequences_fullrange"),
    # Fusion arms (protocol 3, 8, 9, 10). Different trainer, same folds, same
    # selection rule, so their numbers sit in one table with the arms above.
    "arm3_appearance": dict(fusion_arm=True, root="grounding_data/llmstu_sequences_branch_c"),
    "arm8_uniform": dict(fusion_arm=True, root="grounding_data/llmstu_sequences_branch_c"),
    "arm9_learned": dict(fusion_arm=True, root="grounding_data/llmstu_sequences_branch_c"),
    "arm10_full": dict(fusion_arm=True, root="grounding_data/llmstu_sequences_branch_c"),
    # Leakage-free reruns. arm9/arm10 read the quality block through the
    # reliability head, and three of its columns are teacher-side annotation
    # fields; zeroing them at inference collapsed arm9 from 0.657 to 0.257
    # (FINDINGS 12.20). arm3/arm8 never read quality and are unaffected.
    "arm9c_learned_clean": dict(fusion_arm=True, arm_impl="arm9_learned",
                                root="grounding_data/llmstu_sequences_branch_c_clean"),
    "arm10c_full_clean": dict(fusion_arm=True, arm_impl="arm10_full",
                              root="grounding_data/llmstu_sequences_branch_c_clean"),
}
SEEDS = (42, 43, 44)


def write_fold_manifests() -> dict:
    """One manifest per fold. Same npz files, different split assignment."""
    folds = json.loads(FOLDS.read_text())
    base = json.loads(SEQ_MANIFEST.read_text())
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for fd in folds["folds"]:
        k = fd["fold"]
        tr, va = set(fd["inner_train_videos"]), set(fd["inner_val_videos"])
        te = set(fd["outer_test_videos"])
        samples = []
        for s in base["samples"]:
            v = s["video_id"]
            split = "train" if v in tr else "val" if v in va else "test" if v in te else None
            if split is None:
                raise SystemExit(f"video {v} is in no fold partition")
            samples.append({**s, "split": split})
        n = {x: sum(1 for s in samples if s["split"] == x) for x in ("train", "val", "test")}
        payload = {
            "source_manifest": str(SEQ_MANIFEST.relative_to(REPO)),
            "fold": k,
            "protocol": "branch_c_nested_grouped_cv",
            "fold_manifest_sha256": folds["manifest_sha256"],
            "note": ("Branch-C fold manifest. 'test' rows are the OUTER fold and "
                     "must not be loaded during development."),
            "counts": n,
            "samples": samples,
        }
        p = MANIFEST_DIR / f"fold_{k}.json"
        prov.atomic_write_json(payload, p)
        out[k] = p
        print(f"fold {k}: train {n['train']} / val {n['val']} / test {n['test']} -> {p.name}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="auto")
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifests = write_fold_manifests()
    gpus = (prov.select_free_gpus(n=prov.MAX_CONCURRENT_GPUS)
            if args.gpus == "auto" else [int(g) for g in args.gpus.split(",")])
    print(f"\nGPUs: {gpus} (cap {prov.MAX_CONCURRENT_GPUS}, shared host)\n")

    jobs = []
    for arm in args.arms.split(","):
        for k in [int(f) for f in args.folds.split(",")]:
            for seed in SEEDS:
                eid = f"{arm}_f{k}_s{seed}"
                d = OUT_ROOT / arm / eid
                if (d / "run_record.json").exists():
                    print(f"skip (done): {eid}")
                    continue
                cfg = ARMS[arm]
                if cfg.get("fusion_arm"):
                    cmd = [sys.executable, "-m", "attention.branch_c.train_fusion",
                           "--arm", cfg.get("arm_impl", arm), "--fold", str(k), "--seed", str(seed),
                           "--epochs", "60", "--batch-size", "32",
                           "--output-dir", str(d), "--manifest", str(manifests[k]),
                           "--sequence-root", str(REPO / cfg["root"])]
                else:
                    cmd = [sys.executable, "-m", "attention.thesis_eval.train",
                           "--experiment-id", eid, "--model", cfg["model"],
                           "--feature-config", cfg["feature_config"], "--seed", str(seed),
                           "--epochs", str(args.epochs), "--output-dir", str(d),
                           "--manifest", str(manifests[k]),
                           "--sequence-root", str(REPO / cfg["root"])]
                jobs.append((eid, d, arm, k, seed, cmd))
    print(f"{len(jobs)} runs to dispatch\n")
    if args.dry_run:
        for eid, _, _, _, _, cmd in jobs[:3]:
            print(eid, " ".join(cmd[-8:]))
        return

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_dir = OUT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    pending, running, free = list(jobs), [], list(gpus)
    t0 = time.time()
    done = failed = 0
    while pending or running:
        while pending and free:
            eid, d, arm, fold, seed, cmd = pending.pop(0)
            g = free.pop(0)
            d.mkdir(parents=True, exist_ok=True)
            log = open(log_dir / f"{eid}.log", "w")
            p = subprocess.Popen(cmd + ["--device", f"cuda:{g}"], cwd=str(LLMDET),
                                 stdout=log, stderr=subprocess.STDOUT)
            running.append((p, g, eid, log, time.time(), arm, fold, seed))
            print(f"[{time.time()-t0:6.0f}s] launch {eid} on cuda:{g}", flush=True)
        time.sleep(5)
        for item in list(running):
            p, g, eid, log, st, arm, fold, seed = item
            if p.poll() is None:
                continue
            log.close()
            running.remove(item)
            free.append(g)
            wall = time.time() - st
            ok = p.returncode == 0
            done += ok
            failed += (not ok)
            print(f"[{time.time()-t0:6.0f}s] {eid}: "
                  f"{'ok' if ok else f'FAILED rc={p.returncode}'} ({wall/60:.1f} min)",
                  flush=True)
            prov.append_record(prov.RunRecord(
                # Carried through from job construction, never re-parsed out of the
                # run id: an arm named "..._ff_f0_s42" splits on "_f" three times.
                run_id=eid, arm=arm, stage="train",
                command=["python", "-m", "attention.thesis_eval.train", "..."],
                seed=seed, fold=fold,
                device=f"cuda:{g}", wall_clock_s=wall, returncode=p.returncode,
                split_manifest_sha256=prov.sha256_file(FOLDS),
                notes="Branch-C arm sweep, inner train/val only; outer folds untouched",
            ))
    print(f"\n{done} ok, {failed} failed, {(time.time()-t0)/60:.1f} min total")


if __name__ == "__main__":
    main()
