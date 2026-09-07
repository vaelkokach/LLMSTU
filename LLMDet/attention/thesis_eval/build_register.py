"""Build the machine-readable and human-readable final results registers.

Every number the thesis may cite gets one entry with: metric, value, confidence
interval, evaluation split, evaluator version, checkpoint, config, evidence
file, caveat, and an explicit ``citable`` flag. Numbers that are *not* citable
(DDP shard averages, the contaminated matching value, the confounded transfer
result) are recorded here **with their reason**, so that "why can't I use the
0.4400?" has a written answer instead of being rediscovered.

    python -m attention.thesis_eval.build_register --out ../outputs
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from attention.thesis_eval import EVALUATOR_VERSION

REPO = Path(__file__).resolve().parents[3]


def entry(metric: str, value, *, split: str, evidence: str, citable: bool,
          ci: Optional[List[float]] = None, checkpoint: str = "",
          config: str = "", caveat: str = "", evaluator: str = EVALUATOR_VERSION,
          branch: str = "", unit: str = "") -> Dict:
    return {"metric": metric, "value": value, "unit": unit,
            "confidence_interval": ci, "split": split, "evaluator": evaluator,
            "checkpoint": checkpoint, "config": config, "evidence": evidence,
            "caveat": caveat, "citable": citable, "branch": branch}


def from_table_a(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    t = json.loads(path.read_text())
    split = t["split"]
    out = []
    for key, c in t["configs"].items():
        o = c["over_seeds"]
        bs = c.get("seed42_bootstrap") or {}
        for metric in ("macro_f1", "balanced_accuracy", "accuracy", "macro_auprc", "ece"):
            s = o[metric]
            b = bs.get(metric, {})
            out.append(entry(
                f"{key} · {metric}", round(s["mean"], 4), split=split,
                ci=[round(s.get("ci_low", float("nan")), 4),
                    round(s.get("ci_high", float("nan")), 4)],
                checkpoint=c["runs"][0] + "/checkpoints/best.pth",
                config=f"feature_config={c['feature_config']}, model={c['model']}",
                evidence=str(path),
                caveat=(f"mean over {s['n']} seeds (sd {s['std']:.4f}); "
                        f"seed-42 video-bootstrap CI "
                        f"[{b.get('ci_low', float('nan')):.4f}, "
                        f"{b.get('ci_high', float('nan')):.4f}]"),
                citable=True, branch="B (visible cues)"))
    for name, c in t.get("contrasts", {}).items():
        n_sig, n = c["n_seeds_significant_macro_f1"], c["n_seeds"]
        out.append(entry(
            f"Δ macro-F1 · {c['label']}", round(c["delta_macro_f1_seed_means"], 4),
            split=split, evidence=str(path), citable=True, branch="B (visible cues)",
            config=f"{c['a']} − {c['b']}",
            caveat=(f"paired video-level bootstrap; significant in {n_sig}/{n} seeds. "
                    + ("Supported." if n and n_sig == n else
                       "NOT statistically supported — report as within noise."))))
    return out


def from_cmose(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    d = json.loads(path.read_text())
    a = d["audit"]
    out = [entry(
        "CMOSE · subjects appearing in >1 official split",
        f"{a['subjects_in_more_than_one_official_split']}/{a['n_subjects']}",
        split="CMOSE official release", evidence=str(path), citable=True,
        branch="external (CMOSE, separate task)", evaluator="thesis_eval/cmose.py",
        caveat=a["note"])]
    for proto, blk in d["protocols"].items():
        for m, v in blk["test_over_seeds"].items():
            out.append(entry(
                f"CMOSE · {m} ({proto} split)", round(v["mean"], 4),
                split=f"CMOSE test ({proto})", evidence=str(path), citable=True,
                branch="external (CMOSE, separate task)",
                evaluator="thesis_eval/cmose.py",
                config=f"MLP over 1024-d I3D; {blk['split_sizes']}; "
                       f"{blk['subject_overlap_train_test']} subjects shared train/test",
                caveat=(f"mean over 3 seeds (sd {v['std']:.4f}). SEPARATE TASK — "
                        "four-level ordinal engagement. Never place beside the "
                        "six-class visible-cue macro-F1, the grounding R@1 or SCB mAP.")))
    return out


def from_events(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    e = json.loads(path.read_text())
    out = []
    for tag in ("gold_raw", "gold_dedup"):
        block = e["results"].get(tag)
        if not block:
            continue
        n_gold = block["n_gold_events"]
        dedup_note = ("de-duplicated gold: `inactivity` is an alias of `head_down` in "
                      "attention/events.py and one return_to_task marker is duplicated"
                      if tag == "gold_dedup" else
                      "historic raw gold — contains 7 duplicate `inactivity` episodes "
                      "and 1 duplicate marker; kept only for continuity with FINDINGS")
        for name, r in block["per_system"].items():
            ev = r["events"]["by_tiou"][f"{e['primary_iou']:.2f}"]
            seg = r["segmentation"]
            base = dict(split="human-gold diagnostic (984 crops / 10 tracks)",
                        evidence=str(path), branch="B (events)",
                        config=f"tIoU {e['primary_iou']}, {tag}")
            cav = (f"{dedup_note}. Diagnostic set — small and deliberately selected "
                   f"(2 segments, 8 gold tracks); NOT a classroom-wide estimate. "
                   f"Detector and tracker excluded by design.")
            out += [
                entry(f"event recall · {name} ({tag})", round(ev["recall"], 4),
                      citable=(tag == "gold_dedup"), caveat=cav,
                      unit=f"{ev['matched']}/{n_gold}", **base),
                entry(f"event F1 · {name} ({tag})", round(ev["f1"], 4),
                      citable=(tag == "gold_dedup"), caveat=cav, **base),
                entry(f"false alerts/hour · {name} ({tag})",
                      round(ev["false_alerts_per_hour"], 3),
                      citable=(tag == "gold_dedup"), caveat=cav, unit="per hour", **base),
                entry(f"segmental F1@25 · {name} ({tag})", round(seg["f1@25"], 4),
                      citable=(tag == "gold_dedup"), caveat=cav, **base),
                entry(f"segmental edit score · {name} ({tag})",
                      round(seg["edit_score"], 2), citable=(tag == "gold_dedup"),
                      caveat=cav, unit="0-100", **base),
                entry(f"frame accuracy vs human · {name} ({tag})",
                      round(r["frame"]["accuracy"], 4), citable=(tag == "gold_dedup"),
                      caveat=cav, **base),
            ]
    return out


#: Numbers that exist in the repository or in FINDINGS.md and that the thesis
#: must handle explicitly — either cite with the stated caveat, or not at all.
STATIC_ENTRIES: List[Dict] = [
    entry("Branch A · R@1 (grounding)", 0.6462, split="TEST (27 held-out videos)",
          ci=None, checkpoint="work_dirs/thesis_bundle/checkpoints/main_llmstu_exact_iter25000_final.pth",
          config="configs/student_llmstu_exact.py, 25k iters",
          evidence="FINDINGS.md §3.4c; TEST_SPLIT_PROTOCOL.md (pre-registered at commit 87bb2db)",
          evaluator="LLMDet eval_test_split.py", citable=True, branch="A (grounding)",
          caveat="Single permitted run under a pre-registered protocol; the protocol is now closed. "
                 "Validation R@1 was 0.6343, so test exceeds validation by 0.0119."),
    entry("Branch A · R@5", 0.9891, split="TEST", evidence="FINDINGS.md §3.4c",
          evaluator="LLMDet eval_test_split.py", citable=True, branch="A (grounding)"),
    entry("Branch A · R@10", 0.9982, split="TEST", evidence="FINDINGS.md §3.4c",
          evaluator="LLMDet eval_test_split.py", citable=True, branch="A (grounding)"),
    entry("Branch A · ablation Δ R@1 (Hungarian − ordinal)", 0.1724,
          split="validation", evidence="FINDINGS.md §3.5",
          evaluator="LLMDet detector eval", citable=True, branch="A (grounding)",
          caveat="Identical frames, captions, boxes, schedule, LR and seed; only the "
                 "unit→box binding differs. R@10 differs by only 0.014, so the gain is "
                 "attribution, not detection."),
    entry("Branch A · ablation Δ R@1 (p≥0.9 filter − unfiltered Hungarian)", -0.0167,
          split="validation", evidence="FINDINGS.md §3.5", citable=True,
          evaluator="LLMDet detector eval", branch="A (grounding)",
          caveat="NEGATIVE result: the confidence filter does not help. Retains 90.3% of "
                 "the Hungarian gain on 59.7% of the training regions."),
    entry("caption→box matching accuracy (Hungarian, content-only)", 0.7896,
          split="72,398 frames / 261,397 assignments",
          evidence="LLMDet/work_dirs/matching/results_lr_sw0.0.json; FINDINGS.md §3.2",
          evaluator="matching benchmark, spatial_weight=0.0", citable=True,
          branch="A (grounding)",
          caveat="CITE THIS, NOT 0.9231. The 92.3% figure used spatial_weight=0.5, whose "
                 "spatial term leaks the ground-truth left-to-right ordering."),
    entry("caption→box matching accuracy (ordinal baseline)", 0.2607,
          split="72,398 frames", evidence="FINDINGS.md §3.2", citable=True,
          evaluator="matching benchmark, spatial_weight=0.0", branch="A (grounding)"),
    entry("caption→box matching accuracy (contaminated)", 0.9231, split="—",
          evidence="FINDINGS.md §3.2", citable=False, branch="A (grounding)",
          caveat="DO NOT CITE. spatial_weight=0.5 feeds the ground-truth ordering into "
                 "the cost matrix."),
    entry("teacher frame agreement with human gold", 0.874,
          split="human-gold diagnostic (754 accepted frames)",
          evidence="FINDINGS.md §5.3", evaluator="gold event evaluator", citable=True,
          branch="B (label quality)",
          caveat="Empirical teacher benchmark, NOT a theoretical ceiling. Corroborated by "
                 "the disjoint scattered sample's 85.9%."),
    entry("pseudo-label field accuracy", 0.939, split="311 held-out gold crops",
          evidence="outputs/gold_eval_summary.md; FINDINGS.md §5.1", citable=True,
          evaluator="gold audit", branch="B (label quality)",
          caveat="Weakest field: activity, 78.9%."),
    entry("facial expression → expert boredom AUROC", 0.544,
          split="DIPSER, 1,176 paired observations, held-out subjects",
          evidence="LLMDet/work_dirs/boredom_validation.json; FINDINGS.md §6d.1",
          evaluator="validate_boredom.py", citable=True, branch="B (external)",
          caveat="Barely above chance (0.5). MEASURED LIMITATION: the 7 basic Ekman "
                 "expressions do not deliver academic-emotion (Pekrun) recognition. "
                 "The thesis must not claim boredom detection."),
    entry("DIPSER cue↔engagement Spearman ρ", 0.172,
          split="DIPSER, 25 subjects, 1,825 paired observations",
          evidence="work_dirs/attention_temporal_hp/dipser_correlation_metadata.json; FINDINGS.md §6c",
          evaluator="dipser_correlation.py", citable=True, branch="B (external)",
          caveat="p<0.0001 by 5,000-permutation test, and the sign is REVERSED versus the "
                 "naive hypothesis. Weak correlation. Evidence that cue meaning is "
                 "setting-dependent — the strongest support for the visible-cue framing."),
    entry("DIPSER correlation using our 556-dim model", -0.023, split="DIPSER, 4 subjects",
          evidence="work_dirs/attention_temporal_hp/dipser_correlation_smoke.json",
          citable=False, branch="B (external)", evaluator="dipser_correlation.py",
          caveat="DO NOT CITE as a validation result. p=0.74; the model predicts off-task "
                 "on 0–17% of DIPSER frames because the geometry features are far out of "
                 "distribution. Measures domain shift, not the cue↔engagement link."),
    entry("SCB zero-shot R@1", 0.0322, split="SCB Turn-Bow-Head val (505 images)",
          evidence="work_dirs/logs/scb_zeroshot.log; FINDINGS.md §6b", citable=False,
          evaluator="eval_scb_zeroshot.py", branch="A (external)",
          caveat="DO NOT CITE as a transfer score. SCB annotates ONLY behaviour-exhibiting "
                 "students, so correctly detecting an ordinary seated student scores as a "
                 "false positive; and the label mapping (BowHead→'sleeping head down') is "
                 "semantically wrong for a lecture hall. Report as a qualitative limitation. "
                 "The least-confounded number is localisation recall 0.392."),
    entry("DDP shard-averaged macro-F1 (552 / 556 / 570)", [0.4096, 0.4364, 0.4400],
          split="rank-0 validation shard", evidence="attention_temporal*/train_summary.json",
          evaluator="train_temporal_ddp.validate() — NO all-gather", citable=False,
          branch="B (visible cues)",
          caveat="DO NOT CITE. Computed on one rank's shard, not the full validation set. "
                 "Under correct measurement the 556/570 ordering reverses."),
    entry("legacy single-process macro-F1 (552 / 556 / 570)", [0.3835, 0.4098, 0.4080],
          split="validation on the sequence builder's own 102/25 split",
          evidence="attention_temporal*/single_process_eval.json; FINDINGS.md §6f",
          evaluator="eval_baseline_chain.py", citable=False, branch="B (visible cues)",
          caveat="Superseded. Correct as far as it goes, but (a) it is a VALIDATION number "
                 "on a split with no test set, (b) single seed, (c) no confidence interval, "
                 "(d) checkpoint selected by argmax over a metric that swings ±0.05 between "
                 "epochs. Use the thesis_eval ladder instead."),
    entry("DEPLOYED real-scene throughput", 5.77, unit="FPS",
          split="0325.mp4, 190 frames, 6.0 tracks mean",
          evidence="work_dirs/profiling/report_stride_3_2.json",
          evaluator="profiling/profile_pipeline.py", citable=True, branch="runtime",
          config="configs/attention_runtime.yaml (MS-TCN, 553_facefound) at "
                 "detector_stride=3, temporal_stride=2",
          caveat="THIS is the deployed configuration as of 2026-08-03. Single A100. "
                 "fps_p50 6.35, frame_ms_mean 173.3, p95 291.7. Still NEAR-real-time: "
                 "91.1% of frames miss a 10 fps budget and 100% miss 25 fps. The speed "
                 "is bought with striding, and stride_equivalence.json prices it: "
                 "against a 1:1 reference, 3:2 holds cue agreement at 0.947 but finds "
                 "13 of 15 episodes (episode agreement 0.80). Quote the throughput and "
                 "that cost together."),
    entry("superseded 2026-08-01 deployed throughput", 3.05, unit="FPS",
          split="0325.mp4, 120 frames, ~6 students",
          evidence="work_dirs/profiling/report_deployed_mstcn556.json; FINDINGS.md §11.12",
          evaluator="profiling/profile_pipeline.py", citable=False, branch="runtime",
          config="configs/attention_runtime.yaml (MS-TCN, 556-dim, MediaPipe head pose)",
          caveat="DO NOT CITE as the system's speed. This was the deployed figure until "
                 "2026-08-03, when deployment moved to the 553_facefound model with 3:2 "
                 "striding; see the 5.77 FPS entry. Retained because the 1 August "
                 "register reported it and the thesis draft may still quote it."),
    entry("DEPLOYED throughput with 30 students", 0.88, unit="FPS",
          split="0325.mp4 with synthesised detections",
          evidence="work_dirs/profiling/report_deployed_mstcn556.json", citable=True,
          evaluator="profiling/profile_pipeline.py", branch="runtime",
          caveat="p99 1557 ms. Per-student feature extraction dominates beyond ~5 students."),
    entry("same-session throughput WITHOUT head pose", 5.32, unit="FPS",
          split="0325.mp4, 120 frames, ~6 students",
          evidence="work_dirs/profiling/report_legacy_552_samesession.json", citable=True,
          evaluator="profiling/profile_pipeline.py", branch="runtime",
          config="configs/attention_temporal.yaml (552-dim, no head pose)",
          caveat="Like-for-like control measured in the same session as the deployed "
                 "figure. The head-pose block therefore costs ~100 ms/frame at ~6 "
                 "students - more than the detector."),
    entry("archived 2026-07-31 throughput", 7.1, unit="FPS", split="0325.mp4, 110 frames",
          evidence="work_dirs/profiling/report_scaling_batched.json; FINDINGS.md §7",
          evaluator="profiling/profile_pipeline.py", citable=False, branch="runtime",
          caveat="DO NOT CITE as the system's speed. Measured with a 552-dim extractor, "
                 "i.e. WITHOUT the head-pose block the deployed model requires, so it is "
                 "not the deployed configuration. It is also not reproducible on this "
                 "shared machine even for its own config (5.32 in-session)."),
    entry("dashboard end-to-end verification (2026-08-01, first attempt)", "INVALID",
          split="0325.mp4", evidence="FINDINGS.md §6d.3, §11.12", citable=False,
          evaluator="tools/dashboard/pipeline_bridge.py", branch="runtime",
          caveat="DO NOT CITE. The config declared input_dim 570 against a 552-dim "
                 "checkpoint; strict=False raises on a size mismatch, a bare except "
                 "swallowed it, and the dashboard served cues from a RANDOMLY "
                 "INITIALISED network. Re-verified 2026-08-01 with MS-TCN-556."),
    entry("dashboard alert threshold", 0.66, split="fitted on validation, frozen",
          evidence="work_dirs/thesis/runtime/mstcn_553_ff_thresholds.json", citable=True,
          evaluator="thesis_eval/runtime.py", branch="runtime",
          config="checkpoint.mstcn_553_ff_s42 (the DEPLOYED classifier per "
                 "artifacts.lock.json)",
          caveat="Lowest threshold with selective accuracy >= 85%: retains 64.6% of "
                 "frames at 85.3% accuracy vs 73.2% at full coverage. Display threshold "
                 "0.46 retains 91.1% at 76.4%. Temperature 0.9236. Neither is tuned on "
                 "test or on the human-gold set, and temperature scaling moves no "
                 "argmax, so accuracy at full coverage is unchanged."),
    entry("superseded 2026-08-01 dashboard alert threshold", 0.64,
          split="fitted on validation, frozen",
          evidence="work_dirs/thesis/runtime/mstcn_556_thresholds.json", citable=False,
          evaluator="thesis_eval/runtime.py", branch="runtime",
          config="mstcn_556_hp_s42 — NOT the deployed classifier",
          caveat="DO NOT CITE. These are the thresholds of mstcn_556_hp, which the "
                 "1 August register cited as the dashboard's calibration. The deployed "
                 "classifier is mstcn_553_ff_s42, whose frozen thresholds are 0.46/0.66; "
                 "see the entry above. Citing 0.64 describes a model the dashboard does "
                 "not run."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="work_dirs/thesis/tables")
    ap.add_argument("--events", default="work_dirs/thesis/events/summary.json")
    ap.add_argument("--cmose", default="work_dirs/thesis/cmose/cmose_results.json")
    ap.add_argument("--out", default="../outputs")
    args = ap.parse_args()

    entries: List[Dict] = []
    entries += from_table_a(Path(args.tables) / "table_a_val.json")
    entries += from_table_a(Path(args.tables) / "table_a_test.json")
    entries += from_events(Path(args.events))
    entries += from_cmose(Path(args.cmose))
    entries += STATIC_ENTRIES

    reg = {
        "generated": str(date.today()),
        "evaluator_version": EVALUATOR_VERSION,
        "reading_rule": (
            "Every entry carries `citable`. A false value is not an oversight — it "
            "records a number that exists in the repository or in FINDINGS.md and "
            "must NOT enter the thesis, together with the reason. Never mix metrics "
            "from different tables into one leaderboard: accuracy, macro-F1, mAP, "
            "MSE, retrieval R@1 and event recall answer different questions."),
        "n_entries": len(entries),
        "n_citable": sum(1 for e in entries if e["citable"]),
        "entries": entries,
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "FINAL_RESULTS_REGISTER.json").write_text(
        json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# Final results register", "",
        f"**Generated:** {reg['generated']} · **Evaluator:** `{EVALUATOR_VERSION}`", "",
        reg["reading_rule"], "",
        f"{reg['n_citable']} of {reg['n_entries']} entries are citable.", "",
    ]
    for branch in sorted({e["branch"] for e in entries}):
        md += [f"## {branch or 'unassigned'}", "",
               "| metric | value | 95% CI | split | citable | caveat |",
               "|---|---|---|---|:--:|---|"]
        for e in [x for x in entries if x["branch"] == branch]:
            v = e["value"]
            v = (", ".join(f"{x:.4f}" for x in v) if isinstance(v, list)
                 else f"{v}" + (f" {e['unit']}" if e["unit"] else ""))
            ci = ("—" if not e["confidence_interval"] or
                  any(not np.isfinite(c) for c in e["confidence_interval"])
                  else f"[{e['confidence_interval'][0]}, {e['confidence_interval'][1]}]")
            mark = "✅" if e["citable"] else "⛔"
            md.append(f"| {e['metric']} | {v} | {ci} | {e['split']} | {mark} | "
                      f"{e['caveat'].replace(chr(10), ' ')} |")
        md.append("")
    # encoding is explicit: the register emits ✅ / ⛔ markers, and
    # Path.write_text defaults to the locale codec — under cp1252 this raises
    # UnicodeEncodeError and leaves a 0-byte register behind.
    (out / "FINAL_RESULTS_REGISTER.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out}/FINAL_RESULTS_REGISTER.{{json,md}} "
          f"({reg['n_citable']}/{reg['n_entries']} citable)")


if __name__ == "__main__":
    main()
