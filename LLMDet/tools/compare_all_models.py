#!/usr/bin/env python3
"""Run EVERY trained detector variant on one image and compare them.

One best checkpoint per variant. E2 is skipped by default because it is
byte-identical to E0 (verified by MD5) — the March "E2 experiment" was never
actually trained; see MARCH_2026_POSTMORTEM.md §2.

All variants share the Grounding-DINO Swin-T architecture, so a single config
constructs the model for every checkpoint; only the weights differ.

Usage:
  PYTHONPATH=/home/jovyan/Computer_vision/LLMDet \
  python tools/compare_all_models.py --image <frame.jpg> \
      --prompts "a student sitting" "using phone" "sleeping head down" \
      --score-thr 0.15 --device cuda:0 --out work_dirs/model_comparison
"""
import argparse
import json
from pathlib import Path

import numpy as np

CKPT_DIR = "work_dirs/thesis_bundle/checkpoints"
CFG = "configs/student_llmstu_exact.py"

# (label, file, val R@1 on the leak-free split, note)
VARIANTS = [
    ("E0 base (MM-GDINO, not fine-tuned by us)", "e0_iter15000.pth", None,
     "pre-fine-tuning baseline"),
    ("E1 fine-tune (REGRESSION)", "e1_best.pth", None,
     "cost 9.6 pts R@5 vs its own baseline"),
    ("ARM A ordinal", "arm_a_ordinal_iter10000.pth", 0.3230,
     "~72% of training labels bound to the wrong student"),
    ("ARM B hungarian", "arm_b_hungarian_iter10000.pth", 0.4954,
     "+17.2 R@1 over ARM A"),
    ("ARM C hungarian+p90", "arm_c_hungarian_p90_iter10000.pth", 0.4787,
     "40% fewer but higher-confidence pairs; negative result"),
    ("MAIN exact-LLMSTU (BEST)", "main_llmstu_exact_iter25000_final.pth", 0.6343,
     "exact correspondences; the citable model"),
]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    i = (x2 - x1) * (y2 - y1)
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompts", nargs="+",
                    default=["a student sitting", "using phone",
                             "sleeping head down", "talking to peer"])
    ap.add_argument("--score-thr", type=float, default=0.15)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gt-jsonl", default=None,
                    help="optional ODVG file; if the image is in it, report "
                         "recall of the annotated students")
    ap.add_argument("--out", default="work_dirs/model_comparison")
    ap.add_argument("--include-e2", action="store_true",
                    help="E2 is byte-identical to E0; off by default")
    args = ap.parse_args()

    import cv2
    from mmdet.apis import init_detector, inference_detector

    im = cv2.imread(args.image)
    if im is None:
        raise SystemExit(f"cannot read {args.image}")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    gt = None
    if args.gt_jsonl:
        stem = Path(args.image).name
        for line in open(args.gt_jsonl):
            d = json.loads(line)
            if Path(d["filename"]).name == stem:
                gt = d
                break

    variants = list(VARIANTS)
    if args.include_e2:
        variants.insert(2, ("E2 (== E0, never trained)", "e2_best.pth", None,
                            "byte-identical to E0, MD5 verified"))

    results = {}
    for label, fname, r1, note in variants:
        ck = Path(CKPT_DIR) / fname
        if not ck.exists():
            print(f"  SKIP {label}: missing {ck}")
            continue
        print(f"\n=== {label}")
        model = init_detector(CFG, str(ck), device=args.device)
        per_prompt = {}
        for prompt in args.prompts:
            o = inference_detector(model, im, text_prompt=prompt,
                                   custom_entities=True).pred_instances
            sc = o.scores.cpu().numpy()
            bx = o.bboxes.cpu().numpy()
            keep = sc >= args.score_thr
            entry = {"n_det": int(keep.sum()),
                     "max_score": float(sc.max()) if sc.size else 0.0}
            if gt is not None:
                gtb = [r["bbox"] for r in gt["grounding"]["regions"]]
                used = set()
                for pb in bx[keep]:
                    for i, g in enumerate(gtb):
                        if i not in used and iou(pb, g) >= 0.5:
                            used.add(i)
                            break
                entry["gt_students"] = len(gtb)
                entry["matched"] = len(used)
                entry["recall"] = len(used) / max(len(gtb), 1)
            per_prompt[prompt] = entry
            r = f" recall {entry['recall']:.2f}" if "recall" in entry else ""
            print(f"    {prompt:24s} n={entry['n_det']:3d} "
                  f"max={entry['max_score']:.3f}{r}")

            vis = im.copy()
            if gt is not None:
                for g in gt["grounding"]["regions"]:
                    x1, y1, x2, y2 = [int(v) for v in g["bbox"]]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
            for pb, s in zip(bx[keep], sc[keep]):
                x1, y1, x2, y2 = [int(v) for v in pb]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 128, 255), 2)
                cv2.putText(vis, f"{s:.2f}", (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
            tag = label.split("(")[0].strip().replace(" ", "_")
            cv2.putText(vis, f"{label} | {prompt}", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.imwrite(str(outdir / f"{tag}__{prompt.replace(' ','_')}.jpg"), vis)
        results[label] = {"checkpoint": str(ck), "val_R@1": r1, "note": note,
                          "prompts": per_prompt}
        del model

    (outdir / "comparison.json").write_text(json.dumps(
        {"image": args.image, "score_thr": args.score_thr,
         "gt_caption": gt["grounding"]["caption"] if gt else None,
         "results": results}, indent=2))
    print(f"\nwritten: {outdir}/comparison.json  (+ annotated jpgs)")


if __name__ == "__main__":
    main()
