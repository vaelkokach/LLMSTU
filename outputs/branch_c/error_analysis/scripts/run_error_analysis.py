"""
Branch-C error analysis. Reads ONLY outputs/branch_c/eval/<eid>/predictions.npz
(inner-validation predictions, verified against outputs/branch_c/splits/branch_c_folds.json
to contain exactly the inner_val_videos / inner_val.frames of each fold -- zero
overlap with outer_test_videos). Writes CSVs/figures under
outputs/branch_c/error_analysis/. Read-only w.r.t. everything else.

Run: python outputs/branch_c/error_analysis/scripts/run_error_analysis.py
"""
import json
import re
import itertools
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/jovyan/Computer_vision")
EVAL_DIR = ROOT / "outputs/branch_c/eval"
OUT_DIR = ROOT / "outputs/branch_c/error_analysis"
FIG_DIR = OUT_DIR / "figures"
CROP_DIR = OUT_DIR / "crops"
FOLDS_PATH = ROOT / "outputs/branch_c/splits/branch_c_folds.json"
LABELS_PATH = ROOT / "grounding_data/llmstu_tools/outputs/labels_tracked.jsonl"
CROPS_ROOT = ROOT / "grounding_data/LLMSTU/crops"

CUE_CLASSES = ["screen_oriented", "looking_away", "head_down", "turned_to_peer", "phone_use", "uncertain"]
ARMS = {
    "arm1": "arm1_mstcn_553_ff",
    "arm2": "arm2_mstcn_556_mp",
    "arm5": "arm5_mstcn_556_fr",
}
FOLDS = [0, 1, 2, 3, 4]
SEEDS = [42, 43, 44]

# ---------------------------------------------------------------------------
# Guard: verify eval dirs are inner-val only, before touching anything else.
# ---------------------------------------------------------------------------
def verify_inner_val_only():
    folds_manifest = json.loads(FOLDS_PATH.read_text())
    problems = []
    for arm_key, arm_id in ARMS.items():
        for f in FOLDS:
            for s in SEEDS:
                eid = f"{arm_id}_f{f}_s{s}"
                p = EVAL_DIR / eid / "predictions.npz"
                d = np.load(p)
                vids = set(d["video_id"].tolist())
                fold_manifest = folds_manifest["folds"][f]
                outer = set(fold_manifest["outer_test_videos"])
                inner_val = set(fold_manifest["inner_val_videos"])
                if vids & outer:
                    problems.append(f"{eid}: OVERLAPS OUTER TEST videos: {vids & outer}")
                if vids != inner_val:
                    problems.append(f"{eid}: video set != inner_val_videos (n={len(vids)} vs {len(inner_val)})")
                if len(d["video_id"]) != fold_manifest["counts"]["inner_val"]["frames"]:
                    problems.append(f"{eid}: frame count {len(d['video_id'])} != inner_val frames {fold_manifest['counts']['inner_val']['frames']}")
    if problems:
        raise RuntimeError("INNER-VAL GUARD FAILED:\n" + "\n".join(problems))
    print(f"[guard] OK: all {len(ARMS)*len(FOLDS)*len(SEEDS)} eval dirs are inner-val only, zero outer-test overlap.")


def load_eval(arm_id, fold, seed):
    eid = f"{arm_id}_f{fold}_s{seed}"
    d = np.load(EVAL_DIR / eid / "predictions.npz")
    df = pd.DataFrame({
        "video_id": d["video_id"],
        "seat_id": d["seat_id"],
        "t": d["t"],
        "seq_key": d["seq_key"],
        "y": d["y"],
        "pred": d["pred"],
    })
    probs = d["probs"]
    for i, c in enumerate(CUE_CLASSES):
        df[f"p_{c}"] = probs[:, i]
    df["conf"] = probs.max(axis=1)
    df["fold"] = fold
    df["seed"] = seed
    df["arm"] = arm_id
    df["y_cls"] = df["y"].map(lambda i: CUE_CLASSES[i])
    df["pred_cls"] = df["pred"].map(lambda i: CUE_CLASSES[i])
    df["correct"] = df["y"] == df["pred"]
    return df


def load_arm_seed(arm_id, seed):
    """Pool across the 5 outer folds for one seed -- each fold's inner_val is a
    disjoint 20-video slice, so pooling gives frame-level coverage without
    double counting any frame within a single (arm, seed)."""
    return pd.concat([load_eval(arm_id, f, seed) for f in FOLDS], ignore_index=True)


def confusion_from_df(df):
    cm = pd.crosstab(df["y_cls"], df["pred_cls"]).reindex(index=CUE_CLASSES, columns=CUE_CLASSES, fill_value=0)
    return cm


def macro_f1_from_cm(cm):
    f1s = []
    for c in CUE_CLASSES:
        tp = cm.loc[c, c]
        fp = cm[c].sum() - tp
        fn = cm.loc[c].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)), dict(zip(CUE_CLASSES, f1s))


# ===========================================================================
# SECTION 1 -- failure taxonomy
# ===========================================================================
def section1():
    print("=== Section 1: failure taxonomy ===")
    results = {}
    for seed in SEEDS:
        df = load_arm_seed(ARMS["arm2"], seed)
        cm = confusion_from_df(df)
        results[seed] = (df, cm)
    df42, cm42 = results[42]
    cm42.to_csv(OUT_DIR / "s1_confusion_matrix_arm2_seed42_pooled.csv")

    n = len(df42)
    n_err = int((~df42["correct"]).sum())
    acc = 1 - n_err / n
    macro_f1, per_class_f1 = macro_f1_from_cm(cm42)
    print(f"arm2 seed42 pooled: n={n} errors={n_err} acc={acc:.4f} macro_f1={macro_f1:.4f}")

    # error pairs, ranked by share of ALL errors
    err_pairs = []
    for t_ in CUE_CLASSES:
        for p_ in CUE_CLASSES:
            if t_ == p_:
                continue
            cnt = int(cm42.loc[t_, p_])
            if cnt > 0:
                err_pairs.append((t_, p_, cnt))
    err_pairs.sort(key=lambda x: -x[2])
    err_df = pd.DataFrame(err_pairs, columns=["true", "pred", "count"])
    err_df["share_of_all_errors"] = err_df["count"] / n_err
    err_df.to_csv(OUT_DIR / "s1_error_pairs_arm2_seed42_pooled.csv", index=False)

    # per-class: where do this class's errors go (row-normalized off-diagonal)
    conf_dest = {}
    for c in CUE_CLASSES:
        row = cm42.loc[c]
        support = row.sum()
        errs = support - row[c]
        dest = (row.drop(index=c) / errs).sort_values(ascending=False) if errs > 0 else row.drop(index=c) * 0
        conf_dest[c] = {"support": int(support), "n_errors": int(errs), "error_rate": float(errs / support) if support else float('nan'),
                         "top_confusions": {k: float(v) for k, v in dest.items()}}
    (OUT_DIR / "s1_per_class_confusion_breakdown_arm2_seed42.json").write_text(json.dumps(conf_dest, indent=2))

    # check pattern holds for seeds 43/44
    stability = {}
    for seed in SEEDS:
        df, cm = results[seed]
        mf1, pcf1 = macro_f1_from_cm(cm)
        top3 = []
        for t_ in CUE_CLASSES:
            for p_ in CUE_CLASSES:
                if t_ != p_ and cm.loc[t_, p_] > 0:
                    top3.append((t_, p_, int(cm.loc[t_, p_])))
        top3.sort(key=lambda x: -x[2])
        stability[seed] = {"macro_f1": mf1, "per_class_f1": pcf1, "top5_error_pairs": top3[:5],
                            "n_errors": int((~df['correct']).sum()), "n": int(len(df))}
    (OUT_DIR / "s1_seed_stability.json").write_text(json.dumps(stability, indent=2))

    # figure: confusion matrix heatmap (counts, row-normalized for readability)
    cm_norm = cm42.div(cm42.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_norm.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(6)); ax.set_xticklabels(CUE_CLASSES, rotation=40, ha="right")
    ax.set_yticks(range(6)); ax.set_yticklabels(CUE_CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("arm2, seed 42, pooled inner-val (row-normalized)")
    for i in range(6):
        for j in range(6):
            v = cm_norm.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                     color="white" if v > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="row fraction")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s1_confusion_matrix_arm2_seed42.png", dpi=150)
    plt.close(fig)
    return results


# ===========================================================================
# SECTION 2 -- arm5 vs arm2 error-set overlap, vs seed noise floor
# ===========================================================================
def _key(df):
    return list(zip(df["video_id"], df["seat_id"].astype(int), np.round(df["t"].values, 3)))


def align_two(dfa, dfb, tag_a, tag_b):
    """Align two prediction frames on (video_id, seat_id, t) and return merged df.
    Identifying columns (video_id, seat_id, t) are kept once, unsuffixed, taken
    from dfa; all other overlapping columns get _tag_a / _tag_b suffixes."""
    dfa = dfa.copy(); dfb = dfb.copy()
    dfa["_k"] = _key(dfa)
    dfb["_k"] = _key(dfb)
    dfa = dfa.drop_duplicates("_k")
    dfb = dfb.drop_duplicates("_k")
    id_cols = ["video_id", "seat_id", "t"]
    dfb_reduced = dfb.drop(columns=id_cols)
    m = dfa.merge(dfb_reduced, on="_k", suffixes=(f"_{tag_a}", f"_{tag_b}"))
    return m


def section2():
    print("=== Section 2: arm2 vs arm5 error-set comparison, vs seed noise floor ===")
    rows = []
    per_fold_detail = []
    for f in FOLDS:
        for seed in SEEDS:
            d2 = load_eval(ARMS["arm2"], f, seed)
            d5 = load_eval(ARMS["arm5"], f, seed)
            m = align_two(d2, d5, "arm2", "arm5")
            n = len(m)
            agree = int((m["pred_cls_arm2"] == m["pred_cls_arm5"]).sum())
            both_correct = int((m["correct_arm2"] & m["correct_arm5"]).sum())
            both_wrong = int((~m["correct_arm2"] & ~m["correct_arm5"]).sum())
            arm5_right_arm2_wrong = int((m["correct_arm5"] & ~m["correct_arm2"]).sum())
            arm2_right_arm5_wrong = int((m["correct_arm2"] & ~m["correct_arm5"]).sum())
            rows.append(dict(fold=f, seed=seed, n=n, agreement_rate=agree / n,
                              both_correct=both_correct, both_wrong=both_wrong,
                              arm5_right_arm2_wrong=arm5_right_arm2_wrong,
                              arm2_right_arm5_wrong=arm2_right_arm5_wrong,
                              net_arm5_minus_arm2=(arm5_right_arm2_wrong - arm2_right_arm5_wrong) / n))
    arm2_vs_arm5 = pd.DataFrame(rows)
    arm2_vs_arm5.to_csv(OUT_DIR / "s2_arm2_vs_arm5_agreement.csv", index=False)

    # noise floor: same arm, two different seeds, same fold
    noise_rows = []
    for arm_key, arm_id in [("arm2", ARMS["arm2"]), ("arm5", ARMS["arm5"])]:
        for f in FOLDS:
            for s_a, s_b in itertools.combinations(SEEDS, 2):
                da = load_eval(arm_id, f, s_a)
                db = load_eval(arm_id, f, s_b)
                m = align_two(da, db, "a", "b")
                n = len(m)
                agree = int((m["pred_cls_a"] == m["pred_cls_b"]).sum())
                noise_rows.append(dict(arm=arm_key, fold=f, seed_a=s_a, seed_b=s_b, n=n, agreement_rate=agree / n))
    noise_df = pd.DataFrame(noise_rows)
    noise_df.to_csv(OUT_DIR / "s2_seed_noise_floor_agreement.csv", index=False)

    summary = {
        "arm2_vs_arm5_mean_agreement_rate": float(arm2_vs_arm5["agreement_rate"].mean()),
        "arm2_vs_arm5_sd_agreement_rate": float(arm2_vs_arm5["agreement_rate"].std()),
        "seed_noise_floor_mean_agreement_rate": float(noise_df["agreement_rate"].mean()),
        "seed_noise_floor_sd_agreement_rate": float(noise_df["agreement_rate"].std()),
        "seed_noise_floor_by_arm": noise_df.groupby("arm")["agreement_rate"].mean().to_dict(),
        "mean_both_correct_frac": float((arm2_vs_arm5["both_correct"] / arm2_vs_arm5["n"]).mean()),
        "mean_both_wrong_frac": float((arm2_vs_arm5["both_wrong"] / arm2_vs_arm5["n"]).mean()),
        "mean_arm5_right_arm2_wrong_frac": float((arm2_vs_arm5["arm5_right_arm2_wrong"] / arm2_vs_arm5["n"]).mean()),
        "mean_arm2_right_arm5_wrong_frac": float((arm2_vs_arm5["arm2_right_arm5_wrong"] / arm2_vs_arm5["n"]).mean()),
    }
    (OUT_DIR / "s2_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return arm2_vs_arm5, noise_df, summary


# ===========================================================================
# SECTION 3 -- confidence / risk-coverage
# ===========================================================================
def section3():
    print("=== Section 3: confidence and risk-coverage ===")
    frames = []
    for seed in SEEDS:
        frames.append(load_arm_seed(ARMS["arm2"], seed))
    df = pd.concat(frames, ignore_index=True)

    # error confidence distribution
    err_conf = df.loc[~df["correct"], "conf"]
    corr_conf = df.loc[df["correct"], "conf"]
    conf_summary = {
        "n_total": int(len(df)),
        "error_conf_mean": float(err_conf.mean()), "error_conf_median": float(err_conf.median()),
        "correct_conf_mean": float(corr_conf.mean()), "correct_conf_median": float(corr_conf.median()),
        "error_conf_deciles": {str(q): float(err_conf.quantile(q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]},
        "frac_errors_conf_gt_0.7": float((err_conf > 0.7).mean()),
        "frac_errors_conf_gt_0.5": float((err_conf > 0.5).mean()),
        "frac_errors_conf_lt_0.3": float((err_conf < 0.3).mean()),
    }

    # risk-coverage: abstain on lowest-confidence frames, measure accuracy on the rest
    df_sorted = df.sort_values("conf", ascending=False).reset_index(drop=True)
    coverages = [1.0, 0.9, 0.75, 0.5, 0.25, 0.1]
    rc_rows = []
    n = len(df_sorted)
    for cov in coverages:
        k = int(round(cov * n))
        sub = df_sorted.iloc[:k]
        acc = float(sub["correct"].mean()) if k > 0 else float("nan")
        # macro-F1 at this coverage, restricted to the retained subset (classes may vanish)
        cm = confusion_from_df(sub)
        mf1, _ = macro_f1_from_cm(cm)
        rc_rows.append(dict(coverage=cov, n=k, accuracy=acc, macro_f1=mf1,
                             conf_threshold=float(df_sorted.iloc[k - 1]["conf"]) if k > 0 else float("nan")))
    rc_df = pd.DataFrame(rc_rows)
    rc_df.to_csv(OUT_DIR / "s3_risk_coverage_arm2_pooled_allseeds.csv", index=False)
    (OUT_DIR / "s3_confidence_summary.json").write_text(json.dumps(conf_summary, indent=2))
    print(rc_df)
    print(json.dumps(conf_summary, indent=2))

    # per-class confidence-of-error, to see which classes fail loudly vs quietly
    per_class_err_conf = df.loc[~df["correct"]].groupby("y_cls")["conf"].agg(["mean", "median", "count"])
    per_class_err_conf = per_class_err_conf.reindex(CUE_CLASSES)
    per_class_err_conf.to_csv(OUT_DIR / "s3_per_class_error_confidence.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(corr_conf, bins=40, alpha=0.6, label="correct", density=True, color="#3b6fa0")
    axes[0].hist(err_conf, bins=40, alpha=0.6, label="error", density=True, color="#c0574a")
    axes[0].set_xlabel("predicted-class confidence (max prob)"); axes[0].set_ylabel("density")
    axes[0].set_title("Confidence: correct vs error frames (arm2, all seeds)")
    axes[0].legend()

    axes[1].plot(rc_df["coverage"], rc_df["accuracy"], marker="o", color="#3b6fa0", label="accuracy")
    axes[1].plot(rc_df["coverage"], rc_df["macro_f1"], marker="s", color="#c0574a", label="macro-F1")
    axes[1].set_xlabel("coverage (fraction retained, highest-confidence first)")
    axes[1].set_ylabel("metric"); axes[1].set_title("Risk-coverage (arm2, all seeds pooled)")
    axes[1].legend(); axes[1].invert_xaxis()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s3_confidence_and_risk_coverage.png", dpi=150)
    plt.close(fig)
    return rc_df, conf_summary


# ===========================================================================
# SECTION 4 -- temporal structure of errors (run lengths)
# ===========================================================================
def run_lengths_for_df(df, modal_dt=0.999, gap_factor=1.6):
    """Within each (video_id, seat_id) track, sort by t, split into contiguous
    segments wherever the time gap exceeds gap_factor*modal_dt (real discontinuity,
    not a sampling artifact), then compute run lengths of consecutive error frames
    within each contiguous segment."""
    runs = []
    for (vid, seat), g in df.groupby(["video_id", "seat_id"]):
        g = g.sort_values("t")
        t = g["t"].values
        err = (~g["correct"].values).astype(int)
        if len(t) < 2:
            continue
        gaps = np.diff(t)
        seg_breaks = np.where(gaps > gap_factor * modal_dt)[0] + 1
        segs = np.split(np.arange(len(t)), seg_breaks)
        for seg in segs:
            e = err[seg]
            # run-length encode
            i = 0
            while i < len(e):
                j = i
                while j < len(e) and e[j] == e[i]:
                    j += 1
                runs.append((vid, seat, bool(e[i]), j - i))
                i = j
    return pd.DataFrame(runs, columns=["video_id", "seat_id", "is_error", "run_length"])


def section4():
    print("=== Section 4: temporal structure of errors ===")
    df = load_arm_seed(ARMS["arm2"], 42)
    runs = run_lengths_for_df(df)
    err_runs = runs[runs["is_error"]]
    ok_runs = runs[~runs["is_error"]]
    err_runs.to_csv(OUT_DIR / "s4_error_run_lengths_arm2_seed42.csv", index=False)

    dist = err_runs["run_length"].value_counts().sort_index()
    summary = {
        "n_error_runs": int(len(err_runs)),
        "n_error_frames": int(err_runs["run_length"].sum()),
        "frac_isolated_len1": float((err_runs["run_length"] == 1).sum() / len(err_runs)),
        "frac_frames_in_isolated_runs": float(err_runs.loc[err_runs["run_length"] == 1, "run_length"].sum() / err_runs["run_length"].sum()),
        "frac_runs_len_ge3": float((err_runs["run_length"] >= 3).sum() / len(err_runs)),
        "frac_frames_in_runs_ge3": float(err_runs.loc[err_runs["run_length"] >= 3, "run_length"].sum() / err_runs["run_length"].sum()),
        "frac_runs_len_ge5": float((err_runs["run_length"] >= 5).sum() / len(err_runs)),
        "frac_frames_in_runs_ge5": float(err_runs.loc[err_runs["run_length"] >= 5, "run_length"].sum() / err_runs["run_length"].sum()),
        "mean_error_run_length": float(err_runs["run_length"].mean()),
        "median_error_run_length": float(err_runs["run_length"].median()),
        "max_error_run_length": int(err_runs["run_length"].max()),
        "mean_correct_run_length": float(ok_runs["run_length"].mean()),
        "median_correct_run_length": float(ok_runs["run_length"].median()),
        "run_length_histogram": {str(k): int(v) for k, v in dist.items()},
    }
    (OUT_DIR / "s4_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "run_length_histogram"}, indent=2))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    max_show = 15
    counts = [int((err_runs["run_length"] == L).sum()) for L in range(1, max_show)]
    counts.append(int((err_runs["run_length"] >= max_show).sum()))
    labels = [str(i) for i in range(1, max_show)] + [f">={max_show}"]
    ax.bar(labels, counts, color="#c0574a")
    ax.set_xlabel("error run length (consecutive mis-classified frames)")
    ax.set_ylabel("number of runs")
    ax.set_title("arm2, seed 42, pooled inner-val: error run-length distribution")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s4_error_run_length_hist.png", dpi=150)
    plt.close(fig)
    return runs, summary


# ===========================================================================
# SECTION 5 -- join to labels_tracked for head_down evidence-quality
# ===========================================================================
_TF_RE = re.compile(r"^t(\d+)_(\d+)_f(\d+)")


def parse_t_from_src_frame(src_frame):
    m = _TF_RE.match(src_frame)
    if not m:
        return None
    sec, millis, _f = m.groups()
    return int(sec) + int(millis) / 1000.0


def build_labels_index(needed_video_ids):
    """Build a dict (video_id, seat_id, round(t,3)) -> record fields, restricted to
    the video_ids we actually need (keeps memory bounded)."""
    idx = {}
    n_read = 0
    n_kept = 0
    with open(LABELS_PATH) as f:
        for line in f:
            n_read += 1
            # cheap pre-filter before full json parse
            d = json.loads(line)
            vid = d["video_id"]
            if vid not in needed_video_ids:
                continue
            t = parse_t_from_src_frame(d["src_frame"])
            if t is None:
                continue
            key = (vid, int(d["seat_id"]), round(t, 3))
            idx[key] = {
                "occluded": d.get("occluded"),
                "det_conf": d.get("det_conf"),
                "head_span_px": d.get("head_span_px"),
                "activity": d.get("activity"),
                "file_name": d.get("file_name"),
                "caption": d.get("caption"),
                "bbox_crop": d.get("bbox_crop"),
            }
            n_kept += 1
    return idx, n_read, n_kept


def section5():
    print("=== Section 5: pose signal location (join to labels_tracked) ===")
    df2 = load_arm_seed(ARMS["arm2"], 42)
    df5 = load_arm_seed(ARMS["arm5"], 42)
    m = align_two(df2, df5, "arm2", "arm5")
    print(f"aligned arm2/arm5 seed42 pooled: {len(m)} frames "
          f"({len(df2)} / {len(df5)} pre-align)")

    needed_vids = set(m["video_id"].unique().tolist())
    idx, n_read, n_kept = build_labels_index(needed_vids)
    print(f"labels_tracked: read {n_read} lines, kept {n_kept} matching needed video_ids")

    def lookup(row):
        key = (row["video_id"], int(row["seat_id"]), round(row["t"], 3))
        return idx.get(key)

    hits = 0
    misses = 0
    recs = []
    for _, row in m.iterrows():
        rec = lookup(row)
        if rec is None:
            misses += 1
            recs.append({})
        else:
            hits += 1
            recs.append(rec)
    join_rate = hits / (hits + misses)
    print(f"join hit rate: {hits}/{hits+misses} = {join_rate:.4f}")

    join_report = {"hits": hits, "misses": misses, "hit_rate": join_rate}
    (OUT_DIR / "s5_join_diagnostic.json").write_text(json.dumps(join_report, indent=2))

    RELIABLE_JOIN_THRESHOLD = 0.90
    if join_rate < RELIABLE_JOIN_THRESHOLD:
        print(f"JOIN NOT RELIABLE (hit rate {join_rate:.3f} < {RELIABLE_JOIN_THRESHOLD}). "
              "Skipping section 5 head-evidence analysis per instructions -- will report 'skip' in the MD.")
        return {"reliable": False, "join_rate": join_rate}

    rec_df = pd.DataFrame(recs)
    m2 = pd.concat([m.reset_index(drop=True), rec_df.reset_index(drop=True)], axis=1)
    m2.to_csv(OUT_DIR / "s5_arm2_arm5_joined_labels.csv", index=False)

    # restrict to true head_down frames
    hd = m2[m2["y_cls_arm2"] == "head_down"].copy()
    hd = hd.dropna(subset=["det_conf", "head_span_px", "occluded"])
    print(f"head_down frames with usable joined evidence: {len(hd)}")

    hd["arm5_wrong"] = ~hd["correct_arm5"]
    hd["arm5_right"] = hd["correct_arm5"]
    grp = hd.groupby("arm5_wrong")[["det_conf", "head_span_px"]].agg(["mean", "median", "count"])
    occ_rate = hd.groupby("arm5_wrong")["occluded"].mean()

    summary = {
        "n_head_down_joined": int(len(hd)),
        "det_conf_mean_arm5_wrong": float(hd.loc[hd["arm5_wrong"], "det_conf"].mean()),
        "det_conf_mean_arm5_right": float(hd.loc[hd["arm5_right"], "det_conf"].mean()),
        "head_span_px_mean_arm5_wrong": float(hd.loc[hd["arm5_wrong"], "head_span_px"].mean()),
        "head_span_px_mean_arm5_right": float(hd.loc[hd["arm5_right"], "head_span_px"].mean()),
        "occluded_rate_arm5_wrong": float(occ_rate.get(True, float("nan"))),
        "occluded_rate_arm5_right": float(occ_rate.get(False, float("nan"))),
        "n_arm5_wrong": int(hd["arm5_wrong"].sum()),
        "n_arm5_right": int(hd["arm5_right"].sum()),
    }
    (OUT_DIR / "s5_head_down_evidence_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(grp)
    return {"reliable": True, "join_rate": join_rate, "summary": summary, "hd": hd}


# ===========================================================================
# SECTION 6 -- qualitative examples (blurred crops)
# ===========================================================================
def blur_head_region(img, frac=0.55):
    """Blur the top `frac` of the crop (head/face region) heavily. img: BGR np array."""
    import cv2
    h, w = img.shape[:2]
    cut = max(1, int(h * frac))
    region = img[:cut, :, :]
    k = max(15, (min(region.shape[0], region.shape[1]) // 2) * 2 + 1)
    blurred = cv2.GaussianBlur(region, (k, k), 0)
    out = img.copy()
    out[:cut, :, :] = blurred
    return out


def section6(join_result):
    print("=== Section 6: qualitative examples ===")
    import cv2
    df2 = load_arm_seed(ARMS["arm2"], 42)

    examples_spec = [
        ("turned_to_peer", "screen_oriented", "turned_to_peer misread as screen_oriented"),
        ("looking_away", "screen_oriented", "looking_away misread as screen_oriented"),
        ("head_down", "uncertain", "head_down / uncertain confusion"),
        ("uncertain", "head_down", "uncertain / head_down confusion (other direction)"),
    ]

    needed_vids = set(df2["video_id"].unique().tolist())
    idx, n_read, n_kept = build_labels_index(needed_vids)

    all_examples = []
    for true_c, pred_c, label in examples_spec:
        sub = df2[(df2["y_cls"] == true_c) & (df2["pred_cls"] == pred_c)]
        if len(sub) == 0:
            continue
        sub = sub.sort_values("conf", ascending=False)
        picked = sub.head(3)
        for _, row in picked.iterrows():
            key = (row["video_id"], int(row["seat_id"]), round(row["t"], 3))
            rec = idx.get(key)
            ex = {
                "failure_type": label,
                "true_cue": true_c,
                "pred_cue": pred_c,
                "video_id": row["video_id"],
                "seat_id": int(row["seat_id"]),
                "t_seconds": float(row["t"]),
                "confidence": float(row["conf"]),
                "prob_vector": {c: float(row[f"p_{c}"]) for c in CUE_CLASSES},
            }
            if rec is not None:
                ex["labels_tracked_activity"] = rec["activity"]
                ex["occluded"] = rec["occluded"]
                ex["det_conf"] = rec["det_conf"]
                ex["head_span_px"] = rec["head_span_px"]
                ex["caption"] = rec["caption"]
                ex["file_name"] = rec["file_name"]
            else:
                ex["join_failed"] = True
            all_examples.append(ex)

    # write blurred crops where we have a file_name
    crop_out_records = []
    for ex in all_examples:
        fn = ex.get("file_name")
        if not fn:
            continue
        src = CROPS_ROOT / fn
        if not src.exists():
            ex["crop_saved"] = False
            continue
        img = cv2.imread(str(src))
        if img is None:
            ex["crop_saved"] = False
            continue
        blurred = blur_head_region(img, frac=0.55)
        out_name = f"{ex['failure_type'].replace(' ', '_').replace('/', '-')}__{ex['video_id']}__seat{ex['seat_id']}__t{ex['t_seconds']:.3f}.jpg".replace(" ", "_")
        out_path = CROP_DIR / out_name
        cv2.imwrite(str(out_path), blurred)
        ex["crop_saved"] = True
        ex["crop_path"] = str(out_path.relative_to(ROOT))
        crop_out_records.append(ex)

    (OUT_DIR / "s6_qualitative_examples.json").write_text(json.dumps(all_examples, indent=2, default=str))
    print(f"Saved {len(crop_out_records)} blurred example crops.")
    for ex in all_examples:
        print(ex.get("failure_type"), ex.get("video_id"), ex.get("seat_id"), ex.get("t_seconds"),
              "conf=", round(ex.get("confidence", 0), 3), "crop_saved=", ex.get("crop_saved", False))
    return all_examples


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    verify_inner_val_only()
    s1 = section1()
    s2 = section2()
    s3 = section3()
    s4 = section4()
    s5 = section5()
    s6 = section6(s5)
    print("DONE")
