# Cue rules v2 — protocol

**Status: RUN, FAILED, and closed.** Predictions in §4 were written before any
training run and are the standard the outcome is read against. §4b records the
result: the gate failed, and the line of work was stopped on 2026-09-10.
The diagnosis in §1 stands and is worth citing; the *intervention* does not.

---

## 1. The finding

`looking_away` is not a behaviour class. It is, roughly half the time, a
synonym for "the head is down", produced by a gap in the annotation vocabulary.

**`attention_target` is a recoding of `gaze_direction`, not an observation.**
On the 1,000-crop stratified gold-candidate sample it is **96.0%** predictable
from gaze alone: `laptop`→`device` 100%, `own_desk`→`own_work` 100%,
`peer`→`peer` 100%, `phone`→`device` 100%, `teacher_or_board`→`instruction`
98.8%, `away_or_window`→`distracted` 98.4%.

**Its one non-redundant cell is the defect.** `gaze_direction == "down"` maps to
`attention_target == "distracted"` on **94.1%** of records, and `distracted` is
a v1 trigger for `looking_away`. So *looking down* fires *looking away*.

Of the 372 records where the v1 `looking_away` rule fires, **359 (96.5%)** are
driven by `distracted` and **181 (48.7%)** by `distracted` **alone**. Of those
181: gaze is `down` on 85.1% and `away_or_window` on **0.0%**; activity is
`head_down_sleeping` on 70.7%; and **81.2%** end up labelled `head_down` once
precedence is applied.

**The human annotator does the same thing**, so this is a property of the
vocabulary rather than of the VLM. On the 754 usable crops in
`event_gold_bundle/gold_annotations_Admin.jsonl`: `gaze == down` → `distracted`
on **231/232 = 99.6%**; the v1 `looking_away` rule fires on 31.7% of crops and
**100% of those firings are driven by `distracted`** (0.8% by actual gaze
evidence); `looking_away` is true by its own rule **119.5×** more often than
precedence lets it be the label; and **231/232 = 99.6%** of human `head_down`
records also carry `looking_away` as a candidate.

There is no `attention_target` value meaning "looking down at own work", so a
head-down student can only be called `distracted`. The cue rule turns that into
`looking_away`, which sits directly below `head_down` in precedence and
therefore inherits exactly the frames it cannot be told apart from.

### What one mechanism explains

| observation | explanation |
|---|---|
| `looking_away` F1 0.226 at AUROC 0.767 | the class is a mixture of two visually opposite configurations |
| head yaw (553→556) buys only +0.041 | yaw separates averted-from-screen; half the class is head-*down* |
| PRODEN → `head_down` 0.000, "772 of 773" | reproduced here at 99.6% **on human labels** — a rule defect, not a PLL identifiability quirk |
| 13.7% multi-fire | 31.7% on the human set; it is the `down`/`distracted` collision |
| `_reliable` taxonomies had to abstain on these two classes | they were abstaining on a mis-specified class, not a hard one |
| three branches of feature engineering returning nulls | no encoder resolves a class whose definition is internally contradictory |

### Caveats that bound the claim

* Sample numbers come from a **stratified** manifest that oversamples rare
  activities. They are not corpus prevalences. §5 step 1 re-measures corpus-wide
  and **no number above may be cited until it does**.
* The human set is **one video, one annotator, 754 crops**, sampled as
  contiguous track segments rather than randomly.
* The annotation tool **pre-fills pseudo-labels**, so human agreement on
  `down → distracted` is partly acceptance rather than independent judgement.
  Either reading supports the conclusion: the field carries no independent
  information.

---

## 2. What v2 changes

`attention_target` is dropped from every cue rule. Nothing else moves.

| rule | v1 | v2 |
|---|---|---|
| `looking_away` | gaze==away_or_window **or activity==looking_away or target==distracted** | gaze==away_or_window or activity==looking_away |
| `turned_to_peer` | activity==talking_to_peer or talking or gaze==peer **or target==peer** | activity==talking_to_peer or talking or gaze==peer |
| `screen_oriented` | activity in TASK **or (gaze in TASK_GAZES and target in TASK_TARGETS)** | activity in TASK or gaze in TASK_GAZES |
| no-signal gate | gaze/**target**/engagement/activity all unknown-ish | gaze/engagement/activity unknown-ish |
| `phone_use`, `head_down` | — | **unchanged** (never read the field) |

`target==peer` is dropped as exactly redundant with `gaze==peer` (129/129
co-occurrence). `screen_oriented` loses a conjunct that only restated the gaze
test. Dropping `target=="unknown"` from the gate sends unreadable crops to
`uncertain` instead of to `looking_away` — which is what the annotation
guideline says they are.

Measured on the stratified sample: **26/1000 hard labels change**;
`looking_away` purity (share whose gaze actually is `away_or_window`) rises
**74.1% → 90.6%**; `head_down` records ambiguous with `looking_away` fall
**100% → 26.1%**; the under-crediting ratio falls **2.60× → 1.63×**. On the
human set, **0 hard labels change** and the ambiguity falls **99.6% → 0.0%**.

**v1 is frozen.** It is the rule set every published number and every built
sequence was produced under. `map_record` still defaults to v1;
`test_taxonomy_v2.py` asserts v1 against the inlined legacy oracle on all 1,984
real records available offline.

---

## 3. Comparability — the part that must not be fudged

v2 is a **different target**. A macro-F1 against it is not, on its own,
comparable to 0.479, and presenting it as "the model improved" would be the
same error as quoting a coarse-taxonomy score as a six-class result.

Two things make the comparison defensible where the coarse taxonomies could not
be:

1. It is still **six classes at 100% coverage**, and only ~2.6% of hard labels
   move. The corpus-wide rate is measured and reported alongside every number.
2. Features are **not rebuilt**. The label set is a sidecar
   (`build_cue_labels.py`) applied at load time, so a v1 run and a v2 run read
   the *identical* feature file. The target is provably the only difference.

That enables a four-cell cross-evaluation, and **the headline contrast is the
one that removes the incomparability entirely**:

| | scored on v1 | scored on v2 |
|---|---|---|
| **trained on v1** | `A→A` = the 0.479 baseline | `A→B` — how much of B→B is just an easier target? |
| **trained on v2** | **`B→A` — the comparable contrast** | `B→B` = the repaired task |

`B→A` vs `A→A` is measured on the **original** target, with the same features,
the same architecture and the same seeds; only the *training* labels differ. It
is also **conservative**: B is penalised on the frames where v1 and v2
legitimately disagree. If `B→A > A→A` under that handicap, the repair improved
the model at the *original* task, and no comparability caveat is needed.

`run_eval.py` defaults to the label set recorded in the checkpoint's spec, so a
model is scored against its own target unless a cross-evaluation is asked for
explicitly, and it prints a CROSS-EVALUATION banner when it is.
`aggregate.discover` keys on the label set, so v1 and v2 runs cannot be pooled
into one mean-over-seeds by accident.

---

## 4. Pre-registered predictions

Written before any run. **P3 is the gate.**

| # | prediction | falsified if |
|---|---|---|
| P1 | corpus-wide, `attention_target` is ≥90% predictable from gaze, and `gaze==down → distracted` ≥85% | below those — §1's justification weakens and v2 must be re-argued before training |
| P2 | `head_down` ∩ `looking_away` ambiguity falls from >95% to <40% corpus-wide | above 40% |
| **P3** | **`B→A` − `A→A` ≥ +0.01, CI excluding zero**, expected +0.01 to +0.04 | **≤ 0, or CI spans zero** |
| P4 | `looking_away` F1 on the v2 target ≥ 0.30, against 0.226 | below 0.30 |
| P5 | `turned_to_peer` does **not** materially improve (< +0.02) | — registered so a null there is not read as a failure; v2 only drops one redundant disjunct from its rule |
| P6 | v2 raises pseudo-human agreement on the gold set (Stage 2, **model-free**) | falls — then v2 is wrong and P3 must not be believed |
| P7 | PRODEN on v2 no longer collapses `head_down` to 0.000 | it still collapses |

### 4a. Outcome of P1 and P2 — corpus-wide, 283,913 records

`outputs/attention_target_audit.json`. Recorded **before any training run**, so
the revised expectation below cannot be a post-hoc rationalisation.

**P1 — PASSES, but one clause only barely.**

| clause | threshold | sample | **corpus** |
|---|---|---|---|
| `attention_target` predictable from gaze | ≥90% | 96.0% | **98.1%** |
| `gaze==down` → `distracted` | ≥85% | 94.1% | **85.3%** |

The first clause passes comfortably and is *stronger* corpus-wide than in the
sample. The second passes **by 0.3 points**. The stratified sample overstated it
(94.1% vs 85.3%) and that must be said whenever the figure is quoted: the
`down → distracted` collapse is real and dominant, but it is not the near-total
map the sample implied.

**P2 — PASSES decisively.** `head_down` ∩ `looking_away` ambiguity:
**99.3% → 0.6%**, against a predicted <40%. A 165× reduction, and the largest
number in the audit.

**The sample overstated the label churn by 5.5×**, exactly as its stratification
warned it would:

| metric | sample | **corpus** |
|---|---|---|
| hard labels changed | 2.60% | **0.47%** (1,336 / 283,913) |
| multi-fire rate | 25.5% | 8.4% |
| `looking_away` purity | 74.1% → 90.6% | 82.1% → 88.2% |
| under-crediting ratio | 2.60× → 1.63× | 2.21× → 1.14× |
| `turned_to_peer` frames moved | 0 | −5 (P5's premise confirmed) |

**Revised expectation for P3, written now.** The cross-entropy loss reads `y`,
not `y_cand`, and only **0.47%** of `y` moves. `looking_away` loses 1,328 of
19,067 frames — its least coherent 7% — which raises purity 82.1% → 88.2%. That
is one class, one sixth of the macro average. So the honest expectation for
`B→A − A→A` is now **+0.005 to +0.015**, not the +0.01 to +0.04 registered in
§4. **The P3 threshold is NOT moved** — it stands at ≥ +0.01 with a CI
excluding zero, and it now sits at the top of the plausible range rather than
the middle. It is more likely than not to fail.

**Where the value moved instead: P7.** The documented PRODEN collapse —
`head_down` F1 **0.000 in every run**, because 772 of its 773 records also
carried `looking_away` as a candidate — is *mechanistically removed* by a
finding derived independently of it. Partial-label learning reads `y_cand`,
which is where the 99.3% → 0.6% change lands, so the arm that barely moves
under cross-entropy is the one with a large, pre-explained prior here. PRODEN
scored 0.378 ± 0.015 with one class pinned at zero; recovering `head_down` to
anywhere near its cross-entropy value (~0.54) is worth ~+0.09 macro on that arm
alone. **P7 is now the high-expected-value run, not P3**, and a v1 PRODEN arm is
added so the contrast is matched under one trainer rather than against the
archived number.

**If P3 fails**, v2 is reported as a **diagnostic finding only** — the mechanism
is real either way, and it still explains PRODEN, the yaw null and the
abstaining taxonomies — and **not** as a performance improvement. That outcome
is written down here so it cannot be re-described later.

---

### 4b. OUTCOME — the gate fails

12 runs, MS-TCN, `556_hp`, `llmstu_sequences_full_det`, seeds 42/43/44,
validation, 100% coverage. Raw: `work_dirs/thesis/cue_v2/*/eval_val*/metrics.json`.

| arm (macro-F1) | s42 | s43 | s44 | mean |
|---|---|---|---|---|
| `v1_base` → v1 (**A→A**) | 0.5109 | 0.4952 | 0.4777 | 0.4946 |
| `v2_base` → v1 (**B→A**, the gate) | 0.4941 | 0.4804 | 0.4856 | 0.4867 |
| `v1_base` → v2 (A→B) | 0.5067 | 0.4913 | 0.4744 | 0.4908 |
| `v2_base` → v2 (B→B) | 0.4920 | 0.4776 | 0.4825 | 0.4840 |
| `v1_proden` | 0.4904 | 0.4912 | 0.4680 | 0.4832 |
| `v2_proden` | 0.4721 | 0.4851 | 0.4443 | 0.4672 |

**P3 FAILS.** `B→A − A→A` paired within seed: −0.0168, −0.0148, +0.0079 →
**−0.0079, 1/3 seeds**, against a registered ≥ +0.01 with a CI excluding zero.
Negative, not merely null. Training on v2 does not help on v2 either:
`B→B − A→B` is −0.0147, −0.0137, +0.0081 → **−0.0068, 1/3**. Calibration
degrades (ECE 0.017–0.027 → 0.029–0.046).

**P4 FAILS, in the opposite direction to the prediction.** Per-class F1 for
`looking_away`:

| | `looking_away` |
|---|---|
| `v1_base` → v1 | **0.2317** |
| `v1_base` → v2 (target change only) | 0.2209 |
| `v2_base` → v1 (training change, same target) | **0.1570** |
| `v2_base` → v2 | **0.1521** |

P4 predicted ≥ 0.30; the result is 0.1521, a 34% relative drop. The
decomposition is clean: re-scoring the v1-trained model against v2 costs only
−0.011, so the target moving is not the cause; *training* on v2 costs −0.075 on
the same v1 target.

**Why, and it is the useful part.** v1's `looking_away` contained ~7% frames
whose gaze was `down` and whose activity was `head_down_sleeping`. Those are
visually distinctive — face not visible, head lowered — so the model could
actually get them right. v2 strips exactly those and leaves only genuine averted
gaze, which 556-dim CLIP features demonstrably cannot resolve. Purity rose
82.1% → 88.2%; learnability fell. `v2_base` accuracy *rises* (0.717 → 0.765 at
s42) while balanced accuracy *falls* (0.542 → 0.502): the model leans harder on
the majority class.

**So the published 0.226 for `looking_away` was partly propped up by
mislabelled head-down frames.** Genuine averted-gaze detection scores ~0.15.
That makes the §1 diagnosis worse, not better, and it is the one durable result
here.

**P5 HOLDS as registered.** `turned_to_peer` 0.1895 → 0.1799 — no material
move, as predicted, because v2 drops only one redundant disjunct from its rule.

**P7 INCONCLUSIVE — the control is invalid.** `v1_proden` scores `head_down`
**0.6827**, not the documented **0.000**, and macro 0.4832 against the archived
0.378. Probable cause: `patch_pose_columns.py` does not write `y_cand`, and
`llmstu_sequences_full_det` is a patched build, so `data.py` falls back to
one-hot candidates and `--partial-labels` silently degrades to cross-entropy.
If so, `v1_proden` never ran PRODEN and the arm measured "no candidates vs
candidates", not "v1 rules vs v2 rules".

**This is unverified.** Confirm before citing the archived PRODEN number
anywhere:

```bash
cd LLMDet && python -c "
import numpy as np
from pathlib import Path
ps = sorted(Path('../grounding_data/llmstu_sequences_full_det').glob('*/sample_*.npz'))[:200]
print(sum('y_cand' in np.load(p).files for p in ps), '/', len(ps), 'carry y_cand')"
```

`0/200` would mean every PRODEN number produced on this build — including
**FINDINGS.md's 0.378 / `head_down` 0.000** — is a mislabelled cross-entropy
run. `train.py` should abort when `--partial-labels` is passed and no sequence
carries a multi-candidate row, rather than degrading silently; that guard is
**not yet implemented**.

### 4c. Registered outcome

Per §4's own terms, v2 is reported as a **diagnostic finding only** and **not**
as a performance improvement. The vocabulary defect is real, corpus-wide, and
reproduced in human labels; its blast radius on hard labels is 0.47%; removing
it *lowers* measured performance because the frames it mislabelled were the ones
the model could actually see.

Combined with Branch C's four registered nulls, both "better features" and
"better label rules" are now measured dead ends for this task. What remains
unmeasured is label *quality* — the ceiling in §7, which no experiment in this
project has ever established.

---

## 5. Commands (HPC)

Everything below is CPU-only except the training runs. No sequence rebuild, no
feature extraction, no new weights.

**Step 1 — corpus-wide audit. Do this first; P1 gates the rest.**

```bash
python tools/audit_attention_target.py \
  --labels grounding_data/llmstu_tools/outputs/labels_tracked.jsonl \
  --human event_gold_bundle/gold_annotations_Admin.jsonl \
  --out outputs/attention_target_audit.json
```

**Step 2 — build the label sidecars.** One per sequence root; the tool binds to
the root and validates against it. Each build re-derives the v1 labels and
**aborts unless they equal the labels already stored in every npz**, which
proves the replay alignment and this code's reading of the rules before any v2
label is trusted.

```bash
cd LLMDet
python -m attention.thesis_eval.build_cue_labels \
  --sequence-root ../grounding_data/llmstu_sequences_full_det \
  --manifest ../grounding_data/llmstu_seq_split_manifest.json \
  --ruleset v2 --out ../grounding_data/cue_labels_v2_full_det.npz
```

**Step 3 — train, 3 seeds each.** `A` is a re-run of the baseline as a control
on this code; it must reproduce 0.479 ± 0.007.

```bash
cd LLMDet
CUE=../grounding_data/cue_labels_v2_full_det.npz
ROOT=../grounding_data/llmstu_sequences_full_det
for S in 42 43 44; do
  CUDA_VISIBLE_DEVICES=$((S-42)) python -m attention.thesis_eval.train \
    --experiment-id v1_base_s$S --model mstcn --feature-config 556_hp \
    --taxonomy cue6 --seed $S --epochs 90 --sequence-root $ROOT \
    --output-dir work_dirs/thesis/cue_v2/v1_base_s$S --device cuda:0 &
done; wait

for S in 42 43 44; do
  CUDA_VISIBLE_DEVICES=$((S-39)) python -m attention.thesis_eval.train \
    --experiment-id v2_base_s$S --model mstcn --feature-config 556_hp \
    --taxonomy cue6 --seed $S --epochs 90 --sequence-root $ROOT \
    --cue-labels $CUE \
    --output-dir work_dirs/thesis/cue_v2/v2_base_s$S --device cuda:0 &
done; wait
```

**Step 4 — the four-cell cross-evaluation.** `--cue-labels ''` forces the
labels stored in the sequences (v1); omitting the flag uses whatever the
checkpoint was trained on.

```bash
cd LLMDet
for S in 42 43 44; do
  D=work_dirs/thesis/cue_v2
  # A->A and B->B: each model on its own target
  python -m attention.thesis_eval.run_eval --ckpt $D/v1_base_s$S/checkpoints/best.pth \
    --split val --out $D/v1_base_s$S/eval_val --device cuda:0
  python -m attention.thesis_eval.run_eval --ckpt $D/v2_base_s$S/checkpoints/best.pth \
    --split val --out $D/v2_base_s$S/eval_val --device cuda:0
  # B->A: THE GATE. v2-trained model scored on the ORIGINAL target.
  python -m attention.thesis_eval.run_eval --ckpt $D/v2_base_s$S/checkpoints/best.pth \
    --split val --cue-labels '' --out $D/v2_base_s$S/eval_val_on_v1 --device cuda:0
  # A->B: how much of B->B is the target simply being easier?
  python -m attention.thesis_eval.run_eval --ckpt $D/v1_base_s$S/checkpoints/best.pth \
    --split val --cue-labels "$CUE" --out $D/v1_base_s$S/eval_val_on_v2 --device cuda:0
done
```

**Step 5 — PRODEN on the repaired candidates (P7).** The documented collapse was
caused by `head_down` having essentially no unambiguous example; v2 is expected
to remove that.

```bash
for S in 42 43 44; do
  python -m attention.thesis_eval.train \
    --experiment-id v2_proden_s$S --model mstcn --feature-config 556_hp \
    --taxonomy cue6 --seed $S --epochs 90 --sequence-root $ROOT \
    --cue-labels $CUE --partial-labels \
    --output-dir work_dirs/thesis/cue_v2/v2_proden_s$S --device cuda:0
done
```

Optional, same shape: repeat steps 2–4 against `llmstu_sequences_head` with
`--feature-config 1074_hp_head` to see whether the repair composes with the
+0.036 head stream.

Run `python -m pytest attention/tests/ -q` from `LLMDet/` first. `test_thesis_eval.py`
and `test_branch_c_provenance.py` need `mmcv`.

---

## 6. Tarball for the HPC

```
LLMDet/attention/taxonomy.py                            (modified)
LLMDet/attention/sequence_builder.py                    (modified)
LLMDet/attention/thesis_eval/patch_pose_columns.py      (modified)
LLMDet/attention/thesis_eval/data.py                    (modified)
LLMDet/attention/thesis_eval/train.py                   (modified)
LLMDet/attention/thesis_eval/run_eval.py                (modified)
LLMDet/attention/thesis_eval/aggregate.py               (modified)
LLMDet/attention/thesis_eval/build_cue_labels.py        (new)
LLMDet/attention/tests/test_taxonomy_v2.py              (new)
LLMDet/attention/tests/test_cue_labels.py               (new)
tools/audit_attention_target.py                         (new)
tools/gold_annotator/measure_ceiling.py                 (new)
docs/CUE_RULES_V2.md                                    (new)
```

No model weights. No data. Extract at the repo root.

---

## 7. Stage 2 — the ceiling

Independent of the runs above and worth more than any of them: **no frame-level
human agreement number exists for the six cue classes**, so no result in this
project has a denominator.

The stratified 1,000-crop manifest is sampled and **unannotated** — it shares
only 3 `file_name`s with the existing human file, which covers one video and was
drawn as contiguous track segments for event evaluation. The tool is built,
pre-fills pseudo-labels and is keyboard-driven.

```bash
cd tools/gold_annotator
python serve.py --manifest ../../grounding_data/llmstu_tools/outputs/gold_candidates.jsonl \
                --annotator wael --port 8765
# ssh -L 8765:localhost:8765 <hpc>   then open http://localhost:8765/

python measure_ceiling.py \
  --human gold_annotations_wael.jsonl \
  --pseudo ../../grounding_data/llmstu_tools/outputs/gold_candidates.jsonl \
  --out ../../outputs/label_ceiling.json
```

`measure_ceiling.py` scores the pseudo-labels **as if they were a model**, with
`thesis_eval.metrics`, so the result sits on the same axis as every model
number. It reports the ceiling under **both** rule versions — P6, a test of the
repair with no training run involved.

A second annotator on the same manifest gives Cohen's kappa via the existing
`compute_agreement.py`. Until that exists, the ceiling is one annotator's and
must be quoted as such.

**This decides Stage 3.** If the pseudo-labels agree with a human at ~0.60
macro-F1, then 0.479 is already ~80% of achievable, no amount of A100 time
reaches 0.8, and the defensible result is that the task is label-limited. If
agreement is high, the headroom is real and end-to-end perception training is
worth the compute.
