# Label inventory

Every label in the system, in one place. There are **four layers**, and only
one of them is the six cue classes most of the writing talks about. They are
not alternatives — a single student-crop carries a label at every layer.

Nothing here is authored. Each table is transcribed from the file named above
it, and `attention/tests/test_labels_doc.py` fails if this file and that file
disagree. Change the code, not this document, and re-run the test.

| layer | what | defined in |
|---|---|---|
| 0 | detector target | `LLMDet/configs/attention_runtime.yaml` |
| 1 | annotation schema (the actual dataset labels) | `tools/gold_annotator/vocab.py` |
| 2 | six cue classes (derived) | `LLMDet/attention/taxonomy.py` |
| 2b | nine cue classes (derived, a **second** projection) | `LLMDet/attention/taxonomy.py` |
| 3 | regrouped taxonomies | `LLMDet/attention/taxonomy.py` |

---

## Layer 0 — Detector

One foreground category. The architecture is open-vocabulary
(GroundingDINO/LLMDet), but **the fine-tuned checkpoint is not** — see the
warning below.

| | |
|---|---|
| text prompt | `a student sitting` — **inert, see below** |
| output | boxes only, no class — score >= 0.10, at most 80 per frame |

Source: `attention_runtime.yaml:49`. Test R@1 **0.6462** under the closed
protocol in `TEST_SPLIT_PROTOCOL.md`.

### ⚠ The prompt does nothing (measured 2026-09-11)

This document previously said the "label" is a phrase. It is not, for this
checkpoint. Fine-tuning on the single category collapsed the text conditioning:
the detector returns the **same student boxes whatever it is asked for**.

Measured over 12 frames of `0325.mp4`, share of each prompt's boxes that match
a box from `a student sitting` at IoU >= 0.9:

| prompt | detections | matches a student box | mean IoU |
|---|---|---|---|
| `a mobile phone` | 41 | **98%** | 0.975 |
| `qwertyuiop` | 51 | **92%** | 0.918 |
| `a laptop` | 65 | 89% | 0.897 |
| `a fire hydrant` | 59 | 88% | 0.881 |
| `xyzzy` | 72 | 82% | 0.818 |
| `a potted plant` | 80 | 71% | 0.712 |

A nonsense string retrieves students as well as the real phrase does. **R@1
0.6462 is unaffected** — it was always measured on students — but the detector
must be described as a one-class student detector, not as an open-vocabulary one,
and the prompt in the runtime config is documentation rather than a control.

Consequence: it cannot be re-prompted to find phones or laptops. An open-vocab
object pass needs a *pretrained* checkpoint; three sit unused in
`huggingface/mm_grounding_dino/`. FINDINGS §21.

---

## Layer 1 — Annotation schema

Ten fields per student-crop plus one integer, closed vocabulary. This is what
the VLM pseudo-labeller (`Qwen3.5-27B`) emitted and what the human annotator
edited in the gold tool. **These are the only labels anyone assigned**;
everything below is computed from them.

Source: `tools/gold_annotator/vocab.py` (order = cycling order in the UI).

### Categorical — 6 fields, 43 values

| field | values | n |
|---|---|---|
| `activity` | `listening`, `using_laptop`, `head_down_sleeping`, `looking_away`, `reading`, `using_phone`, `talking_to_peer`, `writing_notes`, `eating_drinking`, `raising_hand`, `other` | 11 |
| `gaze_direction` | `laptop`, `teacher_or_board`, `own_desk`, `down`, `away_or_window`, `phone`, `peer`, `unknown` | 8 |
| `attention_target` | `device`, `instruction`, `own_work`, `distracted`, `peer`, `unknown` | 6 |
| `engagement_level` | `engaged`, `partially_engaged`, `disengaged`, `unknown` | 4 |
| `posture` | `upright`, `leaning_forward`, `leaning_back`, `head_down`, `turned_away`, `slumped`, `unknown` | 7 |
| `hand_state` | `unknown`, `on_face`, `on_desk_idle`, `on_phone`, `gesturing`, `writing`, `raised` | 7 |

### Boolean — 4 fields

`phone_visible`, `laptop_visible`, `talking`, `occluded`

### Integer

`face_kpts` — 0 to 3 visible face keypoints. `face_kpts <= 2` **and**
`occluded` is the "visually unverifiable" test (`UNCERTAIN_FACE_KPTS = 2`).

### Corpus counts

Source: `grounding_data/llmstu_tools/outputs/dedup_report.json` (HPC-only, so
the file is not in the repo — these are transcribed).

The record count moves in three steps, and the histograms sum exactly to each:

```
283,913  input_records
 -1,178  dropped_seat_noise
-13,239  dropped_occluded_face2      (occluded AND face_kpts <= 2)
269,496  activity_hist_before        <- sums exactly
138,531  after_dedup                 <- sums exactly
 84,950  final                       (stride 10.0 s, majority cap 0.50)
```

| activity | before dedup | after dedup | final |
|---|---|---|---|
| `listening` | 109,041 | 53,072 | 23,474 |
| `using_laptop` | 91,821 | 42,984 | 19,001 |
| `looking_away` | 16,239 | 11,692 | 11,692 |
| `head_down_sleeping` | 14,674 | 6,250 | 6,250 |
| `reading` | 10,543 | 6,050 | 6,050 |
| `using_phone` | 8,969 | 4,094 | 4,094 |
| `other` | 8,708 | 7,112 | 7,112 |
| `talking_to_peer` | 5,705 | 4,784 | 4,784 |
| `writing_notes` | 2,157 | 1,187 | 1,187 |
| `eating_drinking` | 1,526 | 1,216 | 1,216 |
| `raising_hand` | 113 | 90 | 90 |

Only `listening` and `using_laptop` lose anything in the final step: the
majority cap is what trims them, and it is why the corpus is less
`screen_oriented`-dominated than the raw video.

Per-field counts exist only for `activity`. There is no corpus histogram for
`gaze_direction`, `posture`, `hand_state` or the booleans — if one is needed,
it has to be measured on the HPC.

---

## Layer 2 — The six cue classes

**Derived, never annotated.** `CUE_CLASSES` in `attention/taxonomy.py` is a
projection of Layer 1 through a precedence rule: the conditions are evaluated
in order and the first one that fires becomes the label.

Precedence: `uncertain` -> `phone_use` -> `head_down` -> `turned_to_peer` ->
`looking_away` -> `screen_oriented`. It encodes "a specific off-task cue beats
a generic on-task one" — a laptop user with a visible phone is `phone_use`.
`uncertain` outranks everything because a student who cannot be seen supports
no cue at all.

| id | cue | fires when (ruleset v1) |
|---|---|---|
| 0 | `screen_oriented` | `activity` in {`using_laptop`, `listening`, `reading`, `writing_notes`} OR (`gaze_direction` in {`laptop`, `teacher_or_board`, `own_desk`, `down`} AND `attention_target` in {`device`, `instruction`, `own_work`}) |
| 1 | `looking_away` | `gaze_direction == away_or_window` OR `activity == looking_away` OR `attention_target == distracted` |
| 2 | `head_down` | `activity == head_down_sleeping` OR `posture` in {`head_down`, `slumped`} |
| 3 | `turned_to_peer` | `activity == talking_to_peer` OR `talking` OR `gaze_direction == peer` OR `attention_target == peer` |
| 4 | `phone_use` | `activity == using_phone` OR `phone_visible` OR `gaze_direction == phone` OR `hand_state == on_phone` |
| 5 | `uncertain` | (`occluded` AND `face_kpts <= 2`) OR no usable orientation signal — and also the fallback when nothing else fires |

### Prevalence and performance

Validation split, 42,702 frames, `ff_det/mstcn_553_ff_s42`. Source:
`LLMDet/work_dirs/thesis/ff_det/mstcn_553_ff_s42/eval_val/metrics.json`.
**The test split is unspent** (`TEST_SPLIT_PROTOCOL.md`); these are not test
numbers.

| cue | support | share | F1 | AUPRC | AUROC |
|---|---|---|---|---|---|
| `screen_oriented` | 32,557 | 76.24% | 0.845 | 0.935 | 0.833 |
| `head_down` | 2,961 | 6.93% | 0.673 | 0.773 | 0.953 |
| `looking_away` | 2,905 | 6.80% | 0.210 | 0.159 | 0.766 |
| `uncertain` | 2,038 | 4.77% | 0.529 | | |
| `phone_use` | 1,313 | 3.07% | 0.525 | | |
| `turned_to_peer` | 928 | 2.17% | 0.167 | | |

Two facts a reader needs alongside this table:

* **`screen_oriented` covers screen AND instructor/board orientation.** In a
  computer-lab lecture both are on-task. It is deliberate, documented in
  `taxonomy.py`, and it makes this the broadest class at 76% of frames.
* **`looking_away` and `turned_to_peer` are not reliable classes.** Their
  AUPRC lift over base rate is 2.3x and 4.7x against 11-14x for the others.
  `docs/CUE_RULES_V2.md` explains the mechanism: `attention_target` is 96%
  predictable from `gaze_direction`, and its one non-redundant cell
  (`gaze == down` -> `distracted`, 94.1%) makes *looking down* fire *looking
  away*. Roughly half of `looking_away` is a synonym for "head is down".

### Rule versions

`RULESETS = ("v1", "v2")`, `DEFAULT_RULESET = "v1"`.

* **v1** — the default. Every published number and every built sequence was
  produced under it and it must not change.
* **v2** — drops `attention_target` from every rule (`looking_away`,
  `turned_to_peer`, `screen_oriented`, and the no-signal gate; `phone_use` and
  `head_down` never read it). **Run, failed, closed on 2026-09-10.** The
  diagnosis in `docs/CUE_RULES_V2.md` stands; the intervention does not.

### Retired class

`idle_other` was id 5 in a 7-class version until 2026-07-29. It fired on 16 of
283,913 records and its inverse-frequency weight destabilised training, so it
folds into `uncertain`. Legacy sequences are remapped at load time by
`LEGACY_7CLASS_REMAP` (5->5, 6->5).

### Partial labels

`candidate_set()` returns *every* cue a record supports, not just the winner.
13.7% of records fire more than one rule, and `looking_away` is true by its own
rule 2.74x more often than precedence lets it be the label (1,313 against 479).
PRODEN over these candidates was tried and failed mechanically — 772 of 773
`head_down` records also carry `looking_away`, so the class has no unambiguous
anchor and collapses to F1 0.000.

---

## Layer 2b — The nine cue classes (`cue9`)

**Derived, never annotated**, like Layer 2 — and a *sibling* of it, not a
refinement. `CUE9_CLASSES` in `attention/taxonomy.py` is a **second projection
of Layer 1** with its own precedence list.

Why it cannot be a Layer-3 taxonomy: every entry in `TAXONOMIES` regroups the
six cue ids, and `screen_oriented` has already discarded the fields that tell
writing from reading. A split needs the annotation record, so it needs its own
label build (`build_cue_labels --label-space cue9`). Features are untouched, so a
cue6-vs-cue9 comparison differs in the target and in nothing else.

`screen_oriented` is replaced by four classes; the other five cue names are kept
unchanged, so every statement about `phone_use` or `turned_to_peer` still means
the same thing. One kept rule changes: **`head_down` also fires on
`gaze_direction == down`**.

| id | cue | fires when |
|---|---|---|
| 0 | `writing_notes` | `activity == writing_notes` |
| 1 | `using_laptop` | `activity == using_laptop` AND `attention_target` in {`device`, `instruction`, `own_work`} |
| 2 | `reading` | `gaze_direction` in {`laptop`, `own_desk`} OR `activity == reading` |
| 3 | `listening` | `gaze_direction == teacher_or_board` OR `activity == listening` |
| 4 | `looking_away` | unchanged from Layer 2 |
| 5 | `head_down` | Layer 2 **plus** `gaze_direction == down` |
| 6 | `turned_to_peer` | unchanged from Layer 2 |
| 7 | `phone_use` | unchanged from Layer 2 |
| 8 | `uncertain` | unchanged from Layer 2, and still the fallback |

Precedence is the Layer-2 order with the four on-task rules in place of
`screen_oriented`: `uncertain` -> `phone_use` -> `head_down` -> `turned_to_peer`
-> `looking_away` -> `writing_notes` -> `using_laptop` -> `reading` ->
`listening`.

### Why `head_down` gains `gaze == down`

This is the repair `RULESET_V2_RATIONALE` argues for, applied where it belongs.
`gaze == down` used to reach `looking_away` through
`attention_target == distracted`, so *looking down* fired *looking away*. Because
`head_down` outranks `looking_away`, moving the condition up fixes it directly;
and `gaze == down` simultaneously leaves the on-task gaze set (`reading` is
{`laptop`, `own_desk`}, not {..., `down`}), so the two changes agree rather than
compete.

### Prevalence

All 283,913 records of `labels_tracked.jsonl`, measured — not estimated.

| cue6 | share | | cue9 | share |
|---|---|---|---|---|
| `screen_oriented` | 75.68% | -> | `using_laptop` | 31.62% |
| | | | `listening` | 30.41% |
| | | | `reading` | 11.93% |
| | | | `writing_notes` | 0.75% |
| `looking_away` | 6.72% | | `looking_away` | 6.03% |
| `head_down` | 5.20% | | `head_down` | 6.87% |
| `uncertain` | 5.03% | | `uncertain` | 5.03% |
| `phone_use` | 4.78% | | `phone_use` | 4.78% |
| `turned_to_peer` | 2.60% | | `turned_to_peer` | 2.59% |

**The largest class falls from 75.7% to 31.6%**, which is the point of the split.
`head_down` gains 4,720 records — 1,934 from `looking_away` and 2,786 from
`screen_oriented` (frames whose gaze was `down` with a task-consistent target).

Two things to watch in any cue9 result:

* **`writing_notes` is 0.75%** — 2,119 records, ~1,187 after dedup. `idle_other`
  was retired at 16 records because its inverse-frequency weight destabilised
  training; this is 130x larger, but it is still the thin class.
* **48.2% of records fire more than one cue9 rule**, against 13.7% for cue6,
  because the four on-task classes genuinely overlap (gaze on the laptop fires
  both `using_laptop` and `reading`). A partial-label objective over cue9 is
  therefore a different proposition from one over cue6, and the PRODEN
  identifiability argument would have to be re-checked before trying it.

---

## Layer 3 — Regrouped taxonomies

`TAXONOMIES` in `attention/taxonomy.py`. Each regroups the six cues; classes a
taxonomy will not predict map to `IGNORE_LABEL = -100`, which removes them from
the loss **and** from the metrics.

| name | classes | abstains on |
|---|---|---|
| `cue6` | `screen_oriented`, `looking_away`, `head_down`, `turned_to_peer`, `phone_use`, `uncertain` | |
| `onoff` | `on_task`, `off_task` | |
| `onoff_reliable` | `on_task`, `off_task` | `looking_away`, `turned_to_peer` |
| `coarse3_reliable` | `screen_oriented`, `down_or_hidden`, `phone_use` | `looking_away`, `turned_to_peer` |
| `cue9` | `writing_notes`, `using_laptop`, `reading`, `listening`, `looking_away`, `head_down`, `turned_to_peer`, `phone_use`, `uncertain` | |
| `cue7` | `writing_notes`, `using_laptop`, `looking_away`, `head_down`, `turned_to_peer`, `phone_use`, `uncertain` | |
| `cue8` | `writing_notes`, `using_laptop`, `engaged`, `looking_away`, `head_down`, `turned_to_peer`, `phone_use`, `uncertain` | |

Groupings:

* `onoff` — `on_task` = {`screen_oriented`}; `off_task` = everything else.
* `onoff_reliable` — `on_task` = {`screen_oriented`}; `off_task` =
  {`head_down`, `phone_use`, `uncertain`}.
* `coarse3_reliable` — `down_or_hidden` = {`head_down`, `uncertain`}; the other
  two are singletons.
* `cue8` — **a regrouping of Layer 2b**: `reading` and `listening` merged into
  a class named `engaged`, `using_laptop` kept separate, everything else a
  singleton. The question it asks is "is this student engaged with the lesson",
  separately from "is this student working on a device" — where `cue7` answers
  only the second by folding all three into `using_laptop`.
  cue9's own per-class numbers are the motivation: `reading` is its worst class
  at 0.337 while `listening` reaches 0.673, so the two are not equally
  recoverable, and merging them tests whether the boundary between them was the
  difficulty. Like `cue7` it needs no new label build.
* `cue7` — **a regrouping of Layer 2b**: `reading` and `listening` merged into
  `using_laptop`, everything else a singleton. Unlike `cue9` it needs no new
  label build — the cue9 sidecar already carries the ids and `taxonomy_lut`
  merges them. Splitting `screen_oriented` needed the annotation record back;
  merging three of its parts needs only the three ids.

  The merge is supported by the measured numbers: at 240 epochs cue9 reaches
  `using_laptop` 0.682 and `listening` 0.673 but `reading` only 0.337, and the
  two are keyed on *different fields* — `using_laptop` on
  `activity == using_laptop`, `reading` on `gaze in {laptop, own_desk}` — so the
  same student at a laptop can land in either depending on which field the
  annotator filled. It restores `listening`'s 30.4% of the corpus to one class,
  making `using_laptop` the majority class again at ~74%.

  **Measured (FINDINGS §22): 0.4656 ± 0.0061 against cue9's 0.4612 ± 0.0104.**
  The "easier average" effect is real in direction and negligible in size —
  +0.0044, under half a seed's standard deviation — because the merge replaces
  three mean-slots with one high one and those nearly cancel.

  What the merge *does* change is the shape: `using_laptop` becomes ~74% of the
  corpus, almost exactly the 75.7% `screen_oriented` held in cue6, and it
  behaves the same way — `writing_notes` falls 0.297 → 0.203 as the thin class
  nearest the attractor. cue7 buys `using_laptop` 0.859 and the tightest seed
  spread here; it costs the on-task detail cue9 was built to recover.
* `cue9` — a **passthrough of Layer 2b**, not a regrouping of Layer 2. It is
  listed here only because `--taxonomy cue9` is how a run selects it; its
  `space` key is what tells the loader its ids index `CUE9_CLASSES`. Training or
  evaluating it **requires** `--cue-labels` pointing at a cue9 label build, and
  `data.load_split` refuses to mix a label set with a taxonomy over a different
  space — the ids would silently mean different classes.

**Any number from an abstaining taxonomy must be quoted with its coverage, and
macro-F1 over 2 or 3 classes is not comparable to macro-F1 over 6.**
`onoff_reliable` reaches 0.8415 ± 0.0072 at 91% coverage against `cue6`'s
0.5200 ± 0.0122 at 100% (both at 240 epochs, FINDINGS §18; 0.768 and 0.479 at the
earlier 90-epoch budget). That is a different, easier task, not a better model.

---

## Auxiliary tables keyed by cue

### `CUE_TASK_SCORE` — `attention/taxonomy.py`

Classroom-**aggregate** overlay only. Explicitly not a claim about any
individual's mental state.

| cue | score |
|---|---|
| `screen_oriented` | 1.00 |
| `uncertain` | 0.50 |
| `looking_away` | 0.30 |
| `turned_to_peer` | 0.30 |
| `head_down` | 0.15 |
| `phone_use` | 0.10 |

### `ALERT_AFTER_S` — `tools/dashboard/server.py`

Sustained dwell before an instructor alert. Alerts fire on episodes, not
single frames, and also require the model's own calibrated confidence.

| cue | seconds |
|---|---|
| `phone_use` | 15 |
| `looking_away` | 20 |
| `head_down` | 30 |
| `turned_to_peer` | 30 |

`screen_oriented` and `uncertain` never alert.

This table and the off-task share are keyed on the **six** cue classes, so they
have to be projected onto whichever taxonomy is running
(`taxonomy_alert_dwell`, `taxonomy_off_task_classes`). The two projections use
deliberately different rules, because the questions differ:

* **off-task share** — a merged class counts when it contains an off-task cue and
  no on-task one;
* **alert dwell** — a merged class may alert only when *every* cue it merges is
  independently alertable, and then waits as long as the slowest of them.

So `onoff_reliable`'s `off_task` (which merges `head_down`, `phone_use` **and**
`uncertain`) contributes to the off-task share and **may raise no alert at all**;
`coarse3_reliable` alerts on `phone_use` only. An `off_task` percentage that
includes `uncertain` is not comparable to the six-cue one, and the dashboard
says so. `cue9` splits only the on-task side, so its off-task four and their
dwells are identical to cue6's.

### `CUE_PHRASES` — `attention/cue_phrases.py`

One natural-language phrase per cue for querying a VLM, each derived from the
rule conditions above rather than written freehand. Every phrase describes
visible behaviour: the standing rule is that the system claims "the student's
head is down", never "the student is not paying attention".

---

## What this document does not cover

* Detector training data (ODVG grounding captions) — `grounding_data/`, HPC-only.
* Per-field corpus histograms other than `activity` — not measured.
* Inter-annotator agreement — none exists; single annotator
  (`THESIS_DEFENSIBILITY_REVIEW.md` threat C).
