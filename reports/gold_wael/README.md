# Gold_annotation_wael — what it measures, and what it does not

Run 2026-09-11. Inputs: `Gold_annotation_wael/gold_annotations_wael.jsonl`
(1,000 records, every one `status: ok`), scored against
`grounding_data/llmstu_tools/outputs/gold_candidates.jsonl` — the manifest the
annotator worked from.

## 1. The label ceiling, re-measured on a fresh sample

`reports/gold_wael/label_ceiling_wael.json`

| ruleset | macro-F1 | 95% CI | accuracy |
|---|---|---|---|
| v1 | **0.8645** | [0.8354, 0.8885] | 0.8710 |
| v2 | **0.8708** | [0.8466, 0.8924] | 0.8730 |

Per-class F1 under v1: `head_down` 0.920, `turned_to_peer` 0.891, `phone_use`
0.865, `screen_oriented` 0.858, `uncertain` 0.830, `looking_away` 0.821.

Two things follow.

**The teacher ceiling replicates.** FINDINGS §5 reports 87.4% frame agreement
from the earlier Admin pairing; this is 87.10% accuracy on a different
1,000-crop sample. That is an independent replication, not a re-measurement of
the same number — see §3.

**v2 is still slightly ahead, model-free.** +0.0063 macro-F1, and it moves in
the direction the rule repair predicts: `uncertain` recall 0.710 → 0.812 and its
support 31 → 48, while `looking_away` support falls 120 → 88. The v2 *training*
intervention failed (`docs/CUE_RULES_V2.md`); the v2 *diagnosis* keeps being
confirmed by label-side evidence. This is the second gold set to say so.

Quality of the sample itself is much better than the Admin round:
**0 crops rejected as unusable, against 230 (23.4%)**, and 1,000/1,000 join the
manifest against 3/984. The ceiling is no longer conditional on a 23% reject
rate.

## 2. A correctness fix in the measurement, which turned out to be a no-op

`measure_ceiling.py` scored `map_record()` on the human record, and the gold tool
does not collect `face_kpts`. The field therefore fell back to its default of 3
on the human side, so the `uncertain` gate (`occluded AND face_kpts <= 2`) could
fire only on the pseudo side — human `uncertain` systematically under-counted,
and every pseudo `uncertain` over an occluded crop scored as a false positive it
had not committed.

`join_measured_fields()` now recovers `face_kpts` from the crop's own pseudo
record by `file_name`. It is a measurement of the image, not an annotator
judgement, so this leaks nothing.

**It filled 1,000 records and changed 0 labels on this gold set** — wherever the
annotator marked `occluded`, the detector had found more than two keypoints. The
numbers above are unaffected. The fix is kept because it is correct and the next
gold set may not be so lucky; the report records the no-op rather than implying
a correction happened.

## 3. Inter-annotator agreement is NOT resolved by this set

`THESIS_DEFENSIBILITY_REVIEW.md` threat C is single-annotator labelling. This
gold set does not close it.

```
Admin gold : 984 unique crops
wael  gold : 1,000 unique crops
overlap    : 3 crops  (2 usable)
```

The two sets are annotations of **essentially disjoint samples** — the candidate
manifest was regenerated between the rounds. `compute_agreement.py` returns
κ = 1.000 over n = 2, which is not a measurement of anything.

So the correct reading is: this is a **second, better, independent gold sample**,
and it replicates the ceiling. It is **not** a second annotation of the first,
and no κ can be quoted from the pair.

To close threat C, one annotator has to re-annotate a shared subset — a few
hundred crops drawn from `gold_candidates.jsonl`, which both the existing gold
files already index. Nothing else in the pipeline needs to change.

## 4. Where the pseudo-labeller is wrong

Field-exact on all 10 fields: 73.9% of items. Per-field agreement:

| field | agreement |
|---|---|
| `activity` | 75.8% |
| `gaze_direction` | 86.9% |
| `attention_target` | 89.3% |
| `engagement_level` | 91.8% |
| `hand_state` | 95.1% |
| `posture` | 97.1% |
| `talking` | 99.0% |
| `phone_visible` | 99.3% |
| `laptop_visible` | 99.8% |
| `occluded` | 99.8% |

`activity` is the weakest field at 75.8%, and it is the field the cue9 split
leans on hardest: `writing_notes` and `using_laptop` are keyed on it. That is
the number to quote as the honest limit on any cue9 result for those two classes.

## 5. Reproduce

```bash
python tools/gold_annotator/measure_ceiling.py \
    --human Gold_annotation_wael/gold_annotations_wael.jsonl \
    --pseudo grounding_data/llmstu_tools/outputs/gold_candidates.jsonl \
    --out reports/gold_wael/label_ceiling_wael.json

python tools/gold_annotator/compute_agreement.py \
    event_gold_bundle/gold_annotations_Admin.jsonl \
    Gold_annotation_wael/gold_annotations_wael.jsonl
```

`events_wael.jsonl` is the annotator's append-log: 1,062 lines, 1,000 distinct
`index` values, 60 re-edited. `gold_annotations_wael.jsonl` is the finalised
de-duplicated set and is the file to use.
