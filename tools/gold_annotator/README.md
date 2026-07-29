# Gold Set Annotator

Fast keyboard-driven verification of LLMSTU pseudo-labels for building the human gold set
(500–1000 crops). Every image comes **pre-filled with the model's pseudo-labels** — you only
fix what's wrong and press space. Stdlib-only, no internet needed.

## Launch

```bash
cd tools/gold_annotator

# until the stratified manifest from llmstu_tools exists, make a test sample:
python make_sample_manifest.py --n 50

python serve.py --manifest gold_candidates_sample.jsonl --annotator wael --port 8765
# real run: python serve.py --manifest ../../grounding_data/llmstu_tools/gold_candidates.jsonl --annotator wael
```

Open http://localhost:8765/ (port-forward if remote: `ssh -L 8765:localhost:8765 ...`).
Progress autosaves after every action; re-launching resumes at the first unannotated item.

## Keys

| Key | Action |
|---|---|
| `1`–`6` | select field (activity, gaze, attention target, engagement, posture, hand state) |
| `j`/`k` or `↓`/`↑` | cycle value of selected field |
| `q` `w` `e` `r` | toggle phone_visible / laptop_visible / talking / occluded |
| `space` / `enter` | **accept & next** (saves current field values) |
| `u` | save as *uncertain* & next |
| `x` | reject image (unusable — e.g. fully hidden behind monitor) & next |
| `←` / `→` | previous / next without saving |
| `g` | jump to next unannotated |
| `f` | toggle source-frame view (red box = this student) — use for occlusion calls |

## Output files (per annotator)

- `events_<name>.jsonl` — append-only log of every action (audit trail, never rewritten)
- `gold_annotations_<name>.jsonl` — latest state per image; input for the scripts below

## Two-annotator protocol

Both annotators run the tool on the same manifest with different `--annotator` names, then:

```bash
python compute_agreement.py gold_annotations_A.jsonl gold_annotations_B.jsonl
```

Reports Cohen's kappa + % agreement per field. If mean kappa < 0.6, simplify the taxonomy
before blaming the model (agreement failure = ambiguous labels, per the advisor's note).

## Finalize

```bash
python finalize_gold.py --primary gold_annotations_A.jsonl \
    --secondary gold_annotations_B.jsonl \
    --manifest gold_candidates.jsonl --calib-frac 0.3
```

Writes `gold_calibration.jsonl` (threshold/confidence tuning ONLY) and `gold_heldout.jsonl`
(final reported numbers ONLY — never tune on it). Split is video-wise when the manifest has
`video_id`, so no student leaks across the two halves. Items where annotators disagree are
kept but flagged `disputed: true`; decide once (exclude or adjudicate) and never revisit.

## Draft annotation guideline (edit to taste, then freeze before labeling)

- **activity** — what the student is doing *most of this instant*. `using_laptop` requires
  gaze at the laptop AND hands near it; typing not required. `head_down_sleeping` = eyes
  closed or head resting regardless of posture token. `looking_away` only when gaze is off
  screen/board/desk for no visible task reason. `other` is a last resort.
- **gaze_direction** — where the eyes/head actually point, not where they "should".
  Face invisible → `unknown`, don't guess from body.
- **attention_target** — interpretation layer: `device` (laptop work), `instruction`
  (teacher/board), `own_work` (desk/notes/reading), `distracted` (phone, window, staring off),
  `peer`. If the task context makes it ambiguous (e.g. head down while coding), prefer the
  benign reading or `unknown`.
- **engagement_level** — `engaged` = actively on task; `partially_engaged` = on task but
  intermittent (glances away, fidgeting); `disengaged` = off task ≥ the whole moment
  (sleeping, phone, staring away). When torn between two, pick the *less* engaged only if
  there is positive evidence.
- **posture** — geometric, not interpretive. `slumped` needs visibly collapsed torso.
- **hand_state** — `unknown` whenever hands are not visible; never infer.
- **booleans** — `phone_visible`/`laptop_visible`: the physical object is visible in THIS
  crop (check frame view with `f` if unsure). `talking`: visible mouth movement/turn toward
  peer. `occluded`: another person/monitor blocks a meaningful part of the student.
- **uncertain (`u`)** — face/body too hidden or behavior unreadable. Do NOT force a guess;
  uncertain items are excluded from (or reported separately in) the main metrics, and that
  choice is fixed before evaluation.
- **reject (`x`)** — the image itself is unusable (wrong person, near-total occlusion, corrupt).
