# Visible-cue taxonomy (Branch B)

> **This is the Branch B design record, not the current inventory.** The
> class table below still lists `idle_other` as a seventh class; it was
> merged into `uncertain` on 2026-07-29 (see the last section). For the
> current labels at every layer, read [`docs/LABELS.md`](../../docs/LABELS.md),
> which is checked against the code by `attention/tests/test_labels_doc.py`.

Single source of truth: `attention/taxonomy.py` (`CUE_CLASSES`, `map_record`).

The pipeline classifies **visible attention cues**, not attention itself. The
class set follows the supervisor's guidance (5–7 cue-based labels) and is
derived deterministically from the LLMSTU structured fields — no keyword
matching against free-text captions (the old `WEAK_LABEL_MAP` path collapsed
all students in a frame to one label and produced zero samples for two of
four classes; it is retained only as a deprecated legacy path).

## Classes and mapping (precedence: first match wins)

| # | cue | LLMSTU condition |
|---|-----|------------------|
| 6 | `uncertain` | `occluded ∧ face_kpts ≤ 2`, or all of gaze/target/engagement/activity unknown |
| 4 | `phone_use` | `activity=using_phone` ∨ `phone_visible` ∨ `gaze=phone` ∨ `hand_state=on_phone` |
| 2 | `head_down` | `activity=head_down_sleeping` ∨ `posture ∈ {head_down, slumped}` |
| 3 | `turned_to_peer` | `activity=talking_to_peer` ∨ `talking` ∨ `gaze=peer` ∨ `target=peer` |
| 1 | `looking_away` | `gaze=away_or_window` ∨ `activity=looking_away` ∨ `target=distracted` |
| 0 | `screen_oriented` | `activity ∈ {using_laptop, listening, reading, writing_notes}` ∨ (`gaze ∈ {laptop, teacher_or_board, own_desk, down}` ∧ `target ∈ {device, instruction, own_work}`) |
| 5 | `idle_other` | everything else (`eating_drinking`, `raising_hand`, `other`, idle hands) |

Notes and rationale:

- **Precedence** encodes "specific off-task cue beats generic on-task cue":
  a laptop user with a visible phone is `phone_use`.
- **`uncertain` outranks everything**: labels on heavily occluded students are
  not verifiable from pixels (audit found `occluded=True` on 31% of crops and
  `face_kpts=2` on 10.3%; the intersection is visually unusable). Per the
  annotation guideline, uncertain samples are excluded from the main metrics
  and reported separately — never force-labeled.
- **`screen_oriented` includes instructor/board orientation.** In a
  computer-lab lecture both screen and instructor gaze are task-oriented; a
  finer split (`screen` vs `instruction`) can be recovered later from
  `attention_target` without relabeling.
- **`head_down` is a posture cue, not "sleeping".** The LLMSTU
  `head_down_sleeping` activity mixes sleeping and resting; we only claim the
  observable posture. Its off-task interpretation happens at the *event*
  level (rule baseline: head_down sustained > 20 s → off-task cue).
- Expected class support from the LLMSTU audit (283,913 crops, before dedup):
  `screen_oriented` dominates (listening + using_laptop + reading + writing ≈ 77%),
  `phone_use` ≈ 3.3% (9,194 using_phone + phone_visible extras),
  `head_down` ≈ 7.3%, `looking_away` ≈ 6%, `turned_to_peer` ≈ 2.5%,
  `idle_other` ≈ 4%, `uncertain` ≈ 10% (occluded∧face_kpts≤2 ∪ all-unknown).
  All 7 classes have thousands of samples — the zero-class degeneracy of the
  old pipeline cannot recur, and `enforce_full_class_coverage` is re-enabled
  to guarantee it.

## Event-level interpretation (Branch B claims)

Frame-level cues are aggregated to windows and events
(`attention/rule_baseline.py`, `attention/events.py`). Off-task *claims* are
only made at event level (sustained duration), per the metric separation:
frame-level cue accuracy ≠ window-level state ≠ event-level false alerts.

## Change 2026-07-29: idle_other merged into uncertain (7 -> 6 classes)

First clean training run showed the idle_other fallback fired on only 16 of
283,913 records (the precedence rules absorb everything else), and its
inverse-frequency class weight (~10,000x screen_oriented's) destabilized
training. The annotation guideline already defines "no determinable cue" as
uncertain, so the fallback now maps there. Legacy 7-class NPZ sequences are
remapped at load time via `taxonomy.LEGACY_7CLASS_REMAP` (5->5, 6->5); class
weights are now sqrt-inverse-frequency. The `inactivity` event channel proxies
via head_down until motion features exist.
