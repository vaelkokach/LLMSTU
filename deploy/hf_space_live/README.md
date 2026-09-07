---
title: Classroom Attention Cues (Live)
emoji: 🎥
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Live instructor dashboard for classroom cues
---

# Classroom Attention Cues — instructor dashboard

Detects, attributes and temporally aggregates **observable behavioural cues** in
wide-shot classroom video: where a student is oriented, whether their head is
down, whether a phone is in use, and how long each state has persisted.

## What this measures, and what it does not

This system reports **visible cues**, not mental states. It can say that a
student's head has been down for 51 seconds. It cannot say whether that student
is bored, confused, disengaged, or learning.

That distinction is not a disclaimer bolted on afterwards — it is the scoping the
whole evaluation is built around. Specifically:

* **Not an attention or emotion detector.** Measured boredom AUROC is 0.544,
  barely above chance. Engagement, comprehension and curiosity are not measured.
* **Not validated beyond one setting.** On an external lecture-hall dataset an
  a-priori head-pose proxy correlates significantly in the *opposite* direction
  to the naive disengagement hypothesis (rho = +0.172, permutation p < 0.0001).
  In a computer lab, stillness toward a monitor is often the on-task pattern. Cue
  meaning changes with the room.
* **Not real time.** The deployed configuration reaches 5.77 FPS at ~6 students
  on one A100. Every frame misses a 10 FPS budget; 91% miss it even at the
  deployed striding. "Near-real-time" is the honest description.
* **Not a measure of a student.** Outputs are auditable observations for an
  instructor, not an assessment. They should never drive an automated decision
  about a person.

## What it does well

* **Attribution.** Binding descriptions to the right student by content-based
  bipartite assignment rather than detector-confidence order raises validation
  R@1 from 0.3230 to 0.4954 — the gain is correct attribution, not better
  localisation (R@10 moves 0.014).
* **Temporal modelling.** Task-appropriate temporal architectures matter more
  than extra per-frame features: MS-TCN improves test macro-F1 by 0.0933 over a
  generic transformer, significant in all three seed-matched comparisons.
  Adding facial-expression and motion features does not help and is nominally
  harmful.
* **Calibrated abstention.** Below the display threshold the UI shows
  `uncertain`; below the alert threshold no episode alert fires. Raw predictions
  are always recorded, so abstention never destroys evidence.
* **Model transparency.** Every deployable checkpoint is selectable, and the
  panel shows which one produced the cues on screen, with its validation scores
  and its alert coverage.

## Honest limits on the numbers shown

The human-referenced event evaluation rests on **16 distinct episodes across ten
tracks, annotated by one person**. It is a diagnostic set, not ground truth and
not a ceiling. Event recall peaks at 0.312 against 0.625 for the pseudo-labelling
teacher, and that gap is unresolved.

## Data

No student video ships with this Space. Sessions are supplied by the operator.
Faces can be blurred in the UI; the pipeline records boxes and cue labels, not
identities, and the labeller is explicitly instructed never to infer identity,
age, gender or ethnicity.

## Citation

Built on LLMDet / Grounding-DINO. See the project repository for the full results
register, in which every citable number carries its split, evaluator version,
evidence file and caveat — and every non-citable one carries the reason it must
not be used.
