"""Visible-cue taxonomy for the classroom attention pipeline.

Maps the structured LLMSTU per-student labels (activity, gaze_direction,
attention_target, posture, hand_state, booleans) onto a closed set of
observable cue classes. The taxonomy deliberately describes *visible
behavior*, not inferred mental state: we claim "the student's head is down",
never "the student is not paying attention".

Cue classes (precedence order used by :func:`map_record`, first match wins):

===================  ==========================================================
cue                  fires when
===================  ==========================================================
uncertain            occluded AND face_kpts <= UNCERTAIN_FACE_KPTS, or the
                     record carries no usable orientation signal
                     (gaze/attention/engagement/activity all unknown-ish)
phone_use            activity == using_phone, phone_visible == True,
                     gaze_direction == phone, or hand_state == on_phone
head_down            activity == head_down_sleeping, or posture in
                     {head_down, slumped}
turned_to_peer       activity == talking_to_peer, talking == True,
                     gaze_direction == peer, attention_target == peer
looking_away         gaze_direction == away_or_window, activity ==
                     looking_away, or attention_target == distracted
screen_oriented      task-oriented orientation: activity in {using_laptop,
                     listening, reading, writing_notes} or gaze_direction in
                     {laptop, teacher_or_board, own_desk, down} with a
                     task-consistent attention_target. NOTE: this class
                     covers both screen AND instructor/board orientation —
                     in a computer-lab lecture both are on-task.
uncertain (fallback) everything else (no orientation cue and no specific
                     activity match). Measured on LLMSTU this fallback fires
                     on only 16/283k records — a separate idle_other class is
                     not supported by the data, and the annotation guideline
                     defines exactly this situation as "uncertain".
===================  ==========================================================

The precedence encodes "specific off-task cue beats generic on-task cue":
a student who is using_laptop but has a visible phone is phone_use.
`uncertain` outranks everything because labels on heavily occluded students
are not verifiable from pixels (see annotation guideline).
"""

from typing import Dict, List, Optional

CUE_CLASSES: List[str] = [
    "screen_oriented",
    "looking_away",
    "head_down",
    "turned_to_peer",
    "phone_use",
    "uncertain",
]
CUE_TO_ID: Dict[str, int] = {c: i for i, c in enumerate(CUE_CLASSES)}
NUM_CUE_CLASSES: int = len(CUE_CLASSES)

# NPZ sequences built with the pre-2026-07-29 7-class taxonomy store
# idle_other as id 5 and uncertain as id 6; both fold into uncertain (5).
LEGACY_7CLASS_REMAP: Dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 5}

# On-task score per cue, used for the aggregate classroom overlay only
# (NOT a claim about individual mental state).
CUE_TASK_SCORE: Dict[str, float] = {
    "screen_oriented": 1.0,
    "looking_away": 0.3,
    "turned_to_peer": 0.3,
    "head_down": 0.15,
    "phone_use": 0.1,
    "uncertain": 0.5,
}

# face_kpts at or below this, combined with occluded=True, is visually
# unverifiable (audit: 10.3% of crops have face_kpts == 2).
UNCERTAIN_FACE_KPTS = 2

_TASK_ACTIVITIES = {"using_laptop", "listening", "reading", "writing_notes"}
_TASK_GAZES = {"laptop", "teacher_or_board", "own_desk", "down"}
_TASK_TARGETS = {"device", "instruction", "own_work"}


def map_record(rec: Dict) -> int:
    """Map one LLMSTU label record (parsed jsonl dict) to a cue class id."""
    activity = rec.get("activity", "other")
    gaze = rec.get("gaze_direction", "unknown")
    target = rec.get("attention_target", "unknown")
    posture = rec.get("posture", "unknown")
    hand = rec.get("hand_state", "unknown")
    occluded = bool(rec.get("occluded", False))
    face_kpts = int(rec.get("face_kpts", 3))
    phone_visible = bool(rec.get("phone_visible", False))
    talking = bool(rec.get("talking", False))
    engagement = rec.get("engagement_level", "unknown")

    if occluded and face_kpts <= UNCERTAIN_FACE_KPTS:
        return CUE_TO_ID["uncertain"]
    if (
        gaze == "unknown"
        and target == "unknown"
        and engagement == "unknown"
        and activity == "other"
    ):
        return CUE_TO_ID["uncertain"]

    if activity == "using_phone" or phone_visible or gaze == "phone" or hand == "on_phone":
        return CUE_TO_ID["phone_use"]

    if activity == "head_down_sleeping" or posture in ("head_down", "slumped"):
        return CUE_TO_ID["head_down"]

    if activity == "talking_to_peer" or talking or gaze == "peer" or target == "peer":
        return CUE_TO_ID["turned_to_peer"]

    if gaze == "away_or_window" or activity == "looking_away" or target == "distracted":
        return CUE_TO_ID["looking_away"]

    if activity in _TASK_ACTIVITIES or (gaze in _TASK_GAZES and target in _TASK_TARGETS):
        return CUE_TO_ID["screen_oriented"]

    return CUE_TO_ID["uncertain"]


def parse_stem_time(stem: str) -> Optional[float]:
    """Seconds-within-video from a frame stem like ``t000033_856_f000680...``."""
    if not stem.startswith("t"):
        return None
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0][1:]) + int(parts[1]) / 1000.0
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Coarser taxonomies
# ---------------------------------------------------------------------------
#
# The 6 cue classes are a *projection* of the 10-field LLMSTU schema, not the
# schema itself, and two of them are projections the labels cannot support:
#
#   looking_away    fires on gaze == away_or_window OR activity == looking_away
#                   OR attention_target == distracted
#   turned_to_peer  fires on activity == talking_to_peer OR talking OR
#                   gaze == peer OR attention_target == peer
#
# Each is a disjunction of semantically different conditions, and both rest on
# the fine gaze distinctions a VLM pseudo-labeller is worst at. The measured
# consequences, on ff_det/mstcn_553_ff_s42 validation:
#
#   * F1 0.210 and 0.167, against 0.53-0.85 for every other class;
#   * AUPRC lift over base rate 2.3x and 4.7x, against 11-14x elsewhere;
#   * adding the causally-correct feature -- head yaw/pitch/roll, 553 -> 556 --
#     moves them by +0.041 and +0.021, and 14 further dims (570) move them by
#     -0.002 and -0.010. A causally-correct feature that does not help means the
#     target is noisy, not that the model lacks information.
#
# So `_reliable` taxonomies map those two frames to IGNORE_LABEL rather than
# forcing them into a class. They then leave the loss AND the metrics, which is
# the honest accounting: the model is not scored on them because it is not
# asked to predict them. At runtime the deployed system must abstain on them
# too -- reporting on 91.0% of frames -- rather than silently dropping them.
# Coverage is therefore reported alongside every _reliable number.

#: Matches attention.thesis_eval.data.IGNORE_INDEX (torch's default
#: CrossEntropyLoss ignore_index), so excluded frames vanish from the loss.
IGNORE_LABEL: int = -100

TAXONOMIES: Dict[str, Dict] = {
    "cue6": {
        "classes": CUE_CLASSES,
        "groups": {c: [c] for c in CUE_CLASSES},
        "note": "the original 6-class projection",
    },
    "onoff": {
        "classes": ["on_task", "off_task"],
        "groups": {
            "on_task": ["screen_oriented"],
            "off_task": ["looking_away", "head_down", "turned_to_peer",
                         "phone_use", "uncertain"],
        },
        "note": "binary, every frame kept; the on/off boundary IS the "
                "looking_away boundary, so this inherits its label noise",
    },
    "onoff_reliable": {
        "classes": ["on_task", "off_task"],
        "groups": {
            "on_task": ["screen_oriented"],
            "off_task": ["head_down", "phone_use", "uncertain"],
        },
        "note": "binary with abstention on the two unsupported classes",
    },
    "coarse3_reliable": {
        "classes": ["screen_oriented", "down_or_hidden", "phone_use"],
        "groups": {
            "screen_oriented": ["screen_oriented"],
            "down_or_hidden": ["head_down", "uncertain"],
            "phone_use": ["phone_use"],
        },
        "note": "keeps the actionable distinction between a head down and a "
                "phone, still abstaining on the gaze-ambiguous classes",
    },
}


def taxonomy_classes(name: str) -> List[str]:
    if name not in TAXONOMIES:
        raise KeyError(f"unknown taxonomy {name!r}; "
                       f"known: {', '.join(sorted(TAXONOMIES))}")
    return list(TAXONOMIES[name]["classes"])


def taxonomy_lut(name: str) -> List[int]:
    """6-class id -> new id, or IGNORE_LABEL for a class this taxonomy drops.

    Every one of the 6 source classes must be accounted for: mapped into a
    group, or deliberately excluded. A class that is silently neither would be
    a relabelling bug that shows up only as a quietly better score.
    """
    spec = TAXONOMIES[name] if name in TAXONOMIES else None
    if spec is None:
        raise KeyError(f"unknown taxonomy {name!r}; "
                       f"known: {', '.join(sorted(TAXONOMIES))}")
    lut = [IGNORE_LABEL] * len(CUE_CLASSES)
    for new_id, gname in enumerate(spec["classes"]):
        for src in spec["groups"][gname]:
            lut[CUE_TO_ID[src]] = new_id
    return lut


def taxonomy_excluded(name: str) -> List[str]:
    """Source classes this taxonomy abstains on (mapped to IGNORE_LABEL)."""
    lut = taxonomy_lut(name)
    return [c for c in CUE_CLASSES if lut[CUE_TO_ID[c]] == IGNORE_LABEL]
