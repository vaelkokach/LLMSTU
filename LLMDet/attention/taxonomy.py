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
