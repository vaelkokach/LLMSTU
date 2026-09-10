"""Natural-language phrases for the six cue classes, for querying a VLM.

Every phrase is derived from the conditions in ``taxonomy.cue_conditions`` --
the same rules that produced the labels -- rather than invented. That matters
for two reasons:

* a phrase that describes something the label does not mean would measure
  prompt-writing, not the model;
* the mapping has to be auditable, because a fused system's disagreement rate
  is only interpretable if both halves are answering the same question.

The phrases deliberately describe *visible behaviour*, matching the project's
standing rule: we claim "the student's head is down", never "the student is not
paying attention". A VLM asked to judge attention would answer a different and
less checkable question.

Known limitation, stated rather than hidden: `screen_oriented` covers both
screen and instructor orientation (taxonomy.py documents this -- in a computer
lab both are on-task), so its phrase has to name both. That makes it the
broadest query of the six, and the one most likely to absorb probability mass
from a VLM that is unsure.
"""

from __future__ import annotations

from typing import Dict, List

from .taxonomy import CUE_CLASSES

#: cue -> a phrase describing what is VISIBLE when that cue fires.
#: Each is traceable to the enum values its rule tests in cue_conditions().
CUE_PHRASES: Dict[str, str] = {
    # activity in {using_laptop, listening, reading, writing_notes}
    # or gaze in {laptop, teacher_or_board, own_desk, down} with a task target
    "screen_oriented":
        "a student facing their laptop, the teacher, the board, or their own "
        "desk, working or listening",
    # gaze == away_or_window, activity == looking_away, target == distracted
    "looking_away":
        "a student looking away from their work, turned toward a window or "
        "off into the room",
    # activity == head_down_sleeping, posture in {head_down, slumped}
    "head_down":
        "a student with their head down on the desk, slumped over or asleep",
    # activity == talking_to_peer, talking, gaze == peer, target == peer
    "turned_to_peer":
        "a student turned toward a classmate beside them, talking to them",
    # activity == using_phone, phone_visible, gaze == phone, hand == on_phone
    "phone_use":
        "a student holding a mobile phone or looking down at a phone in "
        "their hands",
    # occluded with no recoverable face signal, or no usable orientation cue
    "uncertain":
        "a student who is blocked from view, turned away, or too unclear to "
        "judge",
}


def phrases_in_class_order() -> List[str]:
    """Phrases ordered to match CUE_CLASSES, so index i is class i.

    The scorer returns a vector aligned with this order and the fusion layer
    indexes it by class id. A mismatch would silently pair every cue with the
    wrong phrase, so the order is derived from CUE_CLASSES rather than from
    the literal above.
    """
    return [CUE_PHRASES[c] for c in CUE_CLASSES]


def describe_mapping() -> str:
    """Human-readable dump, for the dashboard and for the thesis appendix."""
    return "\n".join(f"{c:16} -> {CUE_PHRASES[c]}" for c in CUE_CLASSES)
