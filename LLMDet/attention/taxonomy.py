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

from typing import Dict, List, Optional, Tuple

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


#: Rule versions. ``v1`` is the rule set every published number and every
#: built sequence was produced under and MUST NOT change. ``v2`` is the
#: repaired rule set; see RULESET_V2_RATIONALE.
RULESETS: Tuple[str, ...] = ("v1", "v2")
DEFAULT_RULESET: str = "v1"

RULESET_V2_RATIONALE = """
v2 removes ``attention_target`` from every cue rule.

Measured, on the 1,000-crop stratified sample
(grounding_data/llmstu_tools/outputs/gold_candidates.jsonl):

  * ``attention_target`` is 96.0% predictable from ``gaze_direction`` alone --
    laptop->device 100%, own_desk->own_work 100%, peer->peer 100%,
    phone->device 100%, teacher_or_board->instruction 98.8%,
    away_or_window->distracted 98.4%. It is a recoding of the gaze field, not
    an independent observation.
  * Its ONE non-redundant cell is the defect: ``gaze_direction == "down"``
    maps to ``attention_target == "distracted"`` on 94.1% of records, and
    ``distracted`` is a v1 trigger for ``looking_away``. So *looking down*
    fires *looking away*.
  * Of the 372 records where the v1 ``looking_away`` rule fires, 359 (96.5%)
    are driven by ``distracted`` and 181 (48.7%) by ``distracted`` ALONE. Of
    those 181: gaze is ``down`` on 85.1% and ``away_or_window`` on 0.0%;
    activity is ``head_down_sleeping`` on 70.7%; and 81.2% are labelled
    ``head_down`` once precedence is applied.

The same pathology is present in the human labels, so it is a property of the
annotation vocabulary rather than of the VLM pseudo-labeller. On the 754
usable human-annotated crops in ``event_gold_bundle/gold_annotations_Admin.jsonl``:
``gaze == down`` -> ``distracted`` on 231/232 = 99.6%, and 231/232 = 99.6% of
human ``head_down`` records also carry ``looking_away`` as a candidate --
reproducing the 772-of-773 co-occurrence that collapsed PRODEN (FINDINGS 12.x)
in labels the pseudo-labeller never touched.

There is no ``attention_target`` value meaning "looking down at own work", so a
head-down student can only be called ``distracted``; the cue rule then turns
that into ``looking_away``, which sits directly below ``head_down`` in
precedence and therefore inherits exactly the frames it cannot be told apart
from. ``looking_away`` is, roughly half the time, a synonym for "head is down".

v2 therefore drops the field and keeps only perceptual evidence:

    looking_away     gaze == away_or_window OR activity == looking_away
    turned_to_peer   activity == talking_to_peer OR talking OR gaze == peer
    screen_oriented  activity in TASK_ACTIVITIES OR gaze in TASK_GAZES
    no-signal gate   gaze/engagement/activity unknown-ish (target dropped)

``target == peer`` is dropped as exactly redundant with ``gaze == peer``
(129/129 co-occurrence). ``screen_oriented`` loses the ``target in
TASK_TARGETS`` conjunct, which was only ever a restatement of the gaze test.
Dropping ``target == unknown`` from the no-signal gate makes the gate fire on
crops with no readable orientation that v1 sent to ``looking_away`` via
``distracted``; an unreadable crop becomes ``uncertain``, which is what the
annotation guideline says it is.

Effect on the stratified sample: 26/1000 hard labels change; ``looking_away``
purity (share of the class whose gaze is actually ``away_or_window``) rises
74.1% -> 90.6%; ``head_down`` records ambiguous with ``looking_away`` fall
100% -> 26.1%; the under-crediting ratio falls 2.60x -> 1.63x.

These are SAMPLE numbers, and the sample is stratified (rare activities are
oversampled), so they are not corpus prevalences. Re-measure corpus-wide with
``tools/audit_attention_target.py`` before citing any of them.
"""


def cue_conditions(rec: Dict,
                   ruleset: str = DEFAULT_RULESET) -> "List[Tuple[str, bool]]":
    """Every cue rule and whether it fires, in precedence order.

    The single source of truth for both :func:`map_record` (first match wins)
    and :func:`candidate_set` (all matches). They must not be written twice: a
    drift between the label a frame is given and the set it is credited for
    would be invisible and would silently change what the model is scored on.

    ``("uncertain", True)`` in first position is a GATE, not a candidate among
    others: a student who cannot be seen supports no cue at all, so both
    callers stop there.

    ``ruleset`` selects the rule version. The two versions share this one
    function for the same reason ``map_record`` and ``candidate_set`` do: two
    copies of a precedence list drift silently. v1 is the default everywhere,
    so no existing caller changes behaviour.
    """
    if ruleset not in RULESETS:
        raise KeyError(f"unknown ruleset {ruleset!r}; known: {RULESETS}")
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

    unverifiable = occluded and face_kpts <= UNCERTAIN_FACE_KPTS

    # Rules shared by both versions. phone_use and head_down never referenced
    # attention_target, so v2 leaves them untouched -- the repair is confined
    # to the three rules that read it plus the no-signal gate.
    phone = (activity == "using_phone" or phone_visible
             or gaze == "phone" or hand == "on_phone")
    head_down = (activity == "head_down_sleeping"
                 or posture in ("head_down", "slumped"))

    if ruleset == "v1":
        no_signal = (gaze == "unknown" and target == "unknown"
                     and engagement == "unknown" and activity == "other")
        peer = (activity == "talking_to_peer" or talking
                or gaze == "peer" or target == "peer")
        away = (gaze == "away_or_window" or activity == "looking_away"
                or target == "distracted")
        screen = (activity in _TASK_ACTIVITIES
                  or (gaze in _TASK_GAZES and target in _TASK_TARGETS))
    else:                                            # v2 -- see RULESET_V2_RATIONALE
        no_signal = (gaze == "unknown" and engagement == "unknown"
                     and activity == "other")
        peer = activity == "talking_to_peer" or talking or gaze == "peer"
        away = gaze == "away_or_window" or activity == "looking_away"
        screen = activity in _TASK_ACTIVITIES or gaze in _TASK_GAZES

    return [
        ("uncertain", unverifiable or no_signal),
        ("phone_use", phone),
        ("head_down", head_down),
        ("turned_to_peer", peer),
        ("looking_away", away),
        ("screen_oriented", screen),
    ]


def candidate_set(rec: Dict, ruleset: str = DEFAULT_RULESET) -> "List[int]":
    """Every cue class this record supports -- the PARTIAL label.

    ``map_record`` keeps the highest-precedence firing rule and discards the
    rest. That is not a tie-break, it is a deletion: measured over the 6,816
    LLMSTU pseudo-labels, 13.7% of records fire more than one rule, and
    `looking_away` is true by its own rule 2.74x more often than precedence
    lets it be the label (1313 against 479). The model is then penalised for
    predicting a class that was, by the annotation's own fields, correct.

    Returning the set lets a partial-label objective (PRODEN, Lv et al. ICML
    2020) credit any candidate and let the pixels decide which, instead of a
    hand-written ordering deciding in advance.
    """
    conds = cue_conditions(rec, ruleset)
    if conds[0][1]:                      # the unverifiable/no-signal gate
        return [CUE_TO_ID["uncertain"]]
    fired = [CUE_TO_ID[name] for name, hit in conds[1:] if hit]
    return fired or [CUE_TO_ID["uncertain"]]


def map_record(rec: Dict, ruleset: str = DEFAULT_RULESET) -> int:
    """Map one LLMSTU label record (parsed jsonl dict) to a cue class id.

    First firing rule wins. Kept exactly as it was: every published number and
    every built sequence depends on it. ``candidate_set`` is the partial-label
    view of the same conditions.
    """
    for name, hit in cue_conditions(rec, ruleset):
        if hit:
            return CUE_TO_ID[name]
    return CUE_TO_ID["uncertain"]


def _map_record_legacy(rec: Dict) -> int:
    """The original inlined implementation, kept only as a test oracle."""
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
