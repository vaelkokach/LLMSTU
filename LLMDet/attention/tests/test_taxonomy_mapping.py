from attention.taxonomy import CUE_CLASSES, CUE_TO_ID, map_record, parse_stem_time


def _rec(**kw):
    base = {
        "activity": "listening",
        "gaze_direction": "teacher_or_board",
        "attention_target": "instruction",
        "engagement_level": "engaged",
        "posture": "upright",
        "hand_state": "unknown",
        "phone_visible": False,
        "laptop_visible": False,
        "talking": False,
        "occluded": False,
        "face_kpts": 3,
    }
    base.update(kw)
    return base


def test_class_set():
    assert len(CUE_CLASSES) == 6
    assert CUE_CLASSES[0] == "screen_oriented"


def test_screen_oriented_listening():
    assert map_record(_rec()) == CUE_TO_ID["screen_oriented"]


def test_screen_oriented_laptop():
    r = _rec(activity="using_laptop", gaze_direction="laptop", attention_target="device")
    assert map_record(r) == CUE_TO_ID["screen_oriented"]


def test_phone_beats_laptop():
    r = _rec(activity="using_laptop", phone_visible=True)
    assert map_record(r) == CUE_TO_ID["phone_use"]


def test_phone_from_hand_state():
    assert map_record(_rec(hand_state="on_phone")) == CUE_TO_ID["phone_use"]


def test_head_down_activity():
    r = _rec(activity="head_down_sleeping", posture="leaning_back")
    assert map_record(r) == CUE_TO_ID["head_down"]


def test_head_down_posture_slumped():
    assert map_record(_rec(posture="slumped")) == CUE_TO_ID["head_down"]


def test_turned_to_peer():
    assert map_record(_rec(activity="talking_to_peer")) == CUE_TO_ID["turned_to_peer"]
    assert map_record(_rec(gaze_direction="peer", attention_target="peer")) == CUE_TO_ID["turned_to_peer"]


def test_looking_away():
    r = _rec(activity="looking_away", gaze_direction="away_or_window", attention_target="distracted")
    assert map_record(r) == CUE_TO_ID["looking_away"]


def test_uncertain_occluded_no_face():
    r = _rec(occluded=True, face_kpts=2)
    assert map_record(r) == CUE_TO_ID["uncertain"]


def test_occluded_with_face_still_labeled():
    r = _rec(occluded=True, face_kpts=3, activity="using_laptop")
    assert map_record(r) == CUE_TO_ID["screen_oriented"]


def test_uncertain_all_unknown():
    r = _rec(activity="other", gaze_direction="unknown", attention_target="unknown", engagement_level="unknown")
    assert map_record(r) == CUE_TO_ID["uncertain"]


def test_fallback_eating_maps_to_uncertain():
    # idle_other merged into uncertain (fallback fired on only 16/283k records)
    r = _rec(activity="eating_drinking", gaze_direction="unknown", attention_target="unknown")
    assert map_record(r) == CUE_TO_ID["uncertain"]


def test_uncertain_outranks_phone():
    r = _rec(occluded=True, face_kpts=2, phone_visible=True)
    assert map_record(r) == CUE_TO_ID["uncertain"]


def test_parse_stem_time():
    assert parse_stem_time("t000033_856_f000680") == 33.856
    assert parse_stem_time("t000000_000_f000000_video_0005_0_10_x_y") == 0.0
    assert parse_stem_time("garbage") is None


# ---------------------------------------------------------------------------
# Coarser taxonomies (TAXONOMIES / taxonomy_lut)
# ---------------------------------------------------------------------------

def test_cue6_taxonomy_is_the_identity():
    """The default must not change behaviour: it is applied to every load."""
    from attention.taxonomy import taxonomy_lut, taxonomy_classes, CUE_CLASSES
    assert taxonomy_lut("cue6") == list(range(len(CUE_CLASSES)))
    assert taxonomy_classes("cue6") == list(CUE_CLASSES)


def test_every_source_class_is_mapped_or_deliberately_excluded():
    """No class may fall through silently.

    A source class that is neither grouped nor listed as excluded would be
    relabelled to IGNORE by accident, quietly shrinking the evaluation set and
    improving the score for a reason nobody chose.
    """
    from attention.taxonomy import (CUE_CLASSES, IGNORE_LABEL, TAXONOMIES,
                                    taxonomy_lut, taxonomy_excluded)
    for name, spec in TAXONOMIES.items():
        lut = taxonomy_lut(name)
        assert len(lut) == len(CUE_CLASSES)
        grouped = {c for g in spec["groups"].values() for c in g}
        excluded = set(taxonomy_excluded(name))
        assert grouped | excluded == set(CUE_CLASSES), name
        assert not (grouped & excluded), f"{name}: class both grouped and excluded"
        for new_id in lut:
            assert new_id == IGNORE_LABEL or 0 <= new_id < len(spec["classes"]), name


def test_reliable_taxonomies_abstain_on_the_unsupported_classes():
    """The two classes the labels cannot support must be the excluded ones."""
    from attention.taxonomy import taxonomy_excluded
    for name in ("onoff_reliable", "coarse3_reliable"):
        assert set(taxonomy_excluded(name)) == {"looking_away", "turned_to_peer"}, name
    # ...and the all-frames variants must abstain on nothing.
    for name in ("cue6", "onoff"):
        assert taxonomy_excluded(name) == [], name


def test_group_members_land_on_the_same_new_id():
    from attention.taxonomy import CUE_TO_ID, TAXONOMIES, taxonomy_lut
    for name, spec in TAXONOMIES.items():
        lut = taxonomy_lut(name)
        for gname, members in spec["groups"].items():
            ids = {lut[CUE_TO_ID[m]] for m in members}
            assert len(ids) == 1, f"{name}/{gname} split across ids {ids}"
