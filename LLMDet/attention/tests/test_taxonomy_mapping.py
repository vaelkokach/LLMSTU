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
