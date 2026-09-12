"""The browser-camera path: a frame source that is not a `cv2.VideoCapture`.

The dashboard runs on the GPU host and the camera is in front of whoever opened
the page, so `cv2.VideoCapture(0)` is the wrong thing in both deployments that
matter: on a Hugging Face Space there is no camera attached, and on a shared box
device 0 belongs to somebody else. The browser captures with `getUserMedia` and
POSTs JPEGs instead, and `PushedFrames` stands in for the reader.

What these pin is the part that is silent when wrong: a live source must DROP
frames rather than queue them. If it buffered, the overlay would fall further
behind the room the longer it ran, with nothing on screen to say so — an
instructor would be alerted about a student who put their phone away minutes
ago. That is the failure `LatestFrame` was written for, and `PushedFrames` has
to behave identically.
"""
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "dashboard"))

PB = pytest.importorskip("pipeline_bridge")
SRC = pytest.importorskip("sources")


def test_keeps_only_the_newest_frame():
    """Three frames pushed, one consumed: it must be the LAST one."""
    q = PB.PushedFrames()
    for n in (1, 2, 3):
        q.put(n)
    ok, f = q.read(timeout=1.0)
    assert ok and f == 3, "a live source must skip to the newest frame"
    assert q.dropped == 2


def test_dropped_counts_what_was_skipped():
    q = PB.PushedFrames()
    q.put("a"); q.read(timeout=1.0)
    assert q.dropped == 0
    for c in "bcd":
        q.put(c)
    ok, f = q.read(timeout=1.0)
    assert (ok, f) == (True, "d")
    assert q.dropped == 2


def test_the_same_frame_is_not_returned_twice():
    """Otherwise the pipeline would re-analyse one frame forever when the
    browser stalls, reporting a steady frame rate over a frozen image."""
    q = PB.PushedFrames()
    q.put("x")
    assert q.read(timeout=0.5) == (True, "x")
    ok, f = q.read(timeout=0.2)
    assert f is None, "a consumed frame must not be handed out again"


def test_it_dies_when_the_browser_stops_posting():
    """There is no end-of-stream event for a closed tab.

    Without the idle timeout the worker thread blocks forever on an empty queue,
    holding the GPU while the page still reports 'running'.
    """
    q = PB.PushedFrames(idle_timeout=0.15)
    q.put("x")
    assert q.alive
    time.sleep(0.3)
    assert not q.alive
    ok, f = q.read(timeout=1.0)
    assert ok is False and f is None


def test_release_then_reopen():
    q = PB.PushedFrames()
    q.put("x")
    q.release()
    assert not q.alive
    assert q.read(timeout=0.1) == (False, None)
    q.reopen()
    assert q.alive and q.dropped == 0
    q.put("y")
    assert q.read(timeout=1.0) == (True, "y")


def test_it_matches_the_latest_frame_contract():
    """Both are handed to the same `run_live` loop, so the surface must match.

    `alive` is set in `LatestFrame.__init__` rather than on the class, so it is
    checked on the instance side only; `dropped` is a property and is visible on
    both.
    """
    q = PB.PushedFrames()
    for name in ("read", "release", "dropped", "alive"):
        assert hasattr(q, name), f"PushedFrames is missing {name}"
    for name in ("read", "release", "dropped"):
        assert hasattr(PB.LatestFrame, name), f"LatestFrame is missing {name}"


# --------------------------------------------------------------------------
# source ids
# --------------------------------------------------------------------------

def test_browser_camera_id_parses():
    kind, path = SRC.parse_id(SRC.BROWSER_CAMERA_ID)
    assert (kind, path) == ("camera", "browser")


def test_a_device_index_parses_but_is_a_different_thing():
    """`camera:0` is a capture device ON THE SERVER — useful when the dashboard
    is run locally, and never what a remote viewer means by 'my camera'."""
    assert SRC.parse_id("camera:0") == ("camera", "0")


@pytest.mark.parametrize("bad", ["camera:", "camera:/dev/video0", "camera:eth0",
                                 "camera:../../etc/passwd"])
def test_a_camera_id_is_not_a_path(bad):
    with pytest.raises(ValueError):
        SRC.parse_id(bad)


def test_classify_source_still_reads_a_bare_index_as_a_camera():
    assert PB.classify_source("0") == ("camera", 0)
    assert PB.classify_source("rtsp://cam/stream")[0] == "stream"
    assert PB.classify_source("/tmp/x.mp4")[0] == "file"


# --------------------------------------------------------------------------
# the buffer's lifecycle across runs
#
# `run_live` ends by releasing its reader, and for the browser camera the reader
# IS the process-wide buffer. So the first camera run leaves it permanently
# closed, and the SECOND one attaches to a corpse: read() reports the pusher
# gone on its first call, run_live returns normally, nothing raises, and the
# recorded error stays empty. The page shows `sent` climbing, `analysed` frozen
# and "the camera pipeline is not running" -- with nothing anywhere saying why.
# Deployed, that is what "sent 233, analysed 1" was.
# --------------------------------------------------------------------------

def test_a_released_buffer_is_dead_to_a_second_run():
    """The precondition for the bug. If this ever stops holding, so does it."""
    q = PB.PushedFrames()
    q.put("x")
    q.release()
    q.put("y")                      # the browser keeps posting; put() works
    assert q.seq == 2               # ...and `sent` keeps climbing
    assert q.read(timeout=0.1) == (False, None)   # but nothing can be read
    assert q.taken == 0                           # so `analysed` never moves


def test_starting_a_camera_run_gets_a_live_buffer():
    server = pytest.importorskip("server")
    first = server.camera_buffer(reset=True)
    first.put("x")
    first.release()                 # as run_live does when the first run ends

    second = server.camera_buffer(reset=True)
    assert second is not first, "a run must not inherit the released buffer"
    assert second.alive
    second.put("y")
    assert second.read(timeout=1.0) == (True, "y")


def test_the_frame_endpoint_pushes_into_the_run_s_buffer():
    """POST /api/camera/frame and the worker must share one object.

    They meet only through the module global, so a reset that replaced it
    without the endpoint noticing would drop every frame on the floor.
    """
    server = pytest.importorskip("server")
    server.camera_buffer(reset=True)
    run_buf = server.camera_buffer(reset=True)
    endpoint_buf = server.camera_buffer()        # what _camera_frame() uses
    assert endpoint_buf is run_buf
    endpoint_buf.put("z")
    assert run_buf.read(timeout=1.0) == (True, "z")
