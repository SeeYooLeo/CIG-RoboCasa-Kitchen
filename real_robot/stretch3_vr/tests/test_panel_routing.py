"""Which camera feeds the left panel, and what happens when one stream dies."""

import json
import threading
import time

from stretch3_quest_teleop.backends.ros2 import Ros2StretchBackend
from stretch3_quest_teleop.config import TeleopConfig, default_config_path, merge
from stretch3_quest_teleop.models import Mode


def _backend(mapping=None, mode=None, frames=None, ages=None):
    """A backend with only the camera plumbing, so no ROS or IK is needed."""
    with default_config_path().open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if mapping is not None:
        raw = merge(raw, {"robot": {"left_panel_by_mode": mapping}})
    backend = object.__new__(Ros2StretchBackend)
    backend.config = TeleopConfig(raw)
    backend._last_mode = mode
    backend._camera_lock = threading.Lock()
    backend._camera_frames = dict(frames or {})
    now = time.monotonic()
    backend._camera_stamp = {}
    backend._camera_repeats = {}
    backend._camera_last_encoded = {
        name: now - float((ages or {}).get(name, 0.0)) for name in backend._camera_frames
    }
    return backend


FRAMES = {"head": b"HEAD", "navigation": b"NAV", "gripper": b"GRIP"}


def test_without_a_mapping_the_left_panel_is_the_head_camera():
    backend = _backend(mode=Mode.NAV, frames=FRAMES)
    assert backend._left_panel_source() == "head"
    assert backend.camera_frames()["head"] == b"HEAD"


def test_driving_switches_the_left_panel_to_the_wide_navigation_camera():
    backend = _backend({"NAV": "navigation"}, Mode.NAV, FRAMES)
    assert backend._left_panel_source() == "navigation"
    assert backend.camera_frames()["head"] == b"NAV"


def test_manipulating_keeps_the_head_camera_even_with_the_mapping_set():
    backend = _backend({"NAV": "navigation"}, Mode.MANIP, FRAMES)
    assert backend._left_panel_source() == "head"
    assert backend.camera_frames()["head"] == b"HEAD"


def test_the_mode_is_unknown_before_the_first_command():
    assert _backend({"NAV": "navigation"}, None, FRAMES)._left_panel_source() == "head"


def test_a_dead_stream_hands_the_panel_over_instead_of_freezing():
    """A frozen view of where the robot used to look is worse than the other camera."""
    backend = _backend({"NAV": "navigation"}, Mode.NAV, FRAMES, ages={"navigation": 5.0})
    assert backend.camera_frames()["head"] == b"HEAD"


def test_the_right_panel_is_always_the_wrist_camera():
    for mapping, mode in (({}, Mode.NAV), ({"NAV": "navigation"}, Mode.NAV)):
        assert _backend(mapping, mode, FRAMES).camera_frames()["gripper"] == b"GRIP"


def test_a_camera_with_no_frame_yet_is_simply_absent():
    backend = _backend({"NAV": "navigation"}, Mode.NAV, {"gripper": b"GRIP"})
    assert "head" not in backend.camera_frames()


def test_the_shipped_configuration_leaves_the_panels_as_they_were():
    config = TeleopConfig.load(default_config_path())
    assert config.robot["left_panel_by_mode"] == {}
    assert config.raw["view"]["focus_by_mode"] == {}


class _Stamp:
    def __init__(self, sec, nanosec):
        self.sec, self.nanosec = sec, nanosec


class _Message:
    def __init__(self, sec, nanosec=0):
        self.header = type("H", (), {"stamp": _Stamp(sec, nanosec)})()


def test_a_republished_frame_is_recognised_as_a_stalled_camera():
    """A stalled USB camera resends its last grab, stamp and all, at full rate."""
    backend = _backend()
    backend._node = None
    assert backend._is_repeat("navigation", _Message(10, 5)) is False
    assert backend._is_repeat("navigation", _Message(10, 5)) is True
    assert backend._is_repeat("navigation", _Message(10, 6)) is False
    assert backend._camera_repeats["navigation"] == 0


def test_a_camera_with_no_header_is_not_mistaken_for_stalled():
    backend = _backend()
    assert backend._is_repeat("navigation", object()) is False


def test_each_camera_tracks_its_own_stamps():
    backend = _backend()
    backend._is_repeat("navigation", _Message(10))
    assert backend._is_repeat("head", _Message(10)) is False
