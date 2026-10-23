import types

import pytest

from stretch3_quest_teleop.backends.ros2 import rotation_code
from stretch3_quest_teleop.config import TeleopConfig, default_config_path

# Stand-in for OpenCV so this runs without a ROS or vision install.
FAKE_CV2 = types.SimpleNamespace(
    ROTATE_90_CLOCKWISE="cw",
    ROTATE_180="half",
    ROTATE_90_COUNTERCLOCKWISE="ccw",
)


def test_quarter_turns_map_to_opencv_codes():
    assert rotation_code(FAKE_CV2, 0) is None
    assert rotation_code(FAKE_CV2, 90) == "cw"
    assert rotation_code(FAKE_CV2, 180) == "half"
    assert rotation_code(FAKE_CV2, 270) == "ccw"
    assert rotation_code(FAKE_CV2, -90) == "ccw"
    assert rotation_code(FAKE_CV2, 360) is None


def test_an_unusable_angle_is_rejected_at_start_up():
    """A typo has to fail on the ground, not as a tilted world in the headset."""
    for bad in (45, 91, "left", None):
        with pytest.raises(ValueError):
            rotation_code(FAKE_CV2, bad)


def test_the_shipped_configuration_rotates_nothing():
    rotations = TeleopConfig.load(default_config_path()).robot["camera_rotate_deg"]
    assert set(rotations) == {"navigation", "head", "gripper"}
    assert all(rotation_code(FAKE_CV2, value) is None for value in rotations.values())
