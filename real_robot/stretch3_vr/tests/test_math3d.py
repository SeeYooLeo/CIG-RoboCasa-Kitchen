import math

from stretch3_quest_teleop.math3d import (
    hmd_yaw_pitch_relative,
    map_relative_controller_pose,
    shape_axis,
)
from stretch3_quest_teleop.models import Pose


def test_webxr_translation_maps_forward_left_up():
    anchor = Pose.identity()
    robot = Pose((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
    # WebXR movement: right, up, forward (-Z).
    current = Pose((0.2, 0.3, -0.4), (0.0, 0.0, 0.0, 1.0))
    target = map_relative_controller_pose(anchor, current, robot, 1.0)
    assert target.position == (1.4, 1.8, 3.3)


def test_head_relative_yaw_ignores_translation():
    angle = 0.4
    yaw = (0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2))
    anchor = Pose.identity()
    current = Pose((12.0, -4.0, 7.0), yaw)
    measured_yaw, measured_pitch = hmd_yaw_pitch_relative(anchor, current)
    assert measured_yaw == pytest.approx(-angle)
    assert measured_pitch == pytest.approx(0.0)


def test_axis_deadzone_and_rescale():
    assert shape_axis(0.1, 0.12) == 0.0
    assert shape_axis(-1.0, 0.12) == -1.0


import pytest
