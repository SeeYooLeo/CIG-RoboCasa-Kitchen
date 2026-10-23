import json
from types import SimpleNamespace

import pytest
from stretch3_quest_teleop.legacy_replay import JOINTS, VELOCITIES, load_legacy, apply_action
from stretch3_quest_teleop.robot_access import RobotControlLock


def test_legacy_metadata_and_required_fields(tmp_path):
    action = dict.fromkeys((*JOINTS, *VELOCITIES, "delta_s_robot", "delta_theta"), 0.0)
    (tmp_path / "labels.json").write_text(json.dumps({"0": {"actions": action, "observations": {}}}))
    (tmp_path / "configs.json").write_text(json.dumps({"fps": 15}))
    frames, fps = load_legacy(tmp_path)
    assert fps == 15 and len(frames) == 1
    with pytest.raises(ValueError): load_legacy(tmp_path, fps=0)
    del action["v_gripper"]
    (tmp_path / "labels.json").write_text(json.dumps({"0": {"actions": action}}))
    with pytest.raises(ValueError, match="v_gripper"): load_legacy(tmp_path)


def test_legacy_gripper_retains_actuator_semantics():
    calls = []
    class Motor:
        def move_to(self, *a, **k): calls.append(("position", a))
        def set_velocity(self, *a, **k): calls.append(("velocity", a))
        def move_by(self, *a): calls.append(("gripper_step", a))
        def get_joint(self, name): return self
    robot = SimpleNamespace(lift=Motor(), arm=Motor(), base=Motor(), head=Motor(),
                            end_of_arm=Motor(), push_command=lambda: None)
    action = dict.fromkeys((*JOINTS, *VELOCITIES, "delta_s_robot", "delta_theta"), 0.0)
    action.update(v_gripper=-0.1, delta_s_robot=0.02)
    apply_action(robot, action, "target", 15)
    assert ("gripper_step", (-60.0,)) in calls
    assert ("velocity", (0.3, 0.0)) in calls


def test_robot_control_is_exclusive():
    lock = RobotControlLock()
    try:
        with pytest.raises(RuntimeError, match="owns robot control"):
            RobotControlLock()
    finally:
        lock.close()
    RobotControlLock().close()
