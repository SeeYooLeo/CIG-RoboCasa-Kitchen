from types import SimpleNamespace
import sys
import threading
import types
import time
from concurrent.futures import Future

import pytest

from stretch3_quest_teleop.backends.ros2 import Ros2StretchBackend
from stretch3_quest_teleop.config import TeleopConfig, default_config_path
from stretch3_quest_teleop.models import Mode, RobotCommand, RobotState


def completed(value):
    future = Future()
    future.set_result(value)
    return future


@pytest.fixture
def fake_backend(monkeypatch):
    durations = types.ModuleType("builtin_interfaces.msg")
    durations.Duration = lambda **kwargs: SimpleNamespace(**kwargs)
    trajectories = types.ModuleType("trajectory_msgs.msg")
    trajectories.JointTrajectoryPoint = lambda: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "builtin_interfaces.msg", durations)
    monkeypatch.setitem(sys.modules, "trajectory_msgs.msg", trajectories)
    backend = Ros2StretchBackend.__new__(Ros2StretchBackend)
    backend.config = TeleopConfig.load(default_config_path())
    backend._started = True
    backend._state = RobotState(joint_state_received_monotonic=time.monotonic())
    backend._goal_futures = []
    backend._halted = False
    backend._motion_fault = None
    backend._stop_client = SimpleNamespace(wait_for_service=lambda **kw: True,
        call_async=lambda _: completed(SimpleNamespace(success=True)))
    backend._state_lock = threading.Lock()
    backend._last_mode = Mode.NAV
    backend._explicit_base_active = False
    backend._manip_base_anchor = None
    backend._last_command_time = 0
    backend._types = {"Trigger": SimpleNamespace(Request=lambda: None),
                      "FollowJointTrajectory": SimpleNamespace(
        Goal=lambda: SimpleNamespace(trajectory=SimpleNamespace()))}
    goals = []
    def submit(goal):
        goals.append(goal)
        return completed(SimpleNamespace(accepted=True, status=2,
            cancel_goal_async=lambda: completed(SimpleNamespace(return_code=0))))
    backend._trajectory_client = SimpleNamespace(
        server_is_ready=lambda: True, send_goal_async=submit)
    bases = []
    backend._publish_base = lambda linear, angular: bases.append((linear, angular))
    return backend, goals, bases


def test_recorded_joint_targets_match_submitted_ros_goal(fake_backend):
    backend, goals, bases = fake_backend
    command = RobotCommand(mode=Mode.NAV, base_linear_mps=0.02,
                           head_target=(0.1, -0.5), gripper_open_fraction=0.5)
    receipt = backend.apply(command)
    assert receipt["base_linear_mps"] == bases[-1][0]
    targets = dict(zip(goals[-1].trajectory.joint_names, goals[-1].trajectory.points[0].positions))
    assert receipt["joint_targets"] == targets
    assert targets["gripper_aperture"] == 0.045
    # A rate-limited step records no phantom joint command.
    assert backend.apply(command)["joint_targets"] == {}
    backend.apply_recorded_action(receipt)
    replayed = dict(zip(goals[-1].trajectory.joint_names, goals[-1].trajectory.points[0].positions))
    assert replayed == targets
    assert bases[-1] == (0.02, 0.0)


def test_stop_cancels_pending_acceptance_and_replays_stop(fake_backend):
    backend, _, bases = fake_backend
    future = Future()
    calls = []
    def accept():
        time.sleep(0.02)
        future.set_result(SimpleNamespace(accepted=True, status=2,
            cancel_goal_async=lambda: calls.append("cancel") or completed(SimpleNamespace(return_code=0))))
    thread = threading.Thread(target=accept)
    thread.start()
    backend._goal_futures.append(future)
    receipt = backend.apply(RobotCommand(mode=Mode.NAV, stale_input=True))
    thread.join()
    assert calls == ["cancel"] and bases[-1] == (0, 0)
    assert receipt["stop_motion"] and not receipt["joint_targets"]
    backend._halted = False
    backend._goal_futures.append(future)
    backend.apply_recorded_action(receipt)
    assert calls == ["cancel", "cancel"]


def test_stop_failure_blocks_new_motion_but_still_calls_stop_service(fake_backend):
    backend, _, bases = fake_backend
    calls = []
    backend._goal_futures = [completed(SimpleNamespace(accepted=True, status=2,
        cancel_goal_async=lambda: completed(SimpleNamespace(return_code=1))))]
    backend._stop_client.call_async = lambda _: calls.append("stop") or completed(SimpleNamespace(success=True))
    with pytest.raises(RuntimeError, match="cancellation"):
        backend.apply(RobotCommand(mode=Mode.NAV, stale_input=True))
    assert calls == ["stop"] and bases[-1] == (0, 0)
    with pytest.raises(RuntimeError):
        backend.apply(RobotCommand(mode=Mode.NAV, base_linear_mps=0.1))
    assert bases[-1] == (0, 0)


def test_stale_feedback_stops_before_moving(fake_backend):
    backend, _, bases = fake_backend
    backend._state.joint_state_received_monotonic = time.monotonic() - 1
    with pytest.raises(RuntimeError, match="feedback is stale"):
        backend.apply(RobotCommand(mode=Mode.NAV, base_linear_mps=0.1))
    assert bases == [(0, 0)]


def test_rejected_goal_cannot_be_saved_as_success(fake_backend):
    backend, _, _ = fake_backend
    backend._goal_futures = [completed(SimpleNamespace(accepted=False))]
    with pytest.raises(RuntimeError, match="rejected"):
        backend.apply(RobotCommand(mode=Mode.NAV, stale_input=True))
