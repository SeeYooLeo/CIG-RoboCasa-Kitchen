from stretch3_quest_teleop.config import TeleopConfig, default_config_path, merge
from stretch3_quest_teleop.controller import TeleopController
from stretch3_quest_teleop.models import ControllerState, Mode, Pose, RobotState, VRFrame


def frame(sequence=0, left=None, right=None, hmd=None):
    return VRFrame(
        sequence=sequence,
        client_time_ms=sequence * 16.0,
        hmd_tracked=True,
        hmd_pose=hmd or Pose.identity(),
        left=left or ControllerState(tracked=True),
        right=right or ControllerState(tracked=True),
    )


def short_x(controller, robot, now):
    controller.process(
        frame(left=ControllerState(tracked=True, primary=True)), robot, now
    )
    return controller.process(frame(), robot, now + 0.1)


def test_deadman_stops_base_and_head_only_moves_while_held():
    controller = TeleopController(TeleopConfig.load(default_config_path()))
    robot = RobotState()
    stopped = controller.process(
        frame(left=ControllerState(tracked=True, stick_y=-1.0)), robot, 1.0
    )
    moving = controller.process(
        frame(left=ControllerState(tracked=True, grip=True, stick_y=-1.0)),
        robot,
        1.1,
    )
    assert stopped.base_linear_mps == 0.0
    assert moving.base_linear_mps > 0.0
    assert "head_recentered" in moving.events


def test_modes_precision_and_explicit_base_policy():
    controller = TeleopController(TeleopConfig.load(default_config_path()))
    robot = RobotState()
    assert short_x(controller, robot, 1.0).mode == Mode.ALIGN
    assert short_x(controller, robot, 2.0).mode == Mode.MANIP

    anchor = Pose((0.1, 1.2, -0.3), (0.0, 0.0, 0.0, 1.0))
    controller.process(
        frame(right=ControllerState(tracked=True, grip=True, pose=anchor)),
        robot,
        3.0,
    )
    moved = Pose((0.1, 1.2, -0.4), (0.0, 0.0, 0.0, 1.0))
    ee = controller.process(
        frame(right=ControllerState(tracked=True, grip=True, pose=moved)),
        robot,
        3.1,
    )
    assert ee.ee_target is not None
    assert ee.allow_ik_base_x

    explicit_base = controller.process(
        frame(
            left=ControllerState(tracked=True, grip=True, stick_x=1.0),
            right=ControllerState(tracked=True, grip=True, pose=moved),
        ),
        robot,
        3.2,
    )
    assert explicit_base.base_angular_rps < 0.0
    assert explicit_base.ee_target is None
    assert not explicit_base.allow_ik_base_x

    precision = controller.process(
        frame(right=ControllerState(tracked=True, trigger=0.5, pose=moved)),
        robot,
        3.3,
    )
    assert precision.mode == Mode.PRECISION
    assert precision.base_locked


def test_watchdog_fails_safe():
    config = TeleopConfig.load(default_config_path())
    controller = TeleopController(config)
    controller.process(frame(), RobotState(), 4.0)
    assert controller.watchdog(4.1) is None
    command = controller.watchdog(4.0 + config.network["input_timeout_s"] + 0.01)
    assert command is not None
    assert command.stale_input
    assert command.base_locked


def test_held_trigger_cannot_enter_unlocked_manipulation():
    controller = TeleopController(TeleopConfig.load(default_config_path()))
    robot = RobotState()
    assert short_x(controller, robot, 1.0).mode == Mode.ALIGN
    controller.process(
        frame(
            left=ControllerState(tracked=True, primary=True),
            right=ControllerState(tracked=True, trigger=0.5),
        ),
        robot,
        2.0,
    )
    command = controller.process(
        frame(right=ControllerState(tracked=True, trigger=0.5)),
        robot,
        2.1,
    )
    assert command.mode == Mode.PRECISION
    assert command.base_locked


def test_view_adjustment_inputs_never_reach_the_robot():
    """The right stick resizes the VR panels in the browser only.

    Two controllers are driven through the same sequence; one of them also has
    every unmapped input pinned hard over. If any of them ever started feeding
    the robot, the two command streams would diverge here rather than on a real
    Stretch.
    """
    config = TeleopConfig.load(default_config_path())
    quiet = TeleopController(config)
    stirred = TeleopController(config)
    robot = RobotState()

    # Everything app.js sends but controller.py is not supposed to act on.
    noise = {
        "stick_x": 1.0,
        "stick_y": -1.0,
        "stick_click": True,
    }
    anchor = Pose((0.1, 1.2, -0.3), (0.0, 0.0, 0.0, 1.0))
    moved = Pose((0.1, 1.25, -0.4), (0.0, 0.0, 0.0, 1.0))

    def step(now, left_kwargs, right_kwargs):
        plain = frame(
            left=ControllerState(tracked=True, **left_kwargs),
            right=ControllerState(tracked=True, **right_kwargs),
        )
        noisy = frame(
            left=ControllerState(tracked=True, trigger=1.0, stick_click=True, **left_kwargs),
            right=ControllerState(tracked=True, **{**right_kwargs, **noise}),
        )
        assert (
            quiet.process(plain, robot, now).status_dict()
            == stirred.process(noisy, robot, now).status_dict()
        )

    # Drive right through the phases, including manipulation with the EE
    # clutch held, which is when the right thumb is nearest the stick.
    step(1.0, {}, {})
    step(1.1, {"grip": True, "stick_y": -1.0}, {})
    step(2.0, {"primary": True}, {})
    step(2.1, {}, {})
    step(3.0, {"primary": True}, {})
    step(3.1, {}, {})
    step(4.0, {}, {"grip": True, "pose": anchor})
    step(4.1, {}, {"grip": True, "pose": moved})
    step(4.2, {}, {"grip": True, "pose": moved, "trigger": 0.5})

    assert quiet.mode == stirred.mode == Mode.PRECISION


def test_head_target_starts_from_the_robots_own_head_pose():
    """Connecting a client must not slew the head.

    look_front is only what the target holds before the driver has said
    anything. Once a joint state arrives the target adopts it, so the first
    command a freshly connected operator sends holds the head where it is.
    """
    config = TeleopConfig.load(default_config_path())
    controller = TeleopController(config)
    parked = RobotState(head_pan=-1.4, head_tilt=-0.9)

    first = controller.process(frame(), parked, 1.0)
    assert first.head_target == (-1.4, -0.9)
    assert "head_target_seeded_from_robot" in first.events
    assert first.head_target != tuple(config.head["look_front"])

    # Once only: later joint states must not drag the target around.
    later = controller.process(frame(), RobotState(head_pan=0.9, head_tilt=-0.1), 1.1)
    assert later.head_target == (-1.4, -0.9)
    assert "head_target_seeded_from_robot" not in later.events


def test_head_target_holds_the_default_until_the_driver_is_ready():
    config = TeleopConfig.load(default_config_path())
    controller = TeleopController(config)
    booting = RobotState(head_pan=-1.4, head_tilt=-0.9, ready=False)
    command = controller.process(frame(), booting, 1.0)
    assert command.head_target == tuple(config.head["look_front"])


def test_end_effector_waits_for_a_fresh_grip_after_its_anchor_is_dropped():
    """Driving the base invalidates the anchor; a held grip must not resume.

    The base moves the arm through the world while the operator's hand stays
    put, so the recorded offset stops meaning anything. Picking up again on a
    grip that was never released would move the arm without being asked.
    """
    controller = TeleopController(TeleopConfig.load(default_config_path()))
    robot = RobotState()
    assert short_x(controller, robot, 1.0).mode == Mode.ALIGN
    assert short_x(controller, robot, 2.0).mode == Mode.MANIP

    anchor = Pose((0.1, 1.2, -0.3), (0.0, 0.0, 0.0, 1.0))
    forward = Pose((0.1, 1.2, -0.4), (0.0, 0.0, 0.0, 1.0))
    further = Pose((0.1, 1.2, -0.5), (0.0, 0.0, 0.0, 1.0))

    def held(pose, now, turn=0.0):
        # Turning, not driving: with the head parked on the gripper in MANIP
        # the forward speed is already scaled to zero, so yaw is the only
        # explicit base motion available here.
        return controller.process(
            frame(
                left=ControllerState(tracked=True, grip=turn != 0.0, stick_x=turn),
                right=ControllerState(tracked=True, grip=True, pose=pose),
            ),
            robot,
            now,
        )

    assert held(anchor, 3.0).ee_target is not None
    assert held(forward, 3.1).ee_target is not None

    driving = held(forward, 3.2, turn=-1.0)
    assert driving.base_angular_rps != 0.0
    assert driving.ee_target is None
    assert "ee_frozen_for_explicit_base_motion" in driving.events

    stopped = held(further, 3.3)
    assert stopped.base_angular_rps == 0.0
    assert stopped.ee_target is None
    assert "release_then_press_right_grip_to_reanchor" in stopped.events

    # Still frozen however long the grip is held.
    assert held(further, 3.4).ee_target is None

    released = controller.process(
        frame(right=ControllerState(tracked=True, pose=further)), robot, 3.5
    )
    assert released.ee_target is None
    pressed = held(further, 3.6)
    assert "ee_recentered" in pressed.events
    assert pressed.ee_target is not None


def test_the_no_motion_overlay_leaves_the_base_at_a_standstill():
    """Checklist stage 2 without hand-editing the shipped configuration."""
    root = default_config_path().parent
    config = TeleopConfig.load(
        default_config_path(), [root / "overlay_base_locked.json"]
    )
    controller = TeleopController(config)
    command = controller.process(
        frame(
            left=ControllerState(tracked=True, grip=True, stick_x=1.0, stick_y=-1.0)
        ),
        RobotState(),
        1.0,
    )
    assert command.base_linear_mps == 0.0
    assert command.base_angular_rps == 0.0


def long_x(controller, robot, now, hold=1.5):
    """Press and release left X, held past long_press_s."""
    controller.process(
        frame(left=ControllerState(tracked=True, primary=True)), robot, now
    )
    return controller.process(frame(), robot, now + hold)


def test_the_operator_can_get_back_to_nav():
    """Long press returns from ALIGN to NAV."""
    controller = TeleopController(TeleopConfig.load(default_config_path()))
    robot = RobotState()
    assert short_x(controller, robot, 1.0).mode == Mode.ALIGN
    assert short_x(controller, robot, 2.0).mode == Mode.MANIP
    # The long press funnels through ALIGN rather than dropping straight from
    # MANIP into full driving speed with the arm still out.
    assert long_x(controller, robot, 3.0).mode == Mode.ALIGN
    assert long_x(controller, robot, 6.0).mode == Mode.NAV


def test_a_long_press_in_nav_leaves_the_mode_alone():
    controller = TeleopController(TeleopConfig.load(default_config_path()))
    robot = RobotState()
    assert controller.mode == Mode.NAV
    assert long_x(controller, robot, 1.0).mode == Mode.NAV


def test_a_short_press_still_goes_deeper_not_home():
    controller = TeleopController(TeleopConfig.load(default_config_path()))
    robot = RobotState()
    assert short_x(controller, robot, 1.0).mode == Mode.ALIGN
    assert short_x(controller, robot, 2.0).mode == Mode.MANIP
    assert short_x(controller, robot, 3.0).mode == Mode.ALIGN
    assert short_x(controller, robot, 4.0).mode == Mode.MANIP


def _no_align():
    config = TeleopConfig.load(default_config_path())
    return TeleopController(
        TeleopConfig(merge(config.raw, {"controls": {"align_enabled": False}}))
    )


def test_without_align_a_short_press_toggles_nav_and_manip():
    controller, robot = _no_align(), RobotState()
    assert controller.mode == Mode.NAV
    assert short_x(controller, robot, 1.0).mode == Mode.MANIP
    assert short_x(controller, robot, 2.0).mode == Mode.NAV
    assert short_x(controller, robot, 3.0).mode == Mode.MANIP


def test_without_align_the_long_press_still_goes_home():
    controller, robot = _no_align(), RobotState()
    assert short_x(controller, robot, 1.0).mode == Mode.MANIP
    assert long_x(controller, robot, 2.0).mode == Mode.NAV
    # And a long press in NAV must not toggle into MANIP by accident.
    assert long_x(controller, robot, 5.0).mode == Mode.NAV


def test_without_align_the_abort_button_lands_in_nav():
    controller, robot = _no_align(), RobotState()
    assert short_x(controller, robot, 1.0).mode == Mode.MANIP
    command = controller.process(
        frame(right=ControllerState(tracked=True, secondary=True)), robot, 2.0
    )
    assert command.mode == Mode.NAV


def test_align_stays_reachable_in_the_shipped_configuration():
    controller, robot = TeleopController(
        TeleopConfig.load(default_config_path())
    ), RobotState()
    assert short_x(controller, robot, 1.0).mode == Mode.ALIGN
