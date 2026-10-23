from __future__ import annotations

from dataclasses import replace

from .backends.mock import MockRobotBackend
from .config import TeleopConfig, default_config_path
from .controller import TeleopController
from .math3d import mat_vec, operator_basis, vec_sub
from .models import ControllerState, Mode, Pose, VRFrame


def _frame(
    sequence: int,
    *,
    left: ControllerState | None = None,
    right: ControllerState | None = None,
    hmd: Pose | None = None,
) -> VRFrame:
    return VRFrame(
        sequence=sequence,
        client_time_ms=float(sequence * 16),
        hmd_tracked=True,
        hmd_pose=hmd or Pose.identity(),
        left=left or ControllerState(tracked=True),
        right=right or ControllerState(tracked=True),
    )


def run_smoke(config: TeleopConfig | None = None) -> dict[str, object]:
    config = config or TeleopConfig.load(default_config_path())
    controller = TeleopController(config)
    backend = MockRobotBackend()
    backend.start()
    now = 10.0
    sequence = 0

    def send(frame: VRFrame, dt: float = 0.05):
        nonlocal now
        now += dt
        command = controller.process(frame, backend.state(), now=now)
        backend.apply(command)
        return command

    # Deadman released: no motion.
    command = send(
        _frame(
            sequence,
            left=ControllerState(tracked=True, stick_y=-1.0),
        )
    )
    assert command.base_linear_mps == 0.0
    sequence += 1

    # Deadman held: navigation and head control are enabled.
    command = send(
        _frame(
            sequence,
            left=ControllerState(tracked=True, grip=True, stick_y=-1.0),
        )
    )
    assert command.base_linear_mps > 0.0
    sequence += 1

    # X short press/release: NAV -> ALIGN.
    send(
        _frame(
            sequence,
            left=ControllerState(tracked=True, primary=True),
        )
    )
    sequence += 1
    command = send(_frame(sequence))
    assert command.mode == Mode.ALIGN
    sequence += 1

    # X short press/release: ALIGN -> MANIP.
    send(
        _frame(
            sequence,
            left=ControllerState(tracked=True, primary=True),
        )
    )
    sequence += 1
    command = send(_frame(sequence))
    assert command.mode == Mode.MANIP
    sequence += 1

    # Right grip anchors, then controller movement commands the EE.
    anchor = Pose((0.1, 1.2, -0.3), (0.0, 0.0, 0.0, 1.0))
    send(
        _frame(
            sequence,
            right=ControllerState(tracked=True, grip=True, pose=anchor),
        )
    )
    sequence += 1
    moved = replace(anchor, position=(0.1, 1.25, -0.4))
    command = send(
        _frame(
            sequence,
            right=ControllerState(tracked=True, grip=True, pose=moved),
        )
    )
    assert command.ee_target is not None
    # Pushing the controller away from the operator must move the end effector
    # away from the operator, whichever side of the robot they are standing on.
    # Asserting a fixed +X here would quietly pass for one operator position
    # and mean nothing for any other.
    forward = mat_vec(
        operator_basis(float(config.controls.get("operator_yaw_deg", 0.0))),
        (0.0, 0.0, -1.0),
    )
    travelled = vec_sub(
        command.ee_target.position, backend.commands[-2].ee_target.position
    )
    assert sum(a * b for a, b in zip(travelled, forward)) > 0.0
    sequence += 1

    # Closing the gripper locks the base before contact -- where the site has
    # kept the precision phase. Deployments that drive while carrying disable
    # it, and there the same trigger must leave the base free; asserting one
    # of those two outcomes unconditionally would make this self-test lie
    # about whichever configuration is actually loaded.
    command = send(
        _frame(
            sequence,
            right=ControllerState(
                tracked=True, grip=True, trigger=0.5, pose=moved
            ),
        )
    )
    if bool(config.controls.get("precision_enabled", True)):
        assert command.mode == Mode.PRECISION
        assert command.base_locked
    else:
        assert command.mode == Mode.MANIP
        assert not command.base_locked

    # Input timeout always produces a zero-velocity fail-safe command.
    failsafe = controller.watchdog(
        now=now + float(config.network["input_timeout_s"]) + 0.01
    )
    assert failsafe is not None
    assert failsafe.stale_input
    assert failsafe.base_linear_mps == 0.0
    assert failsafe.base_angular_rps == 0.0
    backend.apply(failsafe)
    backend.stop()
    return {
        "ok": True,
        "commands": len(backend.commands),
        "final_mode": controller.mode.value,
        "failsafe": failsafe.events[0],
    }
