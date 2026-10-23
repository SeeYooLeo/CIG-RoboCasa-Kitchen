from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .config import TeleopConfig
from .math3d import (
    clamp,
    hmd_yaw_pitch_relative,
    map_relative_controller_pose,
    operator_basis,
    shape_axis,
)
from .models import Mode, Pose, RobotCommand, RobotState, VRFrame


@dataclass
class _Buttons:
    left_grip: bool = False
    left_primary: bool = False
    left_secondary: bool = False
    right_grip: bool = False
    right_primary: bool = False
    right_secondary: bool = False
    right_trigger: float = 0.0


class TeleopController:
    """Pure control logic. It has no ROS or network dependencies."""

    def __init__(self, config: TeleopConfig):
        self.config = config
        # Where the operator stands, resolved once: it cannot change mid-session
        # without invalidating every anchor already taken.
        self._basis = operator_basis(
            float(config.controls.get("operator_yaw_deg", 0.0))
        )
        self.mode = Mode.NAV
        self._previous = _Buttons()
        self._last_input_monotonic: float | None = None
        self._left_primary_pressed_at: float | None = None

        self._head_vr_anchor: Pose | None = None
        self._head_robot_anchor: tuple[float, float] | None = None
        self._last_head_target = tuple(float(x) for x in config.head["look_front"])
        self._head_manual_was_enabled = False
        self._head_seeded = False

        self._ee_vr_anchor: Pose | None = None
        self._ee_robot_anchor: Pose | None = None
        self._ee_reanchor_required = False
        self._ee_hint_sent = False
        self._explicit_base_was_active = False

        # Latching gripper. `_gripper_latched` stays None until the operator
        # actually pulls the trigger once, so connecting never commands a
        # gripper that is already holding something to open.
        self._gripper_latched: bool | None = None

    @staticmethod
    def _rising(current: bool, previous: bool) -> bool:
        return current and not previous

    @staticmethod
    def _falling(current: bool, previous: bool) -> bool:
        return previous and not current

    def _button_snapshot(self, frame: VRFrame) -> _Buttons:
        return _Buttons(
            left_grip=frame.left.grip,
            left_primary=frame.left.primary,
            left_secondary=frame.left.secondary,
            right_grip=frame.right.grip,
            right_primary=frame.right.primary,
            right_secondary=frame.right.secondary,
            right_trigger=frame.right.trigger,
        )

    def _align_enabled(self) -> bool:
        """ALIGN is a slow-approach base speed and nothing else.

        It does not enable IK, change the head, or lock anything: the only
        difference from NAV is 0.1 m/s instead of 0.35. Sites that would rather
        toggle straight between driving and manipulating can switch it off.
        """
        return bool(self.config.controls.get("align_enabled", True))

    def _transition_mode(
        self, frame: VRFrame, robot: RobotState, now: float, events: list[str]
    ) -> None:
        buttons = self._button_snapshot(frame)
        if self._rising(buttons.left_primary, self._previous.left_primary):
            self._left_primary_pressed_at = now

        if self._falling(buttons.left_primary, self._previous.left_primary):
            held = (
                now - self._left_primary_pressed_at
                if self._left_primary_pressed_at is not None
                else 0.0
            )
            self._left_primary_pressed_at = None
            long_press = held >= float(self.config.controls["long_press_s"])
            if not self._align_enabled():
                # ALIGN differs from NAV only in base speed, so with it disabled
                # the short press is a plain NAV/MANIP toggle and the long press
                # is the same way home. The cost is that leaving MANIP restores
                # full driving speed in one step, with the arm wherever it was.
                if self.mode == Mode.NAV:
                    if not long_press:
                        self.mode = Mode.MANIP
                        self._require_ee_reanchor()
                        events.append("mode:NAV->MANIP")
                else:
                    self.mode = Mode.NAV
                    self._clear_ee_anchor()
                    events.append("mode:MANIP->NAV")
            elif long_press:
                # Long press returns through ALIGN before enabling NAV speed.
                if self.mode == Mode.ALIGN:
                    self.mode = Mode.NAV
                    self._clear_ee_anchor()
                    events.append("mode:ALIGN->NAV")
                elif self.mode != Mode.NAV:
                    self.mode = Mode.ALIGN
                    self._clear_ee_anchor()
                    events.append("safe_return_requested")
            elif self.mode == Mode.NAV:
                self.mode = Mode.ALIGN
                events.append("mode:NAV->ALIGN")
            elif self.mode == Mode.ALIGN:
                self.mode = Mode.MANIP
                self._require_ee_reanchor()
                events.append("mode:ALIGN->MANIP")
            elif self.mode == Mode.MANIP:
                self.mode = Mode.ALIGN
                self._clear_ee_anchor()
                events.append("mode:MANIP->ALIGN")

        precision_enabled = bool(
            self.config.controls.get("precision_enabled", True)
        )

        if self._rising(buttons.right_primary, self._previous.right_primary):
            if self.mode == Mode.MANIP and precision_enabled:
                self.mode = Mode.PRECISION
                events.append("mode:MANIP->PRECISION")
            elif (
                self.mode == Mode.PRECISION
                and buttons.right_trigger < 0.1
                and not robot.contact_detected
            ):
                self.mode = Mode.MANIP
                self._require_ee_reanchor()
                events.append("mode:PRECISION->MANIP")

        if (
            precision_enabled
            and self.mode == Mode.MANIP
            and bool(self.config.controls["auto_precision_on_gripper_close"])
            and buttons.right_trigger
            >= float(self.config.controls["gripper_close_trigger"])
        ):
            self.mode = Mode.PRECISION
            events.append("auto_precision:gripper_closing")

        if precision_enabled and robot.contact_detected and self.mode == Mode.MANIP:
            self.mode = Mode.PRECISION
            events.append("auto_precision:contact")

        if self._rising(buttons.right_secondary, self._previous.right_secondary):
            if self.mode in (Mode.MANIP, Mode.PRECISION):
                target = Mode.ALIGN if self._align_enabled() else Mode.NAV
                self.mode = target
                self._clear_ee_anchor()
                events.append(f"operator_abort_to_{target.value}")

    def _head_command(
        self, frame: VRFrame, robot: RobotState, events: list[str]
    ) -> tuple[float, float]:
        head_cfg = self.config.head
        controls = self.config.controls
        if not self._head_seeded and robot.ready:
            # Adopt the head's real position the first time the driver reports
            # one. Without this the first command after a client connects slews
            # the head to look_front, which on a robot parked looking at its own
            # gripper is a fast 90 degree swing nobody asked for.
            pan_limits = head_cfg["pan_limits"]
            tilt_limits = head_cfg["tilt_limits"]
            self._last_head_target = (
                clamp(robot.head_pan, float(pan_limits[0]), float(pan_limits[1])),
                clamp(robot.head_tilt, float(tilt_limits[0]), float(tilt_limits[1])),
            )
            self._head_seeded = True
            events.append("head_target_seeded_from_robot")
        if self.mode == Mode.PRECISION:
            self._last_head_target = tuple(float(x) for x in head_cfg["look_at_ee"])
            self._head_manual_was_enabled = False
            return self._last_head_target

        if self.mode == Mode.MANIP:
            manual_enabled = (
                frame.left.tracked
                and frame.hmd_tracked
                and frame.left.grip
                and frame.left.secondary
            )
            default_target = tuple(float(x) for x in head_cfg["look_at_ee"])
        else:
            manual_enabled = (
                frame.left.tracked and frame.hmd_tracked and frame.left.grip
            )
            default_target = self._last_head_target

        if manual_enabled and not self._head_manual_was_enabled:
            self._head_vr_anchor = frame.hmd_pose
            self._head_robot_anchor = (robot.head_pan, robot.head_tilt)
            events.append("head_recentered")

        if manual_enabled and self._head_vr_anchor and self._head_robot_anchor:
            yaw, pitch = hmd_yaw_pitch_relative(
                self._head_vr_anchor, frame.hmd_pose
            )
            pan = self._head_robot_anchor[0] + float(
                controls["head_pan_gain"]
            ) * yaw
            tilt = self._head_robot_anchor[1] + float(
                controls["head_tilt_gain"]
            ) * pitch
            pan_limits = head_cfg["pan_limits"]
            tilt_limits = head_cfg["tilt_limits"]
            self._last_head_target = (
                clamp(pan, float(pan_limits[0]), float(pan_limits[1])),
                clamp(tilt, float(tilt_limits[0]), float(tilt_limits[1])),
            )
        elif self.mode == Mode.MANIP:
            self._last_head_target = default_target

        self._head_manual_was_enabled = manual_enabled
        return self._last_head_target

    def _view_speed_scale(self, head_pan: float) -> float:
        base_cfg = self.config.base
        start = float(base_cfg["head_slowdown_start_rad"])
        stop = float(base_cfg["head_translation_stop_rad"])
        magnitude = abs(head_pan)
        if magnitude <= start:
            return 1.0
        if magnitude >= stop:
            return 0.0
        fraction = (magnitude - start) / (stop - start)
        return 0.5 * (1.0 + math.cos(math.pi * fraction))

    def _base_command(
        self, frame: VRFrame, head_target: tuple[float, float]
    ) -> tuple[float, float, bool]:
        if self.mode == Mode.PRECISION:
            return 0.0, 0.0, False
        if not (frame.left.tracked and frame.left.grip):
            return 0.0, 0.0, False

        deadzone = float(self.config.controls["axis_deadzone"])
        stick_x = shape_axis(frame.left.stick_x, deadzone)
        stick_y = shape_axis(frame.left.stick_y, deadzone)
        if self.mode == Mode.NAV:
            linear_max = float(self.config.base["nav_linear_mps"])
            angular_max = float(self.config.base["nav_angular_rps"])
        elif self.mode == Mode.ALIGN:
            linear_max = float(self.config.base["align_linear_mps"])
            angular_max = float(self.config.base["align_angular_rps"])
        else:
            linear_max = float(self.config.base["manip_linear_mps"])
            angular_max = float(self.config.base["manip_angular_rps"])

        view_scale = self._view_speed_scale(head_target[0])
        linear = -stick_y * linear_max * view_scale
        angular = -stick_x * angular_max
        active = abs(linear) > 1e-6 or abs(angular) > 1e-6
        return linear, angular, active

    def _update_gripper_latch(self, frame: VRFrame, events: list[str]) -> None:
        """Flip the gripper on each fresh trigger pull, instead of tracking it.

        Carrying something across a room otherwise means holding the trigger
        the whole way, which competes with the grip the operator also has to
        hold to steer.
        """
        threshold = float(self.config.controls["gripper_close_trigger"])
        pulled = frame.right.trigger >= threshold
        was_pulled = self._previous.right_trigger >= threshold
        if pulled and not was_pulled:
            self._gripper_latched = not bool(self._gripper_latched)
            events.append(
                "gripper:close" if self._gripper_latched else "gripper:open"
            )

    def _clear_ee_anchor(self) -> None:
        self._ee_vr_anchor = None
        self._ee_robot_anchor = None

    def _require_ee_reanchor(self) -> None:
        """Demand a fresh, deliberate right-grip press before the arm moves."""
        self._ee_reanchor_required = True
        self._ee_hint_sent = False

    def _hint_reanchor(self, events: list[str]) -> None:
        """Say it once. Repeating it every frame buries the other events."""
        if self._ee_reanchor_required and not self._ee_hint_sent:
            events.append("release_then_press_right_grip_to_reanchor")
            self._ee_hint_sent = True

    def _ee_command(
        self,
        frame: VRFrame,
        robot: RobotState,
        explicit_base_active: bool,
        events: list[str],
    ) -> Pose | None:
        in_manipulation = self.mode in (Mode.MANIP, Mode.PRECISION)
        if not in_manipulation or not frame.right.tracked:
            self._clear_ee_anchor()
            return None

        if explicit_base_active:
            if not self._explicit_base_was_active:
                events.append("ee_frozen_for_explicit_base_motion")
            self._require_ee_reanchor()
            self._clear_ee_anchor()
            return None

        right_grip_rising = self._rising(
            frame.right.grip, self._previous.right_grip
        )
        if not frame.right.grip:
            self._clear_ee_anchor()
            self._hint_reanchor(events)
            return None

        if self._ee_reanchor_required and not right_grip_rising:
            # The grip was still held when the anchor was thrown away, by a mode
            # change, by driving the base, or by a dropped link. Re-anchoring on
            # that stale hold would restart the arm without the operator asking
            # again, so wait for a release and a fresh press.
            self._hint_reanchor(events)
            return None

        if right_grip_rising or self._ee_vr_anchor is None:
            self._ee_vr_anchor = frame.right.pose
            self._ee_robot_anchor = robot.ee_pose
            self._ee_reanchor_required = False
            events.append("ee_recentered")

        if not self._ee_vr_anchor or not self._ee_robot_anchor:
            return None

        return map_relative_controller_pose(
            self._ee_vr_anchor,
            frame.right.pose,
            self._ee_robot_anchor,
            float(self.config.controls["ee_translation_scale"]),
            self._basis,
            float(self.config.controls.get("ee_rotation_scale", 1.0)),
        )

    def process(
        self, frame: VRFrame, robot: RobotState, now: float | None = None
    ) -> RobotCommand:
        now = time.monotonic() if now is None else now
        self._last_input_monotonic = now
        events: list[str] = []
        self._transition_mode(frame, robot, now, events)

        head_target = self._head_command(frame, robot, events)
        linear, angular, explicit_base_active = self._base_command(
            frame, head_target
        )
        ee_target = self._ee_command(
            frame, robot, explicit_base_active, events
        )
        self._explicit_base_was_active = explicit_base_active

        latching = str(self.config.controls.get("gripper_mode", "analog")) == "toggle"
        gripper = None
        if latching:
            if frame.right.tracked:
                self._update_gripper_latch(frame, events)
            # Held in every mode, not just MANIP: the point of latching is to
            # keep carrying the object while the operator drives the base.
            if self._gripper_latched is not None:
                gripper = 0.0 if self._gripper_latched else 1.0
        elif self.mode in (Mode.MANIP, Mode.PRECISION) and frame.right.tracked:
            gripper = 1.0 - frame.right.trigger

        command = RobotCommand(
            mode=self.mode,
            base_linear_mps=linear,
            base_angular_rps=angular,
            head_target=head_target,
            ee_target=ee_target,
            gripper_open_fraction=gripper,
            allow_ik_base_x=(self.mode == Mode.MANIP and not explicit_base_active),
            base_locked=(self.mode == Mode.PRECISION),
            events=events,
        )
        self._previous = self._button_snapshot(frame)
        return command

    def watchdog(
        self, now: float | None = None, force: bool = False
    ) -> RobotCommand | None:
        now = time.monotonic() if now is None else now
        timeout = float(self.config.network["input_timeout_s"])
        stale = (
            force
            or self._last_input_monotonic is None
            or now - self._last_input_monotonic > timeout
        )
        if not stale:
            return None
        self._clear_ee_anchor()
        self._require_ee_reanchor()
        return RobotCommand(
            mode=self.mode,
            base_locked=True,
            stale_input=True,
            events=["failsafe_stop:input_timeout" if not force else "failsafe_stop:shutdown"],
        )
