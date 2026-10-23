from __future__ import annotations

import math
import threading
import time
from dataclasses import replace
from typing import Any

from ..config import TeleopConfig
from ..depth_overlay import apply_overlay
from ..kinematics import JointSolution, StretchAIKinematics
from ..math3d import clamp
from ..models import Mode, Pose, RobotCommand, RobotState
from .base import RobotBackend

# Only quarter turns: anything else would need interpolation and would cost
# more on the robot's onboard CPU than the whole JPEG encode.
# Only one of these ever reaches the left panel; the mode decides which.
_LEFT_PANEL_CAMERAS = ("head", "navigation")

_ROTATION_ATTRIBUTES = {
    0: None,
    90: "ROTATE_90_CLOCKWISE",
    180: "ROTATE_180",
    270: "ROTATE_90_COUNTERCLOCKWISE",
}


def rotation_code(cv2_module: Any, degrees: Any) -> Any:
    """Translate a configured angle into an OpenCV rotate code.

    The head D435if is mounted sideways on Stretch, so the raw topic can arrive
    with the world lying on its side. Raising here rather than ignoring a bad
    value means a typo is caught at start-up, on the ground, instead of showing
    up as a tilted world once the operator already has the headset on.
    """
    try:
        normalised = int(degrees) % 360
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"camera_rotate_deg must be a number, got {degrees!r}"
        ) from exc
    attribute = _ROTATION_ATTRIBUTES.get(normalised)
    if normalised not in _ROTATION_ATTRIBUTES:
        raise ValueError(
            "camera_rotate_deg must be 0, 90, 180 or 270 "
            f"(clockwise), got {degrees!r}"
        )
    return None if attribute is None else getattr(cv2_module, attribute)


class Ros2StretchBackend(RobotBackend):
    """Direct Stretch ROS 2 backend.

    The base remains in ROS navigation mode so `/stretch/cmd_vel` can rotate
    the robot in NAV, ALIGN, and MANIP. Non-base joints use the official
    FollowJointTrajectory action server.
    """

    def __init__(self, config: TeleopConfig):
        self.config = config
        self.kinematics = StretchAIKinematics()
        self._state = RobotState(ready=False)
        self._state_lock = threading.Lock()
        self._camera_lock = threading.Lock()
        self._camera_frames: dict[str, bytes] = {}
        self._recording_enabled = False
        self._recording_images: dict[str, dict] = {}
        self._camera_last_encoded: dict[str, float] = {}
        self._camera_rotations: dict[str, Any] = {}
        self._depth_settings: dict[str, dict[str, Any]] = {}
        self._depth_frames: dict[str, tuple[float, Any]] = {}
        self._camera_encoded_count: dict[str, int] = {}
        self._camera_stamp: dict[str, tuple[int, int]] = {}
        self._camera_repeats: dict[str, int] = {}
        self._depth_lock = threading.Lock()
        self._np: Any = None
        self._started = False
        self._last_command_time = 0.0
        self._last_mode: Mode | None = None
        self._manip_base_anchor: tuple[float, float, float] | None = None
        self._explicit_base_active = False

        self._rclpy = None
        self._node = None
        self._executor = None
        self._executor_thread: threading.Thread | None = None
        self._cmd_vel_pub = None
        self._trajectory_client = None
        self._navigation_client = None
        self._stop_client = None
        self._control_lock = None
        self._goal_futures = []
        self._halted = False
        self._motion_fault = None

    def start(self) -> None:
        if self._started:
            return
        from ..robot_access import RobotControlLock
        self._control_lock = RobotControlLock()
        try:
            import rclpy
            from control_msgs.action import FollowJointTrajectory
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
            from rclpy.action import ActionClient
            from rclpy.executors import MultiThreadedExecutor
            from sensor_msgs.msg import JointState
            from std_msgs.msg import String
            from std_srvs.srv import Trigger
        except ImportError as exc:
            raise RuntimeError(
                "ROS2 backend requires a sourced ROS 2 Humble environment "
                "with stretch_ros2 installed."
            ) from exc

        self._types = {
            "FollowJointTrajectory": FollowJointTrajectory,
            "Twist": Twist,
            "Trigger": Trigger,
        }
        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("stretch3_quest_teleop")
        robot_cfg = self.config.robot
        self._cmd_vel_pub = self._node.create_publisher(
            Twist, str(robot_cfg["cmd_vel_topic"]), 10
        )
        self._node.create_subscription(
            JointState,
            str(robot_cfg["joint_states_topic"]),
            self._joint_state_callback,
            20,
        )
        self._node.create_subscription(
            Odometry,
            str(robot_cfg["odom_topic"]),
            self._odom_callback,
            20,
        )
        self._node.create_subscription(
            String,
            str(robot_cfg["mode_topic"]),
            self._mode_callback,
            10,
        )
        self._trajectory_client = ActionClient(
            self._node,
            FollowJointTrajectory,
            str(robot_cfg["trajectory_action"]),
        )
        self._navigation_client = self._node.create_client(
            Trigger, str(robot_cfg["navigation_mode_service"])
        )
        self._stop_client = self._node.create_client(
            Trigger, str(robot_cfg["stop_service"])
        )
        self._configure_camera_subscriptions()

        self._executor = MultiThreadedExecutor(num_threads=3)
        self._executor.add_node(self._node)
        self._executor_thread = threading.Thread(
            target=self._executor.spin,
            name="stretch3-vr-rclpy",
            daemon=True,
        )
        self._executor_thread.start()
        self._started = True
        self._request_navigation_mode()

    def _request_navigation_mode(self) -> None:
        assert self._navigation_client is not None
        service = str(self.config.robot["navigation_mode_service"])
        if not self._navigation_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"Stretch service not available: {service}")
        request = self._types["Trigger"].Request()
        future = self._navigation_client.call_async(request)
        deadline = time.monotonic() + 5.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done():
            raise RuntimeError(f"Timed out calling {service}")
        result = future.result()
        if result is None or not result.success:
            message = getattr(result, "message", "no response")
            raise RuntimeError(f"Could not enter navigation mode: {message}")

    def _joint_state_callback(self, message: Any) -> None:
        joints = {name: float(value) for name, value in zip(message.name, message.position)}
        efforts = {name: float(value) for name, value in zip(message.name, message.effort)}
        with self._state_lock:
            self._state.joints = joints
            self._state.joint_state_received_monotonic = time.monotonic()
            self._state.head_pan = joints.get("joint_head_pan", self._state.head_pan)
            self._state.head_tilt = joints.get("joint_head_tilt", self._state.head_tilt)
            self._state.ready = True
            # Effort is retained for logging and future calibrated contact rules.
            self._state.joints.update({f"effort:{k}": v for k, v in efforts.items()})
            try:
                self._state.ee_pose = self.kinematics.forward(
                    self._state, self._current_virtual_base_x_locked()
                )
            except Exception:
                # Joint states must never stop the ROS executor because FK is optional
                # during startup.
                pass

    def _odom_callback(self, message: Any) -> None:
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
        )
        with self._state_lock:
            self._state.base_pose = (float(position.x), float(position.y), float(yaw))
            self._state.odom_received_monotonic = time.monotonic()

    def _mode_callback(self, message: Any) -> None:
        with self._state_lock:
            self._state.joints["driver_mode_navigation"] = (
                1.0 if message.data == "navigation" else 0.0
            )

    def _configure_camera_subscriptions(self) -> None:
        try:
            import cv2
            from cv_bridge import CvBridge
            from sensor_msgs.msg import Image
        except ImportError:
            if self._node is not None:
                self._node.get_logger().warning(
                    "cv_bridge/OpenCV unavailable; VR camera streaming is disabled"
                )
            return
        self._cv2 = cv2
        self._cv_bridge = CvBridge()
        topics = {
            "navigation": self.config.robot["navigation_camera_topic"],
            "head": self.config.robot["head_camera_topic"],
            "gripper": self.config.robot["gripper_camera_topic"],
        }
        rotations = self.config.robot.get("camera_rotate_deg") or {}
        self._camera_rotations = {
            name: rotation_code(cv2, rotations.get(name, 0))
            for name in topics
        }
        for name, topic in topics.items():
            self._node.create_subscription(
                Image,
                str(topic),
                lambda message, camera_name=name: self._camera_callback(
                    camera_name, message
                ),
                2,
            )
        self._configure_depth_subscriptions(Image)

    def _configure_depth_subscriptions(self, image_type: Any) -> None:
        """Subscribe to the aligned depth stream of every overlaid camera.

        Only the aligned_depth_to_color topic will do: it is published in the
        colour optical frame at colour resolution, which is what makes the
        overlay a plain pixel-wise blend instead of a reprojection.
        """
        overlays = self.config.raw.get("depth_overlay") or {}
        if not isinstance(overlays, dict):
            raise ValueError("depth_overlay must be a JSON object keyed by camera")
        for name, settings in overlays.items():
            if not isinstance(settings, dict) or not settings.get("enabled"):
                continue
            topic = settings.get("depth_topic")
            if not topic:
                raise ValueError(f"depth_overlay.{name} needs a depth_topic")
            if self._np is None:
                try:
                    import numpy
                except ImportError:
                    self._node.get_logger().warning(
                        "numpy unavailable; depth overlays are disabled"
                    )
                    return
                self._np = numpy
            self._depth_settings[name] = dict(settings)
            self._node.create_subscription(
                image_type,
                str(topic),
                lambda message, camera_name=name: self._depth_callback(
                    camera_name, message
                ),
                1,
            )
            self._node.get_logger().info(
                f"depth overlay on {name}: mode={settings.get('mode', 'band')} "
                f"topic={topic}"
            )

    def _depth_callback(self, name: str, message: Any) -> None:
        try:
            depth = self._cv_bridge.imgmsg_to_cv2(message, desired_encoding="16UC1")
        except Exception as exc:
            self._node.get_logger().warning(f"depth decode failed ({name}): {exc}")
            return
        with self._depth_lock:
            self._depth_frames[name] = (time.monotonic(), depth)

    def _draw_depth_overlay(self, name: str, image: Any, now: float) -> None:
        """Composite the depth overlay before the rotate, so the two are independent."""
        settings = self._depth_settings.get(name)
        if settings is None or self._np is None:
            return
        with self._depth_lock:
            entry = self._depth_frames.get(name)
        if entry is None:
            return
        stamped, depth = entry
        if now - stamped > float(settings.get("stale_after_s", 0.25)):
            # A frozen overlay is worse than none: it would keep asserting a
            # grasp plane that the gripper has since moved away from.
            return
        try:
            apply_overlay(image, depth, settings, self._np, self._cv2)
        except Exception as exc:
            self._node.get_logger().warning(f"depth overlay failed ({name}): {exc}")

    def _is_repeat(self, name: str, message: Any) -> bool:
        """True when this message carries the same capture as the last one."""
        header = getattr(message, "header", None)
        stamp = getattr(header, "stamp", None)
        if stamp is None:
            return False
        current = (int(stamp.sec), int(stamp.nanosec))
        if self._camera_stamp.get(name) == current:
            repeats = self._camera_repeats.get(name, 0) + 1
            self._camera_repeats[name] = repeats
            if repeats == 30:
                self._node.get_logger().warning(
                    f"{name} camera is republishing one frame; treating it as "
                    "stopped so the panel falls back to the other camera"
                )
            return True
        self._camera_stamp[name] = current
        self._camera_repeats[name] = 0
        return False

    def _camera_is_fresh(self, name: str, now: float) -> bool:
        window = float(self.config.robot.get("navigation_fallback_after_s", 1.0))
        if window <= 0.0:
            return False
        last = self._camera_last_encoded.get(name)
        return last is not None and now - last < window

    def _left_panel_source(self) -> str:
        """Which camera feeds the left slot, by mode.

        Driving and manipulating want different cameras. Both are mounted on
        link_head_tilt, so both aim wherever the operator looks; the difference
        is the picture. The navigation fisheye is 1280x800 against the head
        D435if's 424x240, which is what makes it the one to drive by. Routing
        here rather than in the headset keeps the wire format at one JPEG per
        slot.
        """
        mapping = self.config.robot.get("left_panel_by_mode") or {}
        if self._last_mode is not None:
            chosen = mapping.get(self._last_mode.value)
            if chosen:
                return str(chosen)
        return "head"

    def _camera_callback(self, name: str, message: Any) -> None:
        now = time.monotonic()
        if self._is_repeat(name, message):
            # A stalled USB camera keeps its publish loop running and re-sends
            # the frame it last managed to grab, header stamp and all. The topic
            # then looks perfectly healthy at full rate -- `ros2 topic hz` and
            # the doctor both pass -- while the panel shows a still photograph.
            # Dropping the repeat lets the frame go stale, which is what makes
            # camera_frames() hand the panel over to the other camera.
            return
        period = 1.0 / max(0.1, float(self.config.robot["camera_encode_hz"]))
        if now - self._camera_last_encoded.get(name, 0.0) < period:
            return
        if name in _LEFT_PANEL_CAMERAS:
            selected = self._left_panel_source()
            if not getattr(self, "_recording_enabled", False) and name != selected and self._camera_is_fresh(selected, now):
                # Only one of these two ever reaches a panel, so JPEG-encoding
                # the other burns the robot's onboard CPU on a picture nobody
                # sees. Encoding resumes if the selected stream stops.
                return
        try:
            image = self._cv_bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            rotate = self._camera_rotations.get(name)
            quality = int(self.config.robot["camera_jpeg_quality"])
            if getattr(self, "_recording_enabled", False):
                raw = self._cv2.rotate(image, rotate) if rotate is not None else image
                ok, encoded_raw = self._cv2.imencode(
                    ".jpg", raw, [int(self._cv2.IMWRITE_JPEG_QUALITY), quality]
                )
                if ok:
                    with self._camera_lock:
                        self._recording_images[name] = {
                            "jpeg": encoded_raw.tobytes(),
                            "source_stamp": [message.header.stamp.sec, message.header.stamp.nanosec],
                            "received_monotonic": now,
                        }
            if name in self._depth_settings:
                image = image.copy()
                self._draw_depth_overlay(name, image, now)
            rotate = self._camera_rotations.get(name)
            if rotate is not None:
                image = self._cv2.rotate(image, rotate)
            quality = int(self.config.robot["camera_jpeg_quality"])
            ok, encoded = self._cv2.imencode(
                ".jpg", image, [int(self._cv2.IMWRITE_JPEG_QUALITY), quality]
            )
            if not ok:
                return
            with self._camera_lock:
                self._camera_frames[name] = encoded.tobytes()
                self._camera_last_encoded[name] = now
                self._camera_encoded_count[name] = (
                    self._camera_encoded_count.get(name, 0) + 1
                )
        except Exception as exc:
            self._node.get_logger().warning(f"camera encode failed ({name}): {exc}")

    def camera_diagnostics(self) -> dict[str, Any]:
        """Enough to tell a frozen panel from a frozen camera, without a client.

        The WebSocket takes one client at a time, so a probe that connects locks
        the headset out. These counters are pollable over /healthz while the
        operator keeps working.
        """
        now = time.monotonic()
        with self._camera_lock:
            counts = dict(self._camera_encoded_count)
            ages = {
                name: round(now - stamp, 2)
                for name, stamp in self._camera_last_encoded.items()
            }
            sizes = {
                name: len(frame) for name, frame in self._camera_frames.items()
            }
        return {
            "repeated_frames": dict(self._camera_repeats),
            "mode_seen_by_cameras": (
                self._last_mode.value if self._last_mode is not None else None
            ),
            "left_panel_source": self._left_panel_source(),
            "encoded": counts,
            "age_s": ages,
            "jpeg_bytes": sizes,
        }

    def state(self) -> RobotState:
        with self._state_lock:
            return replace(self._state, joints=dict(self._state.joints))

    def _reset_manip_anchor(self, state: RobotState) -> None:
        self._manip_base_anchor = state.base_pose

    def _current_virtual_base_x_locked(self) -> float:
        if self._manip_base_anchor is None:
            return 0.0
        x, y, _ = self._state.base_pose
        anchor_x, anchor_y, anchor_theta = self._manip_base_anchor
        return math.cos(anchor_theta) * (x - anchor_x) + math.sin(anchor_theta) * (
            y - anchor_y
        )

    def _current_virtual_base_x(self) -> float:
        with self._state_lock:
            return self._current_virtual_base_x_locked()

    def apply(self, command: RobotCommand) -> dict:
        if not self._started:
            raise RuntimeError("ROS2 backend has not been started")
        state = self.state()
        if command.stale_input or not state.ready:
            # Never move a robot before the first calibrated joint state has
            # confirmed that the driver is alive.
            submitted_at = time.monotonic()
            self._halt_motion()
            return {"base_linear_mps": 0.0, "base_angular_rps": 0.0,
                    "joint_targets": {}, "trajectory_time_s": 0.2, "stop_motion": True,
                    "submitted_at_monotonic": submitted_at}
        self._check_motion_ready(state)
        self._halted = False
        if command.mode != self._last_mode:
            if command.mode in (Mode.MANIP, Mode.PRECISION):
                self._reset_manip_anchor(state)
            self._last_mode = command.mode

        explicit_base = (
            abs(command.base_linear_mps) > 1e-5
            or abs(command.base_angular_rps) > 1e-5
        )
        if self._explicit_base_active and not explicit_base:
            self._reset_manip_anchor(state)
        self._explicit_base_active = explicit_base

        solution: JointSolution | None = None
        current_virtual_x = self._current_virtual_base_x()
        if command.ee_target is not None:
            try:
                solution = self.kinematics.solve(
                    command.ee_target, state, current_virtual_x
                )
            except Exception as exc:
                # Reject an unreachable target while retaining the control connection.
                solution = None
                self._node.get_logger().warning(f"IK raised; dropping EE target: {exc}")
            if solution is None:
                self._node.get_logger().warning("IK failed; dropping EE target")

        linear = command.base_linear_mps
        angular = command.base_angular_rps
        if (
            solution is not None
            and command.allow_ik_base_x
            and not explicit_base
            and not command.base_locked
        ):
            error = solution.virtual_base_x - current_virtual_x
            linear += clamp(
                float(self.config.base["ik_base_kp"]) * error,
                -float(self.config.base["ik_base_max_mps"]),
                float(self.config.base["ik_base_max_mps"]),
            )
        if command.base_locked or command.stale_input:
            linear = 0.0
            angular = 0.0
        submitted_at = time.monotonic()
        self._publish_base(linear, angular)
        targets = self._publish_joint_goal(command, solution, state)
        return {"base_linear_mps": linear, "base_angular_rps": angular,
                "joint_targets": targets,
                "trajectory_time_s": float(self.config.robot["trajectory_time_s"]),
                "stop_motion": False, "submitted_at_monotonic": submitted_at}

    def _check_motion_ready(self, state):
        if self._motion_fault:
            raise RuntimeError(self._motion_fault)
        received = state.joint_state_received_monotonic
        if received is None or time.monotonic() - received > 0.5:
            self._halt_motion()
            raise RuntimeError("robot joint feedback is stale")
        pending = []
        for future in self._goal_futures:
            if not future.done():
                pending.append(future)
            else:
                handle = future.result()
                if not handle.accepted:
                    self._motion_fault = "trajectory goal was rejected; restart the backend"
                    self._halt_motion()
                    raise RuntimeError(self._motion_fault)
                if handle.status < 4:
                    pending.append(future)
        self._goal_futures = pending

    @staticmethod
    def _wait_future(future, deadline):
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.005)
        if not future.done():
            raise RuntimeError("timed out waiting for robot stop acknowledgement")
        return future.result()

    def _halt_motion(self):
        """Cancel submitted goals, including goals still awaiting acceptance."""
        self._publish_base(0.0, 0.0)
        if self._halted:
            return
        deadline = time.monotonic() + 1.0
        error = None
        try:
            handles = [self._wait_future(future, deadline) for future in self._goal_futures]
            cancellations = [(handle, handle.cancel_goal_async()) for handle in handles
                             if handle.accepted and handle.status < 4]
            for handle, future in cancellations:
                response = self._wait_future(future, deadline)
                if response.return_code != 0 and handle.status < 4:
                    raise RuntimeError("robot rejected trajectory cancellation")
            if any(not handle.accepted for handle in handles):
                raise RuntimeError("a submitted trajectory goal was rejected")
        except Exception as exc:
            error = exc
        try:
            # Also stop residual actuator motion after a goal has completed.
            if self._stop_client is None or not self._stop_client.wait_for_service(timeout_sec=0.1):
                raise RuntimeError("robot stop service unavailable")
            response = self._wait_future(
                self._stop_client.call_async(self._types["Trigger"].Request()),
                time.monotonic() + 0.5)
            if response is None or not response.success:
                raise RuntimeError("robot stop service failed")
        except Exception as exc:
            error = error or exc
        if error:
            self._motion_fault = f"stop failed; restart the backend: {error}"
            raise RuntimeError(self._motion_fault) from error
        self._goal_futures = []
        self._halted = True

    def _publish_base(self, linear: float, angular: float) -> None:
        message = self._types["Twist"]()
        message.linear.x = float(linear)
        message.angular.z = float(angular)
        self._cmd_vel_pub.publish(message)

    def _publish_joint_goal(
        self,
        command: RobotCommand,
        solution: JointSolution | None,
        state: RobotState,
    ) -> dict:
        now = time.monotonic()
        period = 1.0 / max(0.1, float(self.config.robot["joint_command_hz"]))
        if now - self._last_command_time < period:
            return {}
        names: list[str] = []
        positions: list[float] = []
        if solution is not None:
            names.extend(
                [
                    "joint_lift",
                    "wrist_extension",
                    "joint_wrist_yaw",
                    "joint_wrist_pitch",
                    "joint_wrist_roll",
                ]
            )
            positions.extend(
                [
                    solution.lift,
                    solution.arm,
                    solution.wrist_yaw,
                    solution.wrist_pitch,
                    solution.wrist_roll,
                ]
            )
        if command.head_target is not None:
            names.extend(["joint_head_pan", "joint_head_tilt"])
            positions.extend(command.head_target)
        if command.gripper_open_fraction is not None:
            aperture = float(self.config.robot["gripper_closed_m"]) + (
                float(self.config.robot["gripper_open_m"])
                - float(self.config.robot["gripper_closed_m"])
            ) * clamp(command.gripper_open_fraction, 0.0, 1.0)
            names.append("gripper_aperture")
            positions.append(aperture)
        if not names:
            return {}

        return self._send_joint_targets(dict(zip(names, positions)),
                                        float(self.config.robot["trajectory_time_s"]))

    def _send_joint_targets(self, targets: dict, duration_s: float) -> dict:
        if not targets:
            return {}

        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectoryPoint

        goal = self._types["FollowJointTrajectory"].Goal()
        goal.trajectory.joint_names = list(targets)
        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in targets.values()]
        seconds = int(duration_s)
        point.time_from_start = Duration(
            sec=seconds, nanosec=int((duration_s - seconds) * 1e9)
        )
        goal.trajectory.points = [point]
        if not self._trajectory_client.server_is_ready():
            if not self._trajectory_client.wait_for_server(timeout_sec=0.1):
                self._node.get_logger().warning("trajectory action server unavailable")
                return {}
        self._goal_futures.append(self._trajectory_client.send_goal_async(goal))
        self._last_command_time = time.monotonic()
        return dict(targets)

    def apply_recorded_action(self, action: dict) -> None:
        if not self._started or not self.state().ready:
            raise RuntimeError("ROS backend is not ready for replay")
        if action.get("stop_motion"):
            self._halt_motion()
            return
        self._check_motion_ready(self.state())
        self._halted = False
        self._publish_base(action["base_linear_mps"], action["base_angular_rps"])
        targets = action["joint_targets"]
        sent = self._send_joint_targets(targets, action["trajectory_time_s"])
        if targets and not sent:
            raise RuntimeError("trajectory server unavailable during replay")

    def set_recording(self, enabled: bool) -> None:
        with self._camera_lock:
            self._recording_enabled = enabled
            if enabled:
                self._recording_images.clear()

    def recording_images(self) -> dict:
        now = time.monotonic()
        with self._camera_lock:
            return {name: {"jpeg": item["jpeg"], "source_stamp": item["source_stamp"],
                           "age_s": max(0.0, now - item["received_monotonic"])}
                    for name, item in self._recording_images.items()}

    def camera_frames(self) -> dict[str, bytes]:
        """Both panels stream continuously; the mode picks the left camera.

        The left slot carries whichever of the head D435if and the wide
        navigation camera the current mode calls for, and falls back to the
        other one. The right slot is always the wrist D405.
        """
        now = time.monotonic()
        selected = self._left_panel_source()
        alternate = "navigation" if selected == "head" else "head"
        with self._camera_lock:
            head = None
            for name in (selected, alternate):
                # Freshness first: a frozen picture of where the robot used to
                # be looking is worse than the other camera's live one.
                if self._camera_frames.get(name) and self._camera_is_fresh(name, now):
                    head = self._camera_frames[name]
                    break
            if head is None:
                head = self._camera_frames.get(selected) or self._camera_frames.get(
                    alternate
                )
            gripper = self._camera_frames.get("gripper")
        frames: dict[str, bytes] = {}
        if head:
            frames["head"] = head
        if gripper:
            frames["gripper"] = gripper
        return frames

    def stop(self) -> None:
        try:
            if self._started:
                self._halt_motion()
        finally:
            try:
                if self._executor is not None:
                    self._executor.shutdown(timeout_sec=1.0)
                if self._executor_thread is not None:
                    self._executor_thread.join(timeout=1.0)
                if self._node is not None:
                    self._node.destroy_node()
            finally:
                self._started = False
                if self._control_lock is not None:
                    self._control_lock.close()
                    self._control_lock = None
