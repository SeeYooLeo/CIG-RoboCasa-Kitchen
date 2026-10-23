from __future__ import annotations

import importlib
import platform
import ssl
import time
from pathlib import Path
from typing import Any

from .config import TeleopConfig

CAMERA_ROLES = ("navigation", "head", "gripper")

# How long to let DDS discovery settle before believing the graph is empty.
DISCOVERY_SETTLE_S = 8.0


def _camera_topics(config: TeleopConfig) -> dict[str, str]:
    return {
        "navigation": str(config.robot["navigation_camera_topic"]),
        "head": str(config.robot["head_camera_topic"]),
        "gripper": str(config.robot["gripper_camera_topic"]),
    }


def _describe_camera(entry: dict[str, Any], rotation: int, elapsed: float) -> str:
    shape = f"{entry['width']}x{entry['height']} {entry['encoding']}"
    orientation = "portrait" if entry["height"] > entry["width"] else "landscape"
    hertz = entry["count"] / max(1e-6, elapsed)
    turned = "" if rotation == 0 else f", bridge rotates {rotation} deg clockwise"
    return f"{shape} ({orientation}), {hertz:.1f} Hz{turned}"


def probe_ros2_graph(config: TeleopConfig, seconds: float) -> list[dict[str, Any]]:
    """Read-only look at the running robot: what is advertised, and what the
    cameras actually publish.

    This subscribes and reads. It creates no publisher, calls no service and
    sends no action goal, so it is safe to run while the robot is powered and
    even while a teleoperation session is live.
    """
    import rclpy
    from rclpy.action import get_action_names_and_types
    from sensor_msgs.msg import Image

    checks: list[dict[str, Any]] = []
    started_rclpy = not rclpy.ok()
    if started_rclpy:
        rclpy.init()
    node = rclpy.create_node("stretch3_quest_teleop_doctor")
    try:
        robot = config.robot
        expected_topics = {
            str(robot[key])
            for key in (
                "cmd_vel_topic",
                "joint_states_topic",
                "odom_topic",
                "mode_topic",
            )
        }
        expected_services = {
            str(robot[key]) for key in ("navigation_mode_service", "stop_service")
        }
        expected_action = str(robot["trajectory_action"])
        # DDS discovery is not instantaneous. Reading the graph in the same
        # breath as creating the node reports only whatever happened to arrive
        # first, which made this check flap between runs on the real robot:
        # cmd_vel present and joint_states "not advertised", then the reverse
        # on the next run, with the driver up and healthy the whole time. A
        # pre-flight check that fails at random teaches the operator to skip
        # it, which is worse than not having it. Wait until everything
        # expected is visible, then report what is genuinely absent.
        deadline = time.monotonic() + DISCOVERY_SETTLE_S
        while True:
            topics = dict(node.get_topic_names_and_types())
            services = dict(node.get_service_names_and_types())
            actions = dict(get_action_names_and_types(node))
            if (
                expected_topics <= topics.keys()
                and expected_services <= services.keys()
                and expected_action in actions
            ) or time.monotonic() >= deadline:
                break
            rclpy.spin_once(node, timeout_sec=0.2)

        for label, key in (
            ("cmd_vel", "cmd_vel_topic"),
            ("joint_states", "joint_states_topic"),
            ("odom", "odom_topic"),
            ("mode", "mode_topic"),
        ):
            name = str(robot[key])
            checks.append(
                {
                    "name": f"ros2_topic:{label}",
                    "ok": name in topics,
                    "required": True,
                    "detail": name if name in topics else f"{name} is not advertised",
                }
            )

        for label, key in (
            ("navigation_mode", "navigation_mode_service"),
            ("stop_the_robot", "stop_service"),
        ):
            name = str(robot[key])
            checks.append(
                {
                    "name": f"ros2_service:{label}",
                    "ok": name in services,
                    "required": True,
                    "detail": name if name in services else f"{name} is not advertised",
                }
            )

        action = str(robot["trajectory_action"])
        checks.append(
            {
                "name": "ros2_action:follow_joint_trajectory",
                "ok": action in actions,
                "required": True,
                "detail": action if action in actions else f"{action} is not advertised",
            }
        )

        if seconds <= 0:
            return checks

        cameras = _camera_topics(config)
        seen: dict[str, dict[str, Any]] = {}

        def make_callback(role: str):
            def callback(message: Any) -> None:
                entry = seen.get(role)
                if entry is None:
                    entry = {
                        "width": int(message.width),
                        "height": int(message.height),
                        "encoding": str(message.encoding),
                        "count": 0,
                    }
                    seen[role] = entry
                entry["count"] += 1

            return callback

        subscriptions = [
            node.create_subscription(Image, topic, make_callback(role), 2)
            for role, topic in cameras.items()
        ]
        began = time.monotonic()
        while time.monotonic() - began < seconds:
            rclpy.spin_once(node, timeout_sec=0.05)
        elapsed = time.monotonic() - began
        for subscription in subscriptions:
            node.destroy_subscription(subscription)

        rotations = robot.get("camera_rotate_deg") or {}
        for role in CAMERA_ROLES:
            entry = seen.get(role)
            checks.append(
                {
                    "name": f"camera:{role}",
                    "ok": entry is not None,
                    # Individually optional: the left panel falls back from the
                    # head camera to the navigation camera, so either one alone
                    # is enough. The pair check below is the one that matters.
                    "required": False,
                    "detail": (
                        _describe_camera(entry, int(rotations.get(role, 0) or 0), elapsed)
                        if entry is not None
                        else f"no frame on {cameras[role]} in {elapsed:.1f} s"
                    ),
                }
            )
        left = "head" if "head" in seen else ("navigation" if "navigation" in seen else None)
        missing = [
            name
            for name, present in (("left panel", left), ("gripper", seen.get("gripper")))
            if not present
        ]
        checks.append(
            {
                "name": "camera:panels",
                "ok": not missing,
                "required": True,
                "detail": (
                    f"left panel from {left}, right panel from gripper"
                    if not missing
                    else "no image for: " + ", ".join(missing)
                ),
            }
        )
        return checks
    finally:
        node.destroy_node()
        if started_rclpy:
            rclpy.shutdown()


def run_doctor(
    config: TeleopConfig,
    *,
    check_ros2: bool = False,
    cert: str | None = None,
    key: str | None = None,
    camera_probe_s: float = 0.0,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, operation, required: bool = True) -> None:
        try:
            detail = operation()
            checks.append(
                {"name": name, "ok": True, "required": required, "detail": str(detail)}
            )
        except Exception as exc:
            checks.append(
                {
                    "name": name,
                    "ok": False,
                    "required": required,
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )

    check("python", lambda: platform.python_version())
    check("config", lambda: f"{len(config.raw)} sections loaded")
    check(
        "config_overlays",
        lambda: ", ".join(config.overlays) if config.overlays else "none",
    )
    check(
        "base_speeds",
        lambda: (
            f"NAV {config.base['nav_linear_mps']} m/s, "
            f"{config.base['nav_angular_rps']} rad/s"
        ),
    )
    check("tls", lambda: ssl.OPENSSL_VERSION)

    if cert or key:
        def load_cert():
            if not cert or not key:
                raise ValueError("--cert and --key must be supplied together")
            if not Path(cert).is_file() or not Path(key).is_file():
                raise FileNotFoundError("certificate or key does not exist")
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(cert, key)
            return Path(cert).resolve()

        check("tls_certificate", load_cert)

    if check_ros2:
        required_modules = (
            "rclpy",
            "control_msgs.action",
            "geometry_msgs.msg",
            "nav_msgs.msg",
            "sensor_msgs.msg",
            "trajectory_msgs.msg",
            "stretch.motion.kinematics",
            "numpy",
        )
        for module in required_modules:
            check(f"import:{module}", lambda name=module: importlib.import_module(name))
        for module in ("cv2", "cv_bridge"):
            check(
                f"camera_optional:{module}",
                lambda name=module: importlib.import_module(name),
                required=False,
            )

        if all(item["ok"] for item in checks if item["name"].startswith("import:")):
            try:
                checks.extend(probe_ros2_graph(config, max(0.0, camera_probe_s)))
            except Exception as exc:
                checks.append(
                    {
                        "name": "ros2_graph",
                        "ok": False,
                        "required": True,
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                )

    required_ok = all(item["ok"] for item in checks if item["required"])
    return {"ok": required_ok, "checks": checks}
