from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any


# The client shows two panels at once, so a bare JPEG is no longer
# self-describing. Binary frames carry a four byte header instead.
CAMERA_FRAME_MAGIC = b"S3"
CAMERA_FRAME_VERSION = 1
CAMERA_SLOTS = ("head", "gripper")


def encode_camera_frame(slot: str, jpeg: bytes) -> bytes:
    try:
        index = CAMERA_SLOTS.index(slot)
    except ValueError as exc:
        raise ValueError(f"unknown camera slot: {slot}") from exc
    return CAMERA_FRAME_MAGIC + bytes((CAMERA_FRAME_VERSION, index)) + jpeg


class Mode(str, Enum):
    NAV = "NAV"
    ALIGN = "ALIGN"
    MANIP = "MANIP"
    PRECISION = "PRECISION"


def _floats(value: Any, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must contain {size} numbers")
    try:
        result = tuple(float(x) for x in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {size} numbers") from exc
    if not all(x == x and abs(x) != float("inf") for x in result):
        raise ValueError(f"{name} contains a non-finite number")
    return result


def _finite_number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]  # x, y, z, w

    @classmethod
    def identity(cls) -> "Pose":
        return cls((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Pose":
        if not isinstance(value, dict):
            raise ValueError("pose must be an object")
        return cls(
            _floats(value.get("position"), 3, "pose.position"),
            _floats(value.get("orientation"), 4, "pose.orientation"),
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "position": list(self.position),
            "orientation": list(self.orientation),
        }


@dataclass(frozen=True)
class ControllerState:
    tracked: bool = False
    pose: Pose = field(default_factory=Pose.identity)
    trigger: float = 0.0
    grip: bool = False
    primary: bool = False
    secondary: bool = False
    stick_x: float = 0.0
    stick_y: float = 0.0
    stick_click: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "ControllerState":
        value = {} if value is None else value
        if not isinstance(value, dict):
            raise ValueError("controller state must be an object")
        pose_value = value.get("pose")
        pose = Pose.from_dict(pose_value) if isinstance(pose_value, dict) else Pose.identity()
        return cls(
            tracked=bool(value.get("tracked", False)),
            pose=pose,
            trigger=max(0.0, min(1.0, _finite_number(value.get("trigger", 0.0), "trigger"))),
            grip=bool(value.get("grip", False)),
            primary=bool(value.get("primary", False)),
            secondary=bool(value.get("secondary", False)),
            stick_x=max(-1.0, min(1.0, _finite_number(value.get("stick_x", 0.0), "stick_x"))),
            stick_y=max(-1.0, min(1.0, _finite_number(value.get("stick_y", 0.0), "stick_y"))),
            stick_click=bool(value.get("stick_click", False)),
        )


@dataclass(frozen=True)
class VRFrame:
    sequence: int
    client_time_ms: float
    hmd_tracked: bool
    hmd_pose: Pose
    left: ControllerState
    right: ControllerState

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "VRFrame":
        if not isinstance(value, dict):
            raise ValueError("VR frame must be an object")
        if value.get("type") not in (None, "vr_frame"):
            raise ValueError("message type must be vr_frame")
        hmd = value.get("hmd", {})
        if not isinstance(hmd, dict):
            raise ValueError("hmd must be an object")
        hmd_pose_value = hmd.get("pose")
        try:
            sequence = int(value.get("sequence", 0))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("sequence must be an integer") from exc
        return cls(
            sequence=sequence,
            client_time_ms=_finite_number(value.get("client_time_ms", 0.0), "client_time_ms"),
            hmd_tracked=bool(hmd.get("tracked", False)),
            hmd_pose=(
                Pose.from_dict(hmd_pose_value)
                if isinstance(hmd_pose_value, dict)
                else Pose.identity()
            ),
            left=ControllerState.from_dict(value.get("left")),
            right=ControllerState.from_dict(value.get("right")),
        )


@dataclass
class RobotState:
    head_pan: float = 0.0
    head_tilt: float = -0.52
    ee_pose: Pose = field(
        default_factory=lambda: Pose((0.45, -0.20, 0.85), (0.0, 0.0, 0.0, 1.0))
    )
    base_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    joints: dict[str, float] = field(default_factory=dict)
    contact_detected: bool = False
    ready: bool = True
    joint_state_received_monotonic: float | None = None
    odom_received_monotonic: float | None = None


@dataclass
class RobotCommand:
    mode: Mode
    base_linear_mps: float = 0.0
    base_angular_rps: float = 0.0
    head_target: tuple[float, float] | None = None
    ee_target: Pose | None = None
    gripper_open_fraction: float | None = None
    allow_ik_base_x: bool = False
    base_locked: bool = False
    stale_input: bool = False
    events: list[str] = field(default_factory=list)

    def status_dict(self) -> dict[str, Any]:
        return {
            "type": "status",
            "mode": self.mode.value,
            "base": {
                "linear_mps": self.base_linear_mps,
                "angular_rps": self.base_angular_rps,
                "locked": self.base_locked,
            },
            "head_target": list(self.head_target) if self.head_target else None,
            "ee_active": self.ee_target is not None,
            "gripper_open_fraction": self.gripper_open_fraction,
            "stale_input": self.stale_input,
            "events": list(self.events),
        }
