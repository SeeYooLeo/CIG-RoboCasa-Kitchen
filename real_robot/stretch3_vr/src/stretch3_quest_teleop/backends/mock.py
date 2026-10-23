from __future__ import annotations

import math
import threading
import time
from dataclasses import replace
from pathlib import Path

from ..models import Mode, Pose, RobotCommand, RobotState
from .base import RobotBackend

CAMERAS = ("navigation", "head", "gripper")


def load_camera_images(directory: str | Path) -> dict[str, list[bytes]]:
    """Read JPEGs for the mock cameras.

    A directory holding `navigation/`, `head/` or `gripper/` subdirectories
    feeds those cameras separately. A flat directory of JPEGs stands in for
    every camera, which is the quick way to force one picture into every mode.
    """
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise ValueError(f"camera directory does not exist: {root}")

    def jpegs(path: Path) -> list[bytes]:
        return [
            item.read_bytes()
            for item in sorted(path.iterdir())
            if item.is_file() and item.suffix.lower() in (".jpg", ".jpeg")
        ]

    cameras: dict[str, list[bytes]] = {}
    for name in CAMERAS:
        subdirectory = root / name
        if subdirectory.is_dir():
            frames = jpegs(subdirectory)
            if frames:
                cameras[name] = frames
    if cameras:
        return cameras

    frames = jpegs(root)
    if not frames:
        raise ValueError(f"no .jpg or .jpeg files under {root}")
    return {name: frames for name in CAMERAS}


class MockRobotBackend(RobotBackend):
    def __init__(self, image_directory: str | Path | None = None) -> None:
        self._state = RobotState()
        self._lock = threading.Lock()
        self._last_apply = time.monotonic()
        self.commands: list[RobotCommand] = []
        self.started = False
        self.cameras = load_camera_images(image_directory) if image_directory else {}
        self._frame_index = 0
        self._last_advance = 0.0

    def start(self) -> None:
        self.started = True
        self._last_apply = time.monotonic()

    def state(self) -> RobotState:
        with self._lock:
            return replace(
                self._state,
                joints=dict(self._state.joints),
            )

    def apply(self, command: RobotCommand) -> dict:
        now = time.monotonic()
        dt = min(0.1, max(0.0, now - self._last_apply))
        self._last_apply = now
        with self._lock:
            x, y, theta = self._state.base_pose
            x += math.cos(theta) * command.base_linear_mps * dt
            y += math.sin(theta) * command.base_linear_mps * dt
            theta += command.base_angular_rps * dt
            self._state.base_pose = (x, y, theta)
            if command.head_target is not None:
                self._state.head_pan, self._state.head_tilt = command.head_target
            if command.ee_target is not None:
                self._state.ee_pose = command.ee_target
            if command.gripper_open_fraction is not None:
                self._state.joints["gripper_open_fraction"] = (
                    command.gripper_open_fraction
                )
            self.commands.append(command)
        targets = {}
        if command.head_target is not None:
            targets.update(zip(("joint_head_pan", "joint_head_tilt"), command.head_target))
        if command.gripper_open_fraction is not None:
            targets["gripper_aperture"] = 0.09 * command.gripper_open_fraction
        return {
            "base_linear_mps": command.base_linear_mps,
            "base_angular_rps": command.base_angular_rps,
            "joint_targets": targets, "trajectory_time_s": 0.2,
            "stop_motion": command.stale_input,
            "mock_ee_pose": command.ee_target.to_dict() if command.ee_target else None,
        }

    def recording_images(self) -> dict:
        return {name: {"jpeg": jpeg, "source_stamp": None, "age_s": None}
                for name, jpeg in self.camera_frames().items()}

    def apply_recorded_action(self, action: dict) -> None:
        targets = action["joint_targets"]
        head = None
        if "joint_head_pan" in targets and "joint_head_tilt" in targets:
            head = (targets["joint_head_pan"], targets["joint_head_tilt"])
        ee = action.get("mock_ee_pose")
        self.apply(RobotCommand(
            mode=Mode.NAV,
            stale_input=action.get("stop_motion", False),
            base_linear_mps=action["base_linear_mps"],
            base_angular_rps=action["base_angular_rps"],
            head_target=head, ee_target=Pose.from_dict(ee) if ee else None,
            gripper_open_fraction=targets.get("gripper_aperture", 0.0) / 0.09
            if "gripper_aperture" in targets else None,
        ))
        with self._lock:
            self._state.joints.update(targets)

    def camera_frames(self) -> dict[str, bytes]:
        """Mirrors the ROS backend's slot mapping so both panels are exercised."""
        if not self.cameras:
            return {}
        now = time.monotonic()
        # One frame per second: enough to prove the stream is live without
        # distracting from the geometry being checked.
        if now - self._last_advance >= 1.0:
            self._last_advance = now
            self._frame_index += 1
        sources = {
            "head": self.cameras.get("head") or self.cameras.get("navigation"),
            "gripper": self.cameras.get("gripper"),
        }
        return {
            slot: frames[self._frame_index % len(frames)]
            for slot, frames in sources.items()
            if frames
        }

    def stop(self) -> None:
        self.apply(
            RobotCommand(
                mode=self.commands[-1].mode if self.commands else Mode.NAV,
                base_locked=True,
                stale_input=True,
                events=["mock_backend_stopped"],
            )
        )
        self.started = False
