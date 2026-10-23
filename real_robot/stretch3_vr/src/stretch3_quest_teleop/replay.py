"""Validate and replay timestamped VR episodes without rerunning inverse kinematics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from .config import TeleopConfig, default_config_path
from .models import Mode, RobotCommand
from .recording import SCHEMA

JOINTS = {
    "joint_lift", "wrist_extension", "joint_wrist_yaw", "joint_wrist_pitch",
    "joint_wrist_roll", "joint_head_pan", "joint_head_tilt", "gripper_aperture",
}


def load_episode(directory: Path, verify_images=True):
    directory = Path(directory).resolve()
    metadata = json.loads((directory / "configs.json").read_text())
    if metadata.get("schema") != SCHEMA:
        raise ValueError("not a VR episode; use legacy-replay only for legacy gamepad-format files")
    if metadata.get("complete") is not True or not (directory / "completed.txt").is_file():
        raise ValueError("episode is incomplete or aborted")
    labels = json.loads((directory / "labels.json").read_text())
    if not isinstance(labels, dict) or set(labels) != {str(i) for i in range(len(labels))}:
        raise ValueError("frame indices must be contiguous")
    frames = [labels[str(i)] for i in range(len(labels))]
    if not frames or metadata.get("num_frames") != len(frames):
        raise ValueError("empty episode or mismatched frame count")
    previous = -1.0
    checked = set()
    for index, frame in enumerate(frames):
        stamp = frame["timestamp_s"]
        if not isinstance(stamp, (float, int)) or not math.isfinite(stamp) or stamp < 0 or stamp <= previous:
            raise ValueError("timestamps must be finite and strictly increasing")
        previous = stamp
        if frame.get("step") != index:
            raise ValueError("frame step does not match its index")
        validate_action(frame["actions"])
        if frame.get("kind") in ("stop", "watchdog") and not frame["actions"].get("stop_motion"):
            raise ValueError("stop events must explicitly stop joint motion")
        for image in frame.get("images", {}).values():
            path = (directory / image["path"]).resolve()
            if not path.is_relative_to(directory):
                raise ValueError("image path escapes the episode directory")
            key = (str(path), image["sha256"])
            if verify_images and key not in checked:
                if hashlib.sha256(path.read_bytes()).hexdigest() != image["sha256"]:
                    raise ValueError("image checksum mismatch")
                checked.add(key)
    final = frames[-1]
    if (final.get("kind") != "stop" or final["actions"]["base_linear_mps"] != 0
            or final["actions"]["base_angular_rps"] != 0 or final["actions"]["joint_targets"]):
        raise ValueError("episode must end with an explicit stop")
    return metadata, frames


def validate_action(action):
    if not isinstance(action, dict) or not isinstance(action.get("joint_targets"), dict):
        raise ValueError("invalid submitted action")
    if set(action["joint_targets"]) - JOINTS:
        raise ValueError("unknown joint in recorded action")
    values = [action.get("base_linear_mps"), action.get("base_angular_rps"),
              action.get("trajectory_time_s"), *action["joint_targets"].values()]
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
        raise ValueError("recorded actions must be finite numbers")
    if not 0 < action["trajectory_time_s"] <= 10:
        raise ValueError("invalid trajectory duration")
    if not isinstance(action.get("stop_motion", False), bool):
        raise ValueError("stop_motion must be boolean")
    if action.get("stop_motion") and (action["base_linear_mps"] or action["base_angular_rps"]
                                     or action["joint_targets"] or action.get("mock_ee_pose")):
        raise ValueError("stop action contains motion targets")


def validate_start(frames, state):
    saved = frames[0]["observations"]["joints"]
    names = {name for frame in frames for name in frame["actions"]["joint_targets"]}
    for name in names:
        # Gripper feedback has robot-specific joint names; no unit conversion is inferred.
        if name == "gripper_aperture":
            continue
        if name not in saved or name not in state.joints:
            raise ValueError(f"missing initial joint feedback: {name}")
        tolerance = 0.03 if name in ("joint_lift", "wrist_extension") else 0.15
        if (not math.isfinite(saved[name]) or not math.isfinite(state.joints[name])
                or abs(saved[name] - state.joints[name]) > tolerance):
            raise ValueError(f"initial pose differs at {name}; position the robot before replay")


def run_replay(backend, frames, *, realtime=True, clock=time.monotonic, sleep=time.sleep):
    start = clock()
    first = frames[0]["timestamp_s"]
    try:
        for frame in frames:
            if realtime:
                due = start + frame["timestamp_s"] - first
                delay = due - clock()
                if delay > 0:
                    sleep(delay)
                if clock() - due > 0.25:
                    raise RuntimeError("replay fell behind; refusing to burst queued motion commands")
            backend.apply_recorded_action(frame["actions"])
    finally:
        backend.apply(RobotCommand(mode=Mode.NAV, base_locked=True, stale_input=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--execute", action="store_true", help="execute after validation; default is inspection only")
    parser.add_argument("--backend", choices=("mock", "ros2"), default="mock")
    parser.add_argument("--config", type=Path, default=default_config_path())
    args = parser.parse_args(argv)
    metadata, frames = load_episode(args.episode)
    print(json.dumps({"schema": metadata["schema"], "backend": metadata["backend"],
                      "frames": len(frames), "duration_s": frames[-1]["timestamp_s"],
                      "images_verified": True, "execute": args.execute}, indent=2))
    if not args.execute:
        return 0
    if args.backend == "ros2":
        if metadata["backend"] != "ros2":
            raise ValueError("mock episodes cannot control hardware")
        from .backends.ros2 import Ros2StretchBackend
        backend = Ros2StretchBackend(TeleopConfig.load(args.config))
    else:
        from .backends.mock import MockRobotBackend
        backend = MockRobotBackend()
    try:
        backend.start()
        if args.backend == "ros2":
            deadline = time.monotonic() + 5
            while not backend.state().ready and time.monotonic() < deadline:
                time.sleep(0.05)
            if not backend.state().ready:
                raise RuntimeError("robot joint feedback is unavailable")
            validate_start(frames, backend.state())
        run_replay(backend, frames)
    finally:
        backend.stop()
    return 0
