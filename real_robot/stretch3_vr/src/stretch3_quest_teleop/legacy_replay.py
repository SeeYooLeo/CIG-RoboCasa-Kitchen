"""Compatibility replay for legacy gamepad-format files, not paper VR collection.

The legacy gripper actuator coordinates are not ROS aperture meters. This
adapter preserves the original velocity/target and move_by semantics.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import time

JOINTS = {
    "joint_lift": "lift", "joint_arm_l0": "arm",
    "joint_wrist_yaw": "wrist_yaw", "joint_wrist_pitch": "wrist_pitch",
    "joint_wrist_roll": "wrist_roll", "joint_head_pan": "head_pan",
    "joint_head_tilt": "head_tilt",
}
VELOCITIES = ("v_lift", "v_arm", "v_wrist_yaw", "v_wrist_pitch", "v_wrist_roll",
              "v_gripper", "v_base_forward", "v_base_yaw")


def load_legacy(directory, fps=None):
    directory = Path(directory)
    labels = json.loads((directory / "labels.json").read_text())
    meta_path = directory / "configs.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get("schema"):
        raise ValueError("use replay for native VR episodes")
    rate = fps if fps is not None else meta.get("fps")
    if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("a positive fps is required from configs.json or --fps")
    if not isinstance(labels, dict) or not labels or set(labels) != {str(i) for i in range(len(labels))}:
        raise ValueError("legacy frame indices must be contiguous and nonempty")
    frames = [labels[str(i)] for i in range(len(labels))]
    for frame in frames:
        action = frame["actions"]
        if action.get("leader") == "robot_dex":
            raise ValueError("Dex records use base_x_joint and different gripper semantics; gamepad replay is incompatible")
        for key in (*JOINTS, *VELOCITIES, "delta_s_robot", "delta_theta"):
            value = action.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"missing or non-finite legacy action: {key}")
    return frames, float(rate)


def apply_action(robot, action, mode, fps):
    if mode == "velocity":
        robot.lift.set_velocity(action["v_lift"], a_m=0.20)
        robot.arm.set_velocity(action["v_arm"], a_m=0.20)
        for name in ("wrist_yaw", "wrist_pitch", "wrist_roll"):
            robot.end_of_arm.set_velocity(name, action[f"v_{name}"], a_r=8.0)
        robot.base.set_velocity(action["v_base_forward"], action["v_base_yaw"])
    else:
        robot.lift.move_to(action["joint_lift"])
        robot.arm.move_to(action["joint_arm_l0"])
        for name in ("wrist_yaw", "wrist_pitch", "wrist_roll"):
            robot.end_of_arm.get_joint(name).move_to(action[f"joint_{name}"])
        robot.base.set_velocity(action["delta_s_robot"] * fps, action["delta_theta"] * fps)
    for name in ("head_pan", "head_tilt"):
        robot.head.get_joint(name).move_to(action[f"joint_{name}"])
    velocity = action["v_gripper"]
    if abs(velocity) > 1e-3:
        robot.end_of_arm.get_joint("stretch_gripper").move_by(math.copysign(60.0, velocity))
    robot.push_command()


def stop_robot(robot):
    robot.lift.set_velocity(0.0, a_m=0.20)
    robot.arm.set_velocity(0.0, a_m=0.20)
    robot.base.set_velocity(0.0, 0.0)
    for name in ("wrist_yaw", "wrist_pitch", "wrist_roll"):
        robot.end_of_arm.set_velocity(name, 0.0, a_r=8.0)
    for name in ("head_pan", "head_tilt"):
        robot.head.set_velocity(name, 0.0, a_r=8.0)
    robot.end_of_arm.get_joint("stretch_gripper").move_by(0.0)
    robot.push_command()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--mode", choices=("velocity", "target"), default="velocity")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    frames, fps = load_legacy(args.episode, args.fps)
    print(json.dumps({"frames": len(frames), "fps": fps, "duration_s": len(frames)/fps,
                      "mode": args.mode, "execute": args.execute,
                      "timing": "estimated from legacy fps; original per-frame timestamps unavailable"}, indent=2))
    if not args.execute:
        return 0
    # The old direct hardware interface must not run alongside a ROS driver.
    try:
        result = subprocess.run(["ros2", "node", "list"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("cannot check ROS ownership; source ROS and retry") from exc
    if result.returncode or any(word in result.stdout for word in ("stretch_driver", "stretch3_quest_teleop")):
        raise RuntimeError("stop the ROS driver and VR server before direct legacy replay")
    from .robot_access import RobotControlLock
    lock = RobotControlLock()
    robot = None
    try:
        import stretch_body.robot as rb
        robot = rb.Robot()
        if not robot.startup() or not robot.is_calibrated():
            raise RuntimeError("robot must already be started and homed")
        live = {"lift": robot.lift.status["pos"], "arm": robot.arm.status["pos"]}
        live.update({name: robot.end_of_arm.motors[name].status["pos"]
                     for name in ("wrist_yaw", "wrist_pitch", "wrist_roll")})
        live.update({name: robot.head.motors[name].status["pos"] for name in ("head_pan", "head_tilt")})
        observations = frames[0]["observations"]
        for name, current in live.items():
            if (name not in observations or not math.isfinite(observations[name])
                    or not math.isfinite(current)
                    or abs(current - observations[name]) > (0.03 if name in ("lift", "arm") else 0.15)):
                raise ValueError(f"initial pose mismatch at {name}; position the robot before replay")
        start = time.monotonic()
        for index, frame in enumerate(frames):
            due = start + index / fps
            delay = due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            if time.monotonic() - due > 0.25:
                raise RuntimeError("replay is late; refusing to burst queued commands")
            apply_action(robot, frame["actions"], args.mode, fps)
        time.sleep(1.0 / fps)
    finally:
        try:
            if robot is not None:
                try:
                    stop_robot(robot)
                finally:
                    robot.stop()
        finally:
            lock.close()
    return 0
