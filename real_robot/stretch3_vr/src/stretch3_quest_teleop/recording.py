"""Timestamped episode storage with the legacy labels.json outer structure."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import queue
import threading
import time
import uuid

SCHEMA = "stretch3-vr-ros2-v1"


def snapshot(state):
    result = asdict(state)
    now = time.monotonic()
    for name in ("joint_state", "odom"):
        received = result.pop(f"{name}_received_monotonic")
        result[f"{name}_age_s"] = None if received is None else max(0.0, now - received)
    return result


class EpisodeRecorder:
    """Write commands and latest camera samples on a bounded background queue.

    Queue overflow or a disk error marks the episode incomplete; it never
    silently drops frames and reports a successful recording.
    """

    def __init__(self, root: Path, backend: str, config: dict):
        self.root = Path(root).resolve()
        self.backend = backend
        self.config = config
        self.directory = None
        self.active = False
        self.error = None
        self.frames = 0
        self.control_frames = 0
        self._thread = None
        self._sealed = False

    def start(self, task: str):
        if self.active or self._sealed:
            raise ValueError("an episode is already being recorded")
        if not isinstance(task, str) or not task.strip() or len(task) > 500:
            raise ValueError("task must contain 1 to 500 characters")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.directory = self.root / f"{stamp}_{uuid.uuid4().hex[:10]}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self._queue = queue.Queue(maxsize=256)
        self._start = time.monotonic()
        self.frames = 0
        self.control_frames = 0
        self.error = None
        self._metadata = {
            "schema": SCHEMA, "backend": self.backend, "task": task.strip(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "complete": False, "success": None,
            "timing": "host monotonic timestamps; latest available asynchronous observations",
            "action_semantics": "submitted ROS base velocities and joint targets; acceptance is not guaranteed",
            "config": self.config,
        }
        self._write_metadata()
        self.active = True
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()
        return self.status()

    def _write_metadata(self):
        temporary = self.directory / "configs.json.tmp"
        temporary.write_text(json.dumps(self._metadata, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.directory / "configs.json")

    def capture(self, observations, actions, images, *, kind="control", client_sequence=None,
                observed_at=None, submitted_at=None, submission_duration_s=0.0):
        if not self.active or self.error:
            return
        if self.backend == "ros2" and kind == "control":
            if any(name not in images or images[name].get("age_s", float("inf")) > 0.5
                   for name in ("head", "gripper")):
                self.error = "head or wrist camera became unavailable or stale"
                return
        observed_at = time.monotonic() if observed_at is None else observed_at
        frame = {
            "timestamp_s": (observed_at if submitted_at is None else submitted_at) - self._start,
            "observation_timestamp_s": observed_at - self._start,
            "submission_duration_s": submission_duration_s,
            "observations": observations, "actions": actions,
            "kind": kind, "client_sequence": client_sequence,
        }
        try:
            self._queue.put_nowait((frame, images))
        except queue.Full:
            self.error = "recording queue overflow"

    def _write_loop(self):
        cached = {}
        try:
            with (self.directory / "frames.jsonl").open("x") as stream:
                while True:
                    item = self._queue.get()
                    if item is None:
                        break
                    frame, images = item
                    step = self.frames
                    frame["step"] = step
                    frame["images"] = {}
                    for name, sample in images.items():
                        if name not in ("head", "navigation", "gripper"):
                            raise ValueError("unknown camera name")
                        jpeg = sample["jpeg"]
                        digest = hashlib.sha256(jpeg).hexdigest()
                        previous = cached.get(name)
                        if previous is None or previous[0] != digest:
                            relative = f"images/{name}/{step:08d}.jpg"
                            path = self.directory / relative
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_bytes(jpeg)
                            cached[name] = (digest, relative)
                        frame["images"][name] = {
                            "path": cached[name][1], "sha256": digest,
                            "source_stamp": sample.get("source_stamp"),
                            "age_s": sample.get("age_s"),
                        }
                    stream.write(json.dumps(frame, allow_nan=False) + "\n")
                    stream.flush()
                    self.frames += 1
                    self.control_frames += frame["kind"] == "control"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def seal(self):
        """Stop accepting events on the control thread before flushing elsewhere."""
        if not self.active:
            raise ValueError("no active recording")
        self.active = False
        self._sealed = True
        self._end = time.monotonic()

    def finish(self, success=None, *, complete=True):
        if self.active:
            self.seal()
        elif not self._sealed:
            raise ValueError("no active recording")
        while self._thread.is_alive():
            try:
                self._queue.put(None, timeout=0.05)
                break
            except queue.Full:
                continue
        self._thread.join()
        # A valid JSON dictionary remains compatible with existing label readers.
        temporary = self.directory / "labels.json.tmp"
        with temporary.open("w") as out:
            out.write("{\n")
            source = self.directory / "frames.jsonl"
            if source.exists():
                with source.open() as stream:
                    for index, line in enumerate(stream):
                        out.write((",\n" if index else "") + json.dumps(str(index)) + ":" + line.strip())
            out.write("\n}\n")
        temporary.replace(self.directory / "labels.json")
        complete = bool(complete and not self.error and self.control_frames)
        self._metadata.update(complete=complete, success=success, num_frames=self.frames,
                              control_frames=self.control_frames,
                              duration_s=self._end - self._start, error=self.error)
        self._write_metadata()
        if complete:
            (self.directory / "completed.txt").write_text("Completed\n")
        self._sealed = False
        return self.status()

    def status(self):
        return {"enabled": True, "active": self.active,
                "episode": self.directory.name if self.directory else None,
                "frames": self.frames, "error": self.error,
                "complete": bool(getattr(self, "_metadata", {}).get("complete", False))}
