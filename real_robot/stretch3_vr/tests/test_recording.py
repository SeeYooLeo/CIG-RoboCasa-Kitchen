import asyncio
import json
from pathlib import Path
import time

import pytest

from stretch3_quest_teleop.backends.mock import MockRobotBackend
from stretch3_quest_teleop.config import TeleopConfig, default_config_path
from stretch3_quest_teleop.models import Mode, RobotCommand
from stretch3_quest_teleop.recording import EpisodeRecorder, snapshot
from stretch3_quest_teleop.replay import load_episode, run_replay
from stretch3_quest_teleop.server import TeleopServer
from test_protocol import masked_frame, read_server_frame


def test_record_save_load_and_replay_through_websocket(tmp_path):
    async def scenario():
        images = tmp_path / "input"
        for name in ("head", "gripper"):
            (images / name).mkdir(parents=True)
            (images / name / "a.jpg").write_bytes(name.encode())
        backend = MockRobotBackend(images)
        server = TeleopServer(TeleopConfig.load(default_config_path()), backend,
                              host="127.0.0.1", port=0, record_dir=tmp_path / "episodes")
        await server.start()
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        async def send(payload):
            writer.write(masked_frame(json.dumps(payload).encode()))
            await writer.drain()
        async def receive(kind):
            while True:
                opcode, data = await asyncio.wait_for(read_server_frame(reader), 2)
                if opcode == 1:
                    item = json.loads(data)
                    if item.get("type") == kind:
                        return item
        try:
            writer.write(b"GET /ws HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n"
                         b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n")
            await writer.drain()
            await reader.readuntil(b"\r\n\r\n")
            assert (await receive("hello"))["recording"]["enabled"]
            await send({"type": "record_control", "operation": "start", "task": "Move a cup"})
            assert (await receive("recording"))["active"]
            pose = {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}
            for index in range(3):
                await send({"type": "vr_frame", "sequence": index,
                            "hmd": {"tracked": True, "pose": pose},
                            "left": {"tracked": True, "grip": True, "stick_y": -0.5}})
                assert (await receive("status"))["recording"]["active"]
                await asyncio.sleep(0.01)
            await send({"type": "record_control", "operation": "save", "success": True})
            status = await receive("recording")
            assert not status["active"] and status["complete"]
            episode = tmp_path / "episodes" / status["episode"]
            metadata, frames = load_episode(episode)
            assert metadata["success"] is True
            assert frames[-1]["kind"] == "stop"
            assert frames[0]["client_sequence"] == 0
            assert frames[0]["actions"]["base_linear_mps"] > 0
            assert set(frames[0]["images"]) == {"head", "gripper"}
            assert frames[0]["observations"]["ready"]
            assert "git_commit" not in metadata and "user" not in metadata
            replay = MockRobotBackend()
            replay.start()
            run_replay(replay, frames, realtime=False)
            assert len(replay.commands) == len(frames) + 1
            for command, saved in zip(replay.commands, frames):
                assert command.base_linear_mps == saved["actions"]["base_linear_mps"]
                assert command.base_angular_rps == saved["actions"]["base_angular_rps"]
            assert replay.commands[-1].stale_input
            image = episode / frames[0]["images"]["head"]["path"]
            image.write_bytes(b"corrupt")
            with pytest.raises(ValueError, match="checksum"):
                load_episode(episode)
            # A broken connection preserves an incomplete episode.
            await send({"type": "record_control", "operation": "start", "task": "Interrupted"})
            interrupted = await receive("recording")
        finally:
            writer.close()
            await writer.wait_closed()
            await asyncio.sleep(0.05)
            await server.close()
        path = tmp_path / "episodes" / interrupted["episode"]
        with pytest.raises(ValueError, match="incomplete"):
            load_episode(path)
    asyncio.run(scenario())


def test_disk_error_does_not_mark_episode_complete(tmp_path, monkeypatch):
    backend = MockRobotBackend()
    recorder = EpisodeRecorder(tmp_path, "mock", {})
    recorder.start("Disk failure")
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(Path, "write_bytes", fail)
    action = backend.apply(RobotCommand(mode=Mode.NAV))
    recorder.capture(snapshot(backend.state()), action, {"head": {"jpeg": b"image"}})
    result = recorder.finish(True)
    assert result["error"] and not result["complete"]
    assert not (recorder.directory / "completed.txt").exists()


def test_replay_preserves_timing_and_stops_on_failure():
    times = [0.0]
    def clock(): return times[0]
    def sleep(seconds): times[0] += seconds
    backend = MockRobotBackend()
    backend.start()
    action = backend.apply(RobotCommand(mode=Mode.NAV, base_linear_mps=0.1))
    frames = [{"timestamp_s": t, "actions": action} for t in (0.2, 0.3, 0.6)]
    run_replay(backend, frames, clock=clock, sleep=sleep)
    assert times[0] == pytest.approx(0.4)
    assert backend.commands[-1].stale_input
    def fail(_): raise RuntimeError("actuator rejected command")
    backend.apply_recorded_action = fail
    with pytest.raises(RuntimeError):
        run_replay(backend, frames, realtime=False)
    assert backend.commands[-1].stale_input


def test_finish_seals_before_background_flush(tmp_path, monkeypatch):
    async def scenario():
        backend = MockRobotBackend()
        server = TeleopServer(TeleopConfig.load(default_config_path()), backend,
                              record_dir=tmp_path)
        backend.start()
        server.recorder.start("Save race")
        server._apply(RobotCommand(mode=Mode.NAV))
        original = server.recorder.finish
        def delayed(*args, **kwargs):
            time.sleep(0.05)
            return original(*args, **kwargs)
        monkeypatch.setattr(server.recorder, "finish", delayed)
        task = asyncio.create_task(server._finish_recording(True))
        await asyncio.sleep(0.01)
        assert not server.recorder.active
        server._apply(server.controller.watchdog(force=True), kind="watchdog")
        with pytest.raises(ValueError):
            server.recorder.start("Premature restart")
        assert (await task)["complete"]
        _, frames = load_episode(server.recorder.directory)
        assert [f["kind"] for f in frames] == ["control", "stop"]
        (server.recorder.directory / "completed.txt").unlink()
        with pytest.raises(ValueError, match="incomplete"):
            load_episode(server.recorder.directory)
    asyncio.run(scenario())


def test_replay_rechecks_time_after_oversleep():
    times = [0.0]
    backend = MockRobotBackend()
    backend.start()
    action = backend.apply(RobotCommand(mode=Mode.NAV, base_linear_mps=0.1))
    backend.commands.clear()
    frames = [{"timestamp_s": t, "actions": action} for t in (0, 0.1)]
    def oversleep(delay): times[0] += delay + 0.3
    with pytest.raises(RuntimeError, match="fell behind"):
        run_replay(backend, frames, clock=lambda: times[0], sleep=oversleep)
    assert len(backend.commands) == 2
    assert backend.commands[-1].stale_input


def test_observations_and_images_are_sampled_before_action(tmp_path):
    backend = MockRobotBackend()
    backend.start()
    server = TeleopServer(TeleopConfig.load(default_config_path()), backend, record_dir=tmp_path)
    server.recorder.start("Ordering")
    original = backend.apply
    current_image = [b"before"]
    backend.recording_images = lambda: {"head": {"jpeg": current_image[0]}}
    def apply(command):
        current_image[0] = b"after"
        time.sleep(0.01)
        return original(command)
    backend.apply = apply
    server._apply(RobotCommand(mode=Mode.NAV, head_target=(0.5, 0.5)))
    asyncio.run(server._finish_recording(True))
    _, frames = load_episode(server.recorder.directory)
    first = frames[0]
    assert first["observations"]["head_pan"] != 0.5
    assert (server.recorder.directory / first["images"]["head"]["path"]).read_bytes() == b"before"
    assert first["submission_duration_s"] >= 0.01


def test_failed_stop_leaves_episode_incomplete(tmp_path):
    async def scenario():
        backend = MockRobotBackend()
        backend.start()
        server = TeleopServer(TeleopConfig.load(default_config_path()), backend, record_dir=tmp_path)
        server.recorder.start("Stop failure")
        server._apply(RobotCommand(mode=Mode.NAV))
        def fail(_): raise RuntimeError("stop service unavailable")
        backend.apply = fail
        result = await server._finish_recording(True)
        assert not result["complete"] and "stop failed" in result["error"]
        with pytest.raises(ValueError, match="incomplete"):
            load_episode(server.recorder.directory)
    asyncio.run(scenario())
