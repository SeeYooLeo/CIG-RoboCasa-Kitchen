import asyncio
import base64
import json
import os
import struct
import time

import pytest

from stretch3_quest_teleop.backends.mock import MockRobotBackend
from stretch3_quest_teleop.config import TeleopConfig, default_config_path
from stretch3_quest_teleop.server import TeleopServer
from stretch3_quest_teleop.websocket import websocket_accept
from stretch3_quest_teleop.models import ControllerState, VRFrame


def masked_frame(payload: bytes, opcode=1):
    mask = b"\x12\x34\x56\x78"
    header = bytearray([0x80 | opcode])
    if len(payload) < 126:
        header.append(0x80 | len(payload))
    else:
        header.append(0x80 | 126)
        header.extend(struct.pack("!H", len(payload)))
    encoded = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    return bytes(header) + mask + encoded


async def read_server_frame(reader):
    first, second = await reader.readexactly(2)
    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", await reader.readexactly(2))[0]
    elif length == 127:
        length = struct.unpack("!Q", await reader.readexactly(8))[0]
    return first & 0x0F, await reader.readexactly(length)


def test_websocket_accept_reference_value():
    assert websocket_accept("dGhlIHNhbXBsZSBub25jZQ==") == (
        "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
    )


def test_http_and_websocket_end_to_end(tmp_path):
    (tmp_path / "head").mkdir()
    (tmp_path / "gripper").mkdir()
    (tmp_path / "head" / "a.jpg").write_bytes(b"head-jpeg")
    (tmp_path / "gripper" / "a.jpg").write_bytes(b"grip-jpeg")

    async def scenario():
        backend = MockRobotBackend(tmp_path)
        server = TeleopServer(
            TeleopConfig.load(default_config_path()),
            backend,
            host="127.0.0.1",
            port=0,
        )
        await server.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
            writer.write(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()
            response = await reader.read()
            assert b"200 OK" in response
            assert b'"ok": true' in response

            reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
            writer.write(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()
            response = await reader.read()
            assert b"200 OK" in response
            assert b"Stretch 3" in response

            reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
            writer.write(b"GET /panel.js HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()
            response = await reader.read()
            assert b"200 OK" in response
            assert b"panelModelMatrix" in response

            reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
            key = base64.b64encode(os.urandom(16)).decode("ascii")
            request = (
                "GET /ws HTTP/1.1\r\n"
                "Host: localhost\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\n"
                "Sec-WebSocket-Version: 13\r\n\r\n"
            )
            writer.write(request.encode("ascii"))
            await writer.drain()
            handshake = await reader.readuntil(b"\r\n\r\n")
            assert b"101 Switching Protocols" in handshake
            opcode, hello = await read_server_frame(reader)
            assert opcode == 1
            greeting = json.loads(hello)
            assert greeting["type"] == "hello"
            # The client draws its camera panel from this section.
            assert greeting["view"]["distance_m"] > 0

            pose = {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}
            message = {
                "type": "vr_frame",
                "sequence": 1,
                "client_time_ms": 16,
                "hmd": {"tracked": True, "pose": pose},
                "left": {
                    "tracked": True,
                    "pose": pose,
                    "grip": True,
                    "stick_x": 0,
                    "stick_y": -1,
                },
                "right": {"tracked": True, "pose": pose},
            }
            writer.write(masked_frame(json.dumps(message).encode()))
            await writer.drain()

            status = None
            slots = {}
            for _ in range(8):
                opcode, payload = await read_server_frame(reader)
                if opcode == 1:
                    parsed = json.loads(payload)
                    if parsed.get("type") == "status":
                        status = parsed
                elif opcode == 2:
                    # "S3" + version + slot index + the JPEG itself.
                    assert payload[:3] == b"S3\x01"
                    slots[payload[3]] = payload[4:]
                if status is not None and len(slots) == 2:
                    break
            assert status is not None
            assert status["base"]["linear_mps"] > 0
            assert slots == {0: b"head-jpeg", 1: b"grip-jpeg"}

            writer.write(masked_frame(b"", opcode=8))
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            await asyncio.sleep(0)
        finally:
            await server.close()
        assert any(command.stale_input for command in backend.commands)

    asyncio.run(scenario())


@pytest.mark.parametrize("field", ["trigger", "stick_x", "stick_y"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), None])
def test_nonfinite_controls_are_rejected(field, value):
    with pytest.raises(ValueError):
        ControllerState.from_dict({field: value})


@pytest.mark.parametrize("payload", [[], {"hmd": []}, {"left": "invalid"}, {"sequence": None}])
def test_invalid_frame_structure_is_rejected(payload):
    with pytest.raises(ValueError):
        VRFrame.from_dict(payload)


@pytest.mark.parametrize("traffic", ["ping", "invalid_json", "partial_frame"])
def test_watchdog_stops_without_valid_control_frames(traffic):
    async def scenario():
        config = TeleopConfig.load(default_config_path())
        config.raw["network"]["input_timeout_s"] = 0.06
        backend = MockRobotBackend()
        server = TeleopServer(config, backend, host="127.0.0.1", port=0)
        await server.start()
        writer = None
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
            writer.write(
                b"GET /ws HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n"
                b"Connection: Upgrade\r\nSec-WebSocket-Version: 13\r\n"
                b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n"
            )
            await writer.drain()
            await reader.readuntil(b"\r\n\r\n")
            await read_server_frame(reader)
            pose = {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}
            message = {
                "type": "vr_frame",
                "hmd": {"tracked": True, "pose": pose},
                "left": {"tracked": True, "grip": True, "stick_y": -1, "pose": pose},
            }
            writer.write(masked_frame(json.dumps(message).encode()))
            await writer.drain()
            _, payload = await read_server_frame(reader)
            assert json.loads(payload)["base"]["linear_mps"] > 0
            split = masked_frame(json.dumps(message).encode())
            if traffic == "partial_frame":
                writer.write(split[:3])
                await writer.drain()
            deadline = time.monotonic() + 0.18
            while time.monotonic() < deadline:
                if traffic == "ping":
                    writer.write(masked_frame(b"probe", opcode=9))
                elif traffic == "invalid_json":
                    writer.write(masked_frame(b"{"))
                await writer.drain()
                await asyncio.sleep(0.01)
            assert backend.commands[-1].stale_input
            assert backend.commands[-1].base_linear_mps == 0
            if traffic == "partial_frame":
                # A watchdog tick must not discard bytes from a partial frame.
                writer.write(split[3:])
                await writer.drain()
                _, payload = await asyncio.wait_for(read_server_frame(reader), 1)
                assert json.loads(payload)["base"]["linear_mps"] > 0
        finally:
            if writer is not None:
                writer.close()
                await writer.wait_closed()
                await asyncio.sleep(0.02)
            await server.close()

    asyncio.run(scenario())
