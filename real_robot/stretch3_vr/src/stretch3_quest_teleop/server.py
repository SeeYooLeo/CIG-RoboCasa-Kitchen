from __future__ import annotations

import asyncio
import contextlib
import json
import mimetypes
import ssl
import time
from importlib.resources import files
from pathlib import PurePosixPath
from typing import Any

from .backends.base import RobotBackend
from .config import TeleopConfig
from .controller import TeleopController
from .models import VRFrame, encode_camera_frame
from .recording import EpisodeRecorder, snapshot
from .websocket import WebSocketError, WebSocketPeer, websocket_accept


class TeleopServer:
    def __init__(
        self,
        config: TeleopConfig,
        backend: RobotBackend,
        host: str | None = None,
        port: int | None = None,
        ssl_context: ssl.SSLContext | None = None,
        record_dir=None,
    ):
        self.config = config
        self.backend = backend
        self.controller = TeleopController(config)
        self.host = host or str(config.network["host"])
        self.port = int(config.network["port"] if port is None else port)
        self.ssl_context = ssl_context
        self._server: asyncio.Server | None = None
        self._client_lock = asyncio.Lock()
        self._input_frames = 0
        self._camera_sent: dict[str, int] = {}
        self._camera_sent_bytes: dict[str, int] = {}
        self._watchdog_task: asyncio.Task | None = None
        self.recorder = EpisodeRecorder(
            record_dir, "ros2" if type(backend).__name__ == "Ros2StretchBackend" else "mock",
            config.raw,
        ) if record_dir is not None else None
        self._connections: set[asyncio.Task] = set()

    async def start(self) -> None:
        self.backend.start()
        self._server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port,
            ssl=self.ssl_context,
        )
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())
        sockets = self._server.sockets or []
        if sockets:
            self.port = int(sockets[0].getsockname()[1])

    async def serve_forever(self) -> None:
        await self.start()
        assert self._server is not None
        protocol = "https" if self.ssl_context else "http"
        print(f"Quest UI: {protocol}://<robot-ip>:{self.port}/")
        print(f"Health:   {protocol}://127.0.0.1:{self.port}/healthz")
        print(f"Backend:  {type(self.backend).__name__}")
        # Report the effective speed limits after applying all overlays.
        base = self.config.base
        print(
            "Overlays: "
            + (", ".join(self.config.overlays) if self.config.overlays else "none")
            + f"  (NAV {base['nav_linear_mps']} m/s, {base['nav_angular_rps']} rad/s)"
        )
        async with self._server:
            await self._server.serve_forever()

    def link_diagnostics(self) -> dict:
        """What actually went out on the wire, per panel slot.

        The camera counters in the backend say a JPEG was produced; these say it
        was handed to the socket. Without both, a frozen panel cannot be pinned
        on the encoder, the link, or the headset -- and the single client slot
        means a debug connection would evict the operator to find out.
        """
        return {
            "input_frames": self._input_frames,
            "sent": dict(self._camera_sent),
            "sent_bytes": dict(self._camera_sent_bytes),
        }

    async def close(self) -> None:
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watchdog_task
            self._watchdog_task = None
        # Preserve startup errors even if the backend cannot accept a stop.
        try:
            failsafe = self.controller.watchdog(force=True)
            if failsafe is not None:
                self.backend.apply(failsafe)
        except Exception as exc:
            print(f"failsafe on shutdown failed: {exc}")
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        for task in list(self._connections):
            task.cancel()
        if self._connections:
            await asyncio.gather(*list(self._connections), return_exceptions=True)
        if self.recorder is not None and self.recorder.active:
            await self._finish_recording(None, complete=False)
        self.backend.stop()

    async def _watchdog_loop(self) -> None:
        """Check input freshness independently of reads and camera writes."""
        period = max(0.01, float(self.config.network["input_timeout_s"]) / 2.0)
        while True:
            await asyncio.sleep(period)
            command = self.controller.watchdog()
            if command is not None:
                try:
                    self._apply(command, kind="watchdog")
                except Exception as exc:
                    if self.recorder is not None and self.recorder.active:
                        self.recorder.error = f"watchdog stop failed: {exc}"
                    print(f"watchdog stop failed: {exc}")

    def _apply(self, command, *, state=None, kind="control", sequence=None):
        observations = snapshot(state if state is not None else self.backend.state())
        images = self.backend.recording_images() if self.recorder and self.recorder.active else {}
        observed_at = time.monotonic()
        submitted = self.backend.apply(command)
        submitted_at = submitted.pop("submitted_at_monotonic", observed_at)
        if self.recorder is not None and self.recorder.active:
            self.recorder.capture(observations, submitted, images, kind=kind,
                                  client_sequence=sequence, observed_at=observed_at,
                                  submitted_at=submitted_at,
                                  submission_duration_s=time.monotonic() - observed_at)

    def _recording_status(self):
        return self.recorder.status() if self.recorder else {"enabled": False, "active": False}

    async def _finish_recording(self, success, *, complete=True):
        try:
            self._apply(self.controller.watchdog(force=True), kind="stop")
        except Exception as exc:
            self.recorder.error = f"stop failed: {exc}"
            complete = False
        self.recorder.seal()
        self.backend.set_recording(False)
        flush = asyncio.create_task(asyncio.to_thread(self.recorder.finish, success, complete=complete))
        try:
            return await asyncio.shield(flush)
        except asyncio.CancelledError:
            await flush
            raise

    async def _record_control(self, payload):
        if self.recorder is None:
            raise ValueError("recording is disabled; start the server with --record-dir")
        operation = payload.get("operation")
        if operation == "start":
            if self.recorder.active:
                raise ValueError("an episode is already being recorded")
            if not self.backend.state().ready:
                raise ValueError("backend is not ready")
            self.backend.set_recording(True)
            try:
                if self.recorder.backend == "ros2":
                    deadline = time.monotonic() + 3
                    while True:
                        images = self.backend.recording_images()
                        if all(name in images and images[name]["age_s"] < 0.5
                               for name in ("head", "gripper")):
                            break
                        if time.monotonic() >= deadline:
                            raise ValueError("fresh head and wrist camera frames are required to record")
                        await asyncio.sleep(0.05)
                return self.recorder.start(payload.get("task", ""))
            except Exception:
                self.backend.set_recording(False)
                raise
        if operation in ("save", "abort"):
            success = payload.get("success")
            if success is not None and not isinstance(success, bool):
                raise ValueError("success must be true, false or null")
            if not self.recorder.active:
                raise ValueError("no active recording")
            return await self._finish_recording(success, complete=operation == "save")
        raise ValueError("unknown recording operation")

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        self._connections.add(task)
        try:
            request_line, headers = await self._read_http_headers(reader)
            method, target, _ = request_line.split(" ", 2)
            path = target.split("?", 1)[0]
            if (
                path == "/ws"
                and headers.get("upgrade", "").lower() == "websocket"
            ):
                await self._upgrade_websocket(reader, writer, headers)
            else:
                await self._serve_http(writer, method, path)
        except (ValueError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            await self._write_response(writer, 400, b"Bad Request\n", "text/plain")
        except ConnectionError:
            writer.close()
        finally:
            self._connections.discard(task)

    async def _read_http_headers(
        self, reader: asyncio.StreamReader
    ) -> tuple[str, dict[str, str]]:
        raw = await reader.readuntil(b"\r\n\r\n")
        if len(raw) > 16384:
            raise ValueError("HTTP headers too large")
        lines = raw.decode("iso-8859-1").split("\r\n")
        request_line = lines[0]
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if not line:
                continue
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
        return request_line, headers

    async def _upgrade_websocket(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        headers: dict[str, str],
    ) -> None:
        key = headers.get("sec-websocket-key")
        if not key:
            await self._write_response(writer, 400, b"Missing WebSocket key\n", "text/plain")
            return
        if self._client_lock.locked():
            await self._write_response(writer, 409, b"Teleop client already connected\n", "text/plain")
            return
        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {websocket_accept(key)}\r\n"
            "\r\n"
        )
        writer.write(response.encode("ascii"))
        await writer.drain()
        peer = WebSocketPeer(
            reader,
            writer,
            int(self.config.network["max_message_bytes"]),
        )
        async with self._client_lock:
            await self._run_teleop_peer(peer)

    async def _run_teleop_peer(self, peer: WebSocketPeer) -> None:
        camera_period = 1.0 / max(0.1, float(self.config.network["camera_send_hz"]))
        last_camera_sent = 0.0
        try:
            await peer.send_text(
                json.dumps(
                    {
                        "type": "hello",
                        "protocol": 2,
                        "mode": self.controller.mode.value,
                        "view": self.config.raw.get("view", {}),
                        "recording": self._recording_status(),
                    }
                )
            )
            while True:
                # Keep a partial WebSocket read intact across watchdog ticks.
                frame = await peer.receive()

                if frame.opcode == 0x8:
                    break
                if frame.opcode == 0x9:
                    await peer.send_pong(frame.payload)
                    continue
                if frame.opcode != 0x1:
                    continue
                try:
                    payload: dict[str, Any] = json.loads(frame.payload.decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("message must be an object")
                    if payload.get("type") == "client_error":
                        print(f"client: {str(payload.get('message'))[:400]}", flush=True)
                        continue
                    if payload.get("type") == "record_control":
                        status = await self._record_control(payload)
                        await peer.send_text(json.dumps({"type": "recording", **status}))
                        continue
                    vr_frame = VRFrame.from_dict(payload)
                    state = self.backend.state()
                    command = self.controller.process(vr_frame, state)
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                    await peer.send_text(
                        json.dumps({"type": "error", "message": str(exc)})
                    )
                    continue

                self._apply(command, state=state, sequence=vr_frame.sequence)
                await peer.send_text(json.dumps({
                    **command.status_dict(), "recording": self._recording_status(),
                }))
                self._input_frames += 1
                now = time.monotonic()
                if now - last_camera_sent >= camera_period:
                    for slot, jpeg in self.backend.camera_frames().items():
                        await peer.send_binary(encode_camera_frame(slot, jpeg))
                        self._camera_sent[slot] = self._camera_sent.get(slot, 0) + 1
                        self._camera_sent_bytes[slot] = len(jpeg)
                    last_camera_sent = now
        except (asyncio.IncompleteReadError, ConnectionError, WebSocketError):
            pass
        finally:
            try:
                if self.recorder is not None and self.recorder.active:
                    await self._finish_recording(None, complete=False)
                else:
                    self._apply(self.controller.watchdog(force=True), kind="stop")
            finally:
                await peer.close()

    async def _serve_http(
        self, writer: asyncio.StreamWriter, method: str, path: str
    ) -> None:
        if method != "GET":
            await self._write_response(writer, 405, b"Method Not Allowed\n", "text/plain")
            return
        if path == "/healthz":
            body = json.dumps(
                {
                    "ok": True,
                    "backend_ready": self.backend.state().ready,
                    "mode": self.controller.mode.value,
                    "cameras": self.backend.camera_diagnostics(),
                    "link": self.link_diagnostics(),
                    "recording": self._recording_status(),
                }
            ).encode("utf-8")
            await self._write_response(writer, 200, body, "application/json")
            return

        route = "/index.html" if path == "/" else path
        relative = PurePosixPath(route.lstrip("/"))
        if ".." in relative.parts or len(relative.parts) != 1:
            await self._write_response(writer, 404, b"Not Found\n", "text/plain")
            return
        resource = files("stretch3_quest_teleop").joinpath("web", relative.name)
        if not resource.is_file():
            await self._write_response(writer, 404, b"Not Found\n", "text/plain")
            return
        body = resource.read_bytes()
        content_type = mimetypes.guess_type(relative.name)[0] or "application/octet-stream"
        await self._write_response(writer, 200, body, content_type)

    async def _write_response(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        body: bytes,
        content_type: str,
    ) -> None:
        reasons = {
            200: "OK",
            400: "Bad Request",
            404: "Not Found",
            405: "Method Not Allowed",
            409: "Conflict",
        }
        response = (
            f"HTTP/1.1 {status} {reasons.get(status, '')}\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Content-Type: {content_type}\r\n"
            "Cache-Control: no-store\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode("ascii")
        writer.write(response + body)
        await writer.drain()
        writer.close()
        try:
            await writer.wait_closed()
        except ConnectionError:
            pass


def make_ssl_context(cert_path: str, key_path: str) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(cert_path, key_path)
    return context
