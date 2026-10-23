from __future__ import annotations

import asyncio
import base64
import hashlib
import struct
from dataclasses import dataclass


WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class WebSocketError(RuntimeError):
    pass


@dataclass(frozen=True)
class WebSocketFrame:
    opcode: int
    payload: bytes
    final: bool = True


def websocket_accept(key: str) -> str:
    digest = hashlib.sha1((key + WEBSOCKET_GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def encode_server_frame(payload: bytes, opcode: int = 0x1) -> bytes:
    length = len(payload)
    header = bytearray([0x80 | (opcode & 0x0F)])
    if length < 126:
        header.append(length)
    elif length <= 0xFFFF:
        header.append(126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", length))
    return bytes(header) + payload


async def read_client_frame(
    reader: asyncio.StreamReader, max_message_bytes: int
) -> WebSocketFrame:
    first_two = await reader.readexactly(2)
    first, second = first_two
    final = bool(first & 0x80)
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    length = second & 0x7F
    if not masked:
        raise WebSocketError("client frames must be masked")
    if length == 126:
        length = struct.unpack("!H", await reader.readexactly(2))[0]
    elif length == 127:
        length = struct.unpack("!Q", await reader.readexactly(8))[0]
    if length > max_message_bytes:
        raise WebSocketError("message exceeds configured size limit")
    mask = await reader.readexactly(4)
    payload = bytearray(await reader.readexactly(length))
    for index in range(length):
        payload[index] ^= mask[index % 4]
    if not final:
        raise WebSocketError("fragmented messages are not supported")
    return WebSocketFrame(opcode=opcode, payload=bytes(payload), final=final)


class WebSocketPeer:
    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        max_message_bytes: int,
    ):
        self.reader = reader
        self.writer = writer
        self.max_message_bytes = max_message_bytes
        self._write_lock = asyncio.Lock()

    async def receive(self) -> WebSocketFrame:
        return await read_client_frame(self.reader, self.max_message_bytes)

    async def send_text(self, text: str) -> None:
        await self._send(text.encode("utf-8"), 0x1)

    async def send_binary(self, payload: bytes) -> None:
        await self._send(payload, 0x2)

    async def send_pong(self, payload: bytes) -> None:
        await self._send(payload, 0xA)

    async def close(self) -> None:
        if self.writer.is_closing():
            return
        try:
            await self._send(b"", 0x8)
        except (ConnectionError, asyncio.CancelledError):
            pass
        self.writer.close()
        try:
            await self.writer.wait_closed()
        except ConnectionError:
            pass

    async def _send(self, payload: bytes, opcode: int) -> None:
        async with self._write_lock:
            self.writer.write(encode_server_frame(payload, opcode))
            await self.writer.drain()
