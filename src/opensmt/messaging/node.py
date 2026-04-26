from __future__ import annotations

import asyncio
import base64
import inspect
import json
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from .scpi import SCPIKind, SCPIMessage, normalize_command, parse_scpi, render_value

TextCallback = Callable[[dict[str, Any], SCPIMessage], Awaitable[None] | None]
BinaryCallback = Callable[[dict[str, Any], bytes], Awaitable[None] | None]


@dataclass(slots=True)
class CallbackRegistry:
    query: dict[str, list[TextCallback]] = field(default_factory=lambda: defaultdict(list))
    set_cmd: dict[str, list[TextCallback]] = field(default_factory=lambda: defaultdict(list))
    response: dict[str, list[TextCallback]] = field(default_factory=lambda: defaultdict(list))
    working: dict[str, list[TextCallback]] = field(default_factory=lambda: defaultdict(list))
    action: dict[str, list[TextCallback]] = field(default_factory=lambda: defaultdict(list))
    text_any: list[TextCallback] = field(default_factory=list)
    binary_any: list[BinaryCallback] = field(default_factory=list)


class BusNode:
    def __init__(self, name: str, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.name = name
        self.host = host
        self.port = port
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._callbacks = CallbackRegistry()

    async def connect(self) -> None:
        self._reader, self._writer = await asyncio.open_connection(self.host, self.port)
        hello = {"type": "hello", "name": self.name}
        self._writer.write((json.dumps(hello) + "\n").encode("utf-8"))
        await self._writer.drain()
        self._receive_task = asyncio.create_task(self._receive_loop(), name=f"bus-node-recv-{self.name}")

    async def close(self) -> None:
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()

    def on_query(self, command: str, callback: TextCallback) -> None:
        self._callbacks.query[normalize_command(command)].append(callback)

    def on_set(self, command: str, callback: TextCallback) -> None:
        self._callbacks.set_cmd[normalize_command(command)].append(callback)

    def on_response(self, command: str, callback: TextCallback) -> None:
        self._callbacks.response[normalize_command(command)].append(callback)

    def on_working(self, command: str, callback: TextCallback) -> None:
        self._callbacks.working[normalize_command(command)].append(callback)

    def on_action(self, command: str, callback: TextCallback) -> None:
        self._callbacks.action[normalize_command(command)].append(callback)

    def on_text(self, callback: TextCallback) -> None:
        self._callbacks.text_any.append(callback)

    def on_binary(self, callback: BinaryCallback) -> None:
        self._callbacks.binary_any.append(callback)

    async def send_query(self, command: str, *, target: str | None = None) -> None:
        cmd = normalize_command(command)
        await self.send_text(f"{cmd}?", kind=SCPIKind.QUERY.value, target=target)

    async def send_set(self, command: str, value: Any, *, target: str | None = None) -> None:
        cmd = normalize_command(command)
        await self.send_text(f"{cmd} {render_value(value)}", kind=SCPIKind.SET.value, target=target)

    async def send_response(self, command: str, value: Any, *, target: str | None = None) -> None:
        cmd = normalize_command(command)
        await self.send_text(f"{cmd}? {render_value(value)}", kind=SCPIKind.RESPONSE.value, target=target)

    async def send_working(self, command: str, *, target: str | None = None) -> None:
        cmd = normalize_command(command)
        await self.send_text(f"{cmd} WORKING", kind=SCPIKind.WORKING.value, target=target)

    async def send_action(self, command: str, *, target: str | None = None) -> None:
        cmd = normalize_command(command)
        await self.send_text(cmd, kind=SCPIKind.ACTION.value, target=target)

    async def send_text(self, text: str, *, kind: str | None = None, target: str | None = None) -> None:
        packet = {
            "type": "message",
            "id": str(uuid.uuid4()),
            "source": self.name,
            "target": target,
            "channel": "text",
            "kind": kind,
            "text": text,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        await self._send_packet(packet)

    async def send_binary(self, payload: bytes, *, topic: str = "BINARY", target: str | None = None) -> None:
        packet = {
            "type": "message",
            "id": str(uuid.uuid4()),
            "source": self.name,
            "target": target,
            "channel": "binary",
            "kind": "binary",
            "topic": topic,
            "binary": base64.b64encode(payload).decode("ascii"),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        await self._send_packet(packet)

    async def _send_packet(self, packet: dict[str, Any]) -> None:
        if not self._writer:
            raise RuntimeError("BusNode is not connected")
        self._writer.write((json.dumps(packet, ensure_ascii=True) + "\n").encode("utf-8"))
        await self._writer.drain()

    async def _receive_loop(self) -> None:
        assert self._reader is not None
        while True:
            line = await self._reader.readline()
            if not line:
                break
            try:
                packet = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if packet.get("type") != "message":
                continue
            await self._dispatch_packet(packet)

    async def _dispatch_packet(self, packet: dict[str, Any]) -> None:
        channel = packet.get("channel")
        if channel == "binary":
            raw = packet.get("binary", "")
            data = base64.b64decode(raw) if raw else b""
            for callback in self._callbacks.binary_any:
                await self._run_callback(callback, packet, data)
            return

        text = str(packet.get("text", ""))
        parsed = parse_scpi(text)

        for callback in self._callbacks.text_any:
            await self._run_callback(callback, packet, parsed)

        if parsed.kind == SCPIKind.UNKNOWN or not parsed.command:
            return

        if parsed.kind == SCPIKind.QUERY:
            await self._dispatch_by_command(self._callbacks.query, parsed.command, packet, parsed)
        elif parsed.kind == SCPIKind.SET:
            await self._dispatch_by_command(self._callbacks.set_cmd, parsed.command, packet, parsed)
        elif parsed.kind == SCPIKind.RESPONSE:
            await self._dispatch_by_command(self._callbacks.response, parsed.command, packet, parsed)
        elif parsed.kind == SCPIKind.WORKING:
            await self._dispatch_by_command(self._callbacks.working, parsed.command, packet, parsed)
        elif parsed.kind == SCPIKind.ACTION:
            await self._dispatch_by_command(self._callbacks.action, parsed.command, packet, parsed)

    async def _dispatch_by_command(
        self,
        callback_map: dict[str, list[TextCallback]],
        command: str,
        packet: dict[str, Any],
        parsed: SCPIMessage,
    ) -> None:
        for key in (command, ":*"):
            for callback in callback_map.get(key, []):
                await self._run_callback(callback, packet, parsed)

    async def _run_callback(self, callback: Callable[..., Any], *args: Any) -> None:
        result = callback(*args)
        if inspect.isawaitable(result):
            await result
