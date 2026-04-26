from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ClientState:
    name: str
    writer: asyncio.StreamWriter


class MessageBroker:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._server: asyncio.AbstractServer | None = None
        self._clients: dict[str, ClientState] = {}
        self._writers: dict[asyncio.StreamWriter, str] = {}
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle_client, self.host, self.port)
        addr = ", ".join(str(sock.getsockname()) for sock in (self._server.sockets or []))
        log.info("Broker listening on %s", addr)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        async with self._lock:
            writers = [state.writer for state in self._clients.values()]
            self._clients.clear()
            self._writers.clear()

        for writer in writers:
            writer.close()
            await writer.wait_closed()

    async def wait_closed(self) -> None:
        if self._server is None:
            return
        async with self._server:
            await self._server.serve_forever()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        client_name = None
        try:
            line = await reader.readline()
            if not line:
                return

            hello = json.loads(line.decode("utf-8"))
            if hello.get("type") != "hello" or not hello.get("name"):
                raise ValueError("invalid hello")

            client_name = str(hello["name"])
            async with self._lock:
                if client_name in self._clients:
                    old_writer = self._clients[client_name].writer
                    old_writer.close()
                    self._writers.pop(old_writer, None)
                self._clients[client_name] = ClientState(name=client_name, writer=writer)
                self._writers[writer] = client_name

            log.info("Client connected: %s (%s)", client_name, peer)

            while True:
                line = await reader.readline()
                if not line:
                    break
                packet = json.loads(line.decode("utf-8"))
                if packet.get("type") != "message":
                    continue
                await self._route_message(packet)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("Client %s disconnected with error: %s", client_name or peer, exc)
        finally:
            await self._remove_client(writer)

    async def _remove_client(self, writer: asyncio.StreamWriter) -> None:
        async with self._lock:
            name = self._writers.pop(writer, None)
            if name and self._clients.get(name, None) and self._clients[name].writer is writer:
                self._clients.pop(name, None)

        writer.close()
        await writer.wait_closed()

    async def _route_message(self, packet: dict[str, Any]) -> None:
        target = packet.get("target")

        async with self._lock:
            if target:
                targets = []
                state = self._clients.get(str(target))
                if state:
                    targets.append(state.writer)
                sender = self._clients.get(str(packet.get("source", "")))
                if sender and sender.writer not in targets:
                    targets.append(sender.writer)
            else:
                targets = [state.writer for state in self._clients.values()]

        payload = (json.dumps(packet, ensure_ascii=True) + "\n").encode("utf-8")
        stale: list[asyncio.StreamWriter] = []
        for writer in targets:
            try:
                writer.write(payload)
                await writer.drain()
            except Exception:
                stale.append(writer)

        for writer in stale:
            await self._remove_client(writer)


async def run_broker(host: str, port: int) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    broker = MessageBroker(host=host, port=port)
    await broker.start()
    await broker.wait_closed()
