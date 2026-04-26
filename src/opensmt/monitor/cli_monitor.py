from __future__ import annotations

import asyncio
from pathlib import Path

from opensmt.messaging import BusNode, SCPIMessage


def _decode_hex_stream(text: str) -> bytes:
    cleaned = text.replace(" ", "")
    return bytes.fromhex(cleaned)


async def _playback_file(node: BusNode, playback_file: str) -> None:
    path = Path(playback_file)
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue

        if item.startswith("SLEEP "):
            delay = float(item.split(maxsplit=1)[1])
            await asyncio.sleep(delay)
            continue

        if item.startswith("BIN "):
            data = _decode_hex_stream(item.split(maxsplit=1)[1])
            await node.send_binary(data, topic="PLAYBACK")
            continue

        await node.send_text(item)


async def run_terminal_monitor(host: str, port: int, name: str, playback_file: str | None = None) -> None:
    node = BusNode(name=name, host=host, port=port)
    await node.connect()

    async def on_text(packet: dict, msg: SCPIMessage) -> None:
        print(f"[{packet.get('source', '?')} -> {packet.get('target', '*')}] {msg.raw}")

    async def on_binary(packet: dict, payload: bytes) -> None:
        print(
            f"[{packet.get('source', '?')} -> {packet.get('target', '*')}] "
            f"BINARY {packet.get('topic', 'BINARY')} {payload.hex()}"
        )

    node.on_text(on_text)
    node.on_binary(on_binary)

    if playback_file:
        asyncio.create_task(_playback_file(node, playback_file))

    print("Monitor verbunden. Befehle: /quit, /play <datei>, /bin <hexbytes>")
    try:
        while True:
            line = await asyncio.to_thread(input, "> ")
            command = line.strip()
            if not command:
                continue

            if command == "/quit":
                break

            if command.startswith("/play "):
                path = command.split(maxsplit=1)[1].strip()
                asyncio.create_task(_playback_file(node, path))
                continue

            if command.startswith("/bin "):
                hex_text = command.split(maxsplit=1)[1]
                await node.send_binary(_decode_hex_stream(hex_text), topic="MONITOR")
                continue

            await node.send_text(command)
    finally:
        await node.close()
