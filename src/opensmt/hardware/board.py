from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import serial_asyncio

log = logging.getLogger(__name__)

_COORD_RE = re.compile(r"\b([A-Z]):(-?\d+(?:\.\d+)?)")
_OK_RE = re.compile(r"^ok\b", re.IGNORECASE)
_BUSY_RE = re.compile(r"echo:busy:", re.IGNORECASE)


def _parse_coords(text: str) -> dict[str, float]:
    """Extract GCode position report fields, ignoring stepper-count suffix."""
    part = text.split("Count")[0]
    return {m.group(1): float(m.group(2)) for m in _COORD_RE.finditer(part)}


@dataclass(slots=True)
class BoardConfig:
    board_id: str
    device: str
    baudrate: int = 115200
    bytesize: int = 8
    parity: str = "N"
    stopbits: int = 1
    xonxoff: bool = False
    rtscts: bool = False
    dsrdtr: bool = False


class SerialBoard:
    """Wraps one physical GCODE controller board over a serial port.

    Responsibilities:
    - Serial connection management and read loop
    - Low-level GCode send + OK / coordinate-response waiting
    - move(), home(), set_digital_out(), set_analog_out(), write_raw()

    Callers (HardwareDriver) own the logical-axis→GCode-letter mapping and
    update the PositionStore after moves complete.
    """

    def __init__(self, config: BoardConfig) -> None:
        self._config = config
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._read_task: asyncio.Task[None] | None = None
        self._line_queue: asyncio.Queue[str] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self.last_rx: str = ""
        self.last_tx: str = ""

    @property
    def board_id(self) -> str:
        return self._config.board_id

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        reader, writer = await serial_asyncio.open_serial_connection(
            url=self._config.device,
            baudrate=self._config.baudrate,
            bytesize=self._config.bytesize,
            parity=self._config.parity,
            stopbits=self._config.stopbits,
            xonxoff=self._config.xonxoff,
            rtscts=self._config.rtscts,
            dsrdtr=self._config.dsrdtr,
        )
        self._reader = reader
        self._writer = writer
        self._read_task = asyncio.create_task(
            self._read_loop(), name=f"board-rx-{self._config.board_id}"
        )
        log.info("Board %s connected on %s", self._config.board_id, self._config.device)

    async def stop(self) -> None:
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
        log.info("Board %s disconnected", self._config.board_id)

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    async def move(self, axis_moves: list[tuple[str, float]], velocity: float) -> None:
        """Move one or more axes simultaneously on this board.

        axis_moves: list of (gcode_letter, target_position)
        velocity:   feed rate in mm/min (already scaled by speed factor)

        Raises RuntimeError on timeout or missing OK.
        """
        async with self._lock:
            terms = " ".join(f"{letter}{value:g}" for letter, value in axis_moves)
            move_line = f"G0 {terms} F{velocity:g}"
            await self._write_line(move_line)

            ok = await self._wait_for_ok(timeout=10.0)
            if not ok:
                raise RuntimeError(
                    f"Board {self._config.board_id}: no OK after '{move_line}'"
                )

            await self._write_line("M400")
            ok = await self._wait_for_ok(timeout=120.0)
            if not ok:
                raise RuntimeError(
                    f"Board {self._config.board_id}: move timeout (M400) for axes "
                    + ", ".join(letter for letter, _ in axis_moves)
                )

    async def home(
        self,
        gcode_letters: list[str],
        homing_velocities: dict[str, float],
    ) -> dict[str, float]:
        """Home the given GCode axes.

        Returns a dict of gcode_letter → position as reported by the firmware.
        Empty dict if the firmware did not send a coordinate report.
        """
        async with self._lock:
            m210_parts = [
                f"{letter}{homing_velocities[letter]:g}"
                for letter in gcode_letters
                if letter in homing_velocities
            ]
            if m210_parts:
                await self._write_line(f"M210 {' '.join(m210_parts)}")

            await self._write_line(f"G28 {' '.join(gcode_letters)}")

            coords = await self._wait_for_coords(timeout=180.0)
            return coords if coords is not None else {}

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    async def set_digital_out(self, local_idx: int, value: int) -> None:
        async with self._lock:
            await self._write_line(f"M106 P{local_idx} S{value}")
            await self._wait_for_ok(timeout=5.0)

    async def set_analog_out(self, local_idx: int, value: int) -> None:
        async with self._lock:
            await self._write_line(f"M106 P{local_idx} S{value}")
            await self._wait_for_ok(timeout=5.0)

    async def set_io(self, pin: int, io_type: str, value: bool | int) -> bool:
        """Set IO pin to a value.

        io_type: "gpio"  → M42 P<pin> S<0|1>
                 "pwm"   → M42 P<pin> S<0-255> (PWM value)
                 "relay" → M42 P<pin> S<0|1>

        Returns True if successful, False otherwise.
        """
        try:
            async with self._lock:
                if io_type in ("gpio", "relay"):
                    # Digital: convert bool to 0/1, or accept int
                    s_value = int(value) if isinstance(value, bool) else int(value)
                elif io_type == "pwm":
                    # PWM: value should be 0-255
                    s_value = int(value)
                    s_value = max(0, min(255, s_value))
                else:
                    log.warning(
                        "Board %s: unknown io_type '%s'", self._config.board_id, io_type
                    )
                    return False

                cmd = f"M42 P{pin} S{s_value}"
                await self._write_line(cmd)

                ok = await self._wait_for_ok(timeout=5.0)
                if not ok:
                    log.warning(
                        "Board %s: no OK after '%s'", self._config.board_id, cmd
                    )
                    return False

                return True
        except Exception as e:
            log.error("Board %s: set_io error: %s", self._config.board_id, e)
            return False

    async def write_raw(self, line: str) -> None:
        """Send a raw GCode line without waiting for a response."""
        async with self._lock:
            await self._write_line(line)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _write_line(self, line: str) -> None:
        if not self._writer:
            raise RuntimeError(f"Board {self._config.board_id} is not connected")
        self.last_tx = line
        self._writer.write((line + "\n").encode("utf-8"))
        await self._writer.drain()
        print(f"[{self._config.board_id}] TX: {line}")
        log.debug("[%s] TX: %s", self._config.board_id, line)

    async def _read_loop(self) -> None:
        assert self._reader is not None
        while True:
            line = await self._reader.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            self.last_rx = text
            log.debug("[%s] RX: %s", self._config.board_id, text)
            await self._line_queue.put(text)

    async def _wait_for_ok(self, timeout: float) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            try:
                line = await asyncio.wait_for(self._line_queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                return False

            if _OK_RE.match(line.strip()):
                return True
            if _BUSY_RE.search(line):
                continue  # firmware busy — keep waiting

    async def _wait_for_coords(self, timeout: float) -> dict[str, float] | None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return None
            try:
                line = await asyncio.wait_for(self._line_queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                return None

            if _BUSY_RE.search(line):
                continue

            coords = _parse_coords(line)
            if coords:
                return coords
