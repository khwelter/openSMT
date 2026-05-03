from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import serial_asyncio

from opensmt.messaging import BusNode, SCPIMessage

from .base import ModuleBase

DeviceCallback = Callable[[str, str], Awaitable[None] | None]

_COORD_RE = re.compile(r"\b([A-Z]):(-?\d+(?:\.\d+)?)")
_BUSY_RE = re.compile(r"echo:busy:", re.IGNORECASE)
_AXIS_PARSE_RE = re.compile(r"(X|Y|Z[1-4]|R[1-4])")

DEFAULT_AXIS_MAP: dict[str, str] = {
    "X": "X",
    "Y": "Y",
    "Z1": "X",
    "Z2": "Y",
    "Z3": "X",
    "Z4": "Y",
    "R1": "A",
    "R2": "B",
    "R3": "A",
    "R4": "B",
}


def _parse_coords(text: str) -> dict[str, float]:
    part = text.split("Count")[0]
    return {m.group(1): float(m.group(2)) for m in _COORD_RE.finditer(part)}


def _parse_numeric(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_position_list(value: Any, expected_count: int) -> list[float] | None:
    if isinstance(value, (int, float)):
        if expected_count != 1:
            return None
        return [float(value)]

    raw = str(value).replace(",", " ")
    parts = [part for part in raw.split() if part]
    if len(parts) != expected_count:
        return None

    try:
        return [float(part) for part in parts]
    except ValueError:
        return None


def _normalize_rotation(axis: str, position: float) -> float:
    if axis.startswith("R"):
        return position % 360.0
    return position


def _fmt_num(value: float) -> str:
    return f"{value:g}"


@dataclass(slots=True)
class SerialPortConfig:
    name: str
    device: str
    baudrate: int = 115200
    bytesize: int = 8
    parity: str = "N"
    stopbits: int = 1
    xonxoff: bool = False
    rtscts: bool = False
    dsrdtr: bool = False
    axes: set[str] = field(default_factory=set)
    axis_map: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class SerialPortState:
    config: SerialPortConfig
    reader: asyncio.StreamReader | None = None
    writer: asyncio.StreamWriter | None = None
    read_task: asyncio.Task[None] | None = None
    last_rx: str = ""
    last_tx: str = ""
    callbacks: list[DeviceCallback] = field(default_factory=list)
    line_queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue)
    command_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SerialGCodeModule(ModuleBase):
    def __init__(self, name: str, config: dict[str, Any], node: BusNode) -> None:
        super().__init__(name, config, node)
        self._ports: dict[str, SerialPortState] = {}
        self._axis_owner: dict[str, str] = {}
        self._axis_velocity: dict[str, float] = {}
        self._homing_velocity: dict[str, float] = {}
        self._speed_factor = float(config.get("speed_factor", 100.0))

        default_velocity = float(config.get("default_velocity", 25000.0))
        configured_axis_velocity = {
            str(axis).upper(): float(speed)
            for axis, speed in config.get("axis_velocity", {}).items()
        }
        configured_homing_velocity = {
            str(axis).upper(): float(speed)
            for axis, speed in config.get("homing_velocity", {}).items()
        }

        for item in config.get("serial_ports", []):
            user_map = {k.upper(): v.upper() for k, v in item.get("axis_map", {}).items()}
            handled_axes = {str(axis).upper() for axis in item.get("axes", [])}

            port_cfg = SerialPortConfig(
                name=str(item["name"]).upper(),
                device=str(item["device"]),
                baudrate=int(item.get("baudrate", 115200)),
                bytesize=int(item.get("bytesize", 8)),
                parity=str(item.get("parity", "N")),
                stopbits=int(item.get("stopbits", 1)),
                xonxoff=bool(item.get("xonxoff", False)),
                rtscts=bool(item.get("rtscts", False)),
                dsrdtr=bool(item.get("dsrdtr", False)),
                axes=handled_axes,
                axis_map={**DEFAULT_AXIS_MAP, **user_map},
            )
            self._ports[port_cfg.name] = SerialPortState(config=port_cfg)

            for axis in handled_axes:
                if axis in self._axis_owner:
                    raise ValueError(
                        f"Axis {axis} assigned to multiple serial ports: "
                        f"{self._axis_owner[axis]} and {port_cfg.name}"
                    )
                self._axis_owner[axis] = port_cfg.name
                self._axis_velocity[axis] = configured_axis_velocity.get(axis, default_velocity)
                self._homing_velocity[axis] = configured_homing_velocity.get(axis, self._axis_velocity[axis])

        for axis, speed in configured_axis_velocity.items():
            if axis not in self._axis_velocity:
                self._axis_velocity[axis] = speed

        for axis, speed in configured_homing_velocity.items():
            if axis not in self._homing_velocity:
                self._homing_velocity[axis] = speed

        self._port_order: list[str] = list(self._ports.keys())
        self._do_state: dict[int, int] = {}
        self._ao_state: dict[int, int] = {}

    def register_device_callback(self, port_name: str, callback: DeviceCallback) -> None:
        key = port_name.upper()
        if key not in self._ports:
            raise KeyError(f"Unknown port: {port_name}")
        self._ports[key].callbacks.append(callback)

    async def start(self) -> None:
        self.node.on_text(self._log_bus_text)
        self.node.on_binary(self._log_bus_binary)
        self.node.on_query("*", self._handle_query)
        self.node.on_set("*", self._handle_set)
        self.node.on_action("*", self._handle_action)

        for state in self._ports.values():
            await self._connect_port(state)

    async def _log_bus_text(self, packet: dict[str, Any], msg: SCPIMessage) -> None:
        src = packet.get("source", "?")
        dst = packet.get("target") or "*"
        print(f"[BUS {self.name}] {src} -> {dst}: {msg.raw}")

    async def _log_bus_binary(self, packet: dict[str, Any], payload: bytes) -> None:
        src = packet.get("source", "?")
        dst = packet.get("target") or "*"
        topic = packet.get("topic", "BINARY")
        print(f"[BUS {self.name}] {src} -> {dst}: {topic} {payload.hex()}")

    async def stop(self) -> None:
        for state in self._ports.values():
            if state.read_task:
                state.read_task.cancel()
                try:
                    await state.read_task
                except asyncio.CancelledError:
                    pass

            if state.writer:
                state.writer.close()
                await state.writer.wait_closed()

    async def _connect_port(self, state: SerialPortState) -> None:
        cfg = state.config
        reader, writer = await serial_asyncio.open_serial_connection(
            url=cfg.device,
            baudrate=cfg.baudrate,
            bytesize=cfg.bytesize,
            parity=cfg.parity,
            stopbits=cfg.stopbits,
            xonxoff=cfg.xonxoff,
            rtscts=cfg.rtscts,
            dsrdtr=cfg.dsrdtr,
        )
        state.reader = reader
        state.writer = writer
        state.read_task = asyncio.create_task(self._read_loop(state), name=f"serial-rx-{self.name}-{cfg.name}")

    async def _write_line(self, state: SerialPortState, line: str) -> None:
        if not state.writer:
            raise RuntimeError("Serial writer is not connected")
        state.last_tx = line
        state.writer.write((line + "\n").encode("utf-8"))
        await state.writer.drain()

    async def _read_loop(self, state: SerialPortState) -> None:
        assert state.reader is not None
        while True:
            line = await state.reader.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            state.last_rx = text
            iface = state.config.device.rsplit("/", maxsplit=1)[-1] or state.config.name
            print(f"[{iface}] RX: {text} | TX: {state.last_tx}")

            for callback in state.callbacks:
                result = callback(state.config.name, text)
                if asyncio.iscoroutine(result):
                    await result

            await self.node.send_response(f":SERIAL:{state.config.name}:RX", text)
            await state.line_queue.put(text)

    async def _handle_query(self, packet: dict[str, Any], msg: SCPIMessage) -> None:
        if packet.get("source") == self.node.name:
            return

        parts = msg.command.strip(":").split(":")
        target = packet.get("source")

        if len(parts) == 2 and parts[0] == self.name:
            verb = parts[1]
            if verb == "STATUS":
                await self.node.send_response(msg.command, "online", target=target)
            elif verb == "SPEEDFACTOR":
                await self.node.send_response(msg.command, self._speed_factor, target=target)
            elif verb == "LASTRX":
                value = "; ".join(
                    f"{state.config.name}={state.last_rx}"
                    for state in self._ports.values()
                    if state.last_rx
                )
                await self.node.send_response(msg.command, value, target=target)
            return

        if len(parts) == 3 and parts[0] == self.name and parts[1] in ("DIGOUT", "ANOUT"):
            try:
                global_idx = int(parts[2])
            except ValueError:
                return
            if parts[1] == "DIGOUT":
                await self.node.send_response(msg.command, self._do_state.get(global_idx, 0), target=target)
            else:
                await self.node.send_response(msg.command, self._ao_state.get(global_idx, 0), target=target)
            return

        if len(parts) < 3 or parts[0] != "SERIAL":
            return

        port_name = parts[1]
        verb = parts[2]
        state = self._ports.get(port_name)
        if not state:
            return

        if verb == "STATUS":
            connected = 1 if state.writer is not None and not state.writer.is_closing() else 0
            await self.node.send_response(msg.command, connected, target=target)
        elif verb == "LASTRX":
            await self.node.send_response(msg.command, state.last_rx, target=target)

    async def _handle_set(self, packet: dict[str, Any], msg: SCPIMessage) -> None:
        if packet.get("source") == self.node.name:
            return

        parts = msg.command.strip(":").split(":")
        target = packet.get("source")

        if len(parts) == 3 and parts[0] == self.name and parts[1] == "VELO":
            await self._handle_set_velocity(msg.command, parts[2], msg.value, target)
            return

        if len(parts) == 2 and parts[0] == self.name and parts[1] == "SPEEDFACTOR":
            await self._handle_set_speed_factor(msg.command, msg.value, target)
            return

        if len(parts) == 3 and parts[0] == self.name and parts[1] == "POS":
            await self._handle_move(msg.command, parts[2], msg.value, target)
            return

        if len(parts) == 3 and parts[0] == self.name and parts[1] == "DIGOUT":
            await self._handle_set_digout(msg.command, parts[2], msg.value, target)
            return

        if len(parts) == 3 and parts[0] == self.name and parts[1] == "ANOUT":
            await self._handle_set_anout(msg.command, parts[2], msg.value, target)
            return

        if len(parts) < 3 or parts[0] != "SERIAL":
            return

        port_name = parts[1]
        command = parts[2]
        state = self._ports.get(port_name)
        if not state or command != "TX":
            return

        await self.node.send_working(msg.command, target=target)

        if not state.writer or state.writer.is_closing():
            await self.node.send_response(msg.command, "PORT_NOT_CONNECTED", target=target)
            return

        line = str(msg.value)
        await self._write_line(state, line)

        await self.node.send_response(msg.command, line, target=target)

    async def _handle_action(self, packet: dict[str, Any], msg: SCPIMessage) -> None:
        if packet.get("source") == self.node.name:
            return

        parts = msg.command.strip(":").split(":")
        if len(parts) < 3 or parts[0] != self.name:
            return

        verb = parts[1]
        axes_str = "".join(parts[2:])
        target = packet.get("source")

        if verb == "HOME":
            logical_axes = _AXIS_PARSE_RE.findall(axes_str.upper())
            if not logical_axes:
                return

            missing = [axis for axis in logical_axes if axis not in self._axis_owner]
            if missing:
                await self.node.send_response(
                    msg.command,
                    f"AXIS_NOT_CONFIGURED:{','.join(missing)}",
                    target=target,
                )
                return

            by_port = self._group_axes_by_port(logical_axes)
            asyncio.create_task(self._execute_home_multi(msg.command, by_port, target))

    def _group_axes_by_port(self, logical_axes: list[str]) -> dict[str, list[tuple[str, str]]]:
        by_port: dict[str, list[tuple[str, str]]] = {}
        for logical_axis in logical_axes:
            port_name = self._axis_owner[logical_axis]
            state = self._ports[port_name]
            by_port.setdefault(port_name, []).append(
                (logical_axis, state.config.axis_map.get(logical_axis, logical_axis))
            )
        return by_port

    async def _handle_set_velocity(
        self,
        command: str,
        axis: str,
        velocity_value: Any,
        target: str | None,
    ) -> None:
        logical_axis = axis.upper()
        if logical_axis not in self._axis_owner:
            await self.node.send_response(command, f"AXIS_NOT_CONFIGURED:{logical_axis}", target=target)
            return

        velocity = _parse_numeric(velocity_value)
        if velocity is None or velocity <= 0:
            await self.node.send_response(command, "INVALID_VELOCITY", target=target)
            return

        self._axis_velocity[logical_axis] = velocity
        await self.node.send_response(command, velocity, target=target)

    async def _handle_set_speed_factor(
        self,
        command: str,
        speed_factor_value: Any,
        target: str | None,
    ) -> None:
        speed_factor = _parse_numeric(speed_factor_value)
        if speed_factor is None or speed_factor < 0 or speed_factor > 100:
            await self.node.send_response(command, "INVALID_SPEED_FACTOR", target=target)
            return

        self._speed_factor = speed_factor
        await self.node.send_response(command, speed_factor, target=target)

    def _route_io_index(self, global_idx: int) -> tuple[str, int] | None:
        port_slot = global_idx // 16
        local_idx = global_idx % 16
        if port_slot >= len(self._port_order):
            return None
        return self._port_order[port_slot], local_idx

    async def _handle_set_digout(
        self,
        command: str,
        index_str: str,
        value: Any,
        target: str | None,
    ) -> None:
        try:
            global_idx = int(index_str)
        except ValueError:
            await self.node.send_response(command, "INVALID_INDEX", target=target)
            return

        route = self._route_io_index(global_idx)
        if route is None:
            await self.node.send_response(command, f"INDEX_OUT_OF_RANGE:{global_idx}", target=target)
            return

        parsed = _parse_numeric(value)
        if parsed is None:
            await self.node.send_response(command, "INVALID_VALUE", target=target)
            return
        int_val = int(parsed)
        if int_val not in (0, 1):
            await self.node.send_response(command, "INVALID_DIGITAL_VALUE", target=target)
            return

        port_name, local_idx = route
        state = self._ports[port_name]
        asyncio.create_task(
            self._execute_digout(command, state, global_idx, local_idx, int_val, target)
        )

    async def _handle_set_anout(
        self,
        command: str,
        index_str: str,
        value: Any,
        target: str | None,
    ) -> None:
        try:
            global_idx = int(index_str)
        except ValueError:
            await self.node.send_response(command, "INVALID_INDEX", target=target)
            return

        route = self._route_io_index(global_idx)
        if route is None:
            await self.node.send_response(command, f"INDEX_OUT_OF_RANGE:{global_idx}", target=target)
            return

        parsed = _parse_numeric(value)
        if parsed is None:
            await self.node.send_response(command, "INVALID_VALUE", target=target)
            return
        int_val = int(parsed)
        if not (0 <= int_val <= 65535):
            await self.node.send_response(command, "INVALID_ANALOG_VALUE", target=target)
            return

        port_name, local_idx = route
        state = self._ports[port_name]
        asyncio.create_task(
            self._execute_anout(command, state, global_idx, local_idx, int_val, target)
        )

    async def _execute_digout(
        self,
        command: str,
        state: SerialPortState,
        global_idx: int,
        local_idx: int,
        value: int,
        target: str | None,
    ) -> None:
        if not state.writer or state.writer.is_closing():
            await self.node.send_response(command, "PORT_NOT_CONNECTED", target=target)
            return

        async with state.command_lock:
            gcode = f"M106 P{local_idx} S{value}"
            await self.node.send_working(command, target=target)
            await self._write_line(state, gcode)

            ok = await self._wait_for_ok(state, timeout=5.0)
            if not ok:
                await self.node.send_response(command, "DIGOUT_NO_OK", target=target)
                return

        self._do_state[global_idx] = value
        await self.node.send_response(command, value, target=target)

    async def _execute_anout(
        self,
        command: str,
        state: SerialPortState,
        global_idx: int,
        local_idx: int,
        value: int,
        target: str | None,
    ) -> None:
        if not state.writer or state.writer.is_closing():
            await self.node.send_response(command, "PORT_NOT_CONNECTED", target=target)
            return

        async with state.command_lock:
            gcode = f"M106 P{local_idx} S{value}"
            await self.node.send_working(command, target=target)
            await self._write_line(state, gcode)

            ok = await self._wait_for_ok(state, timeout=5.0)
            if not ok:
                await self.node.send_response(command, "ANOUT_NO_OK", target=target)
                return

        self._ao_state[global_idx] = value
        await self.node.send_response(command, value, target=target)

    async def _handle_move(
        self,
        command: str,
        axis: str,
        position_value: Any,
        target: str | None,
    ) -> None:
        pair_axes = {
            "XY": ["X", "Y"],
            "Z1R1": ["Z1", "R1"],
            "Z2R2": ["Z2", "R2"],
            "Z3R3": ["Z3", "R3"],
            "Z4R4": ["Z4", "R4"],
            "Z1": ["Z1"],
            "Z2": ["Z2"],
            "Z3": ["Z3"],
            "Z4": ["Z4"],
            "R1": ["R1"],
            "R2": ["R2"],
            "R3": ["R3"],
            "R4": ["R4"],
        }

        logical_axes = pair_axes.get(axis.upper(), [axis.upper()])
        positions = _parse_position_list(position_value, len(logical_axes))
        if positions is None:
            await self.node.send_response(command, "INVALID_POSITION", target=target)
            return

        missing_axes = [logical_axis for logical_axis in logical_axes if logical_axis not in self._axis_owner]
        if missing_axes:
            await self.node.send_response(
                command,
                f"AXIS_NOT_CONFIGURED:{','.join(missing_axes)}",
                target=target,
            )
            return

        if len({self._axis_owner[logical_axis] for logical_axis in logical_axes}) != 1:
            await self.node.send_response(command, "AXES_ON_DIFFERENT_PORTS", target=target)
            return

        normalized_positions = [
            _normalize_rotation(logical_axis, position)
            for logical_axis, position in zip(logical_axes, positions, strict=True)
        ]

        port_name = self._axis_owner[logical_axes[0]]
        state = self._ports[port_name]
        axis_pairs = [
            (logical_axis, state.config.axis_map.get(logical_axis, logical_axis), normalized_position)
            for logical_axis, normalized_position in zip(logical_axes, normalized_positions, strict=True)
        ]
        base_velocity = min(self._axis_velocity.get(logical_axis, 25000.0) for logical_axis in logical_axes)
        effective_velocity = base_velocity * (self._speed_factor / 100.0)

        asyncio.create_task(
            self._execute_move(
                command,
                state,
                axis_pairs,
                effective_velocity,
                target,
            )
        )

    async def _execute_home_multi(
        self,
        command: str,
        by_port: dict[str, list[tuple[str, str]]],
        target: str | None,
    ) -> None:
        tasks = [
            asyncio.create_task(self._execute_home(command, port_name, axis_pairs, target))
            for port_name, axis_pairs in by_port.items()
        ]
        if tasks:
            await asyncio.gather(*tasks)

    async def _execute_home(
        self,
        command: str,
        port_name: str,
        axis_pairs: list[tuple[str, str]],
        target: str | None,
    ) -> None:
        state = self._ports.get(port_name)
        if not state or not state.writer or state.writer.is_closing():
            await self.node.send_response(command, "PORT_NOT_CONNECTED", target=target)
            return

        async with state.command_lock:
            # Build M210 command with homing velocities for the axes being homed
            m210_parts = []
            for logical, gcode in axis_pairs:
                homing_vel = self._homing_velocity.get(logical)
                if homing_vel is not None:
                    m210_parts.append(f"{gcode}{_fmt_num(homing_vel)}")
            
            await self.node.send_working(command, target=target)
            
            # Send M210 to set homing velocities (if any axes have defined homing velocities)
            if m210_parts:
                m210_gcode = f"M210 {' '.join(m210_parts)}"
                await self._write_line(state, m210_gcode)
            
            # Send G28 to perform homing
            gcode_axes = " ".join(gcode for _, gcode in axis_pairs)
            gcode = f"G28 {gcode_axes}"
            await self._write_line(state, gcode)

            coords = await self._wait_for_coords(state, command, target)
            if coords is None:
                await self.node.send_response(command, "TIMEOUT", target=target)
                return

            for logical, gcode_axis in axis_pairs:
                if gcode_axis in coords:
                    await self.node.send_response(
                        f":{self.name}:POS:{logical}",
                        coords[gcode_axis],
                        target=target,
                    )

    async def _execute_move(
        self,
        command: str,
        state: SerialPortState,
        axis_triplets: list[tuple[str, str, float]],
        velocity: float,
        target: str | None,
    ) -> None:
        if not state.writer or state.writer.is_closing():
            await self.node.send_response(command, "PORT_NOT_CONNECTED", target=target)
            return

        async with state.command_lock:
            axis_terms = " ".join(f"{mapped_axis}{_fmt_num(position)}" for _, mapped_axis, position in axis_triplets)
            move_line = f"G0 {axis_terms} F{_fmt_num(velocity)}"
            await self.node.send_working(command, target=target)

            await self._write_line(state, move_line)

            ok = await self._wait_for_ok(state, timeout=10.0)
            if not ok:
                await self.node.send_response(command, "MOVE_NO_OK", target=target)
                return

            await self._write_line(state, "M400")

            done = await self._wait_for_ok(state, timeout=120.0, working_command=command, target=target)
            if not done:
                await self.node.send_response(command, "MOVE_TIMEOUT", target=target)
                return

            for logical_axis, _, position in axis_triplets:
                await self.node.send_response(f":{self.name}:POS:{logical_axis}", position, target=target)

    async def _wait_for_ok(
        self,
        state: SerialPortState,
        timeout: float,
        working_command: str | None = None,
        target: str | None = None,
    ) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False

            try:
                line = await asyncio.wait_for(state.line_queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                return False

            lowered = line.strip().lower()
            if lowered == "ok":
                return True
            if _BUSY_RE.search(line):
                if working_command:
                    await self.node.send_working(working_command, target=target)
                continue

    async def _wait_for_coords(
        self,
        state: SerialPortState,
        command: str,
        target: str | None,
        timeout: float = 120.0,
    ) -> dict[str, float] | None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return None

            try:
                line = await asyncio.wait_for(state.line_queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                return None

            if _BUSY_RE.search(line):
                await self.node.send_working(command, target=target)
                continue

            coords = _parse_coords(line)
            if coords:
                return coords
