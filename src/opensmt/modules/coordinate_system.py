from __future__ import annotations

import asyncio
from typing import Any

from opensmt.messaging import BusNode, SCPIMessage

from .base import ModuleBase


DEFAULT_AXES = ["X", "Y", "Z1", "Z2", "Z3", "Z4", "R1", "R2", "R3", "R4"]
HOME_GROUP_AXES = {
    "XY": ["X", "Y"],
    "Z1Z2": ["Z1", "Z2"],
    "Z3Z4": ["Z3", "Z4"],
}


class CoordinateSystemModule(ModuleBase):
    def __init__(self, name: str, config: dict[str, Any], node: BusNode) -> None:
        super().__init__(name, config, node)
        self._axes = [str(axis).upper() for axis in config.get("axes", DEFAULT_AXES)]
        self._positions: dict[str, float | None] = {axis: None for axis in self._axes}
        self._position_update_count: dict[str, int] = {axis: 0 for axis in self._axes}
        self._default_target = str(config.get("default_target", "GCODE")).upper()
        self._home_target = str(config.get("home_target", self._default_target)).upper()
        self._home_groups = [str(group).upper() for group in config.get("home_groups", ["XY", "Z1Z2", "Z3Z4"])]
        self._home_timeout = float(config.get("home_timeout", 180.0))
        self._axis_targets = {
            str(axis).upper(): str(target).upper()
            for axis, target in config.get("axis_targets", {}).items()
        }
        self._home_lock = asyncio.Lock()

    async def start(self) -> None:
        self.node.on_query("*", self._handle_query)
        self.node.on_set("*", self._handle_set)
        self.node.on_action("*", self._handle_action)
        self.node.on_response("*", self._handle_response)

    async def stop(self) -> None:
        return None

    async def _handle_query(self, packet: dict[str, Any], msg: SCPIMessage) -> None:
        if packet.get("source") == self.node.name:
            return

        parts = msg.command.strip(":").split(":")
        if len(parts) != 3 or parts[0] != self.name:
            return

        scope = parts[1]
        axis = parts[2].upper()
        if scope not in {"ABS", "POS"} or axis not in self._positions:
            return

        value = self._positions[axis]
        if value is None:
            await self.node.send_response(msg.command, "UNKNOWN", target=packet.get("source"))
            return

        await self.node.send_response(msg.command, value, target=packet.get("source"))

    async def _handle_set(self, packet: dict[str, Any], msg: SCPIMessage) -> None:
        if packet.get("source") == self.node.name:
            return

        parts = msg.command.strip(":").split(":")
        if len(parts) != 3 or parts[0] != self.name:
            return

        scope = parts[1]
        axis = parts[2].upper()
        target = packet.get("source")
        if axis not in self._positions:
            return

        if scope == "ABS":
            position = self._parse_numeric(msg.value)
            if position is None:
                await self.node.send_response(msg.command, "INVALID_POSITION", target=target)
                return
            await self._dispatch_absolute(axis, position)
            await self.node.send_response(msg.command, position, target=target)
            return

        if scope == "REL":
            delta = self._parse_numeric(msg.value)
            if delta is None:
                await self.node.send_response(msg.command, "INVALID_POSITION", target=target)
                return

            current = self._positions[axis]
            if current is None:
                await self.node.send_response(msg.command, "UNKNOWN", target=target)
                return

            position = current + delta
            await self._dispatch_absolute(axis, position)
            await self.node.send_response(msg.command, position, target=target)

    async def _handle_response(self, packet: dict[str, Any], msg: SCPIMessage) -> None:
        parts = msg.command.strip(":").split(":")
        if len(parts) != 3 or parts[1] != "POS":
            return

        axis = parts[2].upper()
        if axis not in self._positions:
            return

        value = self._parse_numeric(msg.value)
        if value is None:
            return

        self._positions[axis] = value
        self._position_update_count[axis] += 1

    async def _handle_action(self, packet: dict[str, Any], msg: SCPIMessage) -> None:
        if packet.get("source") == self.node.name:
            return

        parts = msg.command.strip(":").split(":")
        if len(parts) != 2 or parts[0] != self.name or parts[1] != "HOME":
            return

        target = packet.get("source")
        asyncio.create_task(self._execute_home(msg.command, target))

    async def _execute_home(self, command: str, target: str | None) -> None:
        async with self._home_lock:
            requested_axes = [
                axis
                for group in self._home_groups
                for axis in HOME_GROUP_AXES.get(group, [])
                if axis in self._positions
            ]
            snapshot = {axis: self._position_update_count[axis] for axis in requested_axes}

            await self.node.send_working(command, target=target)

            for group in self._home_groups:
                await self.node.send_action(
                    f":{self._home_target}:HOME:{group}",
                    target=self._home_target,
                )

            finished = await self._wait_for_axes_update(snapshot, timeout=self._home_timeout)
            if finished:
                await self.node.send_response(command, "DONE", target=target)
                return

            missing = [
                axis
                for axis, count in snapshot.items()
                if self._position_update_count[axis] <= count
            ]
            await self.node.send_response(command, f"TIMEOUT:{','.join(missing)}", target=target)

    async def _wait_for_axes_update(self, snapshot: dict[str, int], timeout: float) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        while loop.time() < deadline:
            if all(self._position_update_count[axis] > count for axis, count in snapshot.items()):
                return True
            await asyncio.sleep(0.1)

        return False

    async def _dispatch_absolute(self, axis: str, position: float) -> None:
        target_module = self._axis_targets.get(axis, self._default_target)
        await self.node.send_set(f":{target_module}:POS:{axis}", position, target=target_module)

    @staticmethod
    def _parse_numeric(value: Any) -> float | None:
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return None