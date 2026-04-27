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
        self._park_position = self._parse_xy_config(config.get("park_position", {"x": 0.0, "y": 0.0}))
        self._dispose_position = self._parse_xy_config(config.get("dispose_position", {"x": 0.0, "y": 0.0}))
        self._homing_fiducial_main_position = self._parse_xy_config(
            config.get("homing_fiducial_main_position", {"x": 280.0, "y": 180.0})
        )
        self._secondary_fiducial_position = self._parse_xy_config(
            config.get("secondary_fiducial_position", {"x": 300.0, "y": 180.0})
        )
        self._nozzle_change_position = self._parse_xy_config(
            config.get("nozzle_change_position", {"x": 250.0, "y": 50.0})
        )
        self._calibration_spot_position = self._parse_xy_config(
            config.get(
                "calibration_spot_position",
                {
                    "x": self._homing_fiducial_main_position[0] + 100.0,
                    "y": self._homing_fiducial_main_position[1],
                },
            )
        )
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
            if len(parts) == 2 and parts[0] == self.name:
                if parts[1] == "PARK":
                    await self.node.send_response(msg.command, f"{self._park_position[0]:g} {self._park_position[1]:g}", target=packet.get("source"))
                elif parts[1] == "DISPOSE":
                    await self.node.send_response(msg.command, f"{self._dispose_position[0]:g} {self._dispose_position[1]:g}", target=packet.get("source"))
                elif parts[1] in {"HOMINGFIDUCIALMAIN", "FIDUCIALMAIN"}:
                    await self.node.send_response(
                        msg.command,
                        f"{self._homing_fiducial_main_position[0]:g} {self._homing_fiducial_main_position[1]:g}",
                        target=packet.get("source"),
                    )
                elif parts[1] in {"SECONDARYFIDUCIAL", "FIDUCIALSECONDARY"}:
                    await self.node.send_response(
                        msg.command,
                        f"{self._secondary_fiducial_position[0]:g} {self._secondary_fiducial_position[1]:g}",
                        target=packet.get("source"),
                    )
                elif parts[1] == "NOZZLECHANGE":
                    await self.node.send_response(
                        msg.command,
                        f"{self._nozzle_change_position[0]:g} {self._nozzle_change_position[1]:g}",
                        target=packet.get("source"),
                    )
                elif parts[1] in {"CALIBRATIONSPOT", "CALSPOT"}:
                    await self.node.send_response(
                        msg.command,
                        f"{self._calibration_spot_position[0]:g} {self._calibration_spot_position[1]:g}",
                        target=packet.get("source"),
                    )
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
        target = packet.get("source")

        # :COORD:ABS:XY <x> <y>  — simultaneous XY move
        if len(parts) == 3 and parts[0] == self.name and parts[1] == "ABS" and parts[2].upper() == "XY":
            raw = str(msg.value).replace(",", " ").split()
            if len(raw) != 2:
                await self.node.send_response(msg.command, "INVALID_POSITION", target=target)
                return
            x_pos = self._parse_numeric(raw[0])
            y_pos = self._parse_numeric(raw[1])
            if x_pos is None or y_pos is None:
                await self.node.send_response(msg.command, "INVALID_POSITION", target=target)
                return
            await self._dispatch_absolute_xy(x_pos, y_pos)
            await self.node.send_response(msg.command, f"{x_pos:g} {y_pos:g}", target=target)
            return

        if len(parts) != 3 or parts[0] != self.name:
            return

        scope = parts[1]
        axis = parts[2].upper()
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
        if len(parts) < 2 or parts[0] != self.name:
            return

        target = packet.get("source")
        
        if len(parts) == 2:
            # Handle single-word actions like :COORD:HOME, :COORD:PARK, etc.
            if parts[1] == "HOME":
                asyncio.create_task(self._execute_home(msg.command, target))
            elif parts[1] == "PARK":
                asyncio.create_task(self._execute_named_position(msg.command, target, self._park_position))
            elif parts[1] == "DISPOSE":
                asyncio.create_task(self._execute_named_position(msg.command, target, self._dispose_position))
            elif parts[1] in {"HOMINGFIDUCIALMAIN", "FIDUCIALMAIN"}:
                asyncio.create_task(
                    self._execute_named_position(msg.command, target, self._homing_fiducial_main_position)
                )
            elif parts[1] in {"SECONDARYFIDUCIAL", "FIDUCIALSECONDARY"}:
                asyncio.create_task(
                    self._execute_named_position(msg.command, target, self._secondary_fiducial_position)
                )
            elif parts[1] == "NOZZLECHANGE":
                asyncio.create_task(
                    self._execute_named_position(msg.command, target, self._nozzle_change_position)
                )
            elif parts[1] in {"CALIBRATIONSPOT", "CALSPOT"}:
                asyncio.create_task(
                    self._execute_named_position(msg.command, target, self._calibration_spot_position)
                )
        elif len(parts) == 3:
            # Handle group-specific actions like :COORD:HOME:XY
            if parts[1] == "HOME":
                group = parts[2].upper()
                asyncio.create_task(self._execute_home_group(msg.command, target, group))

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

    async def _execute_home_group(self, command: str, target: str | None, group: str) -> None:
        async with self._home_lock:
            axes = HOME_GROUP_AXES.get(group, [])
            requested_axes = [axis for axis in axes if axis in self._positions]
            
            if not requested_axes:
                await self.node.send_response(command, f"UNKNOWN_GROUP:{group}", target=target)
                return
            
            snapshot = {axis: self._position_update_count[axis] for axis in requested_axes}

            await self.node.send_working(command, target=target)
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

    async def _execute_named_position(
        self,
        command: str,
        target: str | None,
        xy: tuple[float, float],
    ) -> None:
        x_pos, y_pos = xy
        await self.node.send_working(command, target=target)
        await self._dispatch_absolute_xy(x_pos, y_pos)
        await self.node.send_response(command, f"DONE {x_pos:g} {y_pos:g}", target=target)

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

    async def _dispatch_absolute_xy(self, x_pos: float, y_pos: float) -> None:
        """Send X and Y to the motion controller as a single simultaneous move."""
        x_target = self._axis_targets.get("X", self._default_target)
        y_target = self._axis_targets.get("Y", self._default_target)
        if x_target == y_target:
            # Same port — send as one combined G0 X... Y... command
            await self.node.send_set(f":{x_target}:POS:XY", f"{x_pos:g} {y_pos:g}", target=x_target)
        else:
            # Different ports — dispatch individually
            await self._dispatch_absolute("X", x_pos)
            await self._dispatch_absolute("Y", y_pos)

    @staticmethod
    def _parse_numeric(value: Any) -> float | None:
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return None

    @classmethod
    def _parse_xy_config(cls, value: Any) -> tuple[float, float]:
        if isinstance(value, dict):
            x_val = cls._parse_numeric(value.get("x"))
            y_val = cls._parse_numeric(value.get("y"))
            if x_val is not None and y_val is not None:
                return x_val, y_val
        return 0.0, 0.0