from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from opensmt.messaging import BusNode, SCPIMessage

from .base import ModuleBase


def _parse_numeric(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class NozzleState:
    name: str
    xr: float
    yr: float
    z_axis: str
    current_z: float


class HeadModule(ModuleBase):
    """HEAD module managing one or more nozzles.

    Supported commands:
    - SET  :HEAD:ABS:<nozzle> <z>
    - SET  :HEAD:REL:<nozzle> <delta_z>
    - ACT  :HEAD:PARK:<nozzle>
    - ACT  :HEAD:PARK
    - QRY  :HEAD:POS:<nozzle>?
    - QRY  :HEAD:NOZZLES?
    """

    def __init__(self, name: str, config: dict[str, Any], node: BusNode) -> None:
        super().__init__(name, config, node)

        self._target = str(config.get("target", "GCODE")).upper()
        self._home_position = float(config.get("home_position", 0.0))
        self._primary_camera = str(config.get("primary_camera", "TOP")).upper()
        self._camera_refs = [str(cam).upper() for cam in config.get("camera_refs", [self._primary_camera])]
        if self._primary_camera not in self._camera_refs:
            self._camera_refs.insert(0, self._primary_camera)

        nozzle_cfg = config.get("nozzles", [])
        self._nozzles: dict[str, NozzleState] = {}
        self._axis_to_nozzle: dict[str, str] = {}
        self._lock = asyncio.Lock()

        for item in nozzle_cfg:
            n_name = str(item.get("name", "")).upper()
            if not n_name:
                continue
            z_axis = str(item.get("z_axis", "")).upper()
            if not z_axis:
                continue
            xr = float(item.get("xr", 0.0))
            yr = float(item.get("yr", 0.0))
            state = NozzleState(
                name=n_name,
                xr=xr,
                yr=yr,
                z_axis=z_axis,
                current_z=self._home_position,
            )
            self._nozzles[n_name] = state
            self._axis_to_nozzle[z_axis] = n_name

        if not self._nozzles:
            # Safe default for development setup.
            defaults = [
                ("N1", "Z1"),
                ("N2", "Z2"),
                ("N3", "Z3"),
                ("N4", "Z4"),
            ]
            for n_name, z_axis in defaults:
                state = NozzleState(
                    name=n_name,
                    xr=0.0,
                    yr=0.0,
                    z_axis=z_axis,
                    current_z=self._home_position,
                )
                self._nozzles[n_name] = state
                self._axis_to_nozzle[z_axis] = n_name

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
        if len(parts) < 2 or parts[0] != self.name:
            return

        target = packet.get("source")

        if len(parts) == 2 and parts[1] == "NOZZLES":
            payload = {
                "primary_camera": self._primary_camera,
                "camera_refs": self._camera_refs,
                "home_position": self._home_position,
                "nozzles": [
                    {
                        "name": n.name,
                        "xr": n.xr,
                        "yr": n.yr,
                        "z_axis": n.z_axis,
                        "z": n.current_z,
                    }
                    for n in self._nozzles.values()
                ],
            }
            await self.node.send_response(msg.command, json.dumps(payload), target=target)
            return

        if len(parts) == 3 and parts[1] == "POS":
            nozzle = self._nozzles.get(parts[2].upper())
            if not nozzle:
                await self.node.send_response(msg.command, "UNKNOWN_NOZZLE", target=target)
                return
            await self.node.send_response(msg.command, nozzle.current_z, target=target)

    async def _handle_set(self, packet: dict[str, Any], msg: SCPIMessage) -> None:
        if packet.get("source") == self.node.name:
            return

        parts = msg.command.strip(":").split(":")
        if len(parts) != 3 or parts[0] != self.name:
            return

        scope = parts[1].upper()
        nozzle_name = parts[2].upper()
        target = packet.get("source")
        nozzle = self._nozzles.get(nozzle_name)
        if not nozzle:
            await self.node.send_response(msg.command, "UNKNOWN_NOZZLE", target=target)
            return

        if scope == "ABS":
            z_target = _parse_numeric(msg.value)
            if z_target is None:
                await self.node.send_response(msg.command, "INVALID_POSITION", target=target)
                return
            await self._move_to(msg.command, nozzle, z_target, target)
            return

        if scope == "REL":
            delta = _parse_numeric(msg.value)
            if delta is None:
                await self.node.send_response(msg.command, "INVALID_POSITION", target=target)
                return
            z_target = nozzle.current_z + delta
            await self._move_to(msg.command, nozzle, z_target, target)

    async def _handle_action(self, packet: dict[str, Any], msg: SCPIMessage) -> None:
        if packet.get("source") == self.node.name:
            return

        parts = msg.command.strip(":").split(":")
        if len(parts) < 2 or parts[0] != self.name:
            return

        target = packet.get("source")
        verb = parts[1].upper()

        if verb != "PARK":
            return

        if len(parts) == 2:
            asyncio.create_task(self._park_all(msg.command, target))
            return

        if len(parts) == 3:
            nozzle = self._nozzles.get(parts[2].upper())
            if not nozzle:
                await self.node.send_response(msg.command, "UNKNOWN_NOZZLE", target=target)
                return
            asyncio.create_task(self._move_to(msg.command, nozzle, self._home_position, target))

    async def _handle_response(self, packet: dict[str, Any], msg: SCPIMessage) -> None:
        parts = msg.command.strip(":").split(":")
        if len(parts) != 3:
            return
        if parts[0] != self._target or parts[1] != "POS":
            return

        axis = parts[2].upper()
        nozzle_name = self._axis_to_nozzle.get(axis)
        if not nozzle_name:
            return

        value = _parse_numeric(msg.value)
        if value is None:
            return

        self._nozzles[nozzle_name].current_z = value

    async def _park_all(self, command: str, target: str | None) -> None:
        await self.node.send_working(command, target=target)
        for nozzle in self._nozzles.values():
            await self.node.send_set(
                f":{self._target}:POS:{nozzle.z_axis}",
                self._home_position,
                target=self._target,
            )
            nozzle.current_z = self._home_position
        await self.node.send_response(command, "DONE", target=target)

    async def _move_to(
        self,
        command: str,
        nozzle: NozzleState,
        z_target: float,
        target: str | None,
    ) -> None:
        async with self._lock:
            # Nozzle travel must not pass above the configured HOME position.
            z_target = min(z_target, self._home_position)

            await self.node.send_working(command, target=target)
            await self.node.send_set(
                f":{self._target}:POS:{nozzle.z_axis}",
                z_target,
                target=self._target,
            )
            nozzle.current_z = z_target
            await self.node.send_response(command, z_target, target=target)
