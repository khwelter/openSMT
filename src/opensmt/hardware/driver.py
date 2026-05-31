from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from opensmt.hardware.board import BoardConfig, SerialBoard
from opensmt.store.location_store import LocationStore
from opensmt.store.nozzle_config import NozzleConfig
from opensmt.store.position_store import PositionStore

log = logging.getLogger(__name__)


@dataclass(slots=True)
class AxisConfig:
    """Maps a logical axis (e.g. 'Z1') to a board and GCode letter."""

    axis: str
    board_id: str
    gcode_letter: str
    velocity: float          # mm/min at 100 % speed
    homing_velocity: float   # mm/min during homing


class HardwareDriver:
    """Central hardware driver.

    All motion and IO commands flow through here. The driver translates
    logical axis names (X, Y, Z1 …) to board + GCode letter using an axis
    table loaded from config. After every completed move it updates the
    PositionStore so all subscribers (e.g. the Dashboard) are notified.

    Config structure expected under the ``driver`` key::

        {
          "speed_factor": 100.0,
          "default_velocity": 25000.0,
          "default_homing_velocity": 5000.0,
          "axes": [
            {"axis": "X",  "board": "XY", "gcode_letter": "X"},
            {"axis": "Z1", "board": "AB", "gcode_letter": "X"},
            ...
          ],
          "axis_velocity":  {"X": 25000.0, "Z1": 5000.0, ...},
          "homing_velocity": {"X": 5000.0,  "Z1": 2500.0, ...},
          "home_groups": {
            "XY":   ["X", "Y"],
            "Z1Z2": ["Z1", "Z2"],
            "Z3Z4": ["Z3", "Z4"]
          }
        }
    """

    def __init__(
        self,
        boards: dict[str, SerialBoard],
        position_store: PositionStore,
        location_store: LocationStore,
        config: dict[str, Any],
    ) -> None:
        self._boards = boards
        self._position_store = position_store
        self._location_store = location_store
        self._speed_factor = float(config.get("speed_factor", 100.0))

        default_vel = float(config.get("default_velocity", 25000.0))
        default_homing_vel = float(config.get("default_homing_velocity", 5000.0))
        axis_velocities = {
            str(k).upper(): float(v) for k, v in config.get("axis_velocity", {}).items()
        }
        homing_velocities = {
            str(k).upper(): float(v) for k, v in config.get("homing_velocity", {}).items()
        }

        self._axes: dict[str, AxisConfig] = {}
        for entry in config.get("axes", []):
            axis = str(entry["axis"]).upper()
            board_id = str(entry["board"]).upper()
            gcode_letter = str(entry["gcode_letter"]).upper()
            vel = axis_velocities.get(axis, default_vel)
            homing_vel = homing_velocities.get(axis, default_homing_vel)
            self._axes[axis] = AxisConfig(
                axis=axis,
                board_id=board_id,
                gcode_letter=gcode_letter,
                velocity=vel,
                homing_velocity=homing_vel,
            )

        self._home_groups: dict[str, list[str]] = {
            str(name).upper(): [str(a).upper() for a in axes]
            for name, axes in config.get("home_groups", {}).items()
        }
        self._homed_axes: set[str] = set()
        self._xy_slack_comp_mm: float = float(config.get("xy_slack_compensation_mm", 1.0))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        for board in self._boards.values():
            await board.start()

    async def stop(self) -> None:
        for board in self._boards.values():
            await board.stop()

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    async def move_axis(self, axis: str, position: float) -> None:
        """Move a single logical axis to an absolute position (mm)."""
        axis = axis.upper()
        if axis in {"X", "Y"} and axis not in self._homed_axes:
            raise RuntimeError(f"Axis {axis} must be homed before movement")
        cfg = self._axes.get(axis)
        if cfg is None:
            raise ValueError(f"Unknown axis: {axis}")
        board = self._boards.get(cfg.board_id)
        if board is None:
            raise ValueError(f"Board not found: {cfg.board_id}")

        velocity = cfg.velocity * self._speed_factor / 100.0
        await board.move([(cfg.gcode_letter, position)], velocity)
        await self._position_store.update(axis, position)

    async def move_axes(self, moves: dict[str, float], *, apply_xy_slack_compensation: bool = True) -> None:
        """Move multiple logical axes.

        Axes on the same board are moved simultaneously (single G0 command).
        Groups on different boards execute in parallel.
        """
        if not moves:
            return
        axes = {k.upper(): v for k, v in moves.items()}
        for axis in axes:
            if axis in {"X", "Y"} and axis not in self._homed_axes:
                raise RuntimeError(f"Axis {axis} must be homed before movement")

        async def _execute_absolute_axes(target_axes: dict[str, float]) -> None:
            # Group by board
            by_board: dict[str, list[tuple[str, str, float]]] = {}
            for axis, position in target_axes.items():
                cfg = self._axes.get(axis)
                if cfg is None:
                    raise ValueError(f"Unknown axis: {axis}")
                by_board.setdefault(cfg.board_id, []).append(
                    (axis, cfg.gcode_letter, position)
                )

            # Execute per-board groups in parallel
            async def _run_board(board_id: str, triplets: list[tuple[str, str, float]]) -> None:
                board = self._boards.get(board_id)
                if board is None:
                    raise ValueError(f"Board not found: {board_id}")
                velocity = (
                    min(self._axes[ax].velocity for ax, _, _ in triplets)
                    * self._speed_factor / 100.0
                )
                axis_moves = [(letter, pos) for _, letter, pos in triplets]
                await board.move(axis_moves, velocity)

            tasks = [
                asyncio.create_task(_run_board(board_id, triplets))
                for board_id, triplets in by_board.items()
            ]
            if tasks:
                await asyncio.gather(*tasks)

            # Update PositionStore (all boards confirmed)
            for axis, position in target_axes.items():
                await self._position_store.update(axis, position)

        # XY slack compensation: when an XY axis moves in negative direction,
        # first overshoot by -1 mm on that axis, then settle at final target.
        only_xy_move = all(axis in {"X", "Y"} for axis in axes)
        if apply_xy_slack_compensation and only_xy_move and self._xy_slack_comp_mm > 0.0:
            pre_target = dict(axes)
            needs_compensation = False
            for axis in ("X", "Y"):
                if axis not in axes:
                    continue
                current = self._position_store.get(axis)
                if current is None:
                    continue
                target = float(axes[axis])
                if target < float(current):
                    pre_target[axis] = target - self._xy_slack_comp_mm
                    needs_compensation = True

            if needs_compensation:
                await _execute_absolute_axes(pre_target)

        await _execute_absolute_axes(axes)

    async def jog_xy(self, dx: float, dy: float) -> None:
        """Move X and Y by relative amounts from their current positions."""
        current_x = self._position_store.get("X") or 0.0
        current_y = self._position_store.get("Y") or 0.0
        await self.move_axes(
            {"X": current_x + dx, "Y": current_y + dy},
            apply_xy_slack_compensation=False,
        )

    async def home_group(self, group: str) -> None:
        """Home a named group of axes (e.g. 'XY', 'Z1Z2', 'Z3Z4')."""
        group = group.upper()
        axes = self._home_groups.get(group)
        if not axes:
            raise ValueError(f"Unknown home group: {group}")

        await self.home_axes(axes)

    async def home_axes(self, axes: list[str]) -> None:
        """Home a list of logical axes.

        Axes are grouped by board so each board receives one homing command
        for its subset of letters.
        """
        if not axes:
            return
        normalized_axes = [str(axis).upper() for axis in axes]

        # Group by board
        by_board: dict[str, list[tuple[str, str]]] = {}
        homing_vels_by_board: dict[str, dict[str, float]] = {}
        for axis in normalized_axes:
            cfg = self._axes.get(axis)
            if cfg is None:
                log.warning("home_axes: axis %s not in axis table, skipping", axis)
                continue
            by_board.setdefault(cfg.board_id, []).append((axis, cfg.gcode_letter))
            homing_vels_by_board.setdefault(cfg.board_id, {})[
                cfg.gcode_letter
            ] = cfg.homing_velocity

        for board_id, axis_pairs in by_board.items():
            board = self._boards.get(board_id)
            if board is None:
                raise ValueError(f"Board not found: {board_id}")
            gcode_letters = [letter for _, letter in axis_pairs]
            coords = await board.home(gcode_letters, homing_vels_by_board[board_id])
            if not coords:
                # Some firmwares respond to G28 without an inline coordinate report.
                # Query the live machine position explicitly as a fallback.
                queried = await board.query_position(timeout=3.0)
                coords = queried if isinstance(queried, dict) else {}
            for axis, gcode_letter in axis_pairs:
                if gcode_letter in coords:
                    await self._position_store.update(axis, coords[gcode_letter])
                    self._homed_axes.add(axis)
                else:
                    # Keep system operable even if firmware does not report coordinates.
                    # Default to 0.0 for newly homed axes when no explicit value is available.
                    current = self._position_store.get(axis)
                    await self._position_store.update(axis, 0.0 if current is None else current)
                    self._homed_axes.add(axis)
                    log.warning(
                        "home_axes: board %s did not report position for %s (%s), marked homed with fallback",
                        board_id,
                        axis,
                        gcode_letter,
                    )

    async def home_all(self) -> None:
        """Home all configured axis groups with Z-axes before XY."""
        groups = list(self._home_groups.keys())

        def _priority(group: str) -> tuple[int, str]:
            axes = self._home_groups.get(group, [])
            has_z = any(axis.startswith("Z") for axis in axes)
            has_xy = any(axis in {"X", "Y"} for axis in axes)
            if has_z:
                return (0, group)
            if has_xy:
                return (2, group)
            return (1, group)

        for group in sorted(groups, key=_priority):
            await self.home_group(group)

    async def home_xy_with_z_prehome(self) -> None:
        """Home Z groups first, then XY groups.

        This is used by the UI's "Home XY" action for machines that require
        Z clearance before XY homing.
        """
        z_groups: list[str] = []
        xy_groups: list[str] = []
        for group, axes in self._home_groups.items():
            has_z = any(str(axis).upper().startswith("Z") for axis in axes)
            has_xy = any(str(axis).upper() in {"X", "Y"} for axis in axes)
            if has_z:
                z_groups.append(group)
            if has_xy:
                xy_groups.append(group)

        seen: set[str] = set()
        for group in sorted(z_groups):
            if group in seen:
                continue
            seen.add(group)
            await self.home_group(group)

        for group in sorted(xy_groups):
            if group in seen:
                continue
            seen.add(group)
            await self.home_group(group)

    async def move_to_location(self, name: str) -> None:
        """Move to a named location from the LocationStore."""
        location = self._location_store.get(name)
        if location is None:
            raise ValueError(f"Unknown location: {name!r}")
        await self.move_axes(location)

    async def jog_nozzle_to_camera_position(
        self,
        nozzle_config: NozzleConfig,
        current_camera_position: tuple[float, float],
        velocity: float | None = None,
    ) -> None:
        """Move nozzle to align with camera's current position.

        Since: nozzle_position = camera_position + offset
        To achieve nozzle at (cx, cy), move machine to (cx - offset_x, cy - offset_y)

        Args:
            nozzle_config: NozzleConfig with offset_x, offset_y
            current_camera_position: (x, y) where camera currently is
            velocity: Optional override velocity (mm/min)
        """
        cam_x, cam_y = current_camera_position
        machine_target_x = cam_x - nozzle_config.offset_x
        machine_target_y = cam_y - nozzle_config.offset_y

        # If velocity override provided, temporarily adjust speed factor
        old_speed_factor = self._speed_factor
        try:
            if velocity is not None:
                # Scale the velocity through speed factor
                self._speed_factor = (velocity / (self._axes["X"].velocity)) * 100.0
            await self.move_axes({"X": machine_target_x, "Y": machine_target_y})
        finally:
            self._speed_factor = old_speed_factor

    async def jog_camera_to_nozzle_position(
        self,
        nozzle_config: NozzleConfig,
        current_nozzle_position: tuple[float, float],
        velocity: float | None = None,
    ) -> None:
        """Move camera (machine) to align with nozzle's current position.

        Since: nozzle_position = camera_position + offset
        The nozzle is currently at (nx, ny).
        To move camera to the nozzle, move machine to (nx, ny).

        Args:
            nozzle_config: NozzleConfig with offset_x, offset_y
            current_nozzle_position: (x, y) absolute position of nozzle
            velocity: Optional override velocity (mm/min)
        """
        nozzle_x, nozzle_y = current_nozzle_position

        old_speed_factor = self._speed_factor
        try:
            if velocity is not None:
                self._speed_factor = (velocity / (self._axes["X"].velocity)) * 100.0
            await self.move_axes({"X": nozzle_x, "Y": nozzle_y})
        finally:
            self._speed_factor = old_speed_factor

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    async def set_digital_out(self, board_id: str, local_idx: int, value: int) -> None:
        board = self._boards.get(board_id.upper())
        if board is None:
            raise ValueError(f"Board not found: {board_id}")
        await board.set_digital_out(local_idx, value)

    async def set_analog_out(self, board_id: str, local_idx: int, value: int) -> None:
        board = self._boards.get(board_id.upper())
        if board is None:
            raise ValueError(f"Board not found: {board_id}")
        await board.set_analog_out(local_idx, value)

    async def set_nozzle_valve(
        self, nozzle_config: NozzleConfig, valve_type: str, on: bool
    ) -> bool:
        """Activate/deactivate a valve on a nozzle.

        Args:
            nozzle_config: NozzleConfig with valve configuration
            valve_type: "vacuum" or "air"
            on: True to activate, False to deactivate

        Returns:
            True if successful, False if valve not configured or command failed
        """
        if valve_type == "vacuum":
            valve_cfg = nozzle_config.vacuum_valve
        elif valve_type == "air":
            valve_cfg = nozzle_config.air_valve
        else:
            log.warning("Unknown valve type: %s", valve_type)
            return False

        if valve_cfg is None:
            log.warning("Nozzle %s does not have %s valve", nozzle_config.name, valve_type)
            return False

        board = self._boards.get(valve_cfg.board.upper())
        if board is None:
            log.error("Board %s not found for %s valve on nozzle %s",
                      valve_cfg.board, valve_type, nozzle_config.name)
            return False

        return await board.set_io(valve_cfg.pin, valve_cfg.io_type, on)

    # ------------------------------------------------------------------
    # Speed control
    # ------------------------------------------------------------------

    @property
    def speed_factor(self) -> float:
        return self._speed_factor

    async def set_speed_factor(self, factor: float) -> None:
        if not 0.0 <= factor <= 100.0:
            raise ValueError("Speed factor must be between 0 and 100")
        self._speed_factor = factor

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def axes(self) -> dict[str, AxisConfig]:
        return dict(self._axes)

    async def query_xy_position_m114(self) -> dict[str, float] | None:
        """Read live XY position directly from firmware via M114."""
        x_cfg = self._axes.get("X")
        y_cfg = self._axes.get("Y")
        if x_cfg is None or y_cfg is None:
            return None
        if x_cfg.board_id != y_cfg.board_id:
            return None

        board = self._boards.get(x_cfg.board_id)
        if board is None:
            return None

        coords = await board.query_position(timeout=2.0)
        if not coords:
            return None

        x_raw = coords.get(x_cfg.gcode_letter)
        y_raw = coords.get(y_cfg.gcode_letter)
        if x_raw is None or y_raw is None:
            return None

        return {"X": float(x_raw), "Y": float(y_raw)}

    @property
    def home_groups(self) -> dict[str, list[str]]:
        return dict(self._home_groups)

    @property
    def boards(self) -> dict[str, SerialBoard]:
        return dict(self._boards)

    def is_axis_homed(self, axis: str) -> bool:
        return str(axis).upper() in self._homed_axes

    def homed_axes(self) -> list[str]:
        return sorted(self._homed_axes)
