from __future__ import annotations

import asyncio
from typing import Any

from opensmt.config import load_config
from opensmt.hardware.board import BoardConfig, SerialBoard
from opensmt.hardware.driver import HardwareDriver
from opensmt.modules.camera_vision import CameraVisionModule
from opensmt.store.location_store import LocationStore
from opensmt.store.nozzle_config import NozzleConfig, NozzleConfigStore, ValveConfig
from opensmt.store.position_store import PositionStore
from opensmt.store.valve_store import ValveStore


async def run_from_config(config_path: str) -> None:
    """Boot the system from a unified system.json config file.

    Expected top-level keys:
        boards    – dict of board_id → serial port config
        driver    – axis table, velocities, home groups, speed_factor
        locations – named coordinate presets
        camera    – web server, nozzles, cameras, pipelines
    """
    config = load_config(config_path)

    # ------------------------------------------------------------------ #
    # Boards
    # ------------------------------------------------------------------ #
    boards: dict[str, SerialBoard] = {}
    for board_id, board_cfg in config.get("boards", {}).items():
        bid = board_id.upper()
        cfg = BoardConfig(
            board_id=bid,
            device=str(board_cfg["device"]),
            baudrate=int(board_cfg.get("baudrate", 115200)),
            bytesize=int(board_cfg.get("bytesize", 8)),
            parity=str(board_cfg.get("parity", "N")),
            stopbits=int(board_cfg.get("stopbits", 1)),
            xonxoff=bool(board_cfg.get("xonxoff", False)),
            rtscts=bool(board_cfg.get("rtscts", False)),
            dsrdtr=bool(board_cfg.get("dsrdtr", False)),
        )
        boards[bid] = SerialBoard(cfg)

    # ------------------------------------------------------------------ #
    # Stores
    # ------------------------------------------------------------------ #
    position_store = PositionStore()
    location_store = LocationStore(config.get("locations", {}))

    # Build NozzleConfigStore from camera config
    nozzle_configs: list[NozzleConfig] = []
    for item in config.get("camera", {}).get("nozzles", []):
        vacuum_cfg_dict = item.get("vacuum_valve", {})
        vacuum_cfg = ValveConfig(
            board=str(vacuum_cfg_dict.get("board", "AB")),
            io_type=str(vacuum_cfg_dict.get("io_type", "gpio")),
            pin=int(vacuum_cfg_dict.get("pin", 0)),
        )

        air_cfg = None
        if "air_valve" in item:
            air_cfg_dict = item["air_valve"]
            air_cfg = ValveConfig(
                board=str(air_cfg_dict.get("board", "AB")),
                io_type=str(air_cfg_dict.get("io_type", "gpio")),
                pin=int(air_cfg_dict.get("pin", 0)),
            )

        nozzle_cfg = NozzleConfig(
            name=str(item["name"]),
            z_axis=str(item["z_axis"]),
            min_z=float(item.get("min_z", -50.0)),
            max_z=float(item.get("max_z", 0.0)),
            offset_x=float(item.get("offset_x", 0.0)),
            offset_y=float(item.get("offset_y", 0.0)),
            vacuum_valve=vacuum_cfg,
            air_valve=air_cfg,
        )
        nozzle_configs.append(nozzle_cfg)

    nozzle_config_store = NozzleConfigStore(nozzle_configs)
    valve_store = ValveStore(nozzle_config_store.names())

    # ------------------------------------------------------------------ #
    # Hardware driver
    # ------------------------------------------------------------------ #
    driver = HardwareDriver(
        boards=boards,
        position_store=position_store,
        location_store=location_store,
        config=config.get("driver", {}),
    )

    # ------------------------------------------------------------------ #
    # Camera / Dashboard
    # ------------------------------------------------------------------ #
    camera = CameraVisionModule(
        name="CAMERA",
        config=config.get("camera", {}),
        driver=driver,
        position_store=position_store,
        location_store=location_store,
        nozzle_config_store=nozzle_config_store,
        valve_store=valve_store,
    )

    await driver.start()
    await camera.start()

    try:
        await asyncio.Event().wait()
    finally:
        await camera.stop()
        await driver.stop()
