from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from opensmt.config import load_config
from opensmt.hardware.board import BoardConfig, SerialBoard
from opensmt.hardware.driver import HardwareDriver
from opensmt.modules.camera_vision import CameraVisionModule
from opensmt.store.feeder_config import FeederConfigStore, feeder_from_dict
from opensmt.store.location_store import LocationStore
from opensmt.store.nozzle_config import NozzleConfig, NozzleConfigStore, ValveConfig
from opensmt.store.position_store import PositionStore
from opensmt.store.valve_store import ValveStore

log = logging.getLogger(__name__)


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
    cfg_path = Path(config_path).expanduser().resolve()

    # Optional override path in config; defaults to sidecar next to system.json.
    # Persisted entries override in-file defaults at startup.
    persist_path_raw = config.get("locations_persist_path")
    if persist_path_raw is not None:
        persist_path = Path(str(persist_path_raw)).expanduser()
        if not persist_path.is_absolute():
            persist_path = (cfg_path.parent / persist_path).resolve()
    else:
        persist_path = cfg_path.with_name(f"{cfg_path.stem}.locations.runtime.json")

    merged_locations = dict(config.get("locations", {}))
    if persist_path.is_file():
        try:
            persisted = json.loads(persist_path.read_text(encoding="utf-8"))
            if isinstance(persisted, dict):
                merged_locations.update(persisted)
            else:
                log.warning("Ignoring persisted locations at %s: root is not an object", persist_path)
        except Exception as exc:
            log.warning("Failed to load persisted locations from %s: %s", persist_path, exc)

    position_store = PositionStore()
    location_store = LocationStore(merged_locations, persist_path=str(persist_path))

    camera_cfg_runtime: dict[str, Any] = dict(config.get("camera", {}))

    # Optional override path in camera config; defaults to sidecar next to system.json.
    offsets_persist_path_raw = camera_cfg_runtime.get("nozzle_offsets_persist_path")
    if offsets_persist_path_raw is not None:
        offsets_persist_path = Path(str(offsets_persist_path_raw)).expanduser()
        if not offsets_persist_path.is_absolute():
            offsets_persist_path = (cfg_path.parent / offsets_persist_path).resolve()
    else:
        offsets_persist_path = cfg_path.with_name(f"{cfg_path.stem}.nozzle_offsets.runtime.json")

    camera_nozzles = list(camera_cfg_runtime.get("nozzles", []))

    if offsets_persist_path.is_file():
        try:
            persisted_offsets = json.loads(offsets_persist_path.read_text(encoding="utf-8"))
            if isinstance(persisted_offsets, dict):
                for item in camera_nozzles:
                    if not isinstance(item, dict):
                        continue
                    n_name = str(item.get("name", "")).upper()
                    persisted = persisted_offsets.get(n_name)
                    if not isinstance(persisted, dict):
                        continue
                    if "offset_x" in persisted:
                        item["offset_x"] = float(persisted["offset_x"])
                    if "offset_y" in persisted:
                        item["offset_y"] = float(persisted["offset_y"])
                    if "tip_id" in persisted:
                        item["tip_id"] = str(persisted["tip_id"])
                    if "standard_down_z" in persisted and persisted["standard_down_z"] is not None:
                        item["standard_down_z"] = float(persisted["standard_down_z"])
            else:
                log.warning("Ignoring persisted nozzle offsets at %s: root is not an object", offsets_persist_path)
        except Exception as exc:
            log.warning("Failed to load persisted nozzle offsets from %s: %s", offsets_persist_path, exc)

    camera_cfg_runtime["nozzles"] = camera_nozzles
    camera_cfg_runtime["_nozzle_offsets_persist_path"] = str(offsets_persist_path)

    camera_res_persist_path_raw = camera_cfg_runtime.get("camera_resolutions_persist_path")
    if camera_res_persist_path_raw is not None:
        camera_res_persist_path = Path(str(camera_res_persist_path_raw)).expanduser()
        if not camera_res_persist_path.is_absolute():
            camera_res_persist_path = (cfg_path.parent / camera_res_persist_path).resolve()
    else:
        camera_res_persist_path = cfg_path.with_name(f"{cfg_path.stem}.camera_resolutions.runtime.json")

    camera_items = list(camera_cfg_runtime.get("cameras", []))
    if camera_res_persist_path.is_file():
        try:
            persisted_res = json.loads(camera_res_persist_path.read_text(encoding="utf-8"))
            if isinstance(persisted_res, dict):
                for item in camera_items:
                    if not isinstance(item, dict):
                        continue
                    cam_name = str(item.get("name", "")).upper()
                    persisted_cam = persisted_res.get(cam_name)
                    if not isinstance(persisted_cam, dict):
                        continue
                    if "resolution_dpcm_x" in persisted_cam:
                        item["resolution_dpcm_x"] = float(persisted_cam["resolution_dpcm_x"])
                    if "resolution_dpcm_y" in persisted_cam:
                        item["resolution_dpcm_y"] = float(persisted_cam["resolution_dpcm_y"])
            else:
                log.warning("Ignoring persisted camera resolutions at %s: root is not an object", camera_res_persist_path)
        except Exception as exc:
            log.warning("Failed to load persisted camera resolutions from %s: %s", camera_res_persist_path, exc)

    camera_cfg_runtime["cameras"] = camera_items
    camera_cfg_runtime["_camera_resolutions_persist_path"] = str(camera_res_persist_path)

    # Build NozzleConfigStore from camera config
    nozzle_configs: list[NozzleConfig] = []
    for item in camera_cfg_runtime.get("nozzles", []):
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
            tip_id=str(item.get("tip_id")) if item.get("tip_id") is not None else None,
            standard_down_z=(float(item.get("standard_down_z")) if item.get("standard_down_z") is not None else None),
            air_valve=air_cfg,
        )
        nozzle_configs.append(nozzle_cfg)

    nozzle_config_store = NozzleConfigStore(nozzle_configs)
    valve_store = ValveStore(nozzle_config_store.names())

    feeders_persist_dir_raw = config.get("feeders_persist_dir")
    if feeders_persist_dir_raw is not None:
        feeders_persist_dir = Path(str(feeders_persist_dir_raw)).expanduser()
        if not feeders_persist_dir.is_absolute():
            feeders_persist_dir = (cfg_path.parent / feeders_persist_dir).resolve()
    else:
        feeders_persist_dir = (cfg_path.parent / "feeders").resolve()

    feeder_items: list[dict[str, Any]] = [
        dict(item) for item in config.get("feeders", []) if isinstance(item, dict)
    ]
    by_id: dict[str, dict[str, Any]] = {}
    for item in feeder_items:
        feeder_id = str(item.get("feeder_id", "")).upper().strip()
        if feeder_id:
            by_id[feeder_id] = item

    if feeders_persist_dir.is_dir():
        for path in sorted(feeders_persist_dir.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning("Failed to read feeder file %s: %s", path, exc)
                continue
            if not isinstance(raw, dict):
                continue
            feeder_id = str(raw.get("feeder_id", "")).upper().strip()
            if not feeder_id:
                continue
            by_id[feeder_id] = raw

    feeder_configs = [feeder_from_dict(item) for item in by_id.values()]
    feeder_config_store = FeederConfigStore(feeder_configs)

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
        config=camera_cfg_runtime,
        driver=driver,
        position_store=position_store,
        location_store=location_store,
        nozzle_config_store=nozzle_config_store,
        feeder_config_store=feeder_config_store,
        feeders_persist_dir=feeders_persist_dir,
        valve_store=valve_store,
    )

    await driver.start()
    await camera.start()

    try:
        await asyncio.Event().wait()
    finally:
        await camera.stop()
        await driver.stop()
