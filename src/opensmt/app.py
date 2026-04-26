from __future__ import annotations

import asyncio
from typing import Any

from opensmt.config import load_config
from opensmt.messaging import BusNode
from opensmt.modules import CameraVisionModule, CoordinateSystemModule, ModuleBase, SerialGCodeModule

MODULE_TYPES: dict[str, type[ModuleBase]] = {
    "camera_vision": CameraVisionModule,
    "coordinate_system": CoordinateSystemModule,
    "serial_gcode": SerialGCodeModule,
}


async def run_from_config(config_path: str) -> None:
    config = load_config(config_path)

    broker_cfg = config.get("broker", {})
    host = str(broker_cfg.get("host", "127.0.0.1"))
    port = int(broker_cfg.get("port", 8765))

    modules_cfg = _normalize_modules(config)
    modules: list[ModuleBase] = []

    for module_cfg in modules_cfg:
        m_type = str(module_cfg["type"])
        m_name = str(module_cfg["name"])
        cls = MODULE_TYPES.get(m_type)
        if not cls:
            raise ValueError(f"Unknown module type: {m_type}")

        node = BusNode(name=m_name, host=host, port=port)
        await node.connect()

        module = cls(name=m_name, config=module_cfg, node=node)
        modules.append(module)

    for module in modules:
        await module.start()

    try:
        await asyncio.Event().wait()
    finally:
        for module in modules:
            await module.stop()
            await module.node.close()


def _normalize_modules(config: dict[str, Any]) -> list[dict[str, Any]]:
    modules = config.get("modules", [])
    if isinstance(modules, list):
        return [item for item in modules if isinstance(item, dict)]
    if isinstance(modules, dict):
        out: list[dict[str, Any]] = []
        for name, section in modules.items():
            if not isinstance(section, dict):
                continue
            merged = dict(section)
            merged.setdefault("name", name)
            out.append(merged)
        return out
    return []
