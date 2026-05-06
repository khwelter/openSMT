from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .final_package import FinalPackage


def package_from_dict(data: dict[str, Any]) -> FinalPackage:
    return FinalPackage(
        name=str(data["name"]),
        footprint=str(data["footprint"]),
        _length_mm=float(data["length_mm"]),
        _width_mm=float(data["width_mm"]),
        _height_mm=float(data["height_mm"]),
        _pin_count=int(data["pin_count"]),
    )


@dataclass(slots=True)
class PackageStore:
    _packages: dict[str, FinalPackage]

    @classmethod
    def from_items(cls, items: list[FinalPackage]) -> "PackageStore":
        table: dict[str, FinalPackage] = {}
        for item in items:
            key = item.name.strip().upper()
            if key in table:
                raise ValueError(f"Duplicate package name: {item.name}")
            table[key] = item
        return cls(table)

    @classmethod
    def from_config_dir(cls, config_dir: str | Path) -> "PackageStore":
        root = Path(config_dir).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"Package config directory not found: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Package config path is not a directory: {root}")

        items: list[FinalPackage] = []
        for path in sorted(root.glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError(f"Invalid package config (expected object): {path}")
            items.append(package_from_dict(raw))
        return cls.from_items(items)

    def get(self, name: str) -> FinalPackage | None:
        return self._packages.get(str(name).strip().upper())

    def all(self) -> list[FinalPackage]:
        return list(self._packages.values())
