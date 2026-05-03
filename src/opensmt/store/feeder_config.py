from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_FEEDER_ID_RE = re.compile(r"^[0-9A-Fa-f]{16,}$")

FEEDER_TYPES = (
    "tray_feeder",
    "auto_feeder",
    "push_pull_feeder",
    "vibration_feeder",
    "label_feeder",
    "tube_feeder",
)


@dataclass(slots=True)
class PickLocation:
    x: float
    y: float


@dataclass(slots=True)
class FeederConfig:
    feeder_id: str
    pick_location: PickLocation
    pick_height: float
    manufacturer_part_number: str
    feeder_type: str

    def __post_init__(self) -> None:
        self.feeder_id = str(self.feeder_id).upper()
        self.feeder_type = str(self.feeder_type).strip().lower()
        self.manufacturer_part_number = str(self.manufacturer_part_number).strip()
        self.pick_height = float(self.pick_height)

        if not _FEEDER_ID_RE.fullmatch(self.feeder_id):
            raise ValueError(
                f"Invalid feeder_id '{self.feeder_id}': expected at least 16 hexadecimal digits"
            )
        if self.feeder_type not in FEEDER_TYPES:
            raise ValueError(
                f"Invalid feeder_type '{self.feeder_type}': expected one of {', '.join(FEEDER_TYPES)}"
            )
        if not self.manufacturer_part_number:
            raise ValueError("manufacturer_part_number must not be empty")

    def to_status(self) -> dict[str, Any]:
        return {
            "feeder_id": self.feeder_id,
            "feeder_type": self.feeder_type,
            "pick_location": {
                "x": self.pick_location.x,
                "y": self.pick_location.y,
            },
            "pick_height": self.pick_height,
            "manufacturer_part_number": self.manufacturer_part_number,
        }


@dataclass(slots=True)
class TrayFeederConfig(FeederConfig):
    feeder_type: str = "tray_feeder"


@dataclass(slots=True)
class AutoFeederConfig(FeederConfig):
    feeder_type: str = "auto_feeder"


@dataclass(slots=True)
class PushPullFeederConfig(FeederConfig):
    feeder_type: str = "push_pull_feeder"


@dataclass(slots=True)
class VibrationFeederConfig(FeederConfig):
    feeder_type: str = "vibration_feeder"


@dataclass(slots=True)
class LabelFeederConfig(FeederConfig):
    feeder_type: str = "label_feeder"


@dataclass(slots=True)
class TubeFeederConfig(FeederConfig):
    feeder_type: str = "tube_feeder"


_FEEDER_TYPE_TO_CLASS: dict[str, type[FeederConfig]] = {
    "tray_feeder": TrayFeederConfig,
    "auto_feeder": AutoFeederConfig,
    "push_pull_feeder": PushPullFeederConfig,
    "vibration_feeder": VibrationFeederConfig,
    "label_feeder": LabelFeederConfig,
    "tube_feeder": TubeFeederConfig,
}


def feeder_from_dict(item: dict[str, Any]) -> FeederConfig:
    feeder_type = str(item.get("feeder_type", "")).strip().lower()
    cls = _FEEDER_TYPE_TO_CLASS.get(feeder_type)
    if cls is None:
        raise ValueError(f"Unknown feeder_type: {feeder_type or '<empty>'}")

    pick_location_raw = item.get("pick_location")
    if not isinstance(pick_location_raw, dict):
        raise ValueError("pick_location must be an object with x/y")

    pick_location = PickLocation(
        x=float(pick_location_raw["x"]),
        y=float(pick_location_raw["y"]),
    )

    return cls(
        feeder_id=str(item["feeder_id"]),
        pick_location=pick_location,
        pick_height=float(item["pick_height"]),
        manufacturer_part_number=str(item["manufacturer_part_number"]),
    )


class FeederConfigStore:
    def __init__(self, feeders: list[FeederConfig]) -> None:
        self._feeders: dict[str, FeederConfig] = {}
        for feeder in feeders:
            if feeder.feeder_id in self._feeders:
                raise ValueError(f"Duplicate feeder_id: {feeder.feeder_id}")
            self._feeders[feeder.feeder_id] = feeder

    def get(self, feeder_id: str) -> FeederConfig | None:
        return self._feeders.get(str(feeder_id).upper())

    def all(self) -> list[FeederConfig]:
        return list(self._feeders.values())

    def by_type(self, feeder_type: str) -> list[FeederConfig]:
        key = str(feeder_type).strip().lower()
        return [feeder for feeder in self._feeders.values() if feeder.feeder_type == key]
