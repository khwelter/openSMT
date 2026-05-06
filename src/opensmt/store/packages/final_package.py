from __future__ import annotations

from dataclasses import dataclass

from .base import Package


@dataclass(frozen=True, slots=True)
class FinalPackage(Package):
    name: str
    footprint: str
    _length_mm: float
    _width_mm: float
    _height_mm: float
    _pin_count: int

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("name must not be empty")
        if not str(self.footprint).strip():
            raise ValueError("footprint must not be empty")
        if self._length_mm <= 0.0:
            raise ValueError("length_mm must be > 0")
        if self._width_mm <= 0.0:
            raise ValueError("width_mm must be > 0")
        if self._height_mm <= 0.0:
            raise ValueError("height_mm must be > 0")
        if self._pin_count <= 0:
            raise ValueError("pin_count must be > 0")

    @property
    def length_mm(self) -> float:
        return float(self._length_mm)

    @property
    def width_mm(self) -> float:
        return float(self._width_mm)

    @property
    def height_mm(self) -> float:
        return float(self._height_mm)

    @property
    def pin_count(self) -> int:
        return int(self._pin_count)

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "footprint": self.footprint,
            "length_mm": self.length_mm,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "pin_count": self.pin_count,
        }
