from __future__ import annotations

from .final_package import FinalPackage


class R1206Package(FinalPackage):
    def __init__(self) -> None:
        super().__init__(
            name="R1206",
            footprint="R_1206_3216Metric",
            _length_mm=3.2,
            _width_mm=1.6,
            _height_mm=0.6,
            _pin_count=2,
        )
