from __future__ import annotations

from .final_package import FinalPackage


class R0603Package(FinalPackage):
    def __init__(self) -> None:
        super().__init__(
            name="R0603",
            footprint="R_0603_1608Metric",
            _length_mm=1.6,
            _width_mm=0.8,
            _height_mm=0.45,
            _pin_count=2,
        )
