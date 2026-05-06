from __future__ import annotations

from .final_package import FinalPackage


class R0402Package(FinalPackage):
    def __init__(self) -> None:
        super().__init__(
            name="R0402",
            footprint="R_0402_1005Metric",
            _length_mm=1.0,
            _width_mm=0.5,
            _height_mm=0.35,
            _pin_count=2,
        )
