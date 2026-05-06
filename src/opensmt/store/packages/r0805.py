from __future__ import annotations

from .final_package import FinalPackage


class R0805Package(FinalPackage):
    def __init__(self) -> None:
        super().__init__(
            name="R0805",
            footprint="R_0805_2012Metric",
            _length_mm=2.0,
            _width_mm=1.25,
            _height_mm=0.55,
            _pin_count=2,
        )
