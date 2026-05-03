from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ValveConfig:
    """Configuration for a single valve (vacuum or air)."""

    board: str  # e.g., "AB", "CD" — which serial board controls this valve
    io_type: str  # "gpio", "pwm", or "relay"
    pin: int  # Pin number on the board


@dataclass
class NozzleConfig:
    """Complete configuration for a single nozzle."""

    name: str  # e.g., "N1", "N2"
    z_axis: str  # e.g., "Z1" — which Z axis this nozzle controls
    min_z: float  # Minimum Z position
    max_z: float  # Maximum Z position
    offset_x: float  # X offset relative to camera position (mm)
    offset_y: float  # Y offset relative to camera position (mm)
    vacuum_valve: ValveConfig  # Always present
    tip_id: str | None = None  # e.g. "501" (Juki 500-series tip id)
    standard_down_z: float | None = None  # PCB placement level for current tip/nozzle
    air_valve: ValveConfig | None = None  # Optional pressurized air valve


class NozzleConfigStore:
    """Immutable nozzle geometry and configuration, loaded from config at startup.

    This is a read-only store of nozzle hardware definitions.
    """

    def __init__(self, nozzles: list[NozzleConfig]) -> None:
        self._nozzles: dict[str, NozzleConfig] = {n.name: n for n in nozzles}

    def get(self, nozzle_name: str) -> NozzleConfig | None:
        """Get nozzle configuration by name."""
        return self._nozzles.get(nozzle_name)

    def all(self) -> dict[str, NozzleConfig]:
        """Get all nozzle configurations."""
        return dict(self._nozzles)

    def names(self) -> list[str]:
        """Get list of all nozzle names."""
        return list(self._nozzles.keys())
