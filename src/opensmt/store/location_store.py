from __future__ import annotations

import json
import pathlib


class LocationStore:
    """Named coordinate presets (park, dispose, fiducials, nozzle-change, etc.).

    Each entry is a dict of axis-name -> mm value (not all axes required).
    Loaded from config at startup; can be updated and optionally persisted
    to a JSON file on change.
    """

    def __init__(
        self,
        initial: dict[str, dict[str, float]],
        persist_path: str | None = None,
    ) -> None:
        self._locations: dict[str, dict[str, float]] = {
            name.lower(): {k.upper(): float(v) for k, v in coords.items()}
            for name, coords in initial.items()
        }
        self._persist_path = pathlib.Path(persist_path) if persist_path else None

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, name: str) -> dict[str, float] | None:
        """Return a copy of the named location, or None if not found."""
        entry = self._locations.get(name.lower())
        return dict(entry) if entry is not None else None

    def all(self) -> dict[str, dict[str, float]]:
        """Return a copy of all locations."""
        return {name: dict(coords) for name, coords in self._locations.items()}

    def names(self) -> list[str]:
        return list(self._locations.keys())

    def persist_path(self) -> str | None:
        """Return persistence file path, if configured."""
        if self._persist_path is None:
            return None
        return str(self._persist_path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def set(self, name: str, coords: dict[str, float]) -> None:
        """Create or replace a named location and optionally persist."""
        self._locations[name.lower()] = {k.upper(): float(v) for k, v in coords.items()}
        self._persist()

    def delete(self, name: str) -> bool:
        """Remove a location. Returns True if it existed."""
        existed = name.lower() in self._locations
        self._locations.pop(name.lower(), None)
        if existed:
            self._persist()
        return existed

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        if self._persist_path:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(
                json.dumps(self._locations, indent=2), encoding="utf-8"
            )
