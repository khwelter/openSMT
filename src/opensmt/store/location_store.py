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
        persist_root_key: str | None = None,
    ) -> None:
        self._locations: dict[str, dict[str, float]] = self._normalize_locations(initial)
        self._persist_path = pathlib.Path(persist_path) if persist_path else None
        self._persist_root_key = str(persist_root_key).strip() if persist_root_key else None
        self._reload_from_persist_if_available()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, name: str) -> dict[str, float] | None:
        """Return a copy of the named location, or None if not found."""
        self._reload_from_persist_if_available()
        entry = self._locations.get(name.lower())
        return dict(entry) if entry is not None else None

    def all(self) -> dict[str, dict[str, float]]:
        """Return a copy of all locations."""
        self._reload_from_persist_if_available()
        return {name: dict(coords) for name, coords in self._locations.items()}

    def names(self) -> list[str]:
        self._reload_from_persist_if_available()
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
        self._reload_from_persist_if_available()
        self._locations[name.lower()] = {k.upper(): float(v) for k, v in coords.items()}
        self._persist()

    def delete(self, name: str) -> bool:
        """Remove a location. Returns True if it existed."""
        self._reload_from_persist_if_available()
        existed = name.lower() in self._locations
        self._locations.pop(name.lower(), None)
        if existed:
            self._persist()
        return existed

    def replace_all(self, locations: dict[str, dict[str, float]]) -> None:
        """Replace all locations in one atomic store update and persist once."""
        self._locations = self._normalize_locations(locations)
        self._persist()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        if self._persist_path:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            if self._persist_root_key:
                payload = {self._persist_root_key: self._locations}
            else:
                payload = self._locations
            self._persist_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _reload_from_persist_if_available(self) -> None:
        if self._persist_path is None or not self._persist_path.is_file():
            return
        try:
            raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
        except Exception:
            return

        source = raw
        if self._persist_root_key:
            source = raw.get(self._persist_root_key) if isinstance(raw, dict) else None
        if not isinstance(source, dict):
            return

        self._locations = self._normalize_locations(source)

    @staticmethod
    def _normalize_locations(locations: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        normalized: dict[str, dict[str, float]] = {}
        for name, coords in locations.items():
            if not isinstance(coords, dict):
                continue
            key = str(name).strip().lower()
            if not key:
                continue
            try:
                normalized[key] = {str(axis).upper(): float(value) for axis, value in coords.items()}
            except Exception:
                continue
        return normalized
