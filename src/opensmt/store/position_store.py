from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable

AXES = ["X", "Y", "Z1", "Z2", "Z3", "Z4", "R1", "R2", "R3", "R4"]

PositionCallback = Callable[[str, float], Awaitable[None] | None]


class PositionStore:
    """Single source of truth for all machine axis positions.

    Components write positions only through ``update()``.
    The Dashboard and other readers call ``get()`` or ``all()``.
    Subscribers are notified immediately after every update.
    """

    def __init__(self) -> None:
        self._positions: dict[str, float | None] = {axis: None for axis in AXES}
        self._callbacks: list[PositionCallback] = []

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def subscribe(self, callback: PositionCallback) -> None:
        """Register a callback that is called whenever any axis is updated."""
        self._callbacks.append(callback)

    def unsubscribe(self, callback: PositionCallback) -> None:
        """Remove a previously registered callback (no-op if not found)."""
        try:
            self._callbacks.remove(callback)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def update(self, axis: str, value: float) -> None:
        """Update one axis position and notify all subscribers."""
        axis = axis.upper()
        if axis not in self._positions:
            return
        self._positions[axis] = value
        for cb in list(self._callbacks):
            result = cb(axis, value)
            if inspect.isawaitable(result):
                await result

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, axis: str) -> float | None:
        """Return the current position of an axis, or None if not yet known."""
        return self._positions.get(axis.upper())

    def all(self) -> dict[str, float | None]:
        """Return a copy of the full position dict."""
        return dict(self._positions)
