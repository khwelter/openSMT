from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

ValveCallback = Callable[[str, str, bool], Awaitable[None] | None]


@dataclass
class NozzleValveState:
    """State of all valves on a single nozzle."""

    nozzle_name: str
    vacuum_on: bool = False
    air_on: bool = False


class ValveStore:
    """Single source of truth for all nozzle valve states.

    Components write valve states only through ``set_vacuum()`` and ``set_air()``.
    The Dashboard and other readers call ``get()`` or ``all()``.
    Subscribers are notified immediately after every state change.
    """

    def __init__(self, nozzle_names: list[str]) -> None:
        self._states: dict[str, NozzleValveState] = {
            name: NozzleValveState(name) for name in nozzle_names
        }
        self._callbacks: list[ValveCallback] = []

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def subscribe(self, callback: ValveCallback) -> None:
        """Register a callback that is called whenever any valve state changes.

        Callback signature: async def(nozzle_name: str, valve_type: str, on: bool) -> None
        where valve_type is 'vacuum' or 'air'.
        """
        self._callbacks.append(callback)

    def unsubscribe(self, callback: ValveCallback) -> None:
        """Remove a previously registered callback (no-op if not found)."""
        try:
            self._callbacks.remove(callback)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def set_vacuum(self, nozzle_name: str, on: bool) -> None:
        """Update vacuum valve state and notify all subscribers."""
        if nozzle_name not in self._states:
            return
        self._states[nozzle_name].vacuum_on = on
        await self._notify(nozzle_name, "vacuum", on)

    async def set_air(self, nozzle_name: str, on: bool) -> None:
        """Update air valve state and notify all subscribers."""
        if nozzle_name not in self._states:
            return
        self._states[nozzle_name].air_on = on
        await self._notify(nozzle_name, "air", on)

    async def _notify(self, nozzle_name: str, valve_type: str, on: bool) -> None:
        """Notify all subscribers of a valve state change."""
        for cb in list(self._callbacks):
            result = cb(nozzle_name, valve_type, on)
            if inspect.isawaitable(result):
                await result

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, nozzle_name: str) -> NozzleValveState | None:
        """Return the current valve state of a nozzle, or None if not found."""
        return self._states.get(nozzle_name)

    def all(self) -> dict[str, NozzleValveState]:
        """Return all nozzle valve states."""
        return dict(self._states)
