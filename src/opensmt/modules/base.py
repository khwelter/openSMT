from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from opensmt.messaging import BusNode


class ModuleBase(ABC):
    def __init__(self, name: str, config: dict[str, Any], node: BusNode) -> None:
        self.name = name
        self.config = config
        self.node = node

    @abstractmethod
    async def start(self) -> None:
        ...

    @abstractmethod
    async def stop(self) -> None:
        ...
