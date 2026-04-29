from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModuleBase(ABC):
    def __init__(self, name: str, config: dict[str, Any], node: object | None = None) -> None:
        self.name = name
        self.config = config
        self.node = node

    @abstractmethod
    async def start(self) -> None:
        ...

    @abstractmethod
    async def stop(self) -> None:
        ...
