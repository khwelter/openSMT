from __future__ import annotations

from abc import ABC, abstractmethod


class Package(ABC):
    @property
    @abstractmethod
    def length_mm(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def width_mm(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def height_mm(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def pin_count(self) -> int:
        raise NotImplementedError
