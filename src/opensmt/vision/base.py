from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class VisionPipelineBase(ABC):
    """Abstract base class for all vision pipelines.

    Subclass this and implement :meth:`process` to create a custom pipeline.
    Register the pipeline type in the module config under ``"pipelines"`` with
    the matching ``"type"`` key, then reference it by name from a camera's
    ``"pipelines"`` list.
    """

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name = name
        self.config = config

    @abstractmethod
    def process(
        self,
        frame: np.ndarray,
        params: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Process a single frame.

        Args:
            frame: BGR image as a NumPy array (H×W×3, uint8).
            params: Runtime parameters supplied by the caller (bus command or
                    web UI). Keys and semantics are pipeline-specific.

        Returns:
            A ``(processed_frame, results)`` tuple where *processed_frame* is
            the annotated/transformed BGR image to display and *results* is an
            arbitrary JSON-serialisable dict with pipeline outputs (coordinates,
            labels, measurements, …).
        """
        ...


class PassthroughPipeline(VisionPipelineBase):
    """Identity pipeline — returns the frame unchanged with no results."""

    def process(
        self,
        frame: np.ndarray,
        params: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return frame, {}
