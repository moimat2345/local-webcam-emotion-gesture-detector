"""Base interface for all vision analyzers."""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import numpy as np
from pydantic import BaseModel

from src.data_models import HolisticResult

T = TypeVar("T", bound=BaseModel)


class BaseAnalyzer(ABC, Generic[T]):
    """Abstract base class for all vision analyzers.

    Each analyzer takes a frame and holistic results, and produces
    a specific type of analysis result (emotion, gesture, etc.).
    """

    @abstractmethod
    def analyze(self, frame: np.ndarray, holistic: HolisticResult) -> T | None:
        """Analyze a frame and return the result.

        Args:
            frame: The video frame as numpy array (BGR format).
            holistic: MediaPipe Holistic detection results.

        Returns:
            Analysis result, or None if analysis could not be performed.
        """
        pass

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Get the current state for LLM context.

        Returns:
            A dict containing relevant state information.
        """
        pass

    def reset(self) -> None:
        """Reset the analyzer's internal state.

        Override this method if your analyzer maintains state
        that needs to be cleared between sessions.
        """
        pass
