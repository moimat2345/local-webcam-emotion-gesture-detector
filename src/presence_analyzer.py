"""Presence detection analyzer - detects if user is present or away.

State machine:
    AWAY --[face detected]--> RETURNING --[face stable for N sec]--> PRESENT
    PRESENT --[face lost for N sec]--> AWAY
    RETURNING --[face lost]--> back to PRESENT/AWAY timer logic
"""

import time
from typing import Any, Optional

import numpy as np

from src.settings import PresenceSettings
from src.base_analyzer import BaseAnalyzer
from src.data_models import HolisticResult, PresenceResult, PresenceState


class PresenceAnalyzer(BaseAnalyzer[PresenceResult]):
    """Detects user presence based on face detection.

    States:
    - PRESENT: Face is detected
    - AWAY: No face detected for longer than threshold
    - RETURNING: Face just reappeared after being away
    """

    def __init__(self, settings: Optional[PresenceSettings] = None) -> None:
        self._settings = settings or PresenceSettings()

        self._current_state = PresenceState.AWAY
        self._last_face_seen_time: Optional[float] = None
        self._face_reappeared_time: Optional[float] = None
        self._last_result: Optional[PresenceResult] = None

    def analyze(self, frame: np.ndarray, holistic: HolisticResult) -> PresenceResult:
        """Analyze presence based on face detection."""
        current_time = time.time()
        face_detected = holistic.has_face

        # State transitions based on face visibility over time
        if face_detected:
            self._last_face_seen_time = current_time

            if self._current_state == PresenceState.AWAY:
                self._face_reappeared_time = current_time
                self._current_state = PresenceState.RETURNING
            elif self._current_state == PresenceState.RETURNING:
                if self._face_reappeared_time is not None:
                    time_since_return = current_time - self._face_reappeared_time
                    if time_since_return >= self._settings.returning_threshold_seconds:
                        self._current_state = PresenceState.PRESENT
                        self._face_reappeared_time = None
        else:
            self._face_reappeared_time = None

            if self._current_state in (PresenceState.PRESENT, PresenceState.RETURNING):
                if self._last_face_seen_time is not None:
                    time_since_seen = current_time - self._last_face_seen_time
                    if time_since_seen >= self._settings.away_threshold_seconds:
                        self._current_state = PresenceState.AWAY

        seconds_since_last_seen = 0.0
        if self._last_face_seen_time is not None:
            seconds_since_last_seen = current_time - self._last_face_seen_time

        result = PresenceResult(
            state=self._current_state,
            face_detected=face_detected,
            seconds_since_last_seen=seconds_since_last_seen,
        )

        self._last_result = result
        return result

    def get_state(self) -> dict[str, Any]:
        """Get current presence state for LLM context."""
        if self._last_result is None:
            return {"presence": "unknown", "face_detected": False}

        return {
            "presence": self._current_state.value,
            "face_detected": self._last_result.face_detected,
            "seconds_away": round(self._last_result.seconds_since_last_seen, 1),
        }

    def reset(self) -> None:
        """Reset presence tracking."""
        self._current_state = PresenceState.AWAY
        self._last_face_seen_time = None
        self._face_reappeared_time = None
        self._last_result = None
