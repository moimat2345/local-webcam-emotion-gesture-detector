"""Posture analysis using body pose landmarks.

Measures forward head angle and shoulder alignment to detect slouching.
Uses MediaPipe Pose landmarks (shoulders, hips, nose) to compute angles.

The slouch score combines neck angle (70% weight) and shoulder tilt (30% weight),
smoothed over a sliding window of recent frames.
"""

import math
from typing import Any, Optional

import numpy as np

from src.settings import PostureSettings
from src.base_analyzer import BaseAnalyzer
from src.data_models import HolisticResult, Landmark, PostureResult


class PostureAnalyzer(BaseAnalyzer[PostureResult]):
    """Analyzes body posture from pose landmarks.

    MediaPipe Pose landmark indices:
    0 = nose
    11 = left shoulder
    12 = right shoulder
    23 = left hip
    24 = right hip
    """

    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    def __init__(self, settings: Optional[PostureSettings] = None) -> None:
        self._settings = settings or PostureSettings()
        self._last_result: Optional[PostureResult] = None
        self._slouch_history: list[float] = []
        self._history_size = 10

    def analyze(self, frame: np.ndarray, holistic: HolisticResult) -> Optional[PostureResult]:
        """Analyze posture from pose landmarks."""
        if not holistic.has_pose or holistic.pose_landmarks is None:
            return None

        landmarks = holistic.pose_landmarks

        if len(landmarks) < 25:
            return None

        neck_angle = self._calculate_neck_angle(landmarks)
        shoulder_alignment = self._calculate_shoulder_alignment(landmarks)
        slouch_score = self._calculate_slouch_score(neck_angle, shoulder_alignment)

        self._slouch_history.append(slouch_score)
        if len(self._slouch_history) > self._history_size:
            self._slouch_history.pop(0)

        smoothed_slouch = sum(self._slouch_history) / len(self._slouch_history)

        is_slouching = neck_angle > self._settings.slouch_threshold_degrees

        result = PostureResult(
            neck_angle=round(neck_angle, 1),
            shoulder_alignment=round(shoulder_alignment, 3),
            slouch_score=round(smoothed_slouch, 2),
            is_slouching=is_slouching,
        )

        self._last_result = result
        return result

    def _calculate_neck_angle(self, landmarks: list[Landmark]) -> float:
        """Calculate the forward head angle in degrees.

        Measures angle between:
        - Vertical line from shoulders
        - Line from shoulder center to nose

        Returns angle in degrees (0 = perfectly upright, positive = forward)
        """
        left_shoulder = landmarks[self.LEFT_SHOULDER]
        right_shoulder = landmarks[self.RIGHT_SHOULDER]
        nose = landmarks[self.NOSE]

        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2

        dx = nose.x - shoulder_center_x
        dy = shoulder_center_y - nose.y

        if dy <= 0:
            return 45.0

        angle_rad = math.atan2(abs(dx), dy)
        angle_deg = math.degrees(angle_rad)

        return angle_deg

    def _calculate_shoulder_alignment(self, landmarks: list[Landmark]) -> float:
        """Calculate shoulder tilt.

        Returns:
            Value between -1 and 1
            -1 = left shoulder much lower
            0 = shoulders level
            1 = right shoulder much lower
        """
        left_shoulder = landmarks[self.LEFT_SHOULDER]
        right_shoulder = landmarks[self.RIGHT_SHOULDER]

        height_diff = right_shoulder.y - left_shoulder.y

        normalized = max(-1.0, min(1.0, height_diff * 10))
        return normalized

    def _calculate_slouch_score(self, neck_angle: float, shoulder_alignment: float) -> float:
        """Calculate overall slouch score (0 = good, 1 = bad)."""
        # Normalize neck angle: 30° forward = max slouch
        neck_component = min(1.0, neck_angle / 30.0)

        # Shoulder tilt contributes less than neck angle
        shoulder_component = abs(shoulder_alignment) * 0.3

        # Weighted combination: neck is the primary indicator
        score = neck_component * 0.7 + shoulder_component * 0.3
        return min(1.0, max(0.0, score))

    def get_state(self) -> dict[str, Any]:
        """Get current posture state for LLM context."""
        if self._last_result is None:
            return {"posture": "unknown", "detected": False}

        state = {
            "detected": True,
            "slouching": self._last_result.is_slouching,
            "slouch_score": self._last_result.slouch_score,
            "neck_angle": self._last_result.neck_angle,
        }

        if self._last_result.is_slouching and self._settings.enable_alerts:
            state["alert"] = "Redresse-toi !"

        return state

    def reset(self) -> None:
        """Reset posture history."""
        self._slouch_history.clear()
        self._last_result = None
