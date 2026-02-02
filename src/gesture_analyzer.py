"""Hand gesture recognition from MediaPipe hand landmarks.

Detects 9 gestures per hand by analyzing finger extension patterns:
- Finger extension is determined by comparing tip, PIP, and MCP joint positions
- Gestures are matched via simple rules (e.g. all fingers up = open palm)
- Each hand is analyzed independently (left and right)

MediaPipe hand landmark layout: wrist(0), thumb(1-4), index(5-8),
middle(9-12), ring(13-16), pinky(17-20). TIP = index 4,8,12,16,20.
"""

import math
from typing import Any, Optional

import numpy as np

from src.base_analyzer import BaseAnalyzer
from src.data_models import (
    GestureResult,
    GestureType,
    HandSide,
    HolisticResult,
    Landmark,
)
from src.utils import landmark_distance


class GestureAnalyzer(BaseAnalyzer[tuple[Optional[GestureResult], Optional[GestureResult]]]):
    """Recognizes hand gestures from hand landmarks.

    MediaPipe hand landmarks indices:
    0 = wrist
    1-4 = thumb (CMC, MCP, IP, TIP)
    5-8 = index finger
    9-12 = middle finger
    13-16 = ring finger
    17-20 = pinky
    """

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    def __init__(self) -> None:
        self._last_left: Optional[GestureResult] = None
        self._last_right: Optional[GestureResult] = None

    def analyze(
        self, frame: np.ndarray, holistic: HolisticResult
    ) -> tuple[Optional[GestureResult], Optional[GestureResult]]:
        """Analyze gestures for both hands."""
        left_result = None
        right_result = None

        if holistic.has_left_hand and holistic.left_hand_landmarks:
            left_result = self._analyze_hand(holistic.left_hand_landmarks, HandSide.LEFT)
            self._last_left = left_result

        if holistic.has_right_hand and holistic.right_hand_landmarks:
            right_result = self._analyze_hand(holistic.right_hand_landmarks, HandSide.RIGHT)
            self._last_right = right_result

        return left_result, right_result

    def _analyze_hand(self, landmarks: list[Landmark], side: HandSide) -> GestureResult:
        """Analyze gesture for a single hand."""
        fingers_extended = self._get_fingers_extended(landmarks, side)
        thumb_up = self._is_thumb_up(landmarks, side)
        thumb_down = self._is_thumb_down(landmarks, side)

        gesture = GestureType.NONE
        confidence = 0.5
        pointing_direction = None

        if thumb_up and not any(fingers_extended[1:]):
            gesture = GestureType.THUMBS_UP
            confidence = 0.9
        elif thumb_down and not any(fingers_extended[1:]):
            gesture = GestureType.THUMBS_DOWN
            confidence = 0.9
        elif all(fingers_extended):
            gesture = GestureType.OPEN_PALM
            confidence = 0.85
        elif not any(fingers_extended):
            gesture = GestureType.FIST
            confidence = 0.85
        elif fingers_extended[1] and fingers_extended[2] and not fingers_extended[3] and not fingers_extended[4]:
            gesture = GestureType.PEACE
            confidence = 0.85
        elif fingers_extended[1] and not any(fingers_extended[2:]):
            gesture = GestureType.POINTING
            confidence = 0.8
            pointing_direction = self._get_pointing_direction(landmarks)
        elif self._is_ok_gesture(landmarks, fingers_extended):
            gesture = GestureType.OK
            confidence = 0.8

        return GestureResult(
            hand=side,
            gesture=gesture,
            confidence=confidence,
            pointing_direction=pointing_direction,
        )

    def _get_fingers_extended(self, landmarks: list[Landmark], side: HandSide) -> list[bool]:
        """Check which fingers are extended with improved accuracy.

        Returns: [thumb, index, middle, ring, pinky]
        """
        wrist = landmarks[self.WRIST]

        # Thumb detection based on distance from palm center
        palm_center_x = (landmarks[self.INDEX_MCP].x + landmarks[self.PINKY_MCP].x) / 2
        thumb_distance = abs(landmarks[self.THUMB_TIP].x - palm_center_x)
        thumb_extended = thumb_distance > 0.1  # Thumb is extended if far from palm

        finger_tips = [self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]
        finger_pips = [self.INDEX_PIP, self.MIDDLE_PIP, self.RING_PIP, self.PINKY_PIP]
        finger_mcps = [self.INDEX_MCP, self.MIDDLE_MCP, self.RING_MCP, self.PINKY_MCP]

        fingers_extended = [thumb_extended]

        # Improved finger extension detection
        for tip_idx, pip_idx, mcp_idx in zip(finger_tips, finger_pips, finger_mcps):
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            mcp = landmarks[mcp_idx]

            # A finger is extended if tip is above both PIP and MCP joints
            # Using a threshold to account for slight angles
            tip_above_pip = tip.y < (pip.y - 0.02)  # Tip clearly above PIP
            pip_above_mcp = pip.y <= mcp.y  # PIP at or above MCP (finger not folded)

            extended = tip_above_pip and pip_above_mcp
            fingers_extended.append(extended)

        return fingers_extended

    def _is_thumb_up(self, landmarks: list[Landmark], side: HandSide) -> bool:
        """Check if thumb is pointing up."""
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_mcp = landmarks[self.THUMB_MCP]
        wrist = landmarks[self.WRIST]

        thumb_pointing_up = thumb_tip.y < thumb_mcp.y < wrist.y

        vertical_distance = abs(thumb_mcp.y - thumb_tip.y)
        horizontal_distance = abs(thumb_mcp.x - thumb_tip.x)

        is_vertical = vertical_distance > horizontal_distance * 0.8

        return thumb_pointing_up and is_vertical

    def _is_thumb_down(self, landmarks: list[Landmark], side: HandSide) -> bool:
        """Check if thumb is pointing down."""
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_mcp = landmarks[self.THUMB_MCP]
        wrist = landmarks[self.WRIST]

        thumb_pointing_down = thumb_tip.y > thumb_mcp.y > wrist.y

        vertical_distance = abs(thumb_mcp.y - thumb_tip.y)
        horizontal_distance = abs(thumb_mcp.x - thumb_tip.x)

        is_vertical = vertical_distance > horizontal_distance * 0.8

        return thumb_pointing_down and is_vertical

    def _is_ok_gesture(self, landmarks: list[Landmark], fingers_extended: list[bool]) -> bool:
        """Check for OK gesture (thumb and index forming circle)."""
        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_TIP]

        distance = landmark_distance(thumb_tip, index_tip)

        tips_close = distance < 0.05

        other_fingers_up = fingers_extended[2] and fingers_extended[3] and fingers_extended[4]

        return tips_close and other_fingers_up

    def _get_pointing_direction(self, landmarks: list[Landmark]) -> tuple[float, float]:
        """Get the direction the index finger is pointing.

        Returns: (dx, dy) normalized direction vector
        """
        index_tip = landmarks[self.INDEX_TIP]
        index_mcp = landmarks[self.INDEX_MCP]

        dx = index_tip.x - index_mcp.x
        dy = index_tip.y - index_mcp.y

        length = math.sqrt(dx * dx + dy * dy)
        if length > 0:
            dx /= length
            dy /= length

        return (round(dx, 3), round(dy, 3))

    def get_state(self) -> dict[str, Any]:
        """Get current gesture state for LLM context."""
        state = {}

        if self._last_left and self._last_left.gesture != GestureType.NONE:
            state["left_hand"] = {
                "gesture": self._last_left.gesture.value,
                "confidence": round(self._last_left.confidence, 2),
            }
            if self._last_left.pointing_direction:
                state["left_hand"]["pointing"] = self._last_left.pointing_direction

        if self._last_right and self._last_right.gesture != GestureType.NONE:
            state["right_hand"] = {
                "gesture": self._last_right.gesture.value,
                "confidence": round(self._last_right.confidence, 2),
            }
            if self._last_right.pointing_direction:
                state["right_hand"]["pointing"] = self._last_right.pointing_direction

        if not state:
            state["detected"] = False

        return state

    def reset(self) -> None:
        """Reset gesture state."""
        self._last_left = None
        self._last_right = None
