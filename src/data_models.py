"""Pydantic data models for vision analysis results.

All models are frozen (immutable) for thread safety and hashability.
FrameAnalysisResult is the top-level container that aggregates all
analyzer outputs for a single frame, and provides to_llm_context()
for compact serialization.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class EmotionType(str, Enum):
    """Detected emotion types."""

    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    SURPRISE = "surprise"
    FEAR = "fear"
    DISGUST = "disgust"
    CONCENTRATION = "concentration"
    CONFUSION = "confusion"
    FRUSTRATION = "frustration"
    NEUTRAL = "neutral"


class PresenceState(str, Enum):
    """User presence states."""

    PRESENT = "present"
    AWAY = "away"
    RETURNING = "returning"


class GestureType(str, Enum):
    """Recognized hand gestures."""

    NONE = "none"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    OPEN_PALM = "open_palm"
    FIST = "fist"
    PEACE = "peace"
    POINTING = "pointing"
    OK = "ok"
    WAVE = "wave"


class HandSide(str, Enum):
    """Which hand."""

    LEFT = "left"
    RIGHT = "right"


class Landmark(BaseModel):
    """A single 3D landmark point.

    Note: MediaPipe can return coordinates slightly outside [0,1] range
    when landmarks are near image edges. We clamp them in post-processing.
    """

    x: float = Field(description="Normalized X coordinate (typically 0-1)")
    y: float = Field(description="Normalized Y coordinate (typically 0-1)")
    z: float = Field(description="Depth (relative to hips)")
    visibility: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    model_config = ConfigDict(frozen=True)


class EmotionResult(BaseModel):
    """Result of emotion analysis."""

    emotion: EmotionType
    confidence: float = Field(ge=0.0, le=1.0)
    secondary_emotion: Optional[EmotionType] = None
    secondary_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(frozen=True)


class PresenceResult(BaseModel):
    """Result of presence detection."""

    state: PresenceState
    face_detected: bool
    seconds_since_last_seen: float = Field(ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(frozen=True)


class GestureResult(BaseModel):
    """Result of gesture recognition for one hand."""

    hand: HandSide
    gesture: GestureType
    confidence: float = Field(ge=0.0, le=1.0)
    pointing_direction: Optional[tuple[float, float]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(frozen=True)


class PostureResult(BaseModel):
    """Result of posture analysis."""

    neck_angle: float = Field(description="Forward head angle in degrees")
    shoulder_alignment: float = Field(
        ge=-1.0, le=1.0, description="Shoulder tilt (-1=left low, 1=right low)"
    )
    slouch_score: float = Field(
        ge=0.0, le=1.0, description="How much slouching (0=good, 1=bad)"
    )
    is_slouching: bool
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(frozen=True)


class HolisticResult(BaseModel):
    """Raw MediaPipe Holistic detection results.

    Stores the landmark arrays for downstream processing.
    """

    face_landmarks: Optional[list[Landmark]] = None
    left_hand_landmarks: Optional[list[Landmark]] = None
    right_hand_landmarks: Optional[list[Landmark]] = None
    pose_landmarks: Optional[list[Landmark]] = None
    face_blendshapes: Optional[dict[str, float]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(frozen=True)

    @property
    def has_face(self) -> bool:
        """Check if face was detected."""
        return self.face_landmarks is not None and len(self.face_landmarks) > 0

    @property
    def has_left_hand(self) -> bool:
        """Check if left hand was detected."""
        return self.left_hand_landmarks is not None and len(self.left_hand_landmarks) > 0

    @property
    def has_right_hand(self) -> bool:
        """Check if right hand was detected."""
        return self.right_hand_landmarks is not None and len(self.right_hand_landmarks) > 0

    @property
    def has_pose(self) -> bool:
        """Check if body pose was detected."""
        return self.pose_landmarks is not None and len(self.pose_landmarks) > 0


class FrameAnalysisResult(BaseModel):
    """Complete analysis result for a single frame."""

    holistic: HolisticResult
    emotion: Optional[EmotionResult] = None
    presence: Optional[PresenceResult] = None
    left_gesture: Optional[GestureResult] = None
    right_gesture: Optional[GestureResult] = None
    posture: Optional[PostureResult] = None
    frame_number: int = Field(ge=0)
    processing_time_ms: float = Field(ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(frozen=True)

    def to_llm_context(self) -> dict:
        """Convert to a compact dict suitable for LLM context."""
        context = {
            "timestamp": self.timestamp.isoformat(),
            "presence": self.presence.state.value if self.presence else "unknown",
        }

        if self.emotion:
            context["emotion"] = {
                "primary": self.emotion.emotion.value,
                "confidence": round(self.emotion.confidence, 2),
            }
            if self.emotion.secondary_emotion:
                context["emotion"]["secondary"] = self.emotion.secondary_emotion.value

        if self.left_gesture and self.left_gesture.gesture != GestureType.NONE:
            context["left_hand_gesture"] = self.left_gesture.gesture.value

        if self.right_gesture and self.right_gesture.gesture != GestureType.NONE:
            context["right_hand_gesture"] = self.right_gesture.gesture.value

        if self.posture:
            context["posture"] = {
                "slouching": self.posture.is_slouching,
                "score": round(self.posture.slouch_score, 2),
            }

        return context
