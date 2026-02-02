"""Configuration settings for the vision module."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class CameraSettings(BaseModel):
    """Camera capture settings."""

    device_id: int = Field(default=0, description="Camera device ID")
    width: int = Field(default=1280, description="Capture width")
    height: int = Field(default=720, description="Capture height")
    fps: int = Field(default=30, description="Target FPS")
    buffer_size: int = Field(default=5, description="Frame buffer size")


class MediaPipeSettings(BaseModel):
    """MediaPipe Tasks API settings."""

    min_detection_confidence: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Min detection confidence"
    )
    min_tracking_confidence: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Min tracking confidence"
    )
    enable_segmentation: bool = Field(default=False, description="Enable segmentation mask")


class PresenceSettings(BaseModel):
    """Presence detection settings."""

    away_threshold_seconds: float = Field(
        default=5.0, ge=1.0, description="Seconds before marking as AWAY"
    )
    returning_threshold_seconds: float = Field(
        default=1.0, ge=0.1, description="Seconds to confirm RETURNING"
    )


class PostureSettings(BaseModel):
    """Posture analysis settings."""

    slouch_threshold_degrees: float = Field(
        default=15.0, ge=5.0, le=45.0, description="Degrees forward to trigger slouch alert"
    )
    enable_alerts: bool = Field(default=True, description="Enable posture alerts")


class VisualizationSettings(BaseModel):
    """Debug visualization settings."""

    enabled: bool = Field(default=True, description="Enable debug window")
    show_face_landmarks: bool = Field(default=True, description="Draw face landmarks")
    show_hand_landmarks: bool = Field(default=True, description="Draw hand landmarks")
    show_pose_landmarks: bool = Field(default=True, description="Draw pose landmarks")
    show_info_overlay: bool = Field(default=True, description="Show text info overlay")
    face_color: tuple[int, int, int] = Field(default=(0, 255, 0), description="Face landmarks color (BGR)")
    left_hand_color: tuple[int, int, int] = Field(default=(255, 0, 0), description="Left hand color (BGR)")
    right_hand_color: tuple[int, int, int] = Field(default=(0, 0, 255), description="Right hand color (BGR)")
    pose_color: tuple[int, int, int] = Field(default=(255, 255, 0), description="Pose landmarks color (BGR)")


class VisionSettings(BaseModel):
    """All vision module settings."""

    camera: CameraSettings = Field(default_factory=CameraSettings)
    mediapipe: MediaPipeSettings = Field(default_factory=MediaPipeSettings)
    presence: PresenceSettings = Field(default_factory=PresenceSettings)
    posture: PostureSettings = Field(default_factory=PostureSettings)
    visualization: VisualizationSettings = Field(default_factory=VisualizationSettings)


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    vision: VisionSettings = Field(default_factory=VisionSettings)
    debug: bool = Field(default=True, description="Enable debug mode")
    language: Literal["fr", "en"] = Field(default="fr", description="Primary language")

    model_config = {
        "env_prefix": "VISION_DETECTOR_",
        "env_nested_delimiter": "__",
    }

    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        """Load settings from YAML file."""
        import yaml

        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)
