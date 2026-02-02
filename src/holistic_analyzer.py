"""MediaPipe wrapper for face, hand, and pose landmark detection.

Uses the MediaPipe Tasks API with three separate models:
- FaceLandmarker: 468 face landmarks + 52 blendshapes (facial action units)
- HandLandmarker: 21 landmarks per hand (up to 2 hands)
- PoseLandmarker: 33 body landmarks with visibility scores

Models are downloaded automatically to ~/.cache/vision-detector/models/
on first run (~40 Mo total). All models run in VIDEO mode for temporal
smoothing between frames.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from src.settings import MediaPipeSettings
from src.data_models import HolisticResult, Landmark

MODEL_DIR = Path.home() / ".cache" / "vision-detector" / "models"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def _ensure_model(filename: str, url: str) -> Path:
    """Download a model if not present."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / filename
    if not path.exists():
        import urllib.request
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, path)
        print(f"Downloaded {filename}")
    return path

class HolisticAnalyzer:
    """Detects face, hand, and pose landmarks using MediaPipe Tasks API.

    Runs three separate task models (FaceLandmarker, HandLandmarker, PoseLandmarker)
    to replace the deprecated mp.solutions.holistic.
    """

    def __init__(self, settings: Optional[MediaPipeSettings] = None) -> None:
        self._settings = settings or MediaPipeSettings()
        self._initialize()

    def _initialize(self) -> None:
        """Create or recreate all landmarker models."""
        self._frame_timestamp_ms = 0
        self._last_result: Optional[HolisticResult] = None

        # Face Landmarker (with blendshapes)
        self._face_landmarker: Optional[vision.FaceLandmarker] = None
        try:
            model_path = _ensure_model("face_landmarker.task", FACE_MODEL_URL)
            self._face_landmarker = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=str(model_path)),
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=False,
                    num_faces=1,
                    min_face_detection_confidence=self._settings.min_detection_confidence,
                    min_face_presence_confidence=self._settings.min_tracking_confidence,
                    running_mode=vision.RunningMode.VIDEO,
                )
            )
        except Exception as e:
            print(f"Warning: FaceLandmarker failed to init: {e}")

        # Pose Landmarker
        self._pose_landmarker: Optional[vision.PoseLandmarker] = None
        try:
            model_path = _ensure_model("pose_landmarker_heavy.task", POSE_MODEL_URL)
            self._pose_landmarker = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=str(model_path)),
                    num_poses=1,
                    min_pose_detection_confidence=self._settings.min_detection_confidence,
                    min_pose_presence_confidence=self._settings.min_tracking_confidence,
                    min_tracking_confidence=self._settings.min_tracking_confidence,
                    output_segmentation_masks=self._settings.enable_segmentation,
                    running_mode=vision.RunningMode.VIDEO,
                )
            )
        except Exception as e:
            print(f"Warning: PoseLandmarker failed to init: {e}")

        # Hand Landmarker
        self._hand_landmarker: Optional[vision.HandLandmarker] = None
        try:
            model_path = _ensure_model("hand_landmarker.task", HAND_MODEL_URL)
            self._hand_landmarker = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=str(model_path)),
                    num_hands=2,
                    min_hand_detection_confidence=self._settings.min_detection_confidence,
                    min_hand_presence_confidence=self._settings.min_tracking_confidence,
                    min_tracking_confidence=self._settings.min_tracking_confidence,
                    running_mode=vision.RunningMode.VIDEO,
                )
            )
        except Exception as e:
            print(f"Warning: HandLandmarker failed to init: {e}")

    def analyze(self, frame: np.ndarray) -> HolisticResult:
        """Run detection on a frame.

        Args:
            frame: BGR image from OpenCV.

        Returns:
            HolisticResult with all detected landmarks and blendshapes.
        """
        # MediaPipe expects RGB, OpenCV provides BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # VIDEO mode requires monotonically increasing timestamps
        ts = self._frame_timestamp_ms
        self._frame_timestamp_ms += 33  # ~30fps increment

        # Face
        face_landmarks = None
        face_blendshapes = None
        if self._face_landmarker:
            try:
                face_result = self._face_landmarker.detect_for_video(mp_image, ts)
                if face_result.face_landmarks:
                    face_landmarks = self._convert_landmarks(face_result.face_landmarks[0])
                if face_result.face_blendshapes:
                    # Convert blendshapes list to {name: score} dict
                    # e.g. {"mouthSmileLeft": 0.8, "browDownRight": 0.2, ...}
                    face_blendshapes = {
                        bs.category_name: bs.score
                        for bs in face_result.face_blendshapes[0]
                    }
            except Exception as e:
                print(f"Face detection error: {e}")

        # Pose
        pose_landmarks = None
        if self._pose_landmarker:
            try:
                pose_result = self._pose_landmarker.detect_for_video(mp_image, ts)
                if pose_result.pose_landmarks:
                    pose_landmarks = self._convert_landmarks(
                        pose_result.pose_landmarks[0], include_visibility=True
                    )
            except Exception as e:
                print(f"Pose detection error: {e}")

        # Hands
        left_hand_landmarks = None
        right_hand_landmarks = None
        if self._hand_landmarker:
            try:
                hand_result = self._hand_landmarker.detect_for_video(mp_image, ts)
                if hand_result.hand_landmarks and hand_result.handedness:
                    for i, handedness_list in enumerate(hand_result.handedness):
                        label = handedness_list[0].category_name
                        lms = self._convert_landmarks(hand_result.hand_landmarks[i])
                        # MediaPipe returns mirrored labels for front camera
                        if label == "Left":
                            right_hand_landmarks = lms
                        else:
                            left_hand_landmarks = lms
            except Exception as e:
                print(f"Hand detection error: {e}")

        result = HolisticResult(
            face_landmarks=face_landmarks,
            left_hand_landmarks=left_hand_landmarks,
            right_hand_landmarks=right_hand_landmarks,
            pose_landmarks=pose_landmarks,
            face_blendshapes=face_blendshapes,
            timestamp=datetime.now(),
        )

        self._last_result = result
        return result

    def get_state(self) -> dict[str, Any]:
        """Get current detection state."""
        if self._last_result is None:
            return {"detected": False}
        return {
            "detected": True,
            "face": self._last_result.has_face,
            "left_hand": self._last_result.has_left_hand,
            "right_hand": self._last_result.has_right_hand,
            "pose": self._last_result.has_pose,
        }

    def reset(self) -> None:
        """Reset analyzer state and recreate models to clear timestamps."""
        self.close()
        self._initialize()

    def close(self) -> None:
        """Release MediaPipe resources."""
        for landmarker in (self._face_landmarker, self._pose_landmarker, self._hand_landmarker):
            if landmarker:
                try:
                    landmarker.close()
                except Exception:
                    pass
        self._face_landmarker = None
        self._pose_landmarker = None
        self._hand_landmarker = None

    @staticmethod
    def _convert_landmarks(
        landmarks,
        include_visibility: bool = False,
    ) -> list[Landmark]:
        """Convert MediaPipe landmarks to Landmark models.

        Clamps coordinates to [0,1] since MediaPipe can return values
        slightly outside bounds for landmarks near image edges.
        """
        result = []
        for lm in landmarks:
            visibility = getattr(lm, "visibility", None) if include_visibility else None
            result.append(
                Landmark(
                    x=max(0.0, min(1.0, lm.x)),
                    y=max(0.0, min(1.0, lm.y)),
                    z=lm.z,
                    visibility=visibility,
                )
            )
        return result

    # DEAD CODE: last_result property - never accessed
    # @property
    # def last_result(self) -> Optional[HolisticResult]:
    #     """Get the most recent detection result."""
    #     return self._last_result

    def __enter__(self) -> "HolisticAnalyzer":
        return self

    def __exit__(self, *args) -> None:
        self.close()
