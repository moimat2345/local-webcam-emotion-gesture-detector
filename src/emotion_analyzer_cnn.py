"""Emotion detection analyzer using EmotiEffLib (HSEmotion) CNN model.

Uses a pre-trained EfficientNet model for 8-class emotion recognition.
The CNN analyzes a cropped face region extracted from MediaPipe landmarks.
Falls back to blendshape-based detection if the CNN model fails to load.

Model: enet_b0_8_best_vgaf (default) — downloads automatically via hsemotion_onnx.
"""

from typing import Any, Optional

import numpy as np

from src.data_models import EmotionResult, EmotionType, HolisticResult
from src.emotion_analyzer import BaseEmotionAnalyzer


class EmotionAnalyzerCNN(BaseEmotionAnalyzer):
    """Detects emotions using EmotiEffLib (HSEmotion) CNN model.

    Provides state-of-the-art emotion recognition with 8 emotion categories:
    - Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise

    Falls back to MediaPipe blendshapes if CNN model fails to load.
    """

    HSEMOTION_TO_EMOTION = {
        0: EmotionType.ANGER,
        1: EmotionType.DISGUST,  # Contempt -> Disgust (closest match)
        2: EmotionType.DISGUST,
        3: EmotionType.FEAR,
        4: EmotionType.JOY,  # Happiness -> Joy
        5: EmotionType.NEUTRAL,
        6: EmotionType.SADNESS,
        7: EmotionType.SURPRISE,
    }

    def __init__(self, model_name: str = "enet_b0_8_best_vgaf") -> None:
        super().__init__(history_size=15)

        self._hsemotion = None
        try:
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

            self._hsemotion = HSEmotionRecognizer(model_name=model_name)
            print(f"EmotiEffLib CNN model loaded: {model_name}")
        except Exception as e:
            print(f"Warning: EmotiEffLib failed to load: {e}")
            print("Falling back to blendshapes-based detection")

    def analyze(
        self, frame: np.ndarray, holistic: HolisticResult
    ) -> Optional[EmotionResult]:
        """Analyze emotion from face using CNN or blendshapes fallback."""
        if not holistic.has_face:
            return None

        # Try CNN-based detection first (most accurate)
        if self._hsemotion is not None:
            result = self._analyze_with_cnn(frame, holistic)
            if result is not None:
                return result

        # Fallback to blendshapes
        if holistic.face_blendshapes:
            return self._analyze_from_blendshapes(holistic.face_blendshapes)

        return None

    def _analyze_with_cnn(
        self, frame: np.ndarray, holistic: HolisticResult
    ) -> Optional[EmotionResult]:
        """Analyze emotion using HSEmotion CNN model."""
        try:
            # Extract face bounding box from landmarks
            face_box = self._get_face_bbox(frame, holistic.face_landmarks)
            if face_box is None:
                return None

            x1, y1, x2, y2 = face_box

            # Crop face with padding
            h, w = frame.shape[:2]
            padding = 0.3  # 30% padding around face for better detection
            pad_x = int((x2 - x1) * padding)
            pad_y = int((y2 - y1) * padding)

            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            face_img = frame[y1:y2, x1:x2]

            if face_img.size == 0 or face_img.shape[0] < 10 or face_img.shape[1] < 10:
                return None

            # Run emotion detection - returns tuple (emotion_idx, scores_array)
            result_tuple = self._hsemotion.predict_emotions(face_img, logits=False)

            # Handle both return formats
            if isinstance(result_tuple, tuple):
                emotion_idx, scores = result_tuple
            else:
                # Fallback if format is different
                return None

            # Convert to int if needed
            if hasattr(emotion_idx, 'item'):
                emotion_idx = int(emotion_idx.item())
            else:
                emotion_idx = int(emotion_idx)

            # Map to our emotion types
            primary_emotion = self.HSEMOTION_TO_EMOTION.get(
                emotion_idx, EmotionType.NEUTRAL
            )

            # Ensure scores is a numpy array
            if not isinstance(scores, np.ndarray):
                scores = np.array(scores)

            primary_confidence = float(scores[emotion_idx])

            # Find secondary emotion
            sorted_indices = np.argsort(scores)[::-1]
            secondary = None

            if len(sorted_indices) > 1:
                sec_idx = int(sorted_indices[1])
                sec_score = float(scores[sec_idx])
                if sec_score > 0.2:
                    secondary = (
                        self.HSEMOTION_TO_EMOTION.get(sec_idx, EmotionType.NEUTRAL),
                        sec_score,
                    )

            return self._build_result(primary_emotion, primary_confidence, secondary)

        except Exception as e:
            # Silently fail and use fallback - don't spam console
            return None

    def _get_face_bbox(
        self, frame: np.ndarray, face_landmarks
    ) -> Optional[tuple[int, int, int, int]]:
        """Extract face bounding box from landmarks."""
        if face_landmarks is None or len(face_landmarks) == 0:
            return None

        h, w = frame.shape[:2]

        # Get min/max coordinates from Landmark objects
        try:
            x_coords = [float(lm.x) * w for lm in face_landmarks]
            y_coords = [float(lm.y) * h for lm in face_landmarks]

            x1 = int(min(x_coords))
            y1 = int(min(y_coords))
            x2 = int(max(x_coords))
            y2 = int(max(y_coords))

            return (x1, y1, x2, y2)
        except (AttributeError, TypeError) as e:
            print(f"Error extracting face bbox: {e}")
            return None

    def _analyze_from_blendshapes(
        self, blendshapes: dict[str, float]
    ) -> EmotionResult:
        """Fallback: analyze from MediaPipe blendshapes."""
        # Extract key blendshape scores
        smile = (
            blendshapes.get("mouthSmileLeft", 0.0)
            + blendshapes.get("mouthSmileRight", 0.0)
        ) / 2
        frown = (
            blendshapes.get("mouthFrownLeft", 0.0)
            + blendshapes.get("mouthFrownRight", 0.0)
        ) / 2
        jaw_open = blendshapes.get("jawOpen", 0.0)
        brow_inner_up = blendshapes.get("browInnerUp", 0.0)
        brow_down = (
            blendshapes.get("browDownLeft", 0.0)
            + blendshapes.get("browDownRight", 0.0)
        ) / 2
        eye_wide = (
            blendshapes.get("eyeWideLeft", 0.0)
            + blendshapes.get("eyeWideRight", 0.0)
        ) / 2
        eye_squint = (
            blendshapes.get("eyeSquintLeft", 0.0)
            + blendshapes.get("eyeSquintRight", 0.0)
        ) / 2

        # Calculate emotion scores
        scores: list[tuple[EmotionType, float]] = []

        scores.append((EmotionType.JOY, smile * 0.8 + (1 - frown) * 0.2))
        scores.append((EmotionType.SADNESS, frown * 0.7 + (1 - smile) * 0.3))
        scores.append(
            (EmotionType.SURPRISE, brow_inner_up * 0.4 + eye_wide * 0.3 + jaw_open * 0.3)
        )
        scores.append(
            (EmotionType.ANGER, brow_down * 0.5 + eye_squint * 0.3 + (1 - smile) * 0.2)
        )

        scores.sort(key=lambda x: x[1], reverse=True)
        primary = scores[0]

        if primary[1] < 0.25:
            emotion = EmotionType.NEUTRAL
            confidence = 0.6
        else:
            emotion = primary[0]
            confidence = min(1.0, 0.5 + primary[1] * 0.5)

        return self._build_result(emotion, confidence)

    def get_state(self) -> dict[str, Any]:
        """Get current emotion state for LLM context."""
        state = super().get_state()
        if state.get("detected"):
            state["model"] = "cnn" if self._hsemotion else "blendshapes"
            if self._last_result and self._last_result.secondary_confidence:
                state["secondary_confidence"] = round(
                    self._last_result.secondary_confidence, 2
                )
        return state
