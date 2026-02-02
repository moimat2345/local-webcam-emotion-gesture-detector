"""Emotion detection analyzer using facial blendshapes from MediaPipe.

Contains:
- BaseEmotionAnalyzer: shared logic (smoothing, state, reset) for all emotion analyzers
- EmotionAnalyzer: blendshape-based detection with landmark fallback

Emotion scoring works by computing weighted sums of blendshape values
(e.g. smile + no frown = joy) and picking the highest-scoring emotion.
Results are smoothed over a sliding window to avoid flickering.
"""

from collections import Counter, deque
from typing import Any, Optional

import numpy as np

from src.base_analyzer import BaseAnalyzer
from src.data_models import EmotionResult, EmotionType, HolisticResult
from src.utils import landmark_distance


class BaseEmotionAnalyzer(BaseAnalyzer[EmotionResult]):
    """Shared logic for emotion analyzers: smoothing, state, reset."""

    def __init__(self, history_size: int = 10) -> None:
        self._last_result: Optional[EmotionResult] = None
        self._emotion_history: deque[EmotionType] = deque(maxlen=history_size)

    def _get_smoothed_emotion(self) -> EmotionType:
        """Get most common emotion from recent history."""
        if not self._emotion_history:
            return EmotionType.NEUTRAL
        counts = Counter(self._emotion_history)
        return counts.most_common(1)[0][0]

    def _build_result(
        self,
        emotion: EmotionType,
        confidence: float,
        secondary: Optional[tuple[EmotionType, float]] = None,
    ) -> EmotionResult:
        """Smooth emotion and build an EmotionResult."""
        self._emotion_history.append(emotion)
        smoothed_emotion = self._get_smoothed_emotion()
        result = EmotionResult(
            emotion=smoothed_emotion,
            confidence=confidence,
            secondary_emotion=secondary[0] if secondary else None,
            secondary_confidence=secondary[1] if secondary else None,
        )
        self._last_result = result
        return result

    def get_state(self) -> dict[str, Any]:
        """Get current emotion state for LLM context."""
        if self._last_result is None:
            return {"emotion": "unknown", "detected": False}
        state = {
            "emotion": self._last_result.emotion.value,
            "confidence": round(self._last_result.confidence, 2),
            "detected": True,
        }
        if self._last_result.secondary_emotion:
            state["secondary"] = self._last_result.secondary_emotion.value
        return state

    def reset(self) -> None:
        """Reset emotion history."""
        self._emotion_history.clear()
        self._last_result = None


class EmotionAnalyzer(BaseEmotionAnalyzer):
    """Detects emotions from facial blendshapes.

    Uses MediaPipe FaceMesh blendshapes (52 facial action units) for accurate
    emotion detection. Falls back to landmark-based heuristics if blendshapes
    are not available.
    """

    def __init__(self) -> None:
        super().__init__(history_size=10)

    def analyze(self, frame: np.ndarray, holistic: HolisticResult) -> Optional[EmotionResult]:
        """Analyze emotion from facial blendshapes or landmarks."""
        if not holistic.has_face:
            return None

        if holistic.face_blendshapes:
            emotion, confidence, secondary = self._analyze_from_blendshapes(
                holistic.face_blendshapes
            )
        elif holistic.face_landmarks:
            emotion, confidence, secondary = self._analyze_from_landmarks(
                holistic.face_landmarks
            )
        else:
            return None

        return self._build_result(emotion, confidence, secondary)

    def _analyze_from_blendshapes(
        self, blendshapes: dict[str, float]
    ) -> tuple[EmotionType, float, Optional[tuple[EmotionType, float]]]:
        """Analyze emotion from MediaPipe face blendshapes.

        Returns: (emotion, confidence, secondary_emotion_tuple or None)
        """
        # Extract key blendshape scores
        smile = (
            blendshapes.get("mouthSmileLeft", 0.0) + blendshapes.get("mouthSmileRight", 0.0)
        ) / 2
        frown = (
            blendshapes.get("mouthFrownLeft", 0.0) + blendshapes.get("mouthFrownRight", 0.0)
        ) / 2
        jaw_open = blendshapes.get("jawOpen", 0.0)
        brow_inner_up = blendshapes.get("browInnerUp", 0.0)
        brow_down = (
            blendshapes.get("browDownLeft", 0.0) + blendshapes.get("browDownRight", 0.0)
        ) / 2
        eye_wide = (
            blendshapes.get("eyeWideLeft", 0.0) + blendshapes.get("eyeWideRight", 0.0)
        ) / 2
        eye_squint = (
            blendshapes.get("eyeSquintLeft", 0.0) + blendshapes.get("eyeSquintRight", 0.0)
        ) / 2
        mouth_pucker = blendshapes.get("mouthPucker", 0.0)

        # Calculate emotion scores
        scores: list[tuple[EmotionType, float]] = []

        # Joy - strong smile
        joy_score = smile * 0.8 + (1 - frown) * 0.2
        scores.append((EmotionType.JOY, joy_score))

        # Sadness - frown, no smile
        sadness_score = frown * 0.7 + (1 - smile) * 0.3
        scores.append((EmotionType.SADNESS, sadness_score))

        # Surprise - raised brows, wide eyes, open jaw
        surprise_score = (brow_inner_up * 0.4 + eye_wide * 0.3 + jaw_open * 0.3)
        scores.append((EmotionType.SURPRISE, surprise_score))

        # Anger - brows down, eyes squinted, no smile
        anger_score = brow_down * 0.5 + eye_squint * 0.3 + (1 - smile) * 0.2
        scores.append((EmotionType.ANGER, anger_score))

        # Concentration - slight brow down, slight squint
        concentration_score = brow_down * 0.4 + eye_squint * 0.2 + (1 - smile) * 0.2 + (1 - jaw_open) * 0.2
        if concentration_score > 0.3 and concentration_score < 0.7:  # Mid-range
            scores.append((EmotionType.CONCENTRATION, concentration_score))
        else:
            scores.append((EmotionType.CONCENTRATION, concentration_score * 0.5))

        # Confusion - asymmetric brows, slight frown
        confusion_score = brow_inner_up * 0.3 + frown * 0.2 + mouth_pucker * 0.2
        scores.append((EmotionType.CONFUSION, confusion_score))

        # Disgust - nose wrinkle, mouth pucker
        disgust_score = mouth_pucker * 0.5 + eye_squint * 0.3
        scores.append((EmotionType.DISGUST, disgust_score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        primary = scores[0]
        secondary = None

        # If top score is very low, default to neutral
        if primary[1] < 0.25:
            return EmotionType.NEUTRAL, 0.7, None

        # Normalize confidence to 0.5-1.0 range
        confidence = min(1.0, 0.5 + primary[1] * 0.5)

        # Add secondary emotion if score is close to primary
        if len(scores) > 1 and scores[1][1] > 0.3 and (primary[1] - scores[1][1]) < 0.2:
            secondary = (scores[1][0], min(1.0, 0.4 + scores[1][1] * 0.4))

        return primary[0], confidence, secondary

    def _analyze_from_landmarks(
        self, landmarks
    ) -> tuple[EmotionType, float, Optional[tuple[EmotionType, float]]]:
        """Fallback: analyze from landmarks (less accurate).

        Returns: (emotion, confidence, secondary_emotion_tuple or None)
        """
        smile_score = self._calculate_smile_score(landmarks)
        eyebrow_score = self._calculate_eyebrow_score(landmarks)
        eye_openness = self._calculate_eye_openness(landmarks)

        emotion, confidence = self._determine_emotion(
            smile_score, eyebrow_score, eye_openness
        )

        secondary = None
        if confidence < 0.7:
            sec_result = self._get_secondary_emotion(
                smile_score, eyebrow_score, eye_openness
            )
            if sec_result[0] is not None:
                secondary = (sec_result[0], sec_result[1])

        return emotion, confidence, secondary

    def _calculate_smile_score(self, landmarks) -> float:
        """Calculate smile score from mouth landmarks (fallback method)."""
        # MediaPipe FaceMesh indices for mouth
        MOUTH_LEFT = 61
        MOUTH_RIGHT = 291
        MOUTH_TOP = 13
        MOUTH_BOTTOM = 14

        try:
            mouth_left = landmarks[MOUTH_LEFT]
            mouth_right = landmarks[MOUTH_RIGHT]
            mouth_top = landmarks[MOUTH_TOP]
            mouth_bottom = landmarks[MOUTH_BOTTOM]

            mouth_width = landmark_distance(mouth_left, mouth_right)
            mouth_height = landmark_distance(mouth_top, mouth_bottom)

            if mouth_height < 0.001:
                return 0.0

            ratio = mouth_width / mouth_height
            mouth_center_y = (mouth_left.y + mouth_right.y) / 2
            corner_lift = mouth_top.y - mouth_center_y

            score = min(1.0, max(0.0, (ratio - 2.0) / 3.0 + corner_lift * 10))
            return score
        except (IndexError, AttributeError):
            return 0.0

    def _calculate_eyebrow_score(self, landmarks) -> float:
        """Calculate eyebrow position (fallback method).

        Positive = raised, Negative = furrowed
        """
        LEFT_EYEBROW_INNER = 107
        RIGHT_EYEBROW_INNER = 336
        NOSE_TIP = 1

        try:
            left_brow = landmarks[LEFT_EYEBROW_INNER]
            right_brow = landmarks[RIGHT_EYEBROW_INNER]
            nose = landmarks[NOSE_TIP]

            avg_height = ((nose.y - left_brow.y) + (nose.y - right_brow.y)) / 2
            normalized = (avg_height - 0.15) / 0.1
            return max(-1.0, min(1.0, normalized))
        except (IndexError, AttributeError):
            return 0.0

    def _calculate_eye_openness(self, landmarks) -> float:
        """Calculate eye openness (fallback method)."""
        LEFT_EYE_TOP = 159
        LEFT_EYE_BOTTOM = 145
        RIGHT_EYE_TOP = 386
        RIGHT_EYE_BOTTOM = 374

        try:
            left_height = landmark_distance(landmarks[LEFT_EYE_TOP], landmarks[LEFT_EYE_BOTTOM])
            right_height = landmark_distance(landmarks[RIGHT_EYE_TOP], landmarks[RIGHT_EYE_BOTTOM])
            avg_height = (left_height + right_height) / 2
            return min(1.0, max(0.0, avg_height / 0.04))
        except (IndexError, AttributeError):
            return 0.5

    def _determine_emotion(
        self, smile: float, eyebrow: float, eye_open: float
    ) -> tuple[EmotionType, float]:
        """Determine primary emotion from facial metrics."""
        if smile > 0.5:
            confidence = min(1.0, 0.5 + smile * 0.5)
            return EmotionType.JOY, confidence

        if eyebrow > 0.5 and eye_open > 0.7:
            confidence = min(1.0, 0.5 + eyebrow * 0.3 + eye_open * 0.2)
            return EmotionType.SURPRISE, confidence

        if eyebrow < -0.3 and eye_open < 0.5:
            confidence = min(1.0, 0.5 + abs(eyebrow) * 0.3 + (1 - eye_open) * 0.2)
            return EmotionType.CONCENTRATION, confidence

        if eyebrow < -0.5:
            confidence = min(1.0, 0.5 + abs(eyebrow) * 0.5)
            return EmotionType.FRUSTRATION, confidence

        if eyebrow > 0.3 and smile < 0.2:
            confidence = min(1.0, 0.4 + eyebrow * 0.3)
            return EmotionType.CONFUSION, confidence

        return EmotionType.NEUTRAL, 0.6

    def _get_secondary_emotion(
        self, smile: float, eyebrow: float, eye_open: float
    ) -> tuple[Optional[EmotionType], Optional[float]]:
        """Get secondary emotion if primary confidence is low."""
        emotions = []

        if smile > 0.3:
            emotions.append((EmotionType.JOY, smile * 0.5))
        if eyebrow > 0.3:
            emotions.append((EmotionType.SURPRISE, eyebrow * 0.4))
        if eyebrow < -0.2:
            emotions.append((EmotionType.CONCENTRATION, abs(eyebrow) * 0.4))

        emotions.sort(key=lambda x: x[1], reverse=True)

        if len(emotions) > 1:
            return emotions[1][0], emotions[1][1]
        return None, None

