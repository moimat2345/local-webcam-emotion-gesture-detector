"""Debug visualization renderer for drawing landmarks and info overlay.

Draws MediaPipe landmarks and analysis results on top of the webcam frame.
All landmark indices follow the MediaPipe conventions:
- Face: 468 landmarks (FaceMesh topology)
- Hands: 21 landmarks per hand (wrist + 4 joints per finger)
- Pose: 33 landmarks (full body skeleton)

Reference: https://ai.google.dev/edge/mediapipe/solutions/vision
"""

from typing import Optional

import cv2
import numpy as np

from src.settings import VisualizationSettings
from src.data_models import (
    EmotionResult,
    FrameAnalysisResult,
    GestureResult,
    GestureType,
    HolisticResult,
    Landmark,
    PostureResult,
    PresenceResult,
    PresenceState,
)


class DebugRenderer:
    """Renders debug visualization with landmarks and info overlays.

    Draws:
    - Face mesh landmarks (green)
    - Hand landmarks with connections (blue/red)
    - Pose skeleton (cyan)
    - Text overlay with emotion, presence, gestures, posture
    """

    # Face oval contour - subset of FaceMesh tesselation for lightweight rendering
    FACE_TESSELATION_INDICES = [
        (10, 338), (338, 297), (297, 332), (332, 284),
        (284, 251), (251, 389), (389, 356), (356, 454),
        (454, 323), (323, 361), (361, 288), (288, 397),
        (397, 365), (365, 379), (379, 378), (378, 400),
        (400, 377), (377, 152), (152, 148), (148, 176),
        (176, 149), (149, 150), (150, 136), (136, 172),
        (172, 58), (58, 132), (132, 93), (93, 234),
        (234, 127), (127, 162), (162, 21), (21, 54),
        (54, 103), (103, 67), (67, 109), (109, 10),
    ]

    # Hand skeleton: wrist(0) → thumb(1-4), index(5-8), middle(9-12), ring(13-16), pinky(17-20)
    # Last 3 entries connect the finger bases across the palm
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),       # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17),             # Palm cross-connections
    ]

    # Pose skeleton connections (upper + lower body)
    # 11/12=shoulders, 13/14=elbows, 15/16=wrists, 23/24=hips, 25-32=legs
    POSE_CONNECTIONS = [
        (11, 12),                              # Shoulder line
        (11, 13), (13, 15),                    # Left arm
        (12, 14), (14, 16),                    # Right arm
        (11, 23), (12, 24),                    # Torso sides
        (23, 24),                              # Hip line
        (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
        (24, 26), (26, 28), (28, 30), (30, 32),  # Right leg
    ]

    def __init__(self, settings: Optional[VisualizationSettings] = None) -> None:
        self._settings = settings or VisualizationSettings()

    def render(
        self,
        frame: np.ndarray,
        result: FrameAnalysisResult,
    ) -> np.ndarray:
        """Render all debug visualizations on the frame.

        Args:
            frame: The video frame to draw on (will be modified in place).
            result: The complete frame analysis result.

        Returns:
            The frame with visualizations drawn.
        """
        if not self._settings.enabled:
            return frame

        output = frame.copy()

        if self._settings.show_face_landmarks and result.holistic.has_face:
            self._draw_face_landmarks(output, result.holistic.face_landmarks)

        if self._settings.show_hand_landmarks:
            if result.holistic.has_left_hand:
                self._draw_hand_landmarks(
                    output,
                    result.holistic.left_hand_landmarks,
                    self._settings.left_hand_color,
                )
            if result.holistic.has_right_hand:
                self._draw_hand_landmarks(
                    output,
                    result.holistic.right_hand_landmarks,
                    self._settings.right_hand_color,
                )

        if self._settings.show_pose_landmarks and result.holistic.has_pose:
            self._draw_pose_landmarks(output, result.holistic.pose_landmarks)

        if self._settings.show_info_overlay:
            self._draw_info_overlay(output, result)

        return output

    def _draw_face_landmarks(
        self,
        frame: np.ndarray,
        landmarks: list[Landmark],
    ) -> None:
        """Draw face mesh landmarks with more detail."""
        h, w = frame.shape[:2]
        color = self._settings.face_color

        # Draw face contour lines
        for idx1, idx2 in self.FACE_TESSELATION_INDICES:
            if idx1 < len(landmarks) and idx2 < len(landmarks):
                pt1 = self._landmark_to_pixel(landmarks[idx1], w, h)
                pt2 = self._landmark_to_pixel(landmarks[idx2], w, h)
                cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA)

        # Draw key facial features with more points
        # Eyes (contour + pupils)
        left_eye = [33, 133, 160, 159, 158, 157, 173, 246]  # Left eye contour
        right_eye = [263, 362, 387, 386, 385, 384, 398, 466]  # Right eye contour

        # Eyebrows
        left_eyebrow = [70, 63, 105, 66, 107, 55, 193]  # Left eyebrow
        right_eyebrow = [300, 293, 334, 296, 336, 285, 417]  # Right eyebrow

        # Mouth (outer contour)
        mouth = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]  # Mouth contour

        # Nose
        nose = [1, 2, 98, 327]  # Nose tip and bridge

        # Face oval
        face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                     397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                     172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

        # Draw all feature points
        all_feature_points = (left_eye + right_eye + left_eyebrow + right_eyebrow +
                             mouth + nose + face_oval)

        for idx in all_feature_points:
            if idx < len(landmarks):
                pt = self._landmark_to_pixel(landmarks[idx], w, h)
                # Bigger circles for eye/mouth/nose, smaller for others
                radius = 3 if idx in (left_eye + right_eye + mouth + nose) else 2
                cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)

    def _draw_hand_landmarks(
        self,
        frame: np.ndarray,
        landmarks: list[Landmark],
        color: tuple[int, int, int],
    ) -> None:
        """Draw hand landmarks with connections."""
        h, w = frame.shape[:2]

        for idx1, idx2 in self.HAND_CONNECTIONS:
            if idx1 < len(landmarks) and idx2 < len(landmarks):
                pt1 = self._landmark_to_pixel(landmarks[idx1], w, h)
                pt2 = self._landmark_to_pixel(landmarks[idx2], w, h)
                cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

        for i, lm in enumerate(landmarks):
            pt = self._landmark_to_pixel(lm, w, h)
            # Fingertips (4, 8, 12, 16, 20) get larger circles
            radius = 5 if i in [4, 8, 12, 16, 20] else 3
            cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)

    def _draw_pose_landmarks(
        self,
        frame: np.ndarray,
        landmarks: list[Landmark],
    ) -> None:
        """Draw pose skeleton."""
        h, w = frame.shape[:2]
        color = self._settings.pose_color

        for idx1, idx2 in self.POSE_CONNECTIONS:
            if idx1 < len(landmarks) and idx2 < len(landmarks):
                lm1 = landmarks[idx1]
                lm2 = landmarks[idx2]

                # Skip joints with low visibility confidence
                if (lm1.visibility or 0) < 0.5 or (lm2.visibility or 0) < 0.5:
                    continue

                pt1 = self._landmark_to_pixel(lm1, w, h)
                pt2 = self._landmark_to_pixel(lm2, w, h)
                cv2.line(frame, pt1, pt2, color, 3, cv2.LINE_AA)

        visible_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        for idx in visible_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                if (lm.visibility or 0) >= 0.5:
                    pt = self._landmark_to_pixel(lm, w, h)
                    cv2.circle(frame, pt, 6, color, -1, cv2.LINE_AA)

    def _draw_info_overlay(
        self,
        frame: np.ndarray,
        result: FrameAnalysisResult,
    ) -> None:
        """Draw text overlay with detailed analysis results."""
        h, w = frame.shape[:2]

        # Semi-transparent black background for text readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 220), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        y_offset = 35
        line_height = 28

        if result.presence:
            presence_text = f"Presence: {result.presence.state.value.upper()}"
            presence_color = (0, 255, 0) if result.presence.state == PresenceState.PRESENT else (0, 165, 255)
            cv2.putText(frame, presence_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, presence_color, 2)
            y_offset += line_height

        if result.emotion:
            # Primary emotion
            emotion_text = f"Emotion: {result.emotion.emotion.value.upper()} ({result.emotion.confidence:.0%})"
            cv2.putText(frame, emotion_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += line_height

            # Secondary emotion if present
            if result.emotion.secondary_emotion and result.emotion.secondary_confidence:
                secondary_text = f"  Secondary: {result.emotion.secondary_emotion.value} ({result.emotion.secondary_confidence:.0%})"
                cv2.putText(frame, secondary_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                y_offset += line_height - 5

        gesture_parts = []
        if result.left_gesture and result.left_gesture.gesture != GestureType.NONE:
            gesture_parts.append(f"L: {result.left_gesture.gesture.value}")
        if result.right_gesture and result.right_gesture.gesture != GestureType.NONE:
            gesture_parts.append(f"R: {result.right_gesture.gesture.value}")

        if gesture_parts:
            gesture_text = "Gestures: " + ", ".join(gesture_parts)
            cv2.putText(frame, gesture_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += line_height

        if result.posture:
            posture_color = (0, 0, 255) if result.posture.is_slouching else (0, 255, 0)
            posture_text = f"Posture: {result.posture.slouch_score:.0%} slouch"
            if result.posture.is_slouching:
                posture_text += " - REDRESSE-TOI!"
            cv2.putText(frame, posture_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, posture_color, 2)
            y_offset += line_height

        fps_text = f"Processing: {result.processing_time_ms:.1f}ms"
        cv2.putText(frame, fps_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

    @staticmethod
    def _landmark_to_pixel(
        landmark: Landmark,
        width: int,
        height: int,
    ) -> tuple[int, int]:
        """Convert normalized landmark to pixel coordinates."""
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        return (x, y)
