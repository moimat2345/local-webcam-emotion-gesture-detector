"""Vision pipeline orchestrator - coordinates capture and analysis.

Main entry point for the processing chain:
Frame → Capture (threaded) → HolisticAnalyzer → Analyzers → DebugRenderer → Output

Usage:
    pipeline = VisionPipeline(config_path=Path("config/config.yaml"))
    pipeline.start()
    while True:
        result = pipeline.process_frame()  # Returns (rendered_frame, analysis)
    pipeline.stop()
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

# import cv2  # DEAD CODE: only used by commented run_with_display()
import numpy as np

from src.emotion_analyzer_cnn import EmotionAnalyzerCNN
from src.gesture_analyzer import GestureAnalyzer
from src.holistic_analyzer import HolisticAnalyzer
from src.posture_analyzer import PostureAnalyzer
from src.presence_analyzer import PresenceAnalyzer
from src.settings import Settings
from src.capture import ThreadedCapture
from src.data_models import FrameAnalysisResult
from src.debug_renderer import DebugRenderer


class VisionPipeline:
    """Main orchestrator for the vision processing pipeline.

    Coordinates:
    - Threaded webcam capture
    - MediaPipe holistic detection
    - All analyzers (emotion, presence, gesture, posture)
    - Debug visualization rendering
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        if config_path and config_path.exists():
            self._settings = Settings.from_yaml(config_path)
        else:
            self._settings = settings or Settings()

        self._vision_settings = self._settings.vision

        self._capture = ThreadedCapture(self._vision_settings.camera)
        self._holistic = HolisticAnalyzer(self._vision_settings.mediapipe)
        self._presence = PresenceAnalyzer(self._vision_settings.presence)
        self._emotion = EmotionAnalyzerCNN()
        self._gesture = GestureAnalyzer()
        self._posture = PostureAnalyzer(self._vision_settings.posture)
        self._renderer = DebugRenderer(self._vision_settings.visualization)

        self._frame_count = 0
        self._is_running = False
        self._callbacks: list[Callable[[FrameAnalysisResult], None]] = []

    def start(self) -> bool:
        """Start the capture thread.

        Returns:
            True if camera started successfully.
        """
        # Reset all analyzers to clear state from previous runs
        self._holistic.reset()
        self._emotion.reset()
        self._presence.reset()
        self._gesture.reset()
        self._posture.reset()

        success = self._capture.start()
        if success:
            self._is_running = True
        return success

    def stop(self) -> None:
        """Stop the pipeline and release resources."""
        self._is_running = False
        try:
            self._capture.stop()
        except Exception as e:
            print(f"Warning: Error stopping capture: {e}")

        try:
            if self._holistic:
                self._holistic.close()
        except Exception as e:
            print(f"Warning: Error closing holistic analyzer: {e}")

    def process_frame(self) -> Optional[tuple[np.ndarray, FrameAnalysisResult]]:
        """Process a single frame through the entire pipeline.

        Returns:
            Tuple of (rendered_frame, analysis_result) or None if no frame available.
        """
        frame = self._capture.read()
        if frame is None:
            return None

        start_time = time.time()

        # Step 1: Run MediaPipe detection (face, hands, pose)
        holistic_result = self._holistic.analyze(frame)

        # Step 2: Run all analyzers in sequence using holistic results
        presence_result = self._presence.analyze(frame, holistic_result)
        emotion_result = self._emotion.analyze(frame, holistic_result)
        left_gesture, right_gesture = self._gesture.analyze(frame, holistic_result)
        posture_result = self._posture.analyze(frame, holistic_result)

        processing_time_ms = (time.time() - start_time) * 1000

        analysis_result = FrameAnalysisResult(
            holistic=holistic_result,
            emotion=emotion_result,
            presence=presence_result,
            left_gesture=left_gesture,
            right_gesture=right_gesture,
            posture=posture_result,
            frame_number=self._frame_count,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.now(),
        )

        self._frame_count += 1

        # Step 3: Notify registered callbacks (e.g. GUI status updates)
        for callback in self._callbacks:
            try:
                callback(analysis_result)
            except Exception:
                pass

        # Step 4: Render debug overlay (landmarks, info text)
        rendered_frame = self._renderer.render(frame, analysis_result)

        return rendered_frame, analysis_result

    def add_callback(self, callback: Callable[[FrameAnalysisResult], None]) -> None:
        """Add a callback to be called after each frame is processed."""
        self._callbacks.append(callback)

    # DEAD CODE: remove_callback() - never called
    # def remove_callback(self, callback: Callable[[FrameAnalysisResult], None]) -> None:
    #     """Remove a previously added callback."""
    #     if callback in self._callbacks:
    #         self._callbacks.remove(callback)

    # DEAD CODE: run_with_display() - never called (GUI uses different approach)
    # def run_with_display(self, window_name: str = "Vision Detector") -> None:
    #     """Run the pipeline with OpenCV window display."""
    #     if not self.start():
    #         print("Failed to start camera!")
    #         return
    #     print(f"Vision pipeline started. Press 'q' to quit.")
    #     print(f"Window: {window_name}")
    #     try:
    #         while self._is_running:
    #             result = self.process_frame()
    #             if result is None:
    #                 time.sleep(0.01)
    #                 continue
    #             rendered_frame, analysis = result
    #             cv2.imshow(window_name, rendered_frame)
    #             key = cv2.waitKey(1) & 0xFF
    #             if key == ord("q"):
    #                 break
    #             elif key == ord("s"):
    #                 filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    #                 cv2.imwrite(filename, rendered_frame)
    #                 print(f"Screenshot saved: {filename}")
    #     finally:
    #         self.stop()
    #         cv2.destroyAllWindows()

    # DEAD CODE: get_current_state() - never called
    # def get_current_state(self) -> dict:
    #     """Get aggregated state from all analyzers for LLM context."""
    #     return {
    #         "holistic": self._holistic.get_state(),
    #         "presence": self._presence.get_state(),
    #         "emotion": self._emotion.get_state(),
    #         "gesture": self._gesture.get_state(),
    #         "posture": self._posture.get_state(),
    #         "capture": {
    #             "fps": self._capture.get_stats().fps,
    #             "running": self._capture.is_running,
    #         },
    #     }

    @property
    def is_running(self) -> bool:
        """Check if pipeline is currently running."""
        return self._is_running

    @property
    def frame_count(self) -> int:
        """Get total frames processed."""
        return self._frame_count

    def __enter__(self) -> "VisionPipeline":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()
