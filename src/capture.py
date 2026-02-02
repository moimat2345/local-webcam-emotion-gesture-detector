"""Threaded webcam capture for non-blocking video acquisition.

Runs a background thread that continuously reads frames from the webcam
and stores them in a bounded circular buffer (deque). The main thread
can read the latest frame at any time without blocking.

Threading model:
    [Capture Thread] --write--> [deque buffer] <--read-- [Main Thread]
                                  (with lock)
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.settings import CameraSettings


@dataclass
class CaptureStats:
    """Statistics about capture performance."""

    fps: float
    dropped_frames: int
    buffer_size: int
    is_running: bool


class ThreadedCapture:
    """Thread-safe webcam capture with frame buffering.

    Captures frames in a separate thread to prevent blocking the main loop.
    Uses a bounded deque as a circular buffer for frames.
    """

    def __init__(self, settings: Optional[CameraSettings] = None) -> None:
        self._settings = settings or CameraSettings()
        self._buffer: deque[np.ndarray] = deque(maxlen=self._settings.buffer_size)
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None

        self._frame_count = 0
        self._dropped_frames = 0
        self._start_time = 0.0
        # DEAD CODE: self._last_frame_time = 0.0

    def start(self) -> bool:
        """Start the capture thread.

        Returns:
            True if camera opened successfully, False otherwise.
        """
        self._cap = cv2.VideoCapture(self._settings.device_id)

        if not self._cap.isOpened():
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._settings.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._settings.height)
        self._cap.set(cv2.CAP_PROP_FPS, self._settings.fps)
        # OpenCV internal buffer = 1 to always get the latest frame
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._stopped.clear()
        self._start_time = time.time()
        self._frame_count = 0
        self._dropped_frames = 0

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        return True

    def stop(self) -> None:
        """Stop the capture thread and release resources."""
        self._stopped.set()

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        with self._lock:
            self._buffer.clear()

    def read(self) -> Optional[np.ndarray]:
        """Get the most recent frame from the buffer.

        Returns:
            The latest frame, or None if buffer is empty.
        """
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1].copy()

    # DEAD CODE: read_all() - never called
    # def read_all(self) -> list[np.ndarray]:
    #     """Get all frames currently in the buffer."""
    #     with self._lock:
    #         return [frame.copy() for frame in self._buffer]

    def get_stats(self) -> CaptureStats:
        """Get capture performance statistics."""
        elapsed = time.time() - self._start_time if self._start_time > 0 else 1.0
        fps = self._frame_count / elapsed if elapsed > 0 else 0.0

        with self._lock:
            buffer_size = len(self._buffer)

        return CaptureStats(
            fps=round(fps, 1),
            dropped_frames=self._dropped_frames,
            buffer_size=buffer_size,
            is_running=not self._stopped.is_set(),
        )

    @property
    def is_running(self) -> bool:
        """Check if capture is currently running."""
        return not self._stopped.is_set()

    # DEAD CODE: frame_size property - never accessed
    # @property
    # def frame_size(self) -> tuple[int, int]:
    #     """Get the frame dimensions (width, height)."""
    #     return self._settings.width, self._settings.height

    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread."""
        target_interval = 1.0 / self._settings.fps

        while not self._stopped.is_set():
            loop_start = time.time()

            if self._cap is None or not self._cap.isOpened():
                break

            ret, frame = self._cap.read()

            if ret and frame is not None:
                with self._lock:
                    # When buffer is full, oldest frame is auto-evicted by deque
                    if len(self._buffer) == self._buffer.maxlen:
                        self._dropped_frames += 1
                    self._buffer.append(frame)

                self._frame_count += 1
                # DEAD CODE: self._last_frame_time = time.time()
            else:
                time.sleep(0.001)
                continue

            # Throttle to target FPS to avoid consuming excessive CPU
            elapsed = time.time() - loop_start
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def __enter__(self) -> "ThreadedCapture":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.stop()
