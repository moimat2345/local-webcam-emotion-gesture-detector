#!/usr/bin/env python3
"""Main entry point for the Vision Detector.

CustomTkinter GUI that controls the vision pipeline:
- Camera selection dropdown (auto-detected)
- Start/stop detection and visualization
- OpenCV window for real-time debug overlay
- Keyboard shortcuts: 'd'=detection, 'v'=visualization, 's'=screenshot, 'q'=quit

Threading model:
    [Main Thread]        - Tkinter GUI + OpenCV display (via root.after polling)
    [Detection Thread]   - Runs VisionPipeline.process_frame() in a loop
    [Capture Thread]     - Webcam capture (inside ThreadedCapture)

Frames flow: Detection Thread → Queue(maxsize=2) → Main Thread → cv2.imshow()
"""

import sys
import threading
from pathlib import Path
from queue import Queue, Empty
from typing import Optional

import cv2
import customtkinter as ctk

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline import VisionPipeline


# Configure CustomTkinter appearance
ctk.set_appearance_mode("dark")  # "dark" ou "light"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"


class VisionGUI:
    """Modern CustomTkinter GUI controller for vision module."""

    def __init__(self):
        self.pipeline: Optional[VisionPipeline] = None
        self.detection_running = False
        self.visualization_running = False
        self.detection_thread: Optional[threading.Thread] = None
        self.viz_window_name = "Vision Detector"
        self._viz_window_created = False
        self._frame_queue: Queue = Queue(maxsize=2)
        self.selected_camera_id = 0  # Default camera

        # Detect available cameras
        self.available_cameras = self._detect_cameras()

        # Create main window
        self.root = ctk.CTk()
        self.root.title("Vision Detector")
        self.root.geometry("900x700")
        self.root.minsize(500, 400)  # Minimum size

        # Configure grid layout (1x1 with padding)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Main container frame
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=15)
        self.main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Header
        self.header = ctk.CTkLabel(
            self.main_frame,
            text="Vision Detector",
            font=ctk.CTkFont(size=28, weight="bold"),
        )
        self.header.grid(row=0, column=0, pady=(20, 10), sticky="ew")

        # Subtitle
        self.subtitle = ctk.CTkLabel(
            self.main_frame,
            text="Détection temps réel : émotions, gestes, posture, présence",
            font=ctk.CTkFont(size=14),
            text_color=("gray60", "gray40"),
        )
        self.subtitle.grid(row=1, column=0, pady=(0, 20), sticky="ew")

        # Camera selection frame
        self.camera_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        self.camera_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        self.camera_label = ctk.CTkLabel(
            self.camera_frame,
            text="Camera:",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.camera_label.grid(row=0, column=0, padx=15, pady=10, sticky="w")

        # Camera dropdown
        camera_options = [f"{idx}: {name}" for idx, name in self.available_cameras]
        if not camera_options:
            camera_options = ["0: Default Camera"]

        self.camera_dropdown = ctk.CTkOptionMenu(
            self.camera_frame,
            values=camera_options,
            command=self._on_camera_selected,
            font=ctk.CTkFont(size=13),
            width=300,
        )
        self.camera_dropdown.grid(row=0, column=1, padx=15, pady=10, sticky="e")
        self.camera_dropdown.set(camera_options[0])
        self.camera_frame.grid_columnconfigure(1, weight=1)

        # Status frame
        self.status_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.status_frame.grid(row=3, column=0, pady=10, sticky="ew")
        self.status_frame.grid_columnconfigure(0, weight=1)

        # Detection status
        self.detection_status_frame = ctk.CTkFrame(self.status_frame, corner_radius=10)
        self.detection_status_frame.grid(row=0, column=0, padx=20, pady=5, sticky="ew")

        self.detection_label = ctk.CTkLabel(
            self.detection_status_frame,
            text="Detection",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.detection_label.grid(row=0, column=0, padx=15, pady=10, sticky="w")

        self.detection_indicator = ctk.CTkLabel(
            self.detection_status_frame,
            text="● STOPPED",
            font=ctk.CTkFont(size=14),
            text_color="red",
        )
        self.detection_indicator.grid(row=0, column=1, padx=15, pady=10, sticky="e")
        self.detection_status_frame.grid_columnconfigure(1, weight=1)

        # Visualization status
        self.viz_status_frame = ctk.CTkFrame(self.status_frame, corner_radius=10)
        self.viz_status_frame.grid(row=1, column=0, padx=20, pady=5, sticky="ew")

        self.viz_label = ctk.CTkLabel(
            self.viz_status_frame,
            text="Visualization",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.viz_label.grid(row=0, column=0, padx=15, pady=10, sticky="w")

        self.viz_indicator = ctk.CTkLabel(
            self.viz_status_frame,
            text="● STOPPED",
            font=ctk.CTkFont(size=14),
            text_color="red",
        )
        self.viz_indicator.grid(row=0, column=1, padx=15, pady=10, sticky="e")
        self.viz_status_frame.grid_columnconfigure(1, weight=1)

        # Buttons frame
        self.buttons_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.buttons_frame.grid(row=4, column=0, pady=20, sticky="ew")
        self.buttons_frame.grid_columnconfigure(0, weight=1)

        # Detection button
        self.detection_btn = ctk.CTkButton(
            self.buttons_frame,
            text="Start Detection",
            command=self.toggle_detection,
            height=50,
            corner_radius=10,
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.detection_btn.grid(row=0, column=0, padx=20, pady=5, sticky="ew")

        # Visualization button
        self.viz_btn = ctk.CTkButton(
            self.buttons_frame,
            text="Start Visualization",
            command=self.toggle_visualization,
            height=50,
            corner_radius=10,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="gray40",
            hover_color="gray30",
            state="disabled",
        )
        self.viz_btn.grid(row=1, column=0, padx=20, pady=5, sticky="ew")

        # Quit button
        self.quit_btn = ctk.CTkButton(
            self.buttons_frame,
            text="Quit",
            command=self.quit_app,
            height=50,
            corner_radius=10,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#e74c3c",
            hover_color="#c0392b",
        )
        self.quit_btn.grid(row=2, column=0, padx=20, pady=5, sticky="ew")

        # Footer info
        self.footer = ctk.CTkLabel(
            self.main_frame,
            text="Keyboard: 's'=screenshot | 'q'=quit | 'v'=toggle viz | 'd'=toggle detection",
            font=ctk.CTkFont(size=12),
            text_color=("gray50", "gray50"),
        )
        self.footer.grid(row=5, column=0, pady=(10, 20), sticky="ew")

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

        # Bind keyboard shortcuts to the main window
        self.root.bind('q', lambda e: self.quit_app())
        self.root.bind('d', lambda e: self.toggle_detection())
        self.root.bind('v', lambda e: self.toggle_visualization())

        self.running = True

    def _detect_cameras(self, max_cameras=10):
        """Detect available cameras on the system."""
        available = []
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to get camera name/info
                backend = cap.getBackendName()
                # Get frame to confirm it works
                ret, _ = cap.read()
                if ret:
                    available.append((i, f"Camera {i} ({backend})"))
                cap.release()
            else:
                # Stop searching after first unavailable camera
                if i > 0 and not available:
                    break
                elif i > 0:
                    # Continue a bit more in case there are gaps
                    if i - available[-1][0] > 2:
                        break

        return available if available else [(0, "Default Camera")]

    def _on_camera_selected(self, selection: str):
        """Handle camera selection change."""
        # Extract camera ID from selection (format: "0: Camera 0 (backend)")
        camera_id = int(selection.split(":")[0])
        self.selected_camera_id = camera_id
        print(f"Camera selected: {selection}")

        # If detection is running, restart it with new camera
        if self.detection_running:
            print("Restarting detection with new camera...")
            self.toggle_detection()  # Stop
            self.toggle_detection()  # Start with new camera

    def update_ui(self):
        """Update UI labels and buttons."""
        if self.detection_running:
            self.detection_indicator.configure(text="● ACTIVE", text_color="#2ecc71")
            self.detection_btn.configure(
                text="Stop Detection",
                fg_color="#e67e22",
                hover_color="#d35400",
            )
            self.viz_btn.configure(state="normal", fg_color="#2ecc71", hover_color="#27ae60")
            self.camera_dropdown.configure(state="disabled")
        else:
            self.detection_indicator.configure(text="● STOPPED", text_color="#e74c3c")
            self.detection_btn.configure(
                text="Start Detection",
                fg_color=("#3B8ED0", "#1F6AA5"),
                hover_color=("#36719F", "#144870"),
            )
            self.viz_btn.configure(state="disabled", fg_color="gray40", hover_color="gray30")
            self.camera_dropdown.configure(state="normal")

        if self.visualization_running:
            self.viz_indicator.configure(text="● ACTIVE", text_color="#2ecc71")
            self.viz_btn.configure(
                text="Stop Visualization",
                fg_color="#e67e22",
                hover_color="#d35400",
            )
        else:
            self.viz_indicator.configure(text="● STOPPED", text_color="#e74c3c")
            if self.detection_running:
                self.viz_btn.configure(
                    text="Start Visualization",
                    fg_color="#2ecc71",
                    hover_color="#27ae60",
                )

    def toggle_detection(self) -> None:
        """Start or stop the detection pipeline."""
        if not self.detection_running:
            # Start detection
            print(f"Starting detection with camera {self.selected_camera_id}...")
            config_path = Path("config/config.yaml")
            self.pipeline = VisionPipeline(config_path=config_path)

            # Override camera device ID with selected camera
            self.pipeline._capture._settings.device_id = self.selected_camera_id

            def on_frame(result):
                if result.frame_number % 30 == 0:
                    state = result.to_llm_context()
                    if state.get("presence") == "present":
                        emotion = state.get("emotion", {}).get("primary", "unknown")
                        print(f"Frame {result.frame_number}: emotion={emotion}")

            self.pipeline.add_callback(on_frame)

            # Reset window flag
            self._viz_window_created = False

            # Run detection in background thread
            self.detection_running = True
            self.detection_thread = threading.Thread(target=self._run_detection, daemon=True)
            self.detection_thread.start()
            print("Detection started!")
            self.update_ui()
        else:
            # Stop detection
            print("Stopping detection...")

            # Close visualization first (before stopping detection flag)
            if self.visualization_running:
                print("Closing visualization window...")
                self.visualization_running = False
                if self._viz_window_created:
                    try:
                        cv2.destroyWindow(self.viz_window_name)
                    except:
                        pass
                    self._viz_window_created = False
                # Clear queue
                while not self._frame_queue.empty():
                    try:
                        self._frame_queue.get_nowait()
                    except:
                        break
                print("Visualization closed!")

            self.detection_running = False

            # Wait for detection thread to finish
            if self.detection_thread and self.detection_thread.is_alive():
                self.detection_thread.join(timeout=2.0)

            if self.pipeline:
                try:
                    self.pipeline.stop()
                except Exception as e:
                    print(f"Warning during pipeline stop: {e}")
                self.pipeline = None

            # Reset window flag
            self._viz_window_created = False
            print("Detection stopped!")
            self.update_ui()

    def _run_detection(self) -> None:
        """Run detection loop in background."""
        if self.pipeline:
            try:
                self.pipeline.start()

                while self.detection_running and self.running:
                    # Use process_frame() which returns (rendered_frame, analysis_result) or None
                    result = self.pipeline.process_frame()
                    if result is None:
                        continue

                    rendered_frame, analysis = result

                    # If visualization is active, put frame in queue for main thread to display
                    if self.visualization_running:
                        try:
                            # Non-blocking put - drop frame if queue is full
                            self._frame_queue.put_nowait((rendered_frame, analysis))
                        except:
                            pass  # Queue full, skip this frame
            except Exception as e:
                print(f"Detection error: {e}")
                import traceback
                traceback.print_exc()

    def toggle_visualization(self) -> None:
        """Toggle visualization window."""
        if not self.detection_running:
            print("Cannot start visualization: detection is not running!")
            return

        if not self.visualization_running:
            # Create OpenCV window in main thread
            print("Opening visualization window...")
            cv2.namedWindow(self.viz_window_name, cv2.WINDOW_NORMAL)
            self._viz_window_created = True
            self.visualization_running = True
            # Start polling for frames
            self._update_visualization()
            print("Visualization active!")
            self.update_ui()
        else:
            # Close visualization window
            print("Closing visualization window...")
            self.visualization_running = False
            if self._viz_window_created:
                try:
                    cv2.destroyWindow(self.viz_window_name)
                except:
                    pass
                self._viz_window_created = False
            # Clear queue
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except:
                    break
            print("Visualization closed!")
            self.update_ui()

    def _update_visualization(self) -> None:
        """Poll for frames from detection thread and display them (runs in main thread)."""
        if not self.visualization_running or not self.running:
            return

        try:
            # Try to get a frame from the queue (non-blocking)
            rendered_frame, analysis = self._frame_queue.get_nowait()
            self._last_displayed_frame = rendered_frame
            self._last_displayed_analysis = analysis

            # Display frame
            cv2.imshow(self.viz_window_name, rendered_frame)
        except Empty:
            # No frame available, just process OpenCV events
            pass

        # Handle keyboard input (always check, not just when frame available)
        key = cv2.waitKey(1) & 0xFF
        last_frame = getattr(self, '_last_displayed_frame', None)
        if key == ord('s') and last_frame is not None:
            analysis = self._last_displayed_analysis
            filename = f"screenshot_{analysis.frame_number}.png"
            cv2.imwrite(filename, last_frame)
            print(f"Screenshot saved: {filename}")
        elif key == ord('q'):
            # Quit the entire application
            self.quit_app()
            return
        elif key == ord('v'):
            # Toggle visualization
            self.toggle_visualization()
            return
        elif key == ord('d'):
            # Toggle detection
            self.toggle_detection()
            return

        # Schedule next update (30 FPS = ~33ms)
        if self.visualization_running:
            self.root.after(33, self._update_visualization)

    def quit_app(self) -> None:
        """Clean quit the application."""
        print("\nQuitting...")
        self.running = False

        if self.detection_running:
            self.detection_running = False

        if self.visualization_running:
            self.visualization_running = False
            try:
                cv2.destroyAllWindows()
            except:
                pass

        if self.pipeline:
            print(f"Total frames processed: {self.pipeline.frame_count}")
            try:
                self.pipeline.stop()
            except Exception as e:
                print(f"Warning during pipeline stop: {e}")

        # Force quit the application
        try:
            self.root.quit()
        except:
            pass

        try:
            self.root.destroy()
        except:
            pass

        # Force exit if GUI doesn't close properly
        import sys
        sys.exit(0)

    def usage(self) -> None:
        """Print usage instructions."""
        print("=== Vision Detector ===")
        print("Keyboard Shortcuts:")
        print("  's' - Save screenshot of current frame")
        print("  'q' - Quit application")
        print("  'v' - Toggle visualization window")
        print("  'd' - Toggle detection on/off")
        print("==============================")

    def run(self) -> int:
        self.usage()
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            self.quit_app()

        return 0


def main() -> int:
    """Run the GUI."""
    # Download models before starting the GUI
    from src.holistic_analyzer import _ensure_model, FACE_MODEL_URL, POSE_MODEL_URL, HAND_MODEL_URL
    print("Checking models...")
    _ensure_model("face_landmarker.task", FACE_MODEL_URL)
    _ensure_model("pose_landmarker_heavy.task", POSE_MODEL_URL)
    _ensure_model("hand_landmarker.task", HAND_MODEL_URL)
    print("Models ready.")

    gui = VisionGUI()
    return gui.run()


if __name__ == "__main__":
    sys.exit(main())
