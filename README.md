<div align="center">

# Local Webcam Emotion & Gesture Detector

**Turn your webcam into an intelligent sensor — entirely offline.**

Real-time emotion recognition, gesture detection, posture analysis and presence tracking,
powered by MediaPipe and a CNN emotion model. No cloud. No telemetry. Just your camera.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-macOS-000000?logo=apple&logoColor=white)
![Status](https://img.shields.io/badge/Status-Alpha-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-00A67E?logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-Private-red)

<a href="README.fr.md"><img src="https://img.shields.io/badge/%F0%9F%87%AB%F0%9F%87%B7_Lire_en_fran%C3%A7ais-blue?style=for-the-badge" alt="Lire en francais"></a>

</div>

---

## What is It

Local Webcam Emotion & Gesture Detector is a **fully local, real-time computer vision application** that turns your webcam into an intelligent sensor. Built on top of **Google's MediaPipe** framework and a **CNN emotion model (HSEmotion/ONNX Runtime)**, it processes every frame through a multi-stage analysis pipeline to extract rich behavioral data — all without sending a single byte to the cloud.

### What it detects

| Capability | Engine | Output |
|:-----------|:-------|:-------|
| **Emotion Recognition** | HSEmotion CNN (8 classes) | Primary + secondary emotion with confidence scores, temporal smoothing across frames |
| **Presence Detection** | MediaPipe Face Landmarker | 3-state machine (`PRESENT` → `AWAY` → `RETURNING`) with configurable timeouts |
| **Gesture Recognition** | MediaPipe Hand Landmarker | 9 gestures per hand (thumbs up/down, peace, fist, pointing, OK, wave, open palm) + pointing direction |
| **Posture Analysis** | MediaPipe Pose Landmarker | Neck forward angle, shoulder alignment, weighted slouch score (0–1) with alerts |

### Why it exists

This project was designed as a **local-first alternative** to cloud-based behavioral analysis tools. Typical use cases include:

- **Developer ergonomics** — get posture alerts while coding for extended sessions
- **Accessibility research** — gesture and presence tracking for hands-free interaction prototypes
- **Emotion-aware interfaces** — feed real-time emotional state into other applications via callbacks
- **Learning and experimentation** — explore MediaPipe, real-time CV pipelines, and threaded capture architectures

### Design principles

- **Privacy by default** — all processing happens on-device; no network calls, no telemetry, no data persistence
- **Single inference pass** — MediaPipe runs once per frame (face + hands + pose), and all analyzers share the result
- **Non-blocking architecture** — webcam capture runs in a dedicated thread with a circular buffer, keeping the GUI responsive
- **Configurable everything** — detection thresholds, FPS targets, visualization colors, and alert behavior are all controlled via `config/config.yaml`

---

## At a Glance

| Capability | Engine | What you get |
|:-----------|:-------|:-------------|
| **Emotion Recognition** | HSEmotion CNN (8 classes) | Primary + secondary emotion, confidence scores, temporal smoothing |
| **Presence Detection** | MediaPipe Face Landmarker | 3-state machine: `PRESENT` → `AWAY` → `RETURNING` |
| **Gesture Recognition** | MediaPipe Hand Landmarker | 9 gestures per hand + pointing direction |
| **Posture Analysis** | MediaPipe Pose Landmarker | Neck angle, shoulder alignment, slouch score (0–1) |

> Every frame goes through a single MediaPipe inference pass shared by all analyzers — keeping latency under 40 ms on a MacBook Pro M1.

---

## Why Local Webcam Emotion & Gesture Detector?

- **Privacy by default** — all processing on-device; nothing leaves your machine, ever.
- **Non-blocking** — webcam capture runs in its own thread with a circular buffer; the GUI stays responsive.
- **Configurable** — thresholds, FPS, colors, alerts: everything lives in `config/config.yaml`.

Built for developers who want posture alerts while coding, researchers prototyping hands-free interaction, or anyone curious about real-time computer vision pipelines.

---

## How It Works

```
┌─────────────┐     ┌───────────────────┐     ┌─────────────────────────────────┐     ┌───────────────┐
│   Webcam    │────▶│  ThreadedCapture  │────▶│        HolisticAnalyzer         │────▶│ DebugRenderer │
│  (camera)   │     │  (background      │     │  (MediaPipe: face 478pts,       │     │ (overlay on   │
│             │     │  thread + buffer) │     │   hands 21pts, pose 33pts)      │     │  video feed)  │
└─────────────┘     └───────────────────┘     └──────────┬──────────────────────┘     └───────┬───────┘
                                                         │                                    │
                                               ┌─────────┴─────────┐                          │
                                               │  Holistic Results │                          │
                                               └─────────┬─────────┘                          │
                                                         │                                    │
                              ┌──────────────────────────┼──────────────────────────┐         │
                              │                          │                          │         │
                    ┌─────────▼─────────┐   ┌────────────▼──────────┐    ┌──────────▼───────┐ │
                    │  EmotionCNN       │   │  GestureAnalyzer      │    │ PostureAnalyzer  │ │
                    │  8 emotions       │   │  9 gestures per hand  │    │ neck angle,      │ │
                    │  (HSEmotion ONNX) │   │  + pointing direction │    │ slouch score     │ │
                    └─────────┬─────────┘   └────────────┬──────────┘    └──────────┬───────┘ │
                              │                          │                          │         │
                    ┌─────────▼─────────┐                │                          │         │
                    │ PresenceAnalyzer  │                │                          │         │
                    │ PRESENT / AWAY /  │                │                          │         │
                    │ RETURNING         │                │                          │         │
                    └─────────┬─────────┘                │                          │         │
                              │                          │                          │         │
                              └──────────────────────────┼──────────────────────────┘         │
                                                         │                                    │
                                              ┌──────────▼──────────┐                         │
                                              │ FrameAnalysisResult │─────────────────────────┘
                                              │ (Pydantic model)    │
                                              └──────────┬──────────┘
                                                         │
                                              ┌──────────▼──────────┐
                                              │   CustomTkinter GUI │
                                              │   (main.py)         │
                                              └─────────────────────┘
```

### Threading Model

```
┌──────────────────┐   ┌────────────────────────┐   ┌──────────────────────┐
│  Capture         │   │  Detection             │   │  Main (GUI)          │
│  Thread          │   │  Thread                │   │  Thread              │
│                  │   │                        │   │                      │
│  Webcam.read()   │──▶│  MediaPipe + Analyzers │──▶│  Tkinter + OpenCV    │
│  → deque buffer  │   │  → FrameAnalysisResult │   │  → display           │
└──────────────────┘   └────────────────────────┘   └──────────────────────┘
```

---

## Quick Start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
python main.py
```

MediaPipe models (~40 MB) are downloaded automatically on first launch to `~/.cache/vision-detector/models/`.

**Keyboard shortcuts:** `d` toggle detection · `v` toggle visualization · `s` screenshot · `q` quit

---

## Project Structure

```
local-webcam-emotion-gesture-detector/
├── main.py                      # GUI entry point (CustomTkinter)
├── config/
│   └── config.yaml              # All user-facing settings
└── src/
    ├── pipeline.py              # Main orchestrator
    ├── capture.py               # Threaded webcam capture
    ├── holistic_analyzer.py     # MediaPipe face + hands + pose
    ├── emotion_analyzer_cnn.py  # HSEmotion CNN (8 classes)
    ├── emotion_analyzer.py      # Blendshapes fallback
    ├── presence_analyzer.py     # Presence state machine
    ├── gesture_analyzer.py      # Hand gesture recognition
    ├── posture_analyzer.py      # Slouch detection
    ├── data_models.py           # Pydantic result models
    ├── debug_renderer.py        # Overlay rendering
    ├── settings.py              # Typed configuration
    └── utils.py                 # Shared helpers
```

---

## Tech Stack

| | Library | Usage |
|:-|:--------|:------|
| ![MediaPipe](https://img.shields.io/badge/MediaPipe-00A67E?logo=google&logoColor=white) | MediaPipe >=0.10 | Face (478 pts), hands (21), pose (33) |
| ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white) | OpenCV >=4.8 | Capture + visualization |
| ![HSEmotion](https://img.shields.io/badge/HSEmotion-FF6F00?logoColor=white) | HSEmotion >=0.2 | CNN emotion model |
| ![ONNX](https://img.shields.io/badge/ONNX_Runtime-7B7B7B?logo=onnx&logoColor=white) | ONNX Runtime >=1.16 | Model inference |
| ![Pydantic](https://img.shields.io/badge/Pydantic-E92063?logo=pydantic&logoColor=white) | Pydantic >=2.0 | Data validation + config |
| ![CustomTkinter](https://img.shields.io/badge/CustomTkinter-1F6FEB?logoColor=white) | CustomTkinter >=5.2 | Dark-mode GUI |

**Requirements:** Python 3.11+ · macOS (Apple Silicon M1/M2) · webcam

---

## Configuration

All settings live in [`config/config.yaml`](config/config.yaml) — camera resolution, FPS target, detection thresholds, landmark colors, posture alerts, and more. See the inline comments for guidance.

---

## Performance

Measured on a MacBook Pro M1:

| Metric | Value |
|:-------|:------|
| FPS | 25–30 |
| Latency | 30–40 ms / frame |
| CPU | ~35–45 % (single core) |
| RAM | ~300–500 MB |

---

## Troubleshooting

| Problem | Fix |
|:--------|:----|
| Camera won't start | Grant camera access in System Preferences > Security > Camera |
| MediaPipe timestamp errors | Restart the app |
| Unstable hand detection | Lower `min_detection_confidence` to `0.3` in config |

---

## Roadmap

- [ ] Local SQLite storage for results
- [ ] Full unit tests
- [ ] Data export (CSV / JSON)
- [ ] Customizable posture alerts
- [ ] Headless mode (no GUI)

---

<p align="center">
  <sub>Built by Mateon — Powered by MediaPipe, OpenCV & CustomTkinter</sub>
</p>
