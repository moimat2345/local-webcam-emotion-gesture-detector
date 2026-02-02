<div align="center">

# Local Webcam Emotion & Gesture Detector

**Transformez votre webcam en capteur intelligent — 100% hors ligne.**

Reconnaissance d'emotions en temps reel, detection de gestes, analyse de posture et suivi de presence,
propulse par MediaPipe et un modele CNN. Pas de cloud. Pas de telemetrie. Juste votre camera.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-macOS-000000?logo=apple&logoColor=white)
![Status](https://img.shields.io/badge/Status-Alpha-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-00A67E?logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-Private-red)

<a href="README.md"><img src="https://img.shields.io/badge/%F0%9F%87%AC%F0%9F%87%A7_Read_in_English-blue?style=for-the-badge" alt="Read in English"></a>

</div>

---

## Qu'est-ce que c'est

Local Webcam Emotion & Gesture Detector est une **application de vision par ordinateur temps reel, entierement locale**, qui transforme votre webcam en capteur intelligent. Construit sur le framework **Google MediaPipe** et un **modele CNN d'emotions (HSEmotion/ONNX Runtime)**, il traite chaque frame a travers un pipeline d'analyse multi-etapes pour extraire des donnees comportementales riches — le tout sans envoyer un seul octet vers le cloud.

### Ce qu'il detecte

| Capacite | Moteur | Sortie |
|:---------|:-------|:-------|
| **Reconnaissance d'emotions** | HSEmotion CNN (8 classes) | Emotion primaire + secondaire avec scores de confiance, lissage temporel entre les frames |
| **Detection de presence** | MediaPipe Face Landmarker | Machine a 3 etats (`PRESENT` → `AWAY` → `RETURNING`) avec timeouts configurables |
| **Reconnaissance de gestes** | MediaPipe Hand Landmarker | 9 gestes par main (pouce haut/bas, paix, poing, pointage, OK, wave, paume ouverte) + direction de pointage |
| **Analyse de posture** | MediaPipe Pose Landmarker | Angle d'inclinaison du cou, alignement des epaules, score de slouch pondere (0–1) avec alertes |

### Pourquoi ce projet existe

Ce projet a ete concu comme une **alternative locale** aux outils d'analyse comportementale bases sur le cloud. Cas d'usage typiques :

- **Ergonomie developpeur** — alertes de posture pendant les longues sessions de code
- **Recherche en accessibilite** — suivi de gestes et de presence pour des prototypes d'interaction mains-libres
- **Interfaces emotion-aware** — alimenter d'autres applications avec l'etat emotionnel en temps reel via des callbacks
- **Apprentissage et experimentation** — explorer MediaPipe, les pipelines CV temps reel et les architectures de capture threadees

### Principes de conception

- **Vie privee par defaut** — tout le traitement se fait sur l'appareil ; aucun appel reseau, aucune telemetrie, aucune persistance de donnees
- **Inference unique** — MediaPipe tourne une seule fois par frame (visage + mains + pose), et tous les analyseurs partagent le resultat
- **Architecture non-bloquante** — la capture webcam tourne dans un thread dedie avec un buffer circulaire, gardant la GUI reactive
- **Tout est configurable** — seuils de detection, FPS cible, couleurs de visualisation et comportement des alertes sont tous controles via `config/config.yaml`

---

## En un coup d'oeil

| Capacite | Moteur | Ce que vous obtenez |
|:---------|:-------|:--------------------|
| **Reconnaissance d'emotions** | HSEmotion CNN (8 classes) | Emotion primaire + secondaire, scores de confiance, lissage temporel |
| **Detection de presence** | MediaPipe Face Landmarker | Machine a 3 etats : `PRESENT` → `AWAY` → `RETURNING` |
| **Reconnaissance de gestes** | MediaPipe Hand Landmarker | 9 gestes par main + direction de pointage |
| **Analyse de posture** | MediaPipe Pose Landmarker | Angle du cou, alignement epaules, score slouch (0–1) |

> Chaque frame passe par une seule inference MediaPipe partagee entre tous les analyseurs — latence sous 40 ms sur MacBook Pro M1.

---

## Pourquoi Local Webcam Emotion & Gesture Detector ?

- **Vie privee par defaut** — tout le traitement sur l'appareil ; rien ne quitte votre machine, jamais.
- **Non-bloquant** — la capture webcam tourne dans son propre thread avec un buffer circulaire ; la GUI reste reactive.
- **Configurable** — seuils, FPS, couleurs, alertes : tout se trouve dans `config/config.yaml`.

Concu pour les developpeurs qui veulent des alertes de posture en codant, les chercheurs prototypant des interactions mains-libres, ou toute personne curieuse des pipelines de vision en temps reel.

---

## Comment ca marche

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────────────────┐     ┌───────────────┐
│   Webcam     │────▶│  ThreadedCapture  │────▶│        HolisticAnalyzer         │────▶│ DebugRenderer │
│  (camera)    │     │  (thread separe   │     │  (MediaPipe: visage 478pts,     │     │ (overlay sur  │
│              │     │   + buffer)        │     │   mains 21pts, pose 33pts)      │     │  flux video)  │
└─────────────┘     └──────────────────┘     └──────────┬──────────────────────┘     └───────┬───────┘
                                                        │                                    │
                                              ┌─────────┴─────────┐                          │
                                              │  Resultats         │                          │
                                              │  Holistic          │                          │
                                              └─────────┬─────────┘                          │
                                                        │                                    │
                              ┌──────────────────────────┼──────────────────────────┐         │
                              │                          │                          │         │
                    ┌─────────▼─────────┐   ┌────────────▼──────────┐   ┌──────────▼───────┐ │
                    │  EmotionCNN       │   │  GestureAnalyzer      │   │ PostureAnalyzer  │ │
                    │  8 emotions       │   │  9 gestes par main    │   │ angle cou,       │ │
                    │  (HSEmotion ONNX) │   │  + direction pointage │   │ score slouch     │ │
                    └─────────┬─────────┘   └────────────┬──────────┘   └──────────┬───────┘ │
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
                                              │ (modele Pydantic)   │
                                              └──────────┬──────────┘
                                                         │
                                              ┌──────────▼──────────┐
                                              │   CustomTkinter GUI │
                                              │   (main.py)         │
                                              └─────────────────────┘
```

### Modele de threading

```
┌──────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│  Capture         │   │  Detection            │   │  Main (GUI)          │
│  Thread          │   │  Thread               │   │  Thread              │
│                  │   │                        │   │                      │
│  Webcam.read()   │──▶│  MediaPipe + Analyzers │──▶│  Tkinter + OpenCV    │
│  → buffer deque  │   │  → FrameAnalysisResult │   │  → affichage         │
└──────────────────┘   └──────────────────────┘   └──────────────────────┘
```

---

## Demarrage rapide

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
python main.py
```

Les modeles MediaPipe (~40 Mo) sont telecharges automatiquement au premier lancement dans `~/.cache/vision-detector/models/`.

**Raccourcis clavier :** `d` detection · `v` visualisation · `s` capture d'ecran · `q` quitter

---

## Structure du projet

```
local-webcam-emotion-gesture-detector/
├── main.py                      # Point d'entree GUI (CustomTkinter)
├── config/
│   └── config.yaml              # Tous les parametres utilisateur
└── src/
    ├── pipeline.py              # Orchestrateur principal
    ├── capture.py               # Capture webcam threadee
    ├── holistic_analyzer.py     # MediaPipe visage + mains + pose
    ├── emotion_analyzer_cnn.py  # HSEmotion CNN (8 classes)
    ├── emotion_analyzer.py      # Fallback blendshapes
    ├── presence_analyzer.py     # Machine a etats presence
    ├── gesture_analyzer.py      # Reconnaissance de gestes
    ├── posture_analyzer.py      # Detection slouching
    ├── data_models.py           # Modeles Pydantic
    ├── debug_renderer.py        # Rendu overlay
    ├── settings.py              # Configuration typee
    └── utils.py                 # Utilitaires partages
```

---

## Stack technique

| | Librairie | Usage |
|:-|:----------|:------|
| ![MediaPipe](https://img.shields.io/badge/MediaPipe-00A67E?logo=google&logoColor=white) | MediaPipe >=0.10 | Visage (478 pts), mains (21), pose (33) |
| ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white) | OpenCV >=4.8 | Capture + visualisation |
| ![HSEmotion](https://img.shields.io/badge/HSEmotion-FF6F00?logoColor=white) | HSEmotion >=0.2 | Modele CNN emotions |
| ![ONNX](https://img.shields.io/badge/ONNX_Runtime-7B7B7B?logo=onnx&logoColor=white) | ONNX Runtime >=1.16 | Inference modele |
| ![Pydantic](https://img.shields.io/badge/Pydantic-E92063?logo=pydantic&logoColor=white) | Pydantic >=2.0 | Validation + config |
| ![CustomTkinter](https://img.shields.io/badge/CustomTkinter-1F6FEB?logoColor=white) | CustomTkinter >=5.2 | GUI mode sombre |

**Prerequis :** Python 3.11+ · macOS (Apple Silicon M1/M2) · webcam

---

## Configuration

Tous les parametres sont dans [`config/config.yaml`](config/config.yaml) — resolution camera, FPS cible, seuils de detection, couleurs des landmarks, alertes posture, etc. Voir les commentaires dans le fichier.

---

## Performance

Mesure sur MacBook Pro M1 :

| Metrique | Valeur |
|:---------|:-------|
| FPS | 25–30 |
| Latence | 30–40 ms / frame |
| CPU | ~35–45 % (single core) |
| RAM | ~300–500 Mo |

---

## Depannage

| Probleme | Solution |
|:---------|:---------|
| Camera ne demarre pas | Autoriser l'acces camera dans Preferences Systeme > Securite > Camera |
| Erreurs timestamp MediaPipe | Redemarrer l'app |
| Detection mains instable | Baisser `min_detection_confidence` a `0.3` dans la config |

---

## Prochaines etapes

- [ ] Stockage SQLite local des resultats
- [ ] Tests unitaires complets
- [ ] Export des donnees (CSV / JSON)
- [ ] Alertes posture personnalisables
- [ ] Mode headless (sans GUI)

---

<p align="center">
  <sub>Built by Mateon — Powered by MediaPipe, OpenCV & CustomTkinter</sub>
</p>
