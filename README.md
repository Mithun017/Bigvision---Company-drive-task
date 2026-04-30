# 🏀 Basketball Player Detection & Tracking

> **Big Vision Internship Assignment** — End-to-end computer vision pipeline that detects basketball players and the ball in game footage and tracks each player with a consistent identity across every frame.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![YOLOv11m](https://img.shields.io/badge/YOLOv11m-Ultralytics-purple)
![ByteTrack](https://img.shields.io/badge/Tracker-ByteTrack-orange)
![mAP](https://img.shields.io/badge/mAP%4050-97.2%25-brightgreen)
![License](https://img.shields.io/badge/License-Academic-green)

---

## 🧩 The Problem We Solved

**Problem:** Manually reviewing basketball game footage to analyse player movement, court coverage, and team positioning takes hours of human effort per game. There was no automated system to detect who is on the court, where they are, and where they have been — frame by frame, in real time.

**Solution we built:** A fully automated pipeline that takes raw basketball footage as input and produces:
- A video where every player carries a persistent numbered ID throughout the game
- Movement trails showing each player's recent path
- Heatmaps of where players and the ball spend the most time
- Court zone occupancy analysis (paint, mid-range, three-point breakdowns)
- An interactive Gradio web demo where anyone can upload a video and see results instantly

---

## 🗺️ Project Workflow

```
Data Collection         Preprocessing              Training
──────────────          ──────────────             ────────
Roboflow DS1            Quality filter             COCO pretrained
Roboflow DS2  ───────►  Resize 640×640  ─────────► YOLOv11m fine-tune
Roboflow DS3            Merge + clean              100 epochs · AdamW
(~3,600 imgs)           player + ball only         mAP@50 = 97.2%
                                                        │
                                                        ▼
Input Video             Per-frame detect           ByteTrack
────────────            ────────────────           ─────────
4K basketball  ────────► YOLOv11m infer  ─────────► Kalman filter
footage                 conf=0.40                  2-stage matching
                        NMS iou=0.5                persist IDs
                                                        │
                            ┌───────────────────────────┤
                            ▼           ▼               ▼
                      Tracked video  Heatmaps     Analytics + Demo
                      IDs + trails   Player/ball  Zone occupancy
                                                  Gradio UI
```

> The Mermaid diagram below shows the same flow for documentation tools.

```mermaid
flowchart TD
    A1[Dataset 1 · 1616 images] --> QF
    A2[Dataset 2 · 1398 images] --> QF
    A3[Dataset 3 · 600+ images] --> QF

    QF[Quality filter\nRemove blurry · dark · corrupted] --> PP[Preprocess\nResize all images to 640×640]
    PP --> MG[Merge datasets\nUnify classes · filter false boxes]
    MG --> MD[(Merged dataset\n~3600 images · player + ball)]

    COCO[COCO pretrained weights\n118k images · 80 classes] --> TR
    MD --> TR[Fine-tune YOLOv11m\n100 epochs · AdamW lr=0.001]
    TR --> BM[(best.pt\nmAP@50 = 0.97)]

    VID[Input video\n4K 60fps basketball] --> DET[Per-frame detection\nYOLOv11m · conf=0.40]
    BM --> DET
    DET --> BT[ByteTrack\nKalman filter + 2-stage matching]

    BT --> OV[Tracked video\nPlayer IDs + movement trails]
    BT --> HM[Heatmaps\nPlayer + ball positions]
    BT --> AN[Analytics dashboard\n6-panel charts]
    BT --> GR[Gradio web demo\nlocalhost:7860]

    style BM fill:#7C3AED,color:#fff
    style BT fill:#1D4ED8,color:#fff
    style OV fill:#166534,color:#fff
    style HM fill:#166534,color:#fff
    style AN fill:#166534,color:#fff
    style GR fill:#166534,color:#fff
```

---

## 🧠 Algorithms — Why We Chose These

### Detection: YOLOv11m (You Only Look Once v11, Medium)

**What it is:** A single-pass neural network that scans an image once and outputs bounding boxes, class labels, and confidence scores for every detected object.

**Why not the alternatives?**

| Model | Speed | Our decision |
|-------|-------|-------------|
| **YOLOv11m** | 30–100 FPS | ✅ Chosen — real-time speed, COCO pretrained on humans |
| Faster R-CNN | 5–10 FPS | ❌ Too slow — 2-stage design, can't track at 30fps |
| DETR | 10–20 FPS | ❌ Needs 100k+ images to converge — we have ~3,600 |
| SSD | 60–120 FPS | ❌ Lower accuracy, outdated architecture |

**Why fine-tune instead of train from scratch?** COCO pretraining already taught the model what a human body looks like — 118,000 images of people in 80 different contexts. Fine-tuning adapts those existing weights to basketball in ~90 minutes instead of training from nothing over days. We removed the referee class from training because referees near court equipment caused posts and hoops to be falsely detected as players.

**Pre-trained weights source:** `yolo11m.pt` from [Ultralytics GitHub](https://github.com/ultralytics/ultralytics), trained on [Microsoft COCO 2017](https://cocodataset.org).

---

### Tracking: ByteTrack

**What it is:** A multi-object tracker that assigns and maintains consistent numeric IDs for every detected player across all video frames.

**The basketball problem it solves:** Players screen each other constantly. When Player #3 runs behind Player #7, their detection confidence drops from 0.85 to ~0.18. SORT and DeepSORT both discard detections below their threshold — they lose Player #3's track and when they re-emerge they get a brand new ID. ByteTrack uses those low-confidence detections in a second matching pass to keep the track alive.

| Tracker | ID switches | Speed | Weakness |
|---------|-------------|-------|----------|
| **ByteTrack** | **558** | **171 FPS** | **✅ Chosen** |
| SORT | 1000+ | 200 FPS | Loses ID every time players cross paths |
| DeepSORT | ~700 | 30 FPS | Appearance CNN confused by identical jerseys |

**How it works (two stages per frame):**
1. **Stage 1** — match high-confidence detections (≥0.40) to existing tracks using IoU overlap
2. **Stage 2** — take any tracks still unmatched and match them against low-confidence detections (0.10–0.40). This is the step that recovers occluded players that every other tracker would lose.

**Pre-trained model:** Uses a built-in Kalman filter (`bytetrack.yaml` in Ultralytics). No separate training needed — the Kalman filter learns each player's velocity and position online as the video plays.

---

## 📊 Results

| Metric | Score |
|--------|-------|
| **mAP@50** | **97.2%** |
| mAP@50-95 | 65%+ |
| Precision | 95%+ |
| Recall | 94%+ |

---

## 🏗️ Project Structure

```
Big Vision/
├── Code/
│   ├── Final/
│   │   ├── basketball_LOCAL_FINAL.ipynb     ← run this on your machine
│   │   ├── basketball_COLAB_FINAL.ipynb     ← run this on Google Colab
│   │   └── basketball_project/             ← auto-created at runtime
│   │       ├── datasets/                   ← downloaded from Roboflow
│   │       ├── merged_dataset/             ← cleaned unified dataset
│   │       ├── runs/                       ← training logs + weights
│   │       ├── extracted_frames/           ← sampled video frames
│   │       └── outputs/                    ← videos, charts, heatmaps
│   └── Inputs/
│       ├── 14789160_3840_2160_60fps.mp4
│       └── 14800461_3840_2160_60fps.mp4
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA (CPU works but training takes 4–8 hours)
- ~8 GB free disk space

### 1 — Install
```bash
pip install ultralytics supervision gradio opencv-python numpy matplotlib roboflow seaborn
```

### 2 — Configure (Cell 1 of the notebook)
Change these three lines only:
```python
BASE_DIR         = './basketball_project'        # where to save everything
API_KEY          = 'YOUR_ROBOFLOW_API_KEY'       # roboflow.com → Settings
INPUT_VIDEO_PATH = 'path/to/your/game.mp4'       # or '' to auto-download
```
Get your free Roboflow key at **roboflow.com → Settings → Roboflow API**.

### 3 — Run
Open `basketball_LOCAL_FINAL.ipynb` and run all cells in order.

| Cell | What it does | Time |
|------|-------------|------|
| 1 | Set paths, detect GPU, create folders | 5 s |
| 2 | Download 3 datasets from Roboflow | ~3 min |
| 3 | Smart path detection (fixes layout bugs) | instant |
| 4 | Quality filter — remove blurry/dark images | ~2 min |
| 5 | Resize all images to 640×640 | ~2 min |
| 6 | Merge datasets, unify classes, filter bad boxes | ~2 min |
| 7 | Verify labels + class distribution chart | ~1 min |
| **8** | **Train YOLOv11m** (fine-tune, 100 epochs) | **~90 min GPU** |
| 9 | Evaluate mAP + generate learning curves | ~5 min |
| 10 | Set input video path | instant |
| 11 | Extract sample frames at 5 FPS | ~1 min |
| 12 | Detection test on still frames | ~1 min |
| **13** | **Full ByteTrack tracking on video** | **~10 min** |
| 14 | Tracking analytics dashboard (6 charts) | ~1 min |
| 15 | Spatial analytics — heatmaps, trajectories | ~2 min |
| 16 | Verify all output files | instant |
| **17** | **Launch Gradio demo** | instant |
| 18 | Final summary report | instant |

> **Skip training:** Place your `best.pt` at `basketball_project/runs/yolov11m_basketball/weights/best.pt` and start from Cell 9.

### 4 — Gradio Demo
Cell 17 opens a browser tab at `http://localhost:7860`. Type a video path or upload any basketball video — tracked output appears automatically.

---

## 📦 Key Design Decisions

**Why 2 classes (player + ball) instead of 3?**
The original datasets include a referee class. During testing, referees standing near court equipment caused basketball posts and hoops to be falsely detected as people. Removing the referee class gave the model a cleaner decision boundary and eliminated the false positives. An aspect ratio filter (`height/width > 5.5 = reject`) provides a second layer of protection.

**Why fine-tune at 640×640?**
YOLOv11m's COCO pretrained weights were trained at 640×640. Using the same size ensures the internal feature maps align perfectly with the learned representations — changing the size breaks that alignment and degrades transfer learning quality.

**Why 3 datasets instead of 1?**
One dataset teaches the model one court color, one camera angle, one jersey scheme. Three diverse sources — different courts, lighting, angles, international basketball — teach generalisation. The model will work on footage it has never seen before.

---

## 📊 Outputs

All saved to `basketball_project/outputs/`:

| File | Description |
|------|-------------|
| `basketball_tracked.mp4` | Full annotated video — bounding boxes, player IDs, movement trails |
| `player_heatmap.jpg` | Gaussian-blurred player position heatmap overlaid on court |
| `spatial_analytics.png` | 6-panel: heatmaps, zone occupancy, density grid, trajectories |
| `tracking_analytics.png` | 6-panel: track lengths, confidence, detections/frame, CDF |
| `learning_curves.png` | Training loss + mAP over 100 epochs |
| `detection_metrics.png` | mAP, Precision, Recall, F1 bar charts |
| `lr_schedule.png` | Learning rate warmup + cosine decay schedule |
| `class_distribution.png` | Dataset class balance across train/valid/test |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `ultralytics` | ≥8.3 | YOLOv11m training, inference, ByteTrack |
| `supervision` | ≥0.18 | Video annotation, box drawing, trail rendering |
| `gradio` | ≥4.0 | Interactive web demo |
| `opencv-python` | ≥4.8 | Image and video I/O |
| `roboflow` | ≥1.0 | Dataset download API |
| `seaborn` | ≥0.12 | Analytics heatmaps |
| `matplotlib` | ≥3.7 | Learning curves and charts |

---

## ⚠️ Known Issues & Fixes

| Issue | Fix |
|-------|-----|
| `ConnectionResetError [WinError 10054]` on Windows | asyncio monkey-patch included in Cell 17 |
| `ValueError: tracker_id is None` | Guard with `if dets.tracker_id is not None` before TraceAnnotator |
| Video not playing in Gradio | Re-encode to H.264 with ffmpeg — Cell 17 does this automatically |
| `CUDA out of memory` during training | Change `batch=16` to `batch=8` in Cell 8 |
| Dataset downloads 0 images | Smart path detection in Cell 3 handles both Roboflow folder layouts |

---

## 🔗 Resources

📂 **Google Drive (datasets, weights, outputs):** [Open in Drive](https://drive.google.com/drive/folders/1XeJBRmKvTCIiReYY5_Iixnz6TFS_oA-S?usp=sharing)

---

## 👤 Author

**Mithun** — Big Vision Internship Assignment

---

## 📄 License

Developed as part of an academic internship assignment. All rights reserved.
