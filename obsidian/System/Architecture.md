---
title: System Architecture
tags: [system, architecture, overview]
created: 2026-04-09
---

# System Architecture

Full picture of how all components connect. See [[Home]] for project list.

---

## Hardware Topology

```
GYM NETWORK (local WiFi / PoE)
│
├── ENTRY PI (door camera)
│     └─ InsightFace ArcFace → member identity → PersonDB
│
├── MACHINE PI × N  (one per machine)
│     ├─ Pi Camera Module 3 Wide (machine view)
│     ├─ Pi Camera Module 3 (barbell sleeve view × 2, free weights only)
│     ├─ YOLO Pose → 17 keypoints
│     ├─ ActivityStateMachine → IDLE / ENGAGED
│     ├─ LSTM Classifier → 8 activity classes
│     ├─ WeightDetector → kg from plate colours
│     ├─ WeightStackTracker → optical flow validates rep
│     ├─ ws_server.py → WebSocket → Tablet on machine
│     ├─ set_reporter.py → HTTP → Supabase / Power Automate
│     └─ clip_reporter.py → uncertain clips → GitHub
│
├── TABLETS (one per machine, Kiosk mode)
│     └─ display/tablet.html → ws://[pi-ip]:8788
│
└── STAFF BROWSER (any device on network)
      └─ display/staff.html → connects to each Pi's WebSocket
```

---

## Data Flow — Live Rep

```
Camera frame
  → YOLO (Hailo NPU, 30fps)
  → 17 keypoints (x, y, confidence)
  → 30-frame ring buffer (30 × 51 = 1530 features)
  → LSTM ONNX inference (~5ms on Pi 5)
  → activity class + confidence
  → ActivityStateMachine (phase gate)
  → rep counted / ignored
  → WebSocket broadcast (10fps) → tablet.html
```

---

## Data Flow — Set Complete

```
Member stands up / long rest detected
  → SetReporter.report_set(reps, form, weight, member_id)
  → HTTP POST → Power Automate (interim) or /api/set (future)
  → Supabase sets table
  → Staff dashboard updates
```

---

## Data Flow — Uncertain Prediction

```
LSTM confidence < 50%
  → ClipReporter saves 30-frame keypoint window + metadata
  → Uploads to GitHub: data/review/{machine}/{date}/
  → Mac Mini: git pull → make review → localhost:8787
  → Human clicks correct class
  → git commit → make train → make deploy
  → Pi now smarter
```

---

## Software Stack

| Layer | Tech |
|-------|------|
| Edge inference | Python 3.11, OpenCV, ultralytics YOLO, onnxruntime |
| Training | PyTorch 2.x, MPS backend (Mac Mini M4) |
| Face recognition | InsightFace (buffalo_sc) |
| Live display | WebSocket (asyncio + websockets) |
| Set reporting | requests → HTTP POST |
| Database | Supabase (Postgres) |
| Frontend | Plain HTML/CSS/JS (tablet.html, staff.html) |
| CI/CD | GitHub Actions |

---

## Component Relationships

- [[System/YOLO Pipeline]] feeds keypoints to [[System/LSTM Model]]
- [[System/LSTM Model]] output gates [[Projects/Rep Tracking]]
- [[Projects/User Tracking]] populates PersonDB, which [[Projects/Rep Tracking]] reads for member name
- [[Projects/Weight ID]] reads barbell weight, [[Projects/E-Weight]] reads cable machine weight
- [[System/WebSocket Layer]] broadcasts all state to [[Hardware/Machine Pi]] tablet
- [[System/Review Loop]] retrains [[System/LSTM Model]] from flagged clips
