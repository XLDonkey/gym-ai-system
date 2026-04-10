---
title: Project 1 — Rep Tracking
tags: [project, rep-tracking, lstm, active]
status: live-rule-based
created: 2026-04-09
---

# Project 1 — Rep Tracking

> Counts every rep, classifies form, gates by engagement phase.

Part of [[Home]]. See also [[System/LSTM Model]], [[System/Activity Classes]], [[System/Review Loop]].

---

## What It Does

The Pi camera watches the person at the machine. Every frame, YOLO extracts 17 body keypoints. A 30-frame window of keypoints is fed to the [[System/LSTM Model]] which outputs one of 8 activity classes. Reps are only counted when the system is confident the person is genuinely working.

---

## Current Status

| Mode | Active? |
|------|---------|
| Rule-based angle counting | **Yes — live** |
| LSTM 8-class classifier | Not yet — needs 300+ training segments |

Rule-based works by measuring elbow/shoulder angle. The LSTM will replace it once data is collected.

---

## The 8-Class Schema

See [[System/Activity Classes]] for full detail.

Key classes:
- `on_machine` (2) → starts session
- `good_rep` (3) → counted
- `bad_rep` (4) → counted + flagged
- `half_rep` (7) → counted + flagged
- `false_rep` (5) → ignored
- `resting` (6) → pause between sets

---

## The Phase Gate

```
IDLE phase   → only classes 0, 1, 2 valid
ENGAGED phase → classes 2–7 valid
```

Needs **10 consecutive `on_machine` frames** to transition IDLE → ENGAGED.
Rep classes (3–7) masked to `-inf` before softmax in IDLE phase — they literally cannot be predicted.

This prevents phantom reps when someone walks past the machine.

---

## Model Architecture

```
Input:  (batch, 30, 51)   30 frames × 51 features (17 keypoints × xyz)
LSTM:   51 → 128 hidden, 1 layer
Dropout: 0.3
Linear: 128 → 64 → ReLU → 8
Softmax → one class wins
```

Full spec in [[System/LSTM Model]].

---

## Key Files

| File | Purpose |
|------|---------|
| `pi/activity_state_machine.py` | IDLE/ENGAGED gate, rep counting |
| `pi/lstm_classifier.py` | ONNX inference wrapper |
| `pi/clip_reporter.py` | Uploads uncertain clips to GitHub |
| `train/train_pytorch.py` | Trains the LSTM on annotated data |
| `pose/label.html` | Annotation tool |
| `pose/review_server.py` | Review portal at localhost:8787 |

---

## Training Pipeline

```
1. Collect video → Pi recordings → rsync with make sync
2. Annotate → pose/label.html → data/annotations/
3. Train → make train → models/weights/activity_v1.onnx
4. Deploy → make deploy PI=pi@IP
```

Data needed: **300+ segments, ≥30 per class**.

---

## Review Loop

When the model is uncertain (confidence < 50%):
1. `clip_reporter.py` saves the 30-frame keypoint window
2. Uploads to `data/review/{machine}/{date}/` on GitHub
3. Mac Mini: `make review` → localhost:8787
4. Click correct class → commit → `make train` → `make deploy`
5. Pi is now smarter

Full detail: [[System/Review Loop]].

---

## Make Commands

```bash
make sync PI=pi@IP       # pull Pi recordings
make annotate            # open annotation tool
make train               # train LSTM + export ONNX
make deploy PI=pi@IP     # push to Pi + restart
make review              # review portal localhost:8787
make stats               # annotation counts per class
make pending             # Pi clips awaiting review
```

---

## Related

- [[System/LSTM Model]] — the model being used
- [[System/Activity Classes]] — the 8 output classes
- [[System/YOLO Pipeline]] — produces the keypoint input
- [[System/Review Loop]] — how the model improves
- [[System/WebSocket Layer]] — broadcasts rep count live to tablet
- [[System/Database Schema]] — where completed sets are stored
- [[Hardware/Machine Pi]] — the hardware this runs on
- [[Data/Training Requirements]] — data needed before LSTM goes live
