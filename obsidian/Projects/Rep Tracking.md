---
title: Project 1 — Rep Tracking
tags: [project, rep-tracking, lstm, active]
status: live-rule-based
created: 2026-04-09
updated: 2026-04-10
---

# Project 1 — Rep Tracking

> Counts every rep, classifies form, gates by engagement phase.

Part of [[Home]]. The core project — everything else feeds into this.

---

## What It Does

1. Pi camera captures 30fps
2. [[System/YOLO Pipeline]] extracts 17 body keypoints per frame
3. [[System/Engagement Detector]] confirms person is genuinely seated and working
4. **Rule-based counting** measures elbow angle → counts rep when full ROM completed
5. **LSTM classifier** (when trained) classifies the last 30 frames as one of [[System/Activity Classes|8 activity classes]]
6. Reps written to Supabase, live count broadcast via [[System/WebSocket Layer]] to tablet

---

## Two Counting Modes

### Mode 1: Rule-Based (Live Now)

Measures the angle between shoulder → elbow → wrist keypoints.

```
Rep counts when:
  elbow_angle crosses ANGLE_BOTTOM (90°)  ← bar pulled down
  then crosses ANGLE_TOP (130°)           ← arms extended
  AND duration between: 300ms – 12,000ms
  AND range of motion ≥ 40°
  AND weight stack confirmed moving (optical flow)
```

State machine: `waiting` → `down` (angle < 90°) → back to `waiting` (angle > 130°) = 1 rep.

### Mode 2: LSTM 8-Class Classifier (Needs Training Data)

30-frame keypoint window → [[System/LSTM Model]] → one of 8 activity classes.
Replaces rule-based once 300+ training segments collected.

| Rep condition | Class |
|---|---|
| Good rep | 3 — counted, green |
| Bad rep | 4 — counted + flagged, red |
| Half rep | 7 — counted + flagged, orange |
| False movement | 5 — ignored |
| Resting | 6 — ignored, session continues |

**Switch**: set `ONNX_MODEL_PATH` in `pi/config.py`. Blank = rule-based.

---

## The Phase Gate

Prevents phantom reps from people walking past the machine.

```
IDLE phase:
  Only classes 0, 1, 2 valid
  Rep classes 3–7 → masked to -∞ before softmax → impossible

ENGAGED phase:
  Classes 2–7 valid
  
Transition IDLE → ENGAGED:
  10 consecutive frames classified as on_machine (class 2)
  AND person confirmed by EngagementDetector (zone + pose)

Transition ENGAGED → IDLE:
  45 consecutive frames of class 0 or 1 (person absent)
  OR SESSION_TIMEOUT_S = 300 (5 min) without activity
```

Full detail: [[System/Engagement Detector]], [[System/Activity Classes]].

---

## Rep Validation (Anti-Phantom)

Even after a rep is detected, it must pass the weight stack check:

```python
if WEIGHT_TRACKING_ENABLED:
    if not weight_tracker.weight_moved_during_rep():
        # Rep rejected — weight didn't actually move
        # Catches mirror-checking, adjusting handles, etc.
```

Uses dense Farneback optical flow on the `WEIGHT_STACK_ROI` region of the frame.
Requires ≥30% of rep frames to show ≥1.5px movement.

---

## Confidence + Review

| Confidence | Action |
|---|---|
| ≥ 0.65 | Accept, high confidence |
| 0.50–0.65 | Accept, log as uncertain |
| < 0.50 | Accept + **flag 30-frame clip to GitHub** |

Flagged clips go to `data/review/{machine_id}/{date}/` for human correction → retrain.
Full detail: [[System/Review Loop]].

---

## Session Lifecycle

```
Person enters frame
  → EngagementDetector: in zone + seated + arms up
  → ENGAGE_FRAMES_REQUIRED (10) frames confirmed
  → Session starts → db.create_session() → Supabase

Person does reps
  → each rep: rom_degrees, duration_s logged
  → set complete when resting detected (class 6) or person stands

Person leaves
  → DISENGAGE_FRAMES_REQUIRED (45) frames of absence
  → SESSION_TIMEOUT_S (300s) hard limit
  → Session closed → db.close_session(total_reps, avg_rom, avg_duration_s)
```

---

## Key Files

| File | Purpose |
|------|---------|
| `pi/main.py` | Main loop — orchestrates everything |
| `pi/activity_state_machine.py` | IDLE/ENGAGED phase gate |
| `pi/onnx_classifier.py` | LSTM ONNX inference wrapper, 30-frame buffer |
| `pi/engagement_detector.py` | Zone + pose confirmation |
| `pi/weight_tracker.py` | Optical flow weight validation |
| `pi/clip_reporter.py` | Flags uncertain clips to GitHub |
| `pi/config.py` | All thresholds and settings |
| `train/train_pytorch.py` | Mac Mini: trains LSTM |
| `pose/label.html` | Annotation tool |
| `pose/review_server.py` | Review portal — localhost:8787 |

---

## Key Thresholds (`pi/config.py`)

| Setting | Value | Purpose |
|---------|-------|---------|
| `ANGLE_TOP` | 130° | Arms extended — rep top |
| `ANGLE_BOTTOM` | 90° | Bar pulled — rep bottom |
| `MIN_ROM_DEGREES` | 40° | Minimum range of motion |
| `MIN_REP_DURATION_MS` | 300 | Fastest valid rep |
| `MAX_REP_DURATION_MS` | 12000 | Slowest valid rep |
| `ONNX_CONFIDENCE_THRESH` | 0.65 | Uncertain threshold |
| `ONNX_REVIEW_THRESH` | 0.50 | Flag for review |
| `IDLE_TIMEOUT_SECONDS` | 30 | Recording idle cutoff |
| `SESSION_TIMEOUT_S` | 300 | Max session without activity |

---

## Training Pipeline

```
1. Pi records sessions → RECORD_SESSIONS = True
2. make sync PI=pi@IP      → pulls recordings to data/raw/
3. make annotate           → pose/label.html (draw time segments, pick class)
4. make extract            → converts video+JSON → data/processed/*.npy
5. make train              → PyTorch LSTM → models/weights/activity_v1.onnx
6. make deploy PI=pi@IP    → scp ONNX → Pi restarts service
7. make review             → annotate flagged clips → repeat
```

Data needed: **300+ segments, ≥30 per class**. See [[Data/Training Requirements]].

---

## Make Commands

```bash
make sync PI=pi@IP       # pull Pi recordings to Mac Mini
make annotate            # open annotation tool (pose/label.html)
make extract             # video + JSON → numpy arrays
make train               # train LSTM + export ONNX
make deploy PI=pi@IP     # push ONNX to Pi + restart service
make review              # review portal localhost:8787
make stats               # annotation count per class
make pending             # Pi clips awaiting review
```

---

## Related

- [[System/LSTM Model]] — neural network spec
- [[System/Activity Classes]] — the 8 output classes
- [[System/YOLO Pipeline]] — produces the keypoint input
- [[System/Engagement Detector]] — session gate
- [[System/Review Loop]] — model improvement loop
- [[System/WebSocket Layer]] — live rep count to tablet
- [[System/Session Recorder]] — video recording for training data
- [[System/Database Schema]] — where sessions and reps are stored
- [[Hardware/Machine Pi]] — the hardware this runs on
- [[Data/Training Requirements]] — what's needed before LSTM goes live
