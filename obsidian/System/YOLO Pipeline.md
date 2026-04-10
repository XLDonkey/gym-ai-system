---
title: YOLO Pipeline
tags: [system, yolo, pose, keypoints, hailo]
created: 2026-04-09
---

# YOLO Pose Pipeline

YOLO runs on every camera frame and extracts 17 body keypoints. These keypoints are the input to everything else — [[System/LSTM Model]], [[Projects/User Tracking]], and angle-based rules.

---

## Model

| Setting | Value |
|---------|-------|
| Model | `yolo11n-pose.pt` (nano — fastest on Pi) |
| Output | 17 keypoints × [x, y, confidence] per person |
| Confidence threshold | 0.3 |
| Device | `hailo` (AI HAT+) or `cpu` |
| FPS | 30fps with Hailo-8 NPU |

---

## The 17 COCO Keypoints

```
0: nose           5: left_shoulder   10: right_wrist
1: left_eye       6: right_shoulder  11: left_hip
2: right_eye      7: left_elbow      12: right_hip
3: left_ear       8: right_elbow     13: left_knee
4: right_ear      9: left_wrist      14: right_knee
                                     15: left_ankle
                                     16: right_ankle
```

For lat pulldown: elbow (7,8) and shoulder (5,6) angles are most important.

---

## Feature Vector

Each frame: 17 keypoints × 3 values = **51 features**
- x coordinate (normalised 0–1 by frame width)
- y coordinate (normalised 0–1 by frame height)
- confidence score

30 frames stacked → `(30, 51)` input to [[System/LSTM Model]].

---

## Hailo AI HAT+

| Spec | Value |
|------|-------|
| Chip | Hailo-8 |
| TOPS | 26 TOPS |
| Form factor | HAT+ for Pi 5 |
| Cost | £70 |
| Interface | PCIe (via HAT+ connector) |

With Hailo: YOLO runs at 30fps, leaving Pi CPU free for LSTM + tracking.
Without Hailo: YOLO on CPU runs ~5–8fps (adequate for testing).

---

## Config

```python
# pi/config.py
YOLO_MODEL      = "yolo11n-pose.pt"
YOLO_CONFIDENCE = 0.3
YOLO_DEVICE     = "hailo"   # or "cpu"
FRAME_RATE      = 30
FRAME_WIDTH     = 1280
FRAME_HEIGHT    = 720
```

---

## Related

- [[System/LSTM Model]] — consumes keypoints
- [[Projects/User Tracking]] — uses bounding boxes for IoU tracking
- [[Hardware/Machine Pi]] — runs YOLO
- [[Hardware/Camera Placement]] — where to mount for best keypoints
