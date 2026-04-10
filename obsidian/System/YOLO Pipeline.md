---
title: YOLO Pipeline
tags: [system, yolo, pose, keypoints, hailo]
created: 2026-04-09
updated: 2026-04-10
---

# YOLO Pose Pipeline

YOLOv11n-pose runs on every camera frame at 30fps and extracts 17 body keypoints. These keypoints feed everything: [[System/LSTM Model]], [[Projects/User Tracking]], [[System/Engagement Detector]], and angle-based rep counting.

---

## Model

| Setting | Value | Config key |
|---------|-------|------------|
| Model | `yolo11n-pose.pt` (nano) | `YOLO_MODEL` |
| Input | 1280×720 (or 640×480) | `FRAME_WIDTH/HEIGHT` |
| FPS | 30 | `FRAME_RATE` |
| Confidence threshold | 0.3 | `YOLO_CONFIDENCE` |
| Device | `hailo` or `cpu` | `YOLO_DEVICE` |

---

## Performance

| Hardware | FPS | Latency |
|----------|-----|---------|
| Hailo-8 AI HAT+ (26 TOPS) | **~30 fps** | ~35ms |
| Pi 5 CPU only | ~5–8 fps | ~80ms |
| Mac Mini M4 | 60+ fps | <10ms |

Hailo offloads YOLO entirely to the NPU — Pi CPU is free for LSTM, tracking, and recording.

---

## The 17 COCO Keypoints

```
0:  nose           5:  left_shoulder   10: right_wrist
1:  left_eye       6:  right_shoulder  11: left_hip
2:  right_eye      7:  left_elbow      12: right_hip
3:  left_ear       8:  right_elbow     13: left_knee
4:  right_ear      9:  left_wrist      14: right_knee
                                       15: left_ankle
                                       16: right_ankle
```

**For lat pulldown:** keypoints 5, 6 (shoulders), 7, 8 (elbows), 9, 10 (wrists) are most important.
**For user tracking:** full bounding box + hip/shoulder position.

---

## Feature Vector (51 per frame)

```
17 keypoints × 3 values = 51 features per frame

[x_0, y_0, c_0,   # nose
 x_1, y_1, c_1,   # left_eye
 ...
 x_16, y_16, c_16] # right_ankle

x, y: normalised 0.0–1.0 by frame dimensions
c:    YOLO keypoint confidence 0.0–1.0
Missing keypoints: [0, 0, 0]
```

30 frames stacked → `(30, 51)` input tensor for [[System/LSTM Model]].

---

## Multi-Person Handling

YOLO returns detections sorted by confidence. Main loop selects the person with the highest overlap with `MACHINE_ZONE_ROI` (the machine's region of interest).

If multiple people are in frame:
- GymTracker assigns a `track_id` to each bounding box
- `closest_track(machine_roi)` returns the track nearest the machine seat
- Only that person's keypoints feed the rep detection pipeline

---

## Hailo AI HAT+ Details

| Spec | Value |
|------|-------|
| Chip | Hailo-8 |
| TOPS | 26 TOPS |
| Form factor | HAT+ for Raspberry Pi 5 |
| Interface | PCIe (via HAT+ GPIO connector) |
| Cost | £70 |
| YOLO model format | Hailo HEF (compiled from ONNX) |

Without Hailo: set `YOLO_DEVICE = "cpu"` — runs at ~5fps on Pi 5 CPU. Fine for development and testing.

---

## Two Demo Modes

| File | Hardware | Use |
|------|----------|-----|
| `pi/skeleton_cpu.py` | Pi 5 CPU only | Testing without Hailo |
| `pi/skeleton_ws_server.py` | Hailo or CPU | Full WebSocket skeleton viewer |

`skeleton_ws_server.py` broadcasts skeleton JSON at ws://pi_ip:8765 and serves a viewer at http://pi_ip:8080. Useful for verifying camera placement and YOLO quality before running the full Overseer.

---

## Config (`pi/config.py`)

```python
YOLO_MODEL      = "yolo11n-pose.pt"
YOLO_CONFIDENCE = 0.3
YOLO_DEVICE     = "hailo"     # "cpu" for testing
FRAME_WIDTH     = 1280
FRAME_HEIGHT    = 720
FRAME_RATE      = 30
CAMERA_INDEX    = 0           # 0 = Pi CSI camera
```

---

## Related

- [[System/LSTM Model]] — consumes 30-frame keypoint windows
- [[System/Engagement Detector]] — uses bounding box + keypoint positions
- [[Projects/User Tracking]] — bounding boxes used by GymTracker IoU matching
- [[Hardware/Machine Pi]] — YOLO runs on Hailo NPU here
- [[Hardware/Camera Placement]] — camera position determines keypoint quality
