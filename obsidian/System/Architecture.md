---
title: System Architecture
tags: [system, architecture, overview]
created: 2026-04-09
updated: 2026-04-10
---

# System Architecture

Full picture of how all components connect. See [[Home]] for the project list.

---

## Hardware Topology

```
GYM LOCAL NETWORK (WiFi / PoE switch)
│
├── ENTRY PI (door camera)
│     ├─ Pi Camera Module 3 (face-on to door)
│     ├─ InsightFace buffalo_sc ArcFace
│     ├─ IdentityWindow (10s evidence accumulation)
│     └─ PersonDB → registers member on entry
│
├── MACHINE PI × N  (one per gym machine)
│     ├─ Pi Camera Module 3 Wide (side-on machine view)
│     ├─ Pi Camera Module 3 × 2 (barbell sleeve, weight ID only)
│     ├─ Hailo-8 AI HAT+ (26 TOPS NPU)
│     ├─ YOLO YOLOv11n-pose → 17 keypoints @ 30fps
│     ├─ EngagementDetector → zone + pose gate
│     ├─ ActivityStateMachine → IDLE / ENGAGED phases
│     ├─ ONNXActivityClassifier → 8-class LSTM (when trained)
│     ├─ RepCounter (rule-based) → angle threshold fallback
│     ├─ WeightStackTracker → optical flow validates movement
│     ├─ WeightDetector → AlphaFit plate colour → kg
│     ├─ GymTracker → IoU tracking, links track to member
│     ├─ SessionRecorder → 10-min MP4 chunks → Google Drive
│     ├─ ClipReporter → confidence < 50% → GitHub data/review/
│     ├─ MachineWSServer (port 8788) → WebSocket → tablet.html
│     └─ SetReporter → HTTP POST → Supabase / Power Automate
│
├── TABLET × N (one per machine, kiosk mode)
│     └─ display/tablet.html → ws://[pi-ip]:8788
│
└── STAFF BROWSER (any device on local network)
      └─ display/staff.html → connects to all Pi WebSockets
```

---

## Main Loop Data Flow (`pi/main.py`)

```
Camera Frame (30fps, 1280×720)
    │
    ▼
YOLO YOLOv11n-pose (Hailo NPU)
    → 17 COCO keypoints per person (x, y, confidence)
    → person bounding box
    │
    ├──→ GymTracker.update(detections)     ← IoU tracking
    │         └─ PersonDB.get_member()     ← member identity
    │
    ├──→ EngagementDetector.update()       ← zone + pose gate
    │         → in_zone AND seated AND wrists up?
    │
    ├──→ WeightStackTracker.update(frame)  ← optical flow on ROI
    │
    ├──→ WeightDetector.identify(frame)    ← plate colour → kg
    │
    └──→ 30-frame keypoint buffer
              │
              ▼
         ONNX Classifier  (if ONNX_MODEL_PATH set)
              → 8-class softmax
              → ActivityStateMachine (phase gate)
              → GatedResult(phase, class_id, confidence)
         OR
         Rule-based RepCounter
              → elbow angle threshold
              → ROM check (≥ 40°)
              → duration check (300ms – 12s)
              │
              ▼
         Rep confirmed?
              → weight_tracker.weight_moved_during_rep()?  ← anti-phantom
              → YES: rep_count++
              │
              ▼
         Confidence < 0.50?
              → ClipReporter → save 30-frame .npy + meta.json
              → upload to GitHub data/review/
              │
              ▼
         MachineWSServer.update_state(...)  → broadcast to tablet (10fps)
              │
              ▼
         Set complete? (member stands / long rest)
              → SetReporter.report_set() → Supabase / Power Automate
              │
              ▼
         Session end
              → db.close_session(total_reps, avg_rom, avg_duration_s)
              → SessionRecorder.end_session() → upload chunk to Google Drive
```

---

## Data Flow — Set Complete Payload

```json
{
  "machine_id":     "lat-pulldown-01",
  "machine_name":   "Nautilus Lat Pulldown",
  "member_id":      "M1089",
  "member_name":    "Matthew",
  "timestamp":      "2026-04-09T14:32:11Z",
  "exercise":       "Lat Pulldown",
  "weight_kg":      52.5,
  "reps":           10,
  "form_breakdown": {"good": 7, "partial": 2, "bad": 1},
  "form_score":     0.78,
  "model_version":  "v1.0-lstm"
}
```

---

## Data Flow — Review Loop

```
LSTM confidence < 0.50
    → ClipReporter saves:
          {uid}_kps.npy      (30×51 float32 keypoints)
          {uid}_meta.json    (machine, timestamp, predicted_class, confidence)
    → GitHub API upload: data/review/{machine_id}/{date}/
    
Mac Mini:
    git pull
    make review          → localhost:8787
    → Human clicks correct class
    → meta.json updated: true_class, review_status=reviewed
    git commit
    make train           → new ONNX
    make deploy PI=pi@IP → Pi now smarter
```

---

## Video Recording Pipeline

```
Pi records session
    → SessionRecorder.write_frame() every loop
    → Chunk rotates every 600s (10 min)
    → Background thread: rclone copy chunk → Google Drive
    → Local chunk deleted after successful upload
    → Filename: {machine_id}_{YYYYMMDD}_{HHMM}_chunk{N}.mp4
```

Cron on Pi: `2am Mon–Fri` uploader.py runs.
`3am`: git pull latest ONNX model (new model active next session).

---

## Training Server (Mac Mini M4)

```
Mac Mini M4 (Apple Silicon, MPS PyTorch backend)
    ├─ make sync PI=pi@IP    → rsync recordings to data/raw/
    ├─ pose/label.html       → annotate videos
    ├─ make extract          → video + JSON → numpy arrays
    ├─ make train            → LSTM → ONNX (MPS accelerated)
    ├─ make review           → review portal localhost:8787
    └─ make deploy           → scp ONNX to Pi + restart
```

Not deployed at the gym — stays in office/home.

---

## Software Stack Summary

| Layer | Tech | File |
|-------|------|------|
| Keypoint extraction | YOLOv11n-pose (Hailo NPU) | `pi/main.py` |
| Activity classification | ONNX LSTM 8-class | `pi/onnx_classifier.py` |
| Phase gate | IDLE/ENGAGED state machine | `pi/activity_state_machine.py` |
| Engagement gate | Zone + pose check | `pi/engagement_detector.py` |
| Weight validation | Farneback optical flow | `pi/weight_tracker.py` |
| Weight identification | HSV colour + YOLO | `weight_id/detector.py` |
| Face recognition | InsightFace buffalo_sc | `face/recognizer.py` |
| Floor tracking | IoU greedy matching | `user_tracking/gym_tracker.py` |
| Video recording | OpenCV + rclone | `pi/session_recorder.py` |
| Clip reporting | GitHub API upload | `pi/clip_reporter.py` |
| Live display | WebSocket asyncio | `display/ws_server.py` |
| Set reporting | HTTP POST requests | `display/set_reporter.py` |
| Database | Supabase PostgreSQL | `members/db_client.py` |
| Training | PyTorch MPS (Mac Mini) | `train/train_pytorch.py` |

---

## Related

- [[Projects/Rep Tracking]] — the primary inference pipeline
- [[Projects/Weight ID]] — plate detection detail
- [[Projects/User Tracking]] — face ID + floor tracking detail
- [[Projects/E-Weight]] — Phase 2 cable machine weights
- [[System/LSTM Model]] — neural network detail
- [[System/Engagement Detector]] — engagement detection detail
- [[System/Review Loop]] — self-improvement loop
- [[System/Session Recorder]] — video recording detail
- [[Hardware/Machine Pi]] — hardware detail
