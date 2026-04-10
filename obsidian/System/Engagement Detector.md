---
title: Engagement Detector
tags: [system, engagement, zone, pose, gate]
created: 2026-04-10
---

# Engagement Detector

> Prevents sessions starting when someone walks past or stands near the machine. The person must be **inside the zone AND in a seated working pose** before tracking begins.

Part of [[System/Architecture]]. Code: `pi/engagement_detector.py`.

---

## Why It Exists

Without engagement detection:
- Person walks past machine → rep counting starts
- Trainer adjusts machine for someone → phantom session
- Member stands chatting next to the machine → session attributed

With engagement detection:
- Two independent checks must **both** pass before session starts
- Requires sustained confirmation (10 frames, ~0.33s)

---

## Two Checks

### Check 1: Zone Overlap
```
MACHINE_ZONE_ROI = (0.15, 0.05, 0.85, 0.95)  # (x1, y1, x2, y2) normalised

person bbox ∩ machine zone
─────────────────────────── ≥ ENGAGEMENT_MIN_OVERLAP (0.40)
    person bbox area

→ At least 40% of person's bounding box must be inside the machine zone
```

### Check 2: Pose Check

Depends on `EXERCISE_TYPE` in `pi/config.py`.

**lat_pulldown / cable_row / seated:**
- Hips below shoulders → person is seated
- Wrists raised above shoulders → arms are working
- Weighted average of keypoint confidences must exceed threshold

**standing (e.g. bicep curl):**
- Body upright
- Arms in working position

Returns `EngagementState`:
```python
@dataclass
class EngagementState:
    engaged:      bool
    in_zone:      bool
    pose_ok:      bool
    zone_ratio:   float   # fraction of bbox inside zone
    hip_y:        float   # hip keypoint y (normalised)
    wrist_raised: bool
```

---

## Confirmation Thresholds

| Setting | Value | Meaning |
|---------|-------|---------|
| `ENGAGE_FRAMES_REQUIRED` | 10 | Frames of `engaged=True` to start session (~0.33s at 30fps) |
| `DISENGAGE_FRAMES_REQUIRED` | 45 | Frames of `engaged=False` to end session (~1.5s at 30fps) |
| `ENGAGEMENT_MIN_OVERLAP` | 0.40 | Minimum zone overlap fraction |

Hysteresis prevents jitter: you need 10 frames in to start, 45 frames out to stop.

---

## Visual Overlay (Debug)

When `SHOW_PREVIEW = True`:
- **Green box**: engaged (both checks pass)
- **Yellow box**: in zone but pose not confirmed
- **Grey box**: not in zone

---

## Calibration

1. Mount camera
2. Set `SHOW_PREVIEW = True` in `pi/config.py`
3. Run `python pi/main.py`
4. Adjust `MACHINE_ZONE_ROI` until the box tightly wraps the machine seat area
5. Sit in the machine — box should turn green
6. Walk past — box should stay grey
7. Set `SHOW_PREVIEW = False` for production

---

## Relationship to ActivityStateMachine

EngagementDetector is the **hardware gate** (physical position + pose).
ActivityStateMachine is the **model gate** (LSTM predicts `on_machine` class).

Both must agree before a session begins:
```
EngagementDetector.engaged = True
AND
ActivityStateMachine.phase = ENGAGED (10× on_machine predicted)
```

---

## Related

- [[System/Activity Classes]] — class 2 (`on_machine`) drives the LSTM side of engagement
- [[System/LSTM Model]] — IDLE/ENGAGED phase gate
- [[System/Architecture]] — where this fits in main loop
- [[Projects/Rep Tracking]] — session gated by this
- [[Hardware/Camera Placement]] — zone ROI depends on camera position
