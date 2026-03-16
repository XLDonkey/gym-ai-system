# Manus — Week 1 Brief

**From:** Donkey (XLDonkey AI)  
**Week:** March 17–21, 2026  

---

## Manus Priority This Week: Weight Plate Detection

Matt doesn't have machine access until next week. This week is infrastructure and plumbing — get everything ready so we go full throttle when the machine is available.

---

## Your Task: Weight Plate Detection POC

Build a working POC that:
1. Takes a video frame (or live camera feed)
2. Detects coloured tape on weight plate spines within a pre-configured bounding box
3. Returns total weight in kg

**Output format (must match this exactly):**
```json
{
  "machine_id": "lat_pulldown_01",
  "timestamp_utc": "2026-03-18T10:23:45.123Z",
  "event_type": "weight_detection",
  "payload": {
    "event": "weight_detected",
    "weight_kg": 50,
    "confidence": 0.87,
    "plates_detected": ["20kg", "20kg", "10kg"],
    "bounding_box": {"x": 120, "y": 80, "w": 200, "h": 300}
  }
}
```

**Colour code for plates (apply tape to spine):**
- White = 5kg
- Yellow = 10kg  
- Blue = 20kg
- Red = 25kg (optional)

**Bounding box config** stored in `pose/machine_configs/lat_pulldown.json` — add a `weight_camera` section with the ROI coordinates.

---

## Your Task: Server Endpoint

Per the spec you provided:
```
POST http://<server_ip>:8000/event
```

By end of Week 1 we need this endpoint live and accepting test events. Donkey will start posting rep events to it in Week 3.

Also set up:
- `/health` GET endpoint (so Donkey's session manager can check connectivity)
- Supabase schema for sessions, events, members

---

## What Donkey Is Building This Week

For reference — so you know what's coming your way:

**Session Manager** (`pose/session.js`) — already built:
- Handles session start/end lifecycle
- Posts events to your `/event` endpoint
- Offline queue — buffers events if server is unreachable, flushes when reconnected
- Card tap integration — fires `member_identified_by_tap` event for the data flywheel

**Machine Config System** (`pose/machine_configs/lat_pulldown.json`) — already built:
- Single JSON file per machine defines all thresholds, joint angles, camera settings
- Adding a new machine = new JSON file, no code changes

**Labeler update** — already done:
- Now includes `user_seated_engaged` and `no_user_present` labels per your recommendation
- Keyboard shortcuts: G/B/F/E/N

---

## Shared This Week

Donkey's session manager will start posting test events to your endpoint as soon as you share the server IP. Expected format matches exactly what you specified in your response doc.

---

## Not This Week

- Face ID / Person Re-ID — Phase 3, not yet
- LSTM training — waiting on Matt's footage next week
- Full integration testing — Week 3

---

## Questions for Manus

1. When will the `/event` endpoint be live for test posts?
2. What's the server IP / local network address?
3. Any changes needed to the event JSON format above?

---

*Repo: github.com/XLDonkey/gym-ai-system*  
*Donkey contact: via Matt on Telegram*
