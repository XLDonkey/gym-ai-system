---
title: Project 3 — User Tracking
tags: [project, user-tracking, face-id, arcface, active]
status: built-needs-enrolment
created: 2026-04-09
---

# Project 3 — User Tracking

> Face recognised at the door → followed to the machine → session attributed automatically.

Part of [[Home]]. Uses InsightFace ArcFace for recognition, IoU bounding box tracking for following.

---

## The Flow

```
Member walks in (door camera)
  → InsightFace ArcFace (10-second IdentityWindow)
  → PersonDB.register_from_entry(member_id, name)
  
Member sits at machine
  → GymTracker.closest_track(machine_roi) → track_id
  → PersonDB.get_member(track_id) → name + UUID
  → db.start_session(member_id, machine_id)
```

---

## Components

### EntryCamera (`user_tracking/entry_camera.py`)
- Background thread, runs on door-facing Pi
- ArcFace recognition every 2 seconds
- 10-second **IdentityWindow** — collects evidence, locks most confident match
- Registers member in PersonDB when window closes

### GymTracker (`user_tracking/gym_tracker.py`)
- IoU bounding box tracker per camera zone
- `closest_track(roi_xyxy)` — finds track closest to machine seat
- Runs optional face recognition on unlocked tracks every 30 frames
- Greedy IoU matching: highest overlap first, threshold 0.30

### PersonDB (`user_tracking/person_db.py`)
- Thread-safe `track_id ↔ member_id` registry
- Negative temp IDs for entry-pending persons (not yet seen at machine)
- Auto-expires tracks older than 300 seconds (5 min)
- Confidence threshold 0.45 to lock an identity

---

## Face Model

| Setting | Value |
|---------|-------|
| Model | `buffalo_sc` (fast) |
| Embedding | 512-dim ArcFace |
| Threshold | 0.40 cosine similarity |
| Check interval | Every 30 frames (~1s at 30fps) |
| Identity window | 10 seconds |

`buffalo_l` is more accurate but slower — fine on Mac Mini, too slow for Pi without Hailo.

---

## Key Files

| File | Purpose |
|------|---------|
| `user_tracking/person_db.py` | Track ↔ member registry |
| `user_tracking/entry_camera.py` | Door camera, ArcFace |
| `user_tracking/gym_tracker.py` | Floor IoU tracker |
| `face/recognizer.py` | InsightFace wrapper, IdentityWindow |
| `data/members/` | Face photos (NOT in git) |

---

## Enrol a Member

```bash
make enrol NAME="Matthew"
# Webcam capture → 512-dim embedding → Supabase members table
```

---

## Status

- [x] PersonDB built and tested
- [x] GymTracker IoU matching
- [x] EntryCamera + IdentityWindow
- [x] Face recognizer wrapper
- [ ] Members enrolled (run `make enrol` for each person)
- [ ] Tested end-to-end: door → machine → session attributed

---

## Related

- [[System/YOLO Pipeline]] — bounding boxes used by GymTracker
- [[System/Database Schema]] — members and sessions tables
- [[System/Architecture]] — where this fits in the full system
- [[Hardware/Machine Pi]] — entry Pi + machine Pi hardware
- [[Hardware/Camera Placement]] — door camera mounting
- [[Decisions/Stack Choices]] — why buffalo_sc, why IoU tracker
- [[Data/Training Requirements]] — member enrolment steps
