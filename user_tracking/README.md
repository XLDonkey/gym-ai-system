# user_tracking — Person Identification & Gym Floor Tracking (Project 3)

Knows who every person is from the moment they walk through the door.

| File | Purpose |
|------|---------|
| `entry_camera.py` | Door-facing Pi: ArcFace face recognition → registers identity in PersonDB |
| `gym_tracker.py` | Machine Pi: IoU bounding box tracker → links tracks to member identities |
| `person_db.py` | Thread-safe registry: `track_id ↔ member_id`, shared across all Pis |

## How it works

```
Member walks in
  → entry_camera.py runs InsightFace over 10s (IdentityWindow)
  → Best match → PersonDB.register_from_entry(member_id, name, confidence)

Member sits at a machine
  → gym_tracker.py assigns track_id to their bounding box each frame
  → GymTracker.closest_track(machine_roi) → track_id
  → PersonDB.get_member(track_id) → member name + Supabase UUID
  → db.start_session(member_id, machine_id) → session logged
```

## Quick start

```bash
# 1. Enrol members (run once per person)
make enrol NAME="Matthew"

# 2. Start entry camera (door-facing Pi)
python3 user_tracking/entry_camera.py

# 3. GymTracker runs automatically inside pi/main.py on each machine Pi
```

## Integration in pi/main.py

```python
from user_tracking.gym_tracker import GymTracker
from user_tracking.person_db import PersonDB

person_db = PersonDB()
tracker   = GymTracker(person_db=person_db, recognizer=face_recognizer, zone=f"machine:{MACHINE_ID}")

# In main loop:
tracks  = tracker.update(frame, yolo_person_detections)
closest = tracker.closest_track(MACHINE_ZONE_ROI)
if closest:
    member = person_db.get_member(closest.track_id)
    if member and member.member_id:
        db.start_session(member.member_id, MACHINE_ID)
```
