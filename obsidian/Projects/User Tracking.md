---
title: Project 3 — User Tracking
tags: [project, user-tracking, face-id, arcface, active]
status: built-needs-enrolment
created: 2026-04-09
updated: 2026-04-10
---

# Project 3 — User Tracking

> Face recognised at the door → followed to the machine → session attributed automatically.

Part of [[Home]]. Uses InsightFace ArcFace for recognition, IoU bounding box tracking for floor following.

---

## The Full Flow

```
Member walks through door (Entry Pi camera)
  → InsightFace ArcFace — runs every 2 seconds
  → IdentityWindow (10s) — accumulates face reads, takes most confident
  → PersonDB.register_from_entry(member_id, name, confidence)
     (negative temp_id until seen at a machine)

Member walks to machine (Machine Pi camera)
  → YOLO detects person bounding box
  → GymTracker.update(frame, detections) → IoU matching → Track
  → GymTracker.closest_track(machine_zone_roi) → this Track
  → PersonDB.get_member(track_id) → TrackedPerson(name, member_id)
  → members_db.create_session(member_id, machine_id) → Supabase

Rep counted → attributed to member_id in Supabase
Session closed → db.close_session() → lifetime stats update
```

---

## Components

### EntryCamera (`user_tracking/entry_camera.py`)
- Runs as background daemon thread on the **door-facing Pi**
- ArcFace recognition called every `recognition_interval_s` (2.0s)
- Uses **IdentityWindow** (10 seconds): collects all face reads in the window, locks the most confident match when window closes
- Registered in PersonDB under negative temp_id (e.g. -1, -2) until the floor GymTracker picks them up

### GymTracker (`user_tracking/gym_tracker.py`)
- Runs on **machine Pi** — one instance per camera zone
- Every frame: `update(frame, detections)` → greedy IoU matching
  - Each new detection matched to nearest existing track (highest IoU first)
  - IoU threshold: 0.30 — below this = new track
  - Missed frames: track kept alive for `max_missed=45` frames before deletion
- `closest_track(roi_xyxy)` → finds the track with centre closest to machine seat ROI
- Every 30 frames: runs face recognition on tracks where `face_locked=False`
- Locks identity once cosine similarity ≥ 0.40

### PersonDB (`user_tracking/person_db.py`)
Thread-safe in-memory `track_id ↔ member_id` registry.

```python
@dataclass
class TrackedPerson:
    track_id:    int              # negative = entry-pending, positive = floor track
    member_id:   Optional[str]    # Supabase UUID
    member_name: Optional[str]
    confidence:  float            # face match confidence
    first_seen:  float            # unix timestamp
    last_seen:   float
    last_zone:   Optional[str]    # "entry" / "floor" / machine_id
    face_locked: bool             # identity confirmed
```

Key methods:
| Method | Purpose |
|--------|---------|
| `register_from_entry(member_id, name, confidence, zone)` | Door camera registers known member |
| `register_unknown(zone)` | Door camera registers unidentified person |
| `update_track(track_id, zone)` | Floor tracker updates last_seen |
| `link_track_to_member(track_id, member_id, name, confidence)` | Face match confirmed |
| `get_member(track_id)` | Returns TrackedPerson or None |
| `expire_old_tracks()` | Removes tracks idle > 300s |

---

## Face Recognition Details

| Setting | Value | Where set |
|---------|-------|-----------|
| Model | `buffalo_sc` | `pi/config.py` → `FACE_MODEL` |
| Embedding | 512-dim ArcFace vector | InsightFace |
| Match threshold | 0.40 cosine similarity | `FACE_THRESHOLD` |
| Run interval | Every 30 frames (~1s) | `FACE_CHECK_INTERVAL` |
| Identity window | 10 seconds | `FACE_IDENTITY_WINDOW_S` |

**buffalo_sc vs buffalo_l:**
- `buffalo_sc` — ~80ms on Pi 5, good accuracy. **Used on Pi.**
- `buffalo_l` — ~300ms on Pi 5, higher accuracy. Fine on Mac Mini, too slow for Pi without Hailo.

**Face embeddings stored:** Supabase `members` table as `FLOAT[]` array (512 values).
Loaded at startup by `FaceRecognizer.load_members(rows)`.

---

## Member Enrolment

```bash
make enrol NAME="Matthew"
# Opens webcam → captures 5 frames → averages embeddings
# → inserts to Supabase members table
# → updates recognizer cache (no restart needed)

# Other options:
python face/enroll_member.py --name "Sarah" --image /path/to/photo.jpg
python face/enroll_member.py --list          # show all enrolled members
python face/enroll_member.py --remove UUID   # deactivate a member
```

One enrolment per member is enough. Re-enrol in different lighting if recognition is poor.

---

## Anonymous Sessions

If `FACE_RECOGNITION_ENABLED = False` in `pi/config.py`:
- Sessions still tracked and reps counted
- Stored under `member_id = "anonymous"` in Supabase
- No member name shown on tablet

Useful for: testing before members are enrolled, machines without face recognition.

---

## Key Files

| File | Purpose |
|------|---------|
| `user_tracking/person_db.py` | Thread-safe track ↔ member registry |
| `user_tracking/entry_camera.py` | Door camera daemon, ArcFace |
| `user_tracking/gym_tracker.py` | Floor IoU tracker |
| `face/recognizer.py` | InsightFace wrapper + IdentityWindow |
| `face/enroll_member.py` | CLI member enrolment tool |
| `members/db_client.py` | Supabase REST client |
| `data/members/` | Face photos — NOT in git |

---

## Status

- [x] PersonDB — thread-safe, expire logic, negative temp_ids
- [x] GymTracker — IoU matching, face recognition integration
- [x] EntryCamera — daemon thread, IdentityWindow
- [x] FaceRecognizer — InsightFace wrapper
- [x] EnrollMember CLI — webcam + still photo modes
- [ ] Members enrolled (`make enrol` for each person)
- [ ] End-to-end test: door → machine → session attributed correctly

---

## Related

- [[System/YOLO Pipeline]] — bounding boxes used by GymTracker
- [[System/Engagement Detector]] — confirms seated pose before attributing session
- [[System/Database Schema]] — members, sessions tables
- [[System/Architecture]] — where this fits in the full system
- [[Hardware/Machine Pi]] — entry Pi (door) + machine Pi (floor)
- [[Hardware/Camera Placement]] — door camera mounting
- [[Decisions/Stack Choices]] — why buffalo_sc, why IoU tracker over DeepSORT
- [[Data/Training Requirements]] — member enrolment steps
