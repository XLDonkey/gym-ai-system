"""
XL Fitness AI — Person Identity Database
Thread-safe in-memory registry mapping tracker IDs to known member identities.

How it fits together:
  1. entry_camera.py recognises a face at the gym door → register_from_entry()
  2. gym_tracker.py assigns track_ids each frame → update_track()
  3. Once a floor track is near the entry track, link_track_to_member() is called
  4. Machine Pi queries get_member(track_id) → member_id → session is logged

Tracks expire after MAX_AGE_S (default 5 minutes) without a frame update.
Call expire_old_tracks() periodically (e.g. once per minute) to clean up.
"""

import time
import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrackedPerson:
    track_id:     int             # ByteTrack / IoU tracker ID
    member_id:    Optional[str]   # Supabase UUID, or None = unknown
    member_name:  Optional[str]   # e.g. "Matthew", or None
    confidence:   float           # face recognition cosine similarity
    first_seen:   float           # time.time() when first detected
    last_seen:    float           # time.time() of last frame update
    last_zone:    Optional[str]   # "entry", "floor", "machine:xlf-pi-001", etc.
    face_locked:  bool            # True = identity confirmed — skip further face checks


class PersonDB:
    """
    Thread-safe registry linking tracker IDs to member identities.

    Lifecycle:
      • Entry camera identifies face → register_from_entry() → negative temp_id
      • Gym tracker assigns a real track_id → update_track() or link_track_to_member()
      • Machine Pi queries → get_member(track_id)
      • Idle ≥ MAX_AGE_S → expire_old_tracks() removes the record

    Thread safety: a single mutex covers all reads and writes.
    """

    MAX_AGE_S  = 300.0   # 5 minutes before a track is considered gone
    LOCK_CONF  = 0.45    # cosine similarity to lock an identity (stop re-checking)

    def __init__(self, max_age_s: float = MAX_AGE_S, lock_conf: float = LOCK_CONF):
        self._lock         = threading.Lock()
        self._by_track:  dict[int, TrackedPerson] = {}
        self._by_member: dict[str, int]           = {}   # member_id → current track_id
        self.max_age_s     = max_age_s
        self.lock_conf     = lock_conf

    # ── Entry camera ───────────────────────────────────────────────────────────

    def register_from_entry(
        self,
        member_id:   str,
        member_name: str,
        confidence:  float,
        zone:        str = "entry",
    ) -> int:
        """
        Called when a face is recognised at the gym door.
        Uses a negative temp_id until gym_tracker assigns a real track_id.
        Returns the temp_id.
        """
        with self._lock:
            temp_id = -(len(self._by_track) + 1)
            now = time.time()
            self._by_track[temp_id] = TrackedPerson(
                track_id    = temp_id,
                member_id   = member_id,
                member_name = member_name,
                confidence  = confidence,
                first_seen  = now,
                last_seen   = now,
                last_zone   = zone,
                face_locked = confidence >= self.lock_conf,
            )
            self._by_member[member_id] = temp_id
            print(f"[person_db] Registered at entry: {member_name}  conf={confidence:.2f}")
            return temp_id

    def register_unknown(self, zone: str = "entry") -> int:
        """Register an unidentified person. Returns temp_id."""
        with self._lock:
            temp_id = -(len(self._by_track) + 1)
            now = time.time()
            self._by_track[temp_id] = TrackedPerson(
                track_id=temp_id, member_id=None, member_name=None,
                confidence=0.0, first_seen=now, last_seen=now,
                last_zone=zone, face_locked=False,
            )
            return temp_id

    # ── Gym tracker integration ────────────────────────────────────────────────

    def update_track(self, track_id: int, zone: str = "floor"):
        """
        Called every frame by gym_tracker for each active track.
        Creates an unknown entry if this track_id is new.
        """
        with self._lock:
            now = time.time()
            if track_id in self._by_track:
                self._by_track[track_id].last_seen = now
                self._by_track[track_id].last_zone  = zone
            else:
                self._by_track[track_id] = TrackedPerson(
                    track_id=track_id, member_id=None, member_name=None,
                    confidence=0.0, first_seen=now, last_seen=now,
                    last_zone=zone, face_locked=False,
                )

    def link_track_to_member(
        self,
        track_id:    int,
        member_id:   str,
        member_name: str,
        confidence:  float,
    ):
        """
        Link an existing floor track to a known member identity.
        No-op if the track is already locked.
        """
        with self._lock:
            if track_id not in self._by_track:
                return
            p = self._by_track[track_id]
            if p.face_locked:
                return
            p.member_id   = member_id
            p.member_name = member_name
            p.confidence  = confidence
            p.face_locked = confidence >= self.lock_conf
            self._by_member[member_id] = track_id
            print(f"[person_db] Linked track {track_id} → {member_name}  conf={confidence:.2f}")

    # ── Queries ────────────────────────────────────────────────────────────────

    def get_member(self, track_id: int) -> Optional[TrackedPerson]:
        """Return TrackedPerson for this track, or None."""
        with self._lock:
            return self._by_track.get(track_id)

    def get_track_for_member(self, member_id: str) -> Optional[int]:
        """Return current track_id for a known member, or None."""
        with self._lock:
            return self._by_member.get(member_id)

    def all_active(self) -> list:
        """Return all non-expired tracked persons."""
        with self._lock:
            now = time.time()
            return [
                p for p in self._by_track.values()
                if (now - p.last_seen) < self.max_age_s
            ]

    # ── Maintenance ────────────────────────────────────────────────────────────

    def expire_old_tracks(self):
        """Remove tracks not seen in max_age_s seconds. Call periodically."""
        with self._lock:
            now     = time.time()
            expired = [
                tid for tid, p in self._by_track.items()
                if (now - p.last_seen) >= self.max_age_s
            ]
            for tid in expired:
                p = self._by_track.pop(tid)
                if p.member_id and self._by_member.get(p.member_id) == tid:
                    del self._by_member[p.member_id]
            if expired:
                print(f"[person_db] Expired {len(expired)} track(s)")

    def clear(self):
        """Reset everything (e.g. gym closing time)."""
        with self._lock:
            self._by_track.clear()
            self._by_member.clear()
