"""
XL Fitness AI — Entry Camera
Monitors the gym entrance and identifies members as they walk in.

A dedicated Pi (or a spare camera on an existing Pi) faces the door.
When a face is detected it runs ArcFace recognition via the IdentityWindow
accumulation strategy: over the first 10 seconds a member is in frame,
it collects multiple reads and returns the highest-confidence match.
The result is registered in PersonDB so every Pi on the gym floor
immediately knows the person's name when they appear at a machine.

Hardware:
  • Any Raspberry Pi 5 with Pi Camera Module 3 (no Hailo needed)
  • Place 1–2m from the door, angled to catch faces as people enter
  • A separate Pi is ideal so recognition runs independently of machine Pis

Usage (standalone):
    python3 user_tracking/entry_camera.py

Usage (integrated):
    camera = EntryCamera(person_db=db, recognizer=recognizer)
    camera.start()     # background thread — returns immediately
    # ... gym runs ...
    camera.stop()
"""

import os
import sys
import threading
import time
from typing import Optional

import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False
    print("[entry_camera] WARNING: opencv-python not installed")

# Re-use existing face recognizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from face.face_recognizer import FaceRecognizer, IdentityWindow
from user_tracking.person_db import PersonDB


class EntryCamera:
    """
    Background thread that watches the entrance camera and populates PersonDB.

    Args:
        person_db:              Shared PersonDB (also used by GymTracker on each Pi).
        recognizer:             Loaded FaceRecognizer with member embeddings cached.
        camera_index:           OpenCV camera index (0 = first USB/Pi camera).
        recognition_interval_s: Seconds between face recognition runs (Pi CPU is slow).
        zone:                   Zone label stored with each entry ("entry").
        enabled:                Set False to disable.
    """

    def __init__(
        self,
        person_db:              PersonDB,
        recognizer:             FaceRecognizer,
        camera_index:           int   = 0,
        capture_w:              int   = 1280,
        capture_h:              int   = 720,
        recognition_interval_s: float = 2.0,
        zone:                   str   = "entry",
        enabled:                bool  = True,
    ):
        self.person_db              = person_db
        self.recognizer             = recognizer
        self.camera_index           = camera_index
        self.capture_w              = capture_w
        self.capture_h              = capture_h
        self.recognition_interval_s = recognition_interval_s
        self.zone                   = zone
        self.enabled                = enabled

        self._stop      = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Open identity windows per-person entering
        self._windows: list[IdentityWindow] = []

        self.total_entries      = 0
        self.identified_entries = 0

    def start(self):
        """Start monitoring in a background daemon thread."""
        if not self.enabled:
            print("[entry_camera] Disabled")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="EntryCamera")
        self._thread.start()
        print(f"[entry_camera] Started  camera={self.camera_index}  zone={self.zone}")

    def stop(self):
        """Signal the thread to stop and wait for it to exit."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        print("[entry_camera] Stopped")

    # ── Background thread ──────────────────────────────────────────────────────

    def _run(self):
        if not _CV2:
            print("[entry_camera] ERROR: opencv not available")
            return

        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.capture_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_h)

        if not cap.isOpened():
            print(f"[entry_camera] ERROR: could not open camera {self.camera_index}")
            return

        last_recog = 0.0

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            now = time.time()
            if now - last_recog >= self.recognition_interval_s:
                last_recog = now
                self._process_frame(frame, now)

            self._flush_windows(now)

        cap.release()

    def _process_frame(self, frame: np.ndarray, now: float):
        """Run face recognition on one frame, feed open identity windows."""
        if not self.recognizer.ready:
            return

        result = self.recognizer.identify_from_frame(frame)
        if not result.face_found:
            return

        # Match to an open window or open a new one
        for w in self._windows:
            if w.open:
                w.add(result)
                return

        # No open window — new person entering
        w = IdentityWindow(duration_s=10.0)
        w.add(result)
        self._windows.append(w)

    def _flush_windows(self, now: float):
        """Finalise expired windows and register results in PersonDB."""
        still_open = []
        for w in self._windows:
            if w.open:
                still_open.append(w)
                continue

            best = w.best()
            if best is None:
                continue

            self.total_entries += 1
            if best.member_id and best.confidence >= 0.35:
                self.identified_entries += 1
                self.person_db.register_from_entry(
                    member_id   = best.member_id,
                    member_name = best.member_name,
                    confidence  = best.confidence,
                    zone        = self.zone,
                )
                print(
                    f"[entry_camera] IN: {best.member_name}  "
                    f"conf={best.confidence:.2f}  "
                    f"total={self.total_entries}"
                )
            else:
                self.person_db.register_unknown(zone=self.zone)
                print(
                    f"[entry_camera] UNKNOWN entered  "
                    f"best_conf={best.confidence:.2f}  "
                    f"total={self.total_entries}"
                )

        self._windows = still_open


# ── Standalone runner ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from members.db_client import SupabaseClient
    import pi.config as cfg

    print("XL Fitness — Entry Camera (standalone)")
    print("Press Ctrl+C to stop\n")

    db          = SupabaseClient(cfg.SUPABASE_URL, cfg.SUPABASE_SERVICE_KEY)
    recognizer  = FaceRecognizer()
    recognizer.load_members(db.get_all_members())

    person_db   = PersonDB()
    camera      = EntryCamera(
        person_db  = person_db,
        recognizer = recognizer,
        zone       = "entry",
    )

    camera.start()

    try:
        while True:
            time.sleep(10)
            active = person_db.all_active()
            print(f"[entry_camera] {len(active)} active persons in gym")
            person_db.expire_old_tracks()
    except KeyboardInterrupt:
        camera.stop()
