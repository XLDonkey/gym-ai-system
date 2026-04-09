"""
XL Fitness AI — Gym Floor Person Tracker
Tracks people across the gym floor and links them to member identities.

Each machine Pi runs its own local GymTracker instance for its camera zone.
PersonDB is the shared store (populated by entry_camera.py at the door).

Tracker design:
  Simple IoU (intersection-over-union) matching — no DeepSORT dependency.
  Good enough for single-camera zones where entry_camera.py already provides
  the identity. For multi-camera tracking, switch to ByteTrack (same API,
  swap _match() — PersonDB is unchanged).

Face recognition mid-floor:
  When a new track appears that wasn't seen at entry, GymTracker runs
  face recognition every FACE_RECOG_EVERY frames until identity is locked.
  This catches members who bypassed the entry camera.

Integration with machine Pi (pi/main.py):
    tracker.update(frame, yolo_detections)
    person = person_db.get_member(closest_track_id)
    if person:
        db.start_session(person.member_id, machine_id=cfg.MACHINE_ID)
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from face.face_recognizer import FaceRecognizer
from user_tracking.person_db import PersonDB


@dataclass
class Track:
    track_id:   int
    bbox_xyxy:  tuple    # (x1, y1, x2, y2)
    confidence: float    # YOLO detection confidence
    age:        int      # frames since first seen
    missed:     int      # consecutive frames without a match


class GymTracker:
    """
    IoU-based person tracker with face recognition for mid-floor identity lookup.

    Args:
        person_db:              Shared PersonDB (same instance as EntryCamera).
        recognizer:             Loaded FaceRecognizer (optional).
        zone:                   Zone label, e.g. "machine:xlf-pi-001".
        iou_threshold:          Minimum IoU to match detection → track (default 0.30).
        max_missed:             Frames before a track is dropped (default 45 = ~1.5s).
        face_recog_every:       Run face recognition every N frames for unknown tracks.
        face_lock_conf:         Cosine similarity to lock identity and stop checking.
        enabled:                Set False to disable (returns empty list every frame).
    """

    def __init__(
        self,
        person_db:         PersonDB,
        recognizer:        Optional[FaceRecognizer] = None,
        zone:              str   = "floor",
        iou_threshold:     float = 0.30,
        max_missed:        int   = 45,
        face_recog_every:  int   = 30,
        face_lock_conf:    float = 0.40,
        enabled:           bool  = True,
    ):
        self.person_db        = person_db
        self.recognizer       = recognizer
        self.zone             = zone
        self.iou_threshold    = iou_threshold
        self.max_missed       = max_missed
        self.face_recog_every = face_recog_every
        self.face_lock_conf   = face_lock_conf
        self.enabled          = enabled

        self._tracks:    list[Track] = []
        self._next_id:   int         = 1
        self._frame_n:   int         = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray, detections: list) -> list[Track]:
        """
        Update tracker with YOLO person detections for one frame.

        Args:
            frame:       Current BGR frame from OpenCV.
            detections:  List of dicts with:
                           "bbox_xyxy": (x1, y1, x2, y2)
                           "confidence": float  (optional)

        Returns:
            List of currently visible (missed=0) Track objects.
        """
        if not self.enabled:
            return []

        self._frame_n += 1
        boxes = [d["bbox_xyxy"] for d in detections]
        confs  = [d.get("confidence", 1.0) for d in detections]

        matched_t, matched_d, lost_t, new_d = self._match(boxes)

        for ti, di in zip(matched_t, matched_d):
            t = self._tracks[ti]
            t.bbox_xyxy  = boxes[di]
            t.confidence = confs[di]
            t.missed     = 0
            t.age       += 1
            self.person_db.update_track(t.track_id, self.zone)

        for ti in lost_t:
            self._tracks[ti].missed += 1

        for di in new_d:
            t = Track(
                track_id   = self._next_id,
                bbox_xyxy  = boxes[di],
                confidence = confs[di],
                age        = 1,
                missed     = 0,
            )
            self._next_id += 1
            self._tracks.append(t)
            self.person_db.update_track(t.track_id, self.zone)

        self._tracks = [t for t in self._tracks if t.missed < self.max_missed]

        if self.recognizer and (self._frame_n % self.face_recog_every == 0):
            self._run_face_recognition(frame)

        return [t for t in self._tracks if t.missed == 0]

    def closest_track(self, roi_xyxy: tuple) -> Optional[Track]:
        """
        Return the track whose centre is closest to the centre of the given ROI.
        Use this in pi/main.py to find which tracked person is at the machine.
        """
        if not self._tracks:
            return None

        rx = (roi_xyxy[0] + roi_xyxy[2]) / 2
        ry = (roi_xyxy[1] + roi_xyxy[3]) / 2
        best, best_dist = None, float("inf")

        for t in self._tracks:
            if t.missed > 0:
                continue
            tx = (t.bbox_xyxy[0] + t.bbox_xyxy[2]) / 2
            ty = (t.bbox_xyxy[1] + t.bbox_xyxy[3]) / 2
            dist = ((tx - rx) ** 2 + (ty - ry) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = t

        return best

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw bounding boxes and identity labels on frame."""
        try:
            import cv2
        except ImportError:
            return frame

        for t in self._tracks:
            if t.missed > 0:
                continue
            x1, y1, x2, y2 = [int(v) for v in t.bbox_xyxy]
            p      = self.person_db.get_member(t.track_id)
            name   = (p.member_name if p and p.member_name else f"#{t.track_id}")
            colour = (30, 200, 30) if (p and p.member_id) else (30, 130, 230)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, name, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

        return frame

    # ── Internal ───────────────────────────────────────────────────────────────

    def _match(self, det_boxes: list) -> tuple:
        """Greedy IoU matching — O(N×M) but N and M are tiny for one camera."""
        if not self._tracks or not det_boxes:
            return [], [], list(range(len(self._tracks))), list(range(len(det_boxes)))

        iou = np.zeros((len(self._tracks), len(det_boxes)))
        for i, t in enumerate(self._tracks):
            for j, b in enumerate(det_boxes):
                iou[i, j] = _iou(t.bbox_xyxy, b)

        matched_t, matched_d, used_t, used_d = [], [], set(), set()
        while True:
            if iou.size == 0:
                break
            ti, di = np.unravel_index(np.argmax(iou), iou.shape)
            if iou[ti, di] < self.iou_threshold:
                break
            matched_t.append(ti); matched_d.append(di)
            used_t.add(ti);       used_d.add(di)
            iou[ti, :] = 0;       iou[:, di] = 0

        lost = [i for i in range(len(self._tracks)) if i not in used_t]
        new  = [j for j in range(len(det_boxes)) if j not in used_d]
        return matched_t, matched_d, lost, new

    def _run_face_recognition(self, frame: np.ndarray):
        """Run face ID on visible tracks not yet locked to a member."""
        for t in self._tracks:
            if t.missed > 0:
                continue
            p = self.person_db.get_member(t.track_id)
            if p and p.face_locked:
                continue
            result = self.recognizer.identify_from_frame(frame, t.bbox_xyxy)
            if result.face_found and result.member_id:
                self.person_db.link_track_to_member(
                    track_id    = t.track_id,
                    member_id   = result.member_id,
                    member_name = result.member_name,
                    confidence  = result.confidence,
                )


def _iou(a: tuple, b: tuple) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0
