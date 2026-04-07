"""
XL Fitness AI Overseer — Machine Engagement Detector
Answers one question: is this person actually ON the machine, or just nearby?

The Problem
───────────
YOLO detects any human in frame — a PT standing next to their client,
someone watching, a person walking past, a bystander. Without engagement
detection, all of these start a session and pollute the data.

The Solution
────────────
Two independent checks must BOTH pass before a session starts:

  1. ZONE CHECK  — Is the person's body centre inside the machine seat zone?
                   Defined as a normalised bounding box in config.py.
                   (x1, y1, x2, y2) where 0,0 = top-left of frame.

  2. POSE CHECK  — Is their pose consistent with sitting on the machine?
                   For seated cable/weight machines:
                     • Hip keypoints are below shoulder keypoints (seated)
                     • At least one wrist is raised above their shoulders
                       (reaching for the bar/handle)
                     • OR: elbow angle is within the working range of the exercise

Both checks use the normalised keypoints from YOLO (0–1 range).

A confirmation buffer prevents false triggers: the person must pass both
checks for N consecutive frames (default 10, ~0.33s) before engagement
is declared. This stops a person briefly entering the zone from firing a session.

Usage
─────
    detector = EngagementDetector(
        zone_norm=(0.15, 0.10, 0.85, 0.95),  # machine seat zone in frame
        exercise="lat_pulldown",
    )

    # In main loop
    engaged = detector.update(kps_normalised, bbox_xyxy_pixels, frame_w, frame_h)

    if engaged and not session_active:
        start_session()

Calibration
───────────
Set MACHINE_ZONE_ROI in config.py to cover the seat + immediate
working space. Use SHOW_PREVIEW=True to see the overlay.

Tip: make the zone generous enough to cover the full ROM of the lift
but not so wide it catches the PT standing 1 metre away.
"""

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional


# ── COCO keypoint indices (same as main.py) ────────────────────────────────────
KP_NOSE        = 0
KP_L_SHOULDER  = 5
KP_R_SHOULDER  = 6
KP_L_ELBOW     = 7
KP_R_ELBOW     = 8
KP_L_WRIST     = 9
KP_R_WRIST     = 10
KP_L_HIP       = 11
KP_R_HIP       = 12

# Minimum keypoint confidence to trust a reading
KP_CONF_MIN = 0.20

# Consecutive frames required to confirm engagement / disengagement
ENGAGE_FRAMES_REQUIRED   = 10   # ~0.33s at 30fps
DISENGAGE_FRAMES_REQUIRED = 45  # ~1.5s — don't drop out on a single missed frame


@dataclass
class EngagementState:
    engaged:        bool   # True = person is on the machine
    in_zone:        bool   # Zone check result
    pose_ok:        bool   # Pose check result
    zone_ratio:     float  # How much of person's bbox overlaps the zone (0–1)
    hip_y:          float  # Normalised hip y position (debug)
    wrist_raised:   bool   # Whether wrists are above shoulders


class EngagementDetector:
    """
    Detects whether the person in frame is actively seated on the machine.

    Call update() every frame when YOLO detects a person.
    Call reset() when the machine becomes idle.
    """

    def __init__(
        self,
        zone_norm:    tuple  = (0.10, 0.05, 0.90, 0.95),
        exercise:     str    = "lat_pulldown",
        min_overlap:  float  = 0.40,   # fraction of bbox that must be inside zone
        enabled:      bool   = True,
    ):
        self.enabled     = enabled
        self.zone_norm   = zone_norm    # (x1,y1,x2,y2) normalised
        self.exercise    = exercise
        self.min_overlap = min_overlap

        self._engage_buf:    deque[bool] = deque(maxlen=ENGAGE_FRAMES_REQUIRED)
        self._disengage_buf: deque[bool] = deque(maxlen=DISENGAGE_FRAMES_REQUIRED)
        self._engaged        = False
        self._last_state:    Optional[EngagementState] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def engaged(self) -> bool:
        return self._engaged if self.enabled else True

    def update(
        self,
        kps:      "np.ndarray",   # shape (17, 3) — normalised x,y,conf
        bbox_px:  tuple,          # (x1, y1, x2, y2) in pixels
        frame_w:  int,
        frame_h:  int,
    ) -> bool:
        """
        Process one frame. Returns True if the person is engaged with the machine.

        kps: normalised keypoints from YOLO — already divided by frame w/h.
        bbox_px: raw pixel bounding box from YOLO boxes.xyxy.
        """
        if not self.enabled:
            return True

        # Normalise bbox to 0–1
        bx1 = bbox_px[0] / frame_w
        by1 = bbox_px[1] / frame_h
        bx2 = bbox_px[2] / frame_w
        by2 = bbox_px[3] / frame_h

        state = EngagementState(
            engaged      = False,
            in_zone      = False,
            pose_ok      = False,
            zone_ratio   = 0.0,
            hip_y        = 0.0,
            wrist_raised = False,
        )

        # ── Check 1: Zone overlap ──────────────────────────────────────────────
        state.zone_ratio = _bbox_overlap_ratio(
            (bx1, by1, bx2, by2), self.zone_norm
        )
        state.in_zone = state.zone_ratio >= self.min_overlap

        # ── Check 2: Pose engagement ───────────────────────────────────────────
        state.pose_ok, state.hip_y, state.wrist_raised = _check_pose_engaged(
            kps, self.exercise
        )

        # Both must pass
        candidate = state.in_zone and state.pose_ok

        # ── Confirmation buffers ───────────────────────────────────────────────
        self._engage_buf.append(candidate)
        self._disengage_buf.append(not candidate)

        if not self._engaged:
            # Require N consecutive positive frames to engage
            if (
                len(self._engage_buf) == ENGAGE_FRAMES_REQUIRED
                and all(self._engage_buf)
            ):
                self._engaged = True
                print(
                    f"[engage] ENGAGED — zone_overlap={state.zone_ratio:.2f}  "
                    f"pose_ok={state.pose_ok}  hip_y={state.hip_y:.2f}"
                )
        else:
            # Require N consecutive negative frames to disengage
            if (
                len(self._disengage_buf) == DISENGAGE_FRAMES_REQUIRED
                and all(self._disengage_buf)
            ):
                self._engaged = False
                print(f"[engage] DISENGAGED — person left or moved off machine")

        state.engaged   = self._engaged
        self._last_state = state
        return self._engaged

    def reset(self):
        """Call when machine goes idle."""
        self._engage_buf.clear()
        self._disengage_buf.clear()
        self._engaged = False

    @property
    def last_state(self) -> Optional[EngagementState]:
        return self._last_state

    # ── Debug overlay ──────────────────────────────────────────────────────────

    def draw_overlay(self, frame: "np.ndarray") -> "np.ndarray":
        """Draw the machine zone and engagement status on the frame."""
        import cv2
        if not self.enabled:
            return frame

        h, w = frame.shape[:2]
        zx1, zy1, zx2, zy2 = self.zone_norm
        px1 = int(zx1 * w)
        py1 = int(zy1 * h)
        px2 = int(zx2 * w)
        py2 = int(zy2 * h)

        if self._engaged:
            colour = (0, 255, 0)      # green = engaged
            label  = "ON MACHINE"
        elif self._last_state and self._last_state.in_zone:
            colour = (0, 200, 255)    # yellow = in zone but pose not right
            label  = "IN ZONE"
        else:
            colour = (80, 80, 80)     # grey = not in zone
            label  = "ZONE"

        cv2.rectangle(frame, (px1, py1), (px2, py2), colour, 2)
        cv2.putText(frame, label, (px1 + 4, py1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

        if self._last_state:
            s = self._last_state
            cv2.putText(
                frame,
                f"zone={s.zone_ratio:.2f}  pose={'OK' if s.pose_ok else 'NO'}  "
                f"wrist={'up' if s.wrist_raised else 'down'}",
                (px1, py2 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1,
            )

        return frame


# ── Pose engagement logic ──────────────────────────────────────────────────────

def _check_pose_engaged(kps, exercise: str) -> tuple[bool, float, bool]:
    """
    Check if the person's pose indicates they are actively using the machine.

    Returns (engaged: bool, hip_y: float, wrist_raised: bool)

    Logic per exercise type:
      lat_pulldown / cable_row / seated_*:
        The person must be SEATED — hips are roughly level with or
        below their shoulders in the normalised frame (higher y value).

        AND at least one wrist must be raised toward the bar/handle
        (wrist y < shoulder y, i.e. wrists are higher in the frame).

      standing_*:
        Hips are above mid-frame, arms in working position.
        (extend as needed for other exercises)
    """
    def kp(idx) -> tuple[float, float, float]:
        """Return (x, y, conf) for keypoint idx."""
        return float(kps[idx][0]), float(kps[idx][1]), float(kps[idx][2])

    # Gather key joints
    ls_x, ls_y, ls_c = kp(KP_L_SHOULDER)
    rs_x, rs_y, rs_c = kp(KP_R_SHOULDER)
    lh_x, lh_y, lh_c = kp(KP_L_HIP)
    rh_x, rh_y, rh_c = kp(KP_R_HIP)
    lw_x, lw_y, lw_c = kp(KP_L_WRIST)
    rw_x, rw_y, rw_c = kp(KP_R_WRIST)

    # Average shoulder and hip y (higher y = lower in frame)
    shoulder_y = _weighted_avg([(ls_y, ls_c), (rs_y, rs_c)])
    hip_y      = _weighted_avg([(lh_y, lh_c), (rh_y, rh_c)])

    # Average wrist y
    wrist_y    = _weighted_avg([(lw_y, lw_c), (rw_y, rw_c)])

    if shoulder_y < 0 or hip_y < 0:
        # Not enough confident keypoints to decide
        return False, -1.0, False

    exercise_lower = exercise.lower()

    if any(k in exercise_lower for k in ("lat_pulldown", "pulldown", "cable_row",
                                          "seated", "row", "chest_press", "shoulder_press",
                                          "leg_press", "leg_extension", "leg_curl",
                                          "pec_deck", "fly")):
        # SEATED MACHINE LOGIC
        # Hips should be lower in frame than shoulders (larger y = lower)
        # Threshold: hips at least 10% of frame height below shoulders
        seated = (hip_y - shoulder_y) > 0.08

        # Wrists raised above shoulders (reaching for bar) — wrist y < shoulder y
        wrist_raised = (wrist_y >= 0 and shoulder_y - wrist_y > 0.05)

        # Must be seated AND (wrists up OR arms in mid-exercise position)
        # Mid-exercise: wrists are between hip level and shoulder level
        arms_working = (wrist_y >= 0 and hip_y > wrist_y > 0)

        engaged = seated and (wrist_raised or arms_working)
        return engaged, round(hip_y, 3), wrist_raised

    else:
        # STANDING EXERCISE FALLBACK
        # Just check that the person is upright and their hands are in use
        upright   = shoulder_y < 0.6          # shoulders in top 60% of frame
        arms_up   = wrist_y >= 0 and wrist_y < shoulder_y + 0.3
        engaged   = upright and arms_up
        return engaged, round(hip_y, 3), arms_up


def _weighted_avg(points: list[tuple[float, float]]) -> float:
    """
    Weighted average of y values, using confidence as weight.
    Returns -1 if no points have sufficient confidence.
    """
    total_w = 0.0
    total_v = 0.0
    for val, conf in points:
        if conf >= KP_CONF_MIN:
            total_w += conf
            total_v += val * conf
    if total_w < KP_CONF_MIN:
        return -1.0
    return total_v / total_w


def _bbox_overlap_ratio(bbox_a: tuple, bbox_b: tuple) -> float:
    """
    Fraction of bbox_a that overlaps with bbox_b.
    Both are normalised (x1,y1,x2,y2) tuples.
    """
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a       = max(1e-9, (ax2 - ax1) * (ay2 - ay1))
    return intersection / area_a
