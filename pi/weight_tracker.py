"""
XL Fitness AI Overseer — Weight Stack Tracker
Confirms a rep is real by verifying the weight stack actually moved.

Problem it solves
─────────────────
Joint angles alone cannot distinguish a real rep from:
  - Moving the bar/handle without engaging the weight pin
  - Performing the motion without sitting in the machine
  - Stretching, adjusting, or posing

Solution
────────
Track a Region of Interest (ROI) covering the weight stack using dense
optical flow. If the weight stack doesn't show significant vertical
displacement during what the angle detector thinks is a rep, the rep
is rejected as a false positive.

Usage
─────
    tracker = WeightStackTracker(
        roi_norm=(0.72, 0.05, 0.92, 0.88),  # normalised (x1,y1,x2,y2)
        frame_w=1280,
        frame_h=720,
    )

    # In main loop — call every frame when a person is present
    tracker.update(frame)

    # When angle state machine fires a candidate rep:
    if tracker.weight_moved_during_rep():
        # count the rep
    else:
        # reject — phantom rep, weight didn't move

Calibration
───────────
Set the ROI in config.py (WEIGHT_STACK_ROI) to cover the visible
weight stack plates/pin. Use SHOW_PREVIEW=True to see the overlay.

ROI format: normalised floats (0.0–1.0) → (x1, y1, x2, y2)
  x1=0.72 means 72% across the frame width
  The default covers the right side of a side-on camera view.

Tuning
──────
WEIGHT_MOVE_PX_THRESHOLD  — minimum optical flow displacement (pixels).
  Increase if phantom reps still get through.
  Decrease if real reps are being rejected (heavy weight moves less).

WEIGHT_MOVE_FRAME_RATIO   — fraction of frames within the rep that must
  show movement. Default 0.3 = at least 30% of frames during the rep
  need to show the weight moving.
"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ── Defaults (override in config.py) ──────────────────────────────────────────

# Normalised ROI covering the weight stack in the camera frame.
# (x1, y1, x2, y2) — all values 0.0–1.0
# Default: right 20% of the frame, full height — adjust per camera placement
DEFAULT_ROI_NORM = (0.72, 0.05, 0.92, 0.88)

# Minimum mean optical flow magnitude (pixels/frame) to count as "weight moving"
WEIGHT_MOVE_PX_THRESHOLD = 1.5

# Fraction of frames in the rep window that must show movement
WEIGHT_MOVE_FRAME_RATIO = 0.30

# Rolling history window — stores per-frame movement magnitude
HISTORY_FRAMES = 150   # 5 seconds at 30fps


@dataclass
class WeightSample:
    magnitude: float   # mean optical flow magnitude in ROI
    moving: bool       # True if magnitude > threshold


class WeightStackTracker:
    """
    Tracks weight stack movement using dense optical flow on an ROI.

    Call update(frame) every frame.
    Call start_rep() when the angle state machine enters the 'down' phase.
    Call weight_moved_during_rep() to validate a candidate rep.
    """

    def __init__(
        self,
        roi_norm:         tuple  = DEFAULT_ROI_NORM,
        frame_w:          int    = 1280,
        frame_h:          int    = 720,
        move_threshold:   float  = WEIGHT_MOVE_PX_THRESHOLD,
        move_frame_ratio: float  = WEIGHT_MOVE_FRAME_RATIO,
        enabled:          bool   = True,
    ):
        self.enabled          = enabled
        self.move_threshold   = move_threshold
        self.move_frame_ratio = move_frame_ratio

        # Convert normalised ROI to pixel coords
        x1n, y1n, x2n, y2n = roi_norm
        self.roi = (
            int(x1n * frame_w),
            int(y1n * frame_h),
            int(x2n * frame_w),
            int(y2n * frame_h),
        )

        self._prev_gray: Optional[np.ndarray] = None
        self._history: deque[WeightSample]    = deque(maxlen=HISTORY_FRAMES)
        self._rep_start_idx: Optional[int]    = None   # index in history when rep began
        self._frame_count: int                = 0

        # For debug overlay
        self.last_magnitude: float = 0.0
        self.last_moving: bool     = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray):
        """
        Process one frame. Call this every frame in the main loop.
        Updates internal history — must be called before weight_moved_during_rep().
        """
        if not self.enabled:
            return

        x1, y1, x2, y2 = self.roi
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        # Downsample for speed on Pi
        gray = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2))

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            self._frame_count += 1
            return

        # Dense optical flow (Farneback) — fast on small ROI
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray,
            None,
            pyr_scale=0.5,
            levels=2,
            winsize=10,
            iterations=2,
            poly_n=5,
            poly_sigma=1.1,
            flags=0,
        )

        # Use vertical component (y) — weight moves up/down
        vert_flow = flow[..., 1]
        magnitude  = float(np.mean(np.abs(vert_flow)))
        moving     = magnitude > self.move_threshold

        self._history.append(WeightSample(magnitude=magnitude, moving=moving))
        self._prev_gray = gray
        self._frame_count += 1

        self.last_magnitude = magnitude
        self.last_moving    = moving

    def start_rep(self):
        """
        Call when the angle state machine enters the 'down' phase.
        Marks the start of the rep window in the history buffer.
        """
        self._rep_start_idx = len(self._history)

    def weight_moved_during_rep(self) -> bool:
        """
        Returns True if the weight stack moved sufficiently since start_rep().

        Always returns True if tracker is disabled (fail-open — don't block
        reps if the ROI is not configured).
        """
        if not self.enabled:
            return True

        if self._rep_start_idx is None:
            # start_rep() was never called — can't validate
            return True

        # Slice the samples collected since rep start
        samples = list(self._history)[self._rep_start_idx:]

        if len(samples) < 3:
            # Too few frames to make a judgement — pass it through
            return True

        moving_frames = sum(1 for s in samples if s.moving)
        ratio         = moving_frames / len(samples)

        moved = ratio >= self.move_frame_ratio

        if not moved:
            print(
                f"[weight] Rep REJECTED — weight barely moved  "
                f"({moving_frames}/{len(samples)} frames  "
                f"ratio={ratio:.2f}  threshold={self.move_frame_ratio:.2f})"
            )

        self._rep_start_idx = None  # reset for next rep
        return moved

    def reset(self):
        """Call at session end or when the machine becomes idle."""
        self._prev_gray    = None
        self._history.clear()
        self._rep_start_idx = None

    # ── Debug overlay ──────────────────────────────────────────────────────────

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the weight stack ROI and current flow magnitude on the frame.
        Only useful when SHOW_PREVIEW=True during calibration.
        """
        if not self.enabled:
            return frame

        x1, y1, x2, y2 = self.roi
        colour = (0, 255, 0) if self.last_moving else (0, 128, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            frame,
            f"Weight: {self.last_magnitude:.2f}px {'MOVING' if self.last_moving else 'still'}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1,
        )
        return frame
