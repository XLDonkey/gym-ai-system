"""
XL Fitness AI — Activity State Machine
Hierarchical phase gating for ONNX activity classifier output.

The problem
───────────
The 8 activity classes are NOT flat — they have a strict hierarchy:

    no_person / user_present
          ↓  (person sits on machine)
       on_machine
          ↓  (person starts a set)
    good_rep / bad_rep / false_rep / resting / half_rep

A "bad_rep" is only physically possible AFTER the user has been confirmed
on_machine.  If the ONNX model predicts "bad_rep" while the user is still
approaching the machine, that output is nonsensical.

The fix: mask invalid class logits to -∞ before softmax
────────────────────────────────────────────────────────
By masking before softmax (not after), the probabilities re-normalise over
only the valid classes.  The model can still "want" to predict a rep class
internally, but the gate forces it to choose the best valid class instead.

Phase IDLE
  Valid:  no_person (0), user_present (1), on_machine (2)
  → class 2 confirmed for ENGAGE_FRAMES consecutive frames → ENGAGED

Phase ENGAGED
  Valid:  on_machine (2), good_rep (3), bad_rep (4),
          false_rep (5), resting (6), half_rep (7)
  Note:   class 2 stays valid — user can settle back between sets
  → class 0 or 1 for DISENGAGE_FRAMES consecutive frames → IDLE

The transition thresholds provide hysteresis — a single bad frame doesn't
flip the phase, and brief occlusion doesn't kill an active session.
"""

import time

import numpy as np

from onnx_classifier import CLASS_NAMES, NUM_CLASSES, ACTIVE_CLASSES, REP_CLASSES

# ── Phase definitions ─────────────────────────────────────────────────────────

_IDLE_VALID     = frozenset({0, 1, 2})    # classes allowed before engagement
_ENGAGED_VALID  = frozenset({2, 3, 4, 5, 6, 7})   # classes allowed once engaged

# Transition thresholds (frames at ~30fps)
ENGAGE_FRAMES    = 10   # consecutive on_machine frames required to start a session
DISENGAGE_FRAMES = 45   # consecutive idle frames required to end a session (~1.5s)


class ActivityStateMachine:
    """
    Wraps the raw ONNX logits with phase-based class gating.

    Usage:
        sm     = ActivityStateMachine()
        result = sm.update(classifier_result.raw_logits)

        result.phase        → "IDLE" | "ENGAGED"
        result.class_id     → 0–7 (gated)
        result.class_name   → str
        result.confidence   → float (normalised over valid classes only)
        result.is_engaged   → bool
        result.is_rep_state → bool
        result.just_engaged → bool  (True on the exact frame engagement started)
        result.just_ended   → bool  (True on the exact frame session ended)
    """

    def __init__(
        self,
        engage_frames: int = ENGAGE_FRAMES,
        disengage_frames: int = DISENGAGE_FRAMES,
    ):
        self.phase             = "IDLE"
        self._engage_frames    = engage_frames
        self._disengage_frames = disengage_frames
        self._engage_count     = 0
        self._disengage_count  = 0
        self._last_result      = None

    def update(self, raw_logits: np.ndarray) -> "GatedResult":
        """
        Apply phase-gated softmax to raw ONNX logits.
        Call every frame (matches ONNXActivityClassifier.update()).

        Args:
            raw_logits: np.ndarray shape (8,) — raw LSTM output before softmax.
                        Pass None to get a safe idle result.
        Returns:
            GatedResult with the constrained single-class prediction.
        """
        if raw_logits is None:
            raw_logits = np.full(NUM_CLASSES, -1e9, dtype=np.float32)

        prev_phase = self.phase

        # ── Determine valid classes for current phase ─────────────────────────
        valid = _IDLE_VALID if self.phase == "IDLE" else _ENGAGED_VALID

        # ── Mask invalid classes (set to -∞ before softmax) ──────────────────
        masked = raw_logits.copy().astype(np.float64)
        for i in range(NUM_CLASSES):
            if i not in valid:
                masked[i] = -np.inf

        # ── Gated softmax ─────────────────────────────────────────────────────
        finite_mask = np.isfinite(masked)
        exp         = np.where(finite_mask, np.exp(masked - masked[finite_mask].max()), 0.0)
        total       = exp.sum()
        probs       = exp / total if total > 0 else exp

        class_id   = int(np.argmax(probs))
        confidence = float(probs[class_id])

        # ── Phase transitions ─────────────────────────────────────────────────
        if self.phase == "IDLE":
            if class_id == 2:          # on_machine
                self._engage_count += 1
                if self._engage_count >= self._engage_frames:
                    self.phase        = "ENGAGED"
                    self._engage_count = 0
            else:
                self._engage_count = 0
            self._disengage_count = 0

        elif self.phase == "ENGAGED":
            if class_id in {0, 1}:    # no_person or user_present → leaving
                self._disengage_count += 1
                if self._disengage_count >= self._disengage_frames:
                    self.phase            = "IDLE"
                    self._disengage_count = 0
            else:
                self._disengage_count = 0

        result = GatedResult(
            class_id    = class_id,
            confidence  = confidence,
            probs       = probs,
            phase       = self.phase,
            just_engaged = (prev_phase == "IDLE"     and self.phase == "ENGAGED"),
            just_ended   = (prev_phase == "ENGAGED"  and self.phase == "IDLE"),
            engage_progress = min(self._engage_count / self._engage_frames, 1.0),
        )
        self._last_result = result
        return result

    def reset(self):
        """Force back to IDLE (e.g. machine reconfiguration)."""
        self.phase            = "IDLE"
        self._engage_count    = 0
        self._disengage_count = 0

    @property
    def is_engaged(self) -> bool:
        return self.phase == "ENGAGED"

    @property
    def last_result(self) -> "GatedResult":
        return self._last_result


class GatedResult:
    """Output from ActivityStateMachine.update()."""

    def __init__(
        self,
        class_id: int,
        confidence: float,
        probs: np.ndarray,
        phase: str,
        just_engaged: bool = False,
        just_ended: bool = False,
        engage_progress: float = 0.0,
    ):
        self.class_id        = class_id
        self.class_name      = CLASS_NAMES[class_id] if class_id < NUM_CLASSES else "unknown"
        self.confidence      = confidence
        self.probs           = probs
        self.phase           = phase           # "IDLE" | "ENGAGED"
        self.just_engaged    = just_engaged    # True on the frame session starts
        self.just_ended      = just_ended      # True on the frame session ends
        self.engage_progress = engage_progress # 0.0→1.0 fill bar while confirming
        self.timestamp       = time.time()

    @property
    def is_engaged(self) -> bool:
        return self.phase == "ENGAGED"

    @property
    def is_rep_state(self) -> bool:
        """True for good_rep (3), bad_rep (4), half_rep (7)."""
        return self.class_id in REP_CLASSES

    @property
    def low_confidence(self) -> bool:
        """True when the model is uncertain — flag for review."""
        return 0 < self.confidence < 0.50

    def draw_overlay(self, frame) -> None:
        """Draw phase indicator on the frame (call from main loop if SHOW_PREVIEW)."""
        try:
            import cv2
        except ImportError:
            return

        h, w = frame.shape[:2]
        colour = (0, 200, 0) if self.is_engaged else (100, 100, 100)

        # Phase pill (top-left)
        label = f"[{self.phase}] {self.class_name}  {self.confidence*100:.0f}%"
        cv2.rectangle(frame, (8, 8), (8 + len(label)*10 + 8, 36), (0, 0, 0), -1)
        cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)

        # Engagement progress bar (fills while confirming on_machine)
        if self.phase == "IDLE" and self.engage_progress > 0:
            bar_w = int((w // 3) * self.engage_progress)
            cv2.rectangle(frame, (w//3, h - 16), (w//3 + bar_w, h - 6), (0, 200, 255), -1)
            cv2.putText(frame, "engaging...", (w//3 + 4, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    def __repr__(self) -> str:
        return (
            f"<GatedResult [{self.phase}] {self.class_name}({self.class_id}) "
            f"conf={self.confidence:.2f}"
            + (" ENGAGED!" if self.just_engaged else "")
            + (" ENDED!"   if self.just_ended   else "")
            + ">"
        )
