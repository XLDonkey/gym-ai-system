"""
XL Fitness AI — Weight Plate Detector
Identifies the weight loaded on a barbell by detecting AlphaFit plate colours.

Camera placement: mounted on the barbell frame looking along the sleeve from
the side at ~45°. Each plate appears as a coloured stripe ring in the frame.

Detection pipeline:
  1. YOLO object detection → bounding boxes around each plate stripe
  2. Colour classification within each box → plate colour → kg value
  3. Sum all plates on both sleeves + 20 kg bar = total weight

Fallback (no YOLO model yet):
  Whole-frame HSV colour band scanning — same as the browser prototype
  in weight/weight_plate_detector.html. Less accurate but zero setup.

Usage:
    detector = WeightDetector(yolo_model="models/weights/weight_id_v1.onnx")
    reading  = detector.identify(frame)
    if reading.confident:
        print(f"{reading.total_kg} kg  [{reading.method}]")
        # → "70.0 kg  [yolo]"

    # For annotation tool / review portal:
    frame = detector.draw_overlay(frame)
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

try:
    from ultralytics import YOLO
    _YOLO = True
except ImportError:
    _YOLO = False

from weight_id.colour_matcher import ColourMatcher, BARBELL_WEIGHT_KG


@dataclass
class WeightReading:
    total_kg:   Optional[float]   # total weight in kg (plates + bar), or None
    confident:  bool              # True when confidence ≥ threshold
    confidence: float             # 0.0–1.0
    method:     str               # "yolo", "colour_scan", or "none"
    plates:     list = field(default_factory=list)
    # Each plate: {"colour": str, "kg": float, "confidence": float, "bbox": [x1,y1,x2,y2]}
    latency_ms: float = 0.0


class WeightDetector:
    """
    Identifies AlphaFit barbell plate weights from a camera frame.

    Args:
        yolo_model:           Path to a fine-tuned YOLO .pt or .onnx for plate detection.
                              If None, falls back to colour-scan mode.
        colour_config:        Path to weight_plate_colours.json (HSV overrides).
        confidence_threshold: Minimum confidence to mark a reading as confident.
        add_bar_weight:       Add BARBELL_WEIGHT_KG (20 kg) to total (default True).
        enabled:              Set False to disable entirely.
    """

    def __init__(
        self,
        yolo_model:           str   = None,
        colour_config:        str   = "configs/weight_plate_colours.json",
        confidence_threshold: float = 0.65,
        add_bar_weight:       bool  = True,
        enabled:              bool  = True,
    ):
        self.threshold      = confidence_threshold
        self.add_bar_weight = add_bar_weight
        self.enabled        = enabled
        self._model         = None
        self._last_reading: Optional[WeightReading] = None

        self.colour_matcher = ColourMatcher(colour_config)

        if yolo_model and _YOLO and os.path.exists(yolo_model):
            try:
                self._model = YOLO(yolo_model)
                print(f"[weight_id] YOLO model loaded: {yolo_model}")
            except Exception as e:
                print(f"[weight_id] YOLO load error: {e} — using colour scan fallback")

        mode = "YOLO + colour" if self._model else "colour scan (no YOLO)"
        if enabled:
            print(f"[weight_id] Active  mode={mode}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def identify(self, frame: np.ndarray) -> WeightReading:
        """
        Identify weight from one video frame.
        Call every ~1 second during a session — no need for every frame.
        """
        if not self.enabled or not _CV2:
            return WeightReading(None, False, 0.0, "none")

        t0 = time.time()
        reading = self._yolo_identify(frame) if self._model else self._colour_identify(frame)
        reading.latency_ms = round((time.time() - t0) * 1000, 1)
        self._last_reading = reading
        return reading

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw weight bounding boxes and total on frame. Call after identify()."""
        if not _CV2 or self._last_reading is None:
            return frame

        r = self._last_reading
        colour = (0, 220, 0) if r.confident else (0, 165, 255)

        # Draw per-plate boxes (YOLO mode only)
        for p in r.plates:
            if "bbox" in p:
                x1, y1, x2, y2 = p["bbox"]
                plate_col = _PLATE_DRAW_COLOUR.get(p["colour"], (200, 200, 200))
                cv2.rectangle(frame, (x1, y1), (x2, y2), plate_col, 2)
                cv2.putText(
                    frame, f"{p['kg']:.0f}kg",
                    (x1 + 4, y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, plate_col, 1,
                )

        # Total weight label
        label = f"{r.total_kg:.1f} kg" if r.total_kg is not None else "? kg"
        label += f"  [{r.method}  {r.confidence:.0%}]"
        cv2.putText(frame, label, (10, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
        return frame

    # ── Detection methods ──────────────────────────────────────────────────────

    def _yolo_identify(self, frame: np.ndarray) -> WeightReading:
        """YOLO bounding box detection + per-box colour classification."""
        try:
            results = self._model(frame, verbose=False, conf=0.25)[0]
        except Exception as e:
            print(f"[weight_id] YOLO error: {e}")
            return WeightReading(None, False, 0.0, "yolo_error")

        plates  = []
        total   = 0.0

        for box in results.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            colour_name, kg, colour_conf = self.colour_matcher.classify_crop(crop)
            yolo_conf = float(box.conf[0])
            conf      = (yolo_conf + colour_conf) / 2.0

            plates.append({
                "colour":     colour_name,
                "kg":         kg,
                "confidence": round(conf, 3),
                "bbox":       [x1, y1, x2, y2],
            })
            total += kg

        if not plates:
            return WeightReading(None, False, 0.0, "yolo")

        if self.add_bar_weight:
            total += BARBELL_WEIGHT_KG

        avg_conf = sum(p["confidence"] for p in plates) / len(plates)
        return WeightReading(
            total_kg   = total,
            confident  = avg_conf >= self.threshold,
            confidence = round(avg_conf, 3),
            method     = "yolo",
            plates     = plates,
        )

    def _colour_identify(self, frame: np.ndarray) -> WeightReading:
        """Whole-frame colour band scan — no YOLO required."""
        plates, total, conf = self.colour_matcher.analyse_frame(
            frame, add_bar_weight=self.add_bar_weight
        )
        return WeightReading(
            total_kg   = total if plates else None,
            confident  = conf >= self.threshold,
            confidence = conf,
            method     = "colour_scan",
            plates     = plates,
        )


# ── Drawing colour map ─────────────────────────────────────────────────────────
# BGR colours for bounding box overlay per plate colour
_PLATE_DRAW_COLOUR = {
    "red":    (50, 50, 220),
    "blue":   (200, 100, 30),
    "yellow": (30, 210, 230),
    "green":  (50, 180, 50),
    "white":  (200, 200, 200),
}
