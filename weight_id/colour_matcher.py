"""
XL Fitness AI — AlphaFit Weight Plate Colour Matcher
Maps plate edge-stripe colours → kg values for XL Fitness AlphaFit bumper plates.

AlphaFit plates are black with a thick coloured stripe around the edge:
  Red    — 25 kg  (high contrast, reliable)
  Blue   — 20 kg  (high contrast, reliable)
  Yellow — 15 kg  (high contrast, reliable)
  Green  — 10 kg  (reliable)
  White  — 5 kg   (can blend with pale backgrounds — check V_low)

The camera looks along the barbell sleeve from the side. Each plate appears
as a coloured ring. YOLO finds the bounding box, this module classifies colour.

Fallback mode (no YOLO): whole-frame colour band scanning — same algorithm
as the browser prototype in weight/weight_plate_detector.html.
"""

import json
import os
import numpy as np
from typing import Optional

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    print("[colour_matcher] WARNING: opencv-python not installed")


# HSV ranges for AlphaFit barbell plates under indoor gym lighting.
# Format: [H_low, H_high, S_low, S_high, V_low, V_high]
# Tune these for your gym's lighting conditions (SHOW_PREVIEW=True in config.py).
DEFAULT_COLOURS = {
    "red": {
        "hsv_ranges": [
            [0,   10,  140, 255, 80, 255],   # lower red hue
            [170, 180, 140, 255, 80, 255],   # upper red hue (wraps at 180°)
        ],
        "kg": 25.0,
        "hex": "#e74c3c",
    },
    "blue": {
        "hsv_ranges": [[100, 130, 130, 255, 60, 255]],
        "kg": 20.0,
        "hex": "#3498db",
    },
    "yellow": {
        "hsv_ranges": [[20, 35, 160, 255, 150, 255]],
        "kg": 15.0,
        "hex": "#f1c40f",
    },
    "green": {
        "hsv_ranges": [[40, 80, 110, 255, 60, 255]],
        "kg": 10.0,
        "hex": "#2ecc71",
    },
    "white": {
        "hsv_ranges": [[0, 180, 0, 50, 175, 255]],   # low saturation, high value
        "kg": 5.0,
        "hex": "#bdc3c7",
    },
}

# Barbell bar weight added to all plate totals.
BARBELL_WEIGHT_KG = 20.0


class ColourMatcher:
    """
    Classifies a cropped plate image or full frame by AlphaFit stripe colour.

    Works in two modes:
      classify_crop(crop)   — single plate bounding box (from YOLO)
      analyse_frame(frame)  — whole-frame scan (no YOLO fallback)
    """

    def __init__(self, config_path: str = "configs/weight_plate_colours.json"):
        self.colours = dict(DEFAULT_COLOURS)
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    overrides = json.load(f)
                self.colours.update(overrides.get("colours", {}))
                print(f"[colour_matcher] Loaded config: {config_path}")
            except Exception as e:
                print(f"[colour_matcher] Config load error: {e} — using defaults")

    # ── Single plate crop ──────────────────────────────────────────────────────

    def classify_crop(self, crop: np.ndarray) -> tuple:
        """
        Classify a single cropped plate image by dominant colour.

        Returns:
            (colour_name, kg_value, confidence)
            confidence: fraction of pixels matching the winning colour [0.0–1.0]
        """
        if not _CV2_AVAILABLE or crop.size == 0:
            return "unknown", 0.0, 0.0

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        total_px = crop.shape[0] * crop.shape[1]

        best_colour, best_kg, best_count = "unknown", 0.0, 0

        for colour_name, spec in self.colours.items():
            mask  = self._build_mask(hsv, spec["hsv_ranges"])
            count = int(np.sum(mask > 0))
            if count > best_count:
                best_count  = count
                best_colour = colour_name
                best_kg     = spec["kg"]

        confidence = best_count / max(total_px, 1)
        return best_colour, best_kg, round(confidence, 3)

    # ── Whole-frame scan (no YOLO) ─────────────────────────────────────────────

    def analyse_frame(
        self,
        frame:              np.ndarray,
        add_bar_weight:     bool  = True,
        exclude_bottom_pct: float = 0.28,  # ignore floor (grey → false white hits)
        centre_zone_pct:    float = 0.35,  # only count centre horizontal strip
    ) -> tuple:
        """
        Full-frame plate detection using horizontal colour band scanning.
        Same algorithm as weight/weight_plate_detector.html.

        Returns:
            (plates, total_kg, confidence)
            plates:    list of {"colour": str, "kg": float}
            total_kg:  sum of all detected plates + bar (if add_bar_weight)
            confidence: fraction of expected plate area matched [0–1]
        """
        if not _CV2_AVAILABLE or frame.size == 0:
            return [], 0.0, 0.0

        h, w = frame.shape[:2]
        active_h = int(h * (1.0 - exclude_bottom_pct))
        cx1 = int(w * (0.5 - centre_zone_pct / 2))
        cx2 = int(w * (0.5 + centre_zone_pct / 2))

        hsv_roi = cv2.cvtColor(frame[:active_h, cx1:cx2], cv2.COLOR_BGR2HSV)
        total_px = hsv_roi.shape[0] * hsv_roi.shape[1]

        plates, total_kg, matched_px = [], 0.0, 0

        for colour_name, spec in self.colours.items():
            mask  = self._build_mask(hsv_roi, spec["hsv_ranges"])
            count = int(np.sum(mask > 0))
            if count < 80:   # skip noise
                continue

            n = self._count_plate_bands(mask)
            for _ in range(n):
                plates.append({"colour": colour_name, "kg": spec["kg"]})
                total_kg += spec["kg"]
            matched_px += count

        if add_bar_weight and plates:
            total_kg += BARBELL_WEIGHT_KG

        confidence = min(1.0, matched_px / max(total_px * 0.04, 1))
        return plates, total_kg, round(confidence, 3)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_mask(self, hsv: np.ndarray, ranges: list) -> np.ndarray:
        """Union of all HSV ranges — handles red hue wrap-around."""
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for r in ranges:
            h_lo, h_hi, s_lo, s_hi, v_lo, v_hi = r
            mask |= cv2.inRange(
                hsv,
                np.array([h_lo, s_lo, v_lo], dtype=np.uint8),
                np.array([h_hi, s_hi, v_hi], dtype=np.uint8),
            )
        return mask

    def _count_plate_bands(self, mask: np.ndarray) -> int:
        """
        Count distinct plate bands in a binary mask using connected components.
        Morphological closing merges nearby pixels belonging to the same plate.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 25))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        n_labels, *_ = cv2.connectedComponentsWithStats(closed)
        return max(0, n_labels - 1)  # subtract background label
