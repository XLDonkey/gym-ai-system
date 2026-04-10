---
title: AlphaFit Plates
tags: [data, weight, colour, alphafit]
created: 2026-04-09
updated: 2026-04-10
---

# AlphaFit Weight Plates

XL Fitness uses **AlphaFit** branded plates. Each weight has a distinct coloured stripe ring. This is the detection basis for [[Projects/Weight ID]].

---

## Colour → Weight Mapping

| Stripe Colour | Weight | Detection reliability |
|---------------|--------|----------------------|
| Red | 25 kg | 99% |
| Blue | 20 kg | 99% |
| Yellow | 15 kg | 99% |
| Green | 10 kg | 98% |
| White | 5 kg | 95% |

**Barbell bar:** 20 kg added automatically (`BARBELL_WEIGHT_KG = 20.0` in `configs/weight_plate_colours.json`).

---

## HSV Ranges (OpenCV — H: 0–180, S: 0–255, V: 0–255)

| Colour | H range | S range | V range | Special |
|--------|---------|---------|---------|---------|
| Red | 0–10 AND 170–180 | 140–255 | 80–255 | Two ranges — hue wraps |
| Blue | 100–130 | 130–255 | 60–255 | |
| Yellow | 20–35 | 160–255 | 150–255 | |
| Green | 40–80 | 110–255 | 60–255 | |
| White | 0–180 | 0–50 | 175–255 | Low saturation |

### Red Plate — Two Ranges

OpenCV's HSV hue goes 0–180. Red wraps around at the ends (both ~0 and ~180 = red).

```python
# Red requires two masks, combined with OR:
mask_red1 = cv2.inRange(hsv, [0,   140, 80], [10,  255, 255])
mask_red2 = cv2.inRange(hsv, [170, 140, 80], [180, 255, 255])
mask_red  = cv2.bitwise_or(mask_red1, mask_red2)
```

### White Plate — Tricky

White can match grey, cream, or silver backgrounds. To reduce false positives:
- Frame bottom 28% excluded (floor is often white/grey)
- Analysis centred on horizontal 35% strip (where plates appear)
- Requires high value (175+) AND very low saturation (< 50)

---

## Config File

`configs/weight_plate_colours.json` — override HSV ranges per gym if lighting is unusual:

```json
{
  "barbell_weight_kg": 20.0,
  "colours": {
    "red":    { "hsv_ranges": [[0,140,80,10,255,255],[170,140,80,180,255,255]], "kg": 25, "hex": "#ef4444" },
    "blue":   { "hsv_ranges": [[100,130,60,130,255,255]], "kg": 20, "hex": "#3b82f6" },
    "yellow": { "hsv_ranges": [[20,160,150,35,255,255]],  "kg": 15, "hex": "#eab308" },
    "green":  { "hsv_ranges": [[40,110,60,80,255,255]],   "kg": 10, "hex": "#22c55e" },
    "white":  { "hsv_ranges": [[0,0,175,180,50,255]],     "kg": 5,  "hex": "#f8fafc" }
  }
}
```

---

## Full-Frame Colour Scan (No Training)

`ColourMatcher.analyse_frame(frame)`:

1. Convert frame to HSV
2. Exclude bottom 28% of frame (floor/mat area)
3. Focus on middle horizontal 35% strip (where barbell plates appear)
4. Apply each colour mask
5. Morphological closing to connect gaps within a plate band
6. Connected components → count distinct plate bands
7. Sum all kg values + 20kg bar

This works **right now** with no YOLO training. Less precise but useful from day one.

---

## Per-Plate YOLO Classification

When YOLO model is deployed:

1. YOLO detects bounding box for each plate stripe visible in frame
2. `ColourMatcher.classify_crop(bbox_crop)`:
   - Crops frame to bbox
   - Applies each HSV mask to crop
   - Returns colour with highest pixel count above threshold
3. Maps colour → kg
4. Sums all plates + 20kg bar

More accurate — ignores background, clothing, lighting gradient outside ROI.

---

## Training Image Tips

For best YOLO training results:
- Photograph in actual gym lighting (overhead fluorescent is common — affects hue)
- Vary: 1 plate, 2 plates, 3+ plates, mixed combinations
- Capture from both left and right sleeve angles
- Include partial plates (edge of frame)
- Include plates with shadows, chalk marks
- **50+ images per colour minimum**

---

## Related

- [[Projects/Weight ID]] — uses these ranges
- [[Hardware/Camera Placement]] — camera angle critical for colour accuracy
- [[Data/Training Requirements]] — 50+ photos per colour needed
