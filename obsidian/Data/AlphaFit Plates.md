---
title: AlphaFit Plates
tags: [data, weight, colour, alphafit]
created: 2026-04-09
---

# AlphaFit Weight Plates

XL Fitness uses **AlphaFit** branded plates. Each weight has a distinct coloured stripe ring around the edge. This is the basis for [[Projects/Weight ID]].

---

## Colour → Weight Mapping

| Stripe Colour | Weight | Reliability |
|---------------|--------|-------------|
| Red | 25 kg | 99% |
| Blue | 20 kg | 99% |
| Yellow | 15 kg | 99% |
| Green | 10 kg | 98% |
| White | 5 kg | 95% |

**Barbell bar:** 20 kg (added automatically — not a plate).

---

## HSV Ranges (OpenCV)

Stored in `configs/weight_plate_colours.json`.

| Colour | H range | S range | V range | Notes |
|--------|---------|---------|---------|-------|
| Red | 0–10 + 165–180 | 100–255 | 60–255 | Two ranges — red wraps hue |
| Blue | 100–130 | 100–255 | 60–255 | |
| Yellow | 20–35 | 100–255 | 100–255 | |
| Green | 40–80 | 60–255 | 60–255 | |
| White | 0–180 | 0–40 | 180–255 | Low saturation, high value |

Red requires two HSV ranges because red hue wraps around 0/180 in OpenCV's 0–180 hue scale. The masks are OR'd together.

---

## Detection Logic

1. YOLO finds bounding box around each plate stripe
2. Crop the bounding box
3. Convert crop to HSV
4. Apply each colour mask → count pixels
5. Winning colour = most pixels above threshold
6. Sum all plates + 20 kg bar = total weight

---

## Colour Scan Fallback

When no YOLO model is available, `ColourMatcher.analyse_frame(frame)` scans the whole frame:
- Applies each colour mask to full frame
- Counts distinct bands using morphological closing + connected components
- Less accurate (can pick up clothing etc.) but works immediately with no training

---

## Training Images Needed

- **50+ photos per colour** in varied lighting conditions
- Save to `data/weight_plates/images/train/`
- See `make train-weight` in [[Projects/Weight ID]]

---

## Related

- [[Projects/Weight ID]] — full project
- [[System/Architecture]] — where weight detection fits
