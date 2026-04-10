---
title: Project 2 — Weight ID
tags: [project, weight-id, yolo, colour, active]
status: built-needs-images
created: 2026-04-09
updated: 2026-04-10
---

# Project 2 — Weight ID (Free Weights)

> Reads AlphaFit barbell plate colours → total weight in kg. Zero manual entry.

Part of [[Home]]. Applies to **free weight barbells only**. For pin-loaded cable machines see [[Projects/E-Weight]].

---

## How It Works

Camera looks along the barbell sleeve from the side (~45°). Each plate appears as a coloured stripe ring.

```
Camera frame
  → YOLO bounding box per plate stripe
  → HSV colour classify each box → kg value
  → sum all plates + 20 kg bar = total weight
  → WeightReading(total_kg, confident, method)
```

**Right now (no training needed):** colour scan of full frame — counts distinct coloured bands. Good enough for approximate weights from day one.

**After training:** YOLO detects exact plate positions → per-plate classification → more reliable.

---

## AlphaFit Plate Colours

Full HSV detail: [[Data/AlphaFit Plates]]

| Colour | Weight | Confidence |
|--------|--------|------------|
| Red stripe | 25 kg | 99% |
| Blue stripe | 20 kg | 99% |
| Yellow stripe | 15 kg | 99% |
| Green stripe | 10 kg | 98% |
| White stripe | 5 kg | 95% |

**Bar:** 20 kg added automatically (`add_bar_weight = True`).

---

## Two Detection Modes

### YOLO Mode (after training)
- YOLOv11 detects each plate stripe as a bounding box
- `ColourMatcher.classify_crop(bbox_crop)` → dominant HSV colour → kg
- Per-plate result: `{"colour": "red", "kg": 25, "confidence": 0.94}`
- More accurate — ignores clothing, background noise

### Colour Scan Mode (works now, zero training)
- `ColourMatcher.analyse_frame(frame)` scans whole frame
- Excludes bottom 28% (floor), centres on horizontal 35% strip
- Counts distinct plate bands via morphological closing + connected components
- Returns approximate total — good enough to start logging

`WeightReading.method` tells you which mode was used: `"yolo"` or `"colour_scan"`.

---

## HSV Colour Ranges

Loaded from `configs/weight_plate_colours.json`. Key handling:
- **Red** requires **two HSV ranges** (hue wraps at 0/180 in OpenCV) — the masks are OR'd
- **White** matches low saturation (0–50) + high value (175–255) — can also match grey/cream

```python
# Red plate (two ranges, union)
lower1 = [0,   140, 80],  upper1 = [10,  255, 255]
lower2 = [170, 140, 80],  upper2 = [180, 255, 255]
```

---

## Camera Placement

See [[Hardware/Camera Placement]] for full mounting guide.

- Mount on barbell rack, looking along sleeve from side (~45°)
- **Two cameras per barbell station** — one per sleeve (left and right)
- At least one side always has an unobstructed view
- Costs: 2 × Pi Camera Module 3 (£35 each) extra per station

---

## Key Files

| File | Purpose |
|------|---------|
| `weight_id/detector.py` | `WeightDetector` — main class, YOLO + colour fallback |
| `weight_id/colour_matcher.py` | `ColourMatcher` — HSV ranges → kg values |
| `weight_id/train.py` | YOLOv11 fine-tuning pipeline |
| `configs/weight_plate_colours.json` | HSV colour config (per gym lighting) |
| `data/weight_plates/images/train/` | Training images (not in git) |

---

## WeightReading dataclass

```python
@dataclass
class WeightReading:
    total_kg:   Optional[float]   # None if no plates detected
    confident:  bool               # True if confidence >= threshold
    confidence: float              # 0.0–1.0
    method:     str                # "yolo", "colour_scan", or "none"
    plates:     list               # per-plate [{colour, kg, confidence}]
    latency_ms: float
```

---

## Training

```bash
# 1. Collect 50+ photos per plate colour in gym lighting
#    → data/weight_plates/images/train/
#    → data/weight_plates/images/val/

# 2. Train YOLOv11 on plate dataset
make train-weight
# → models/weights/weight_id_v1.onnx

# 3. Test colour scan immediately (no training needed)
make test-weight
```

Classes: `red_plate` (25kg), `blue_plate` (20kg), `yellow_plate` (15kg), `green_plate` (10kg), `white_plate` (5kg).

---

## Status

- [x] `ColourMatcher` — HSV colour scan (works now)
- [x] `WeightDetector` — YOLO + colour fallback pipeline
- [x] `train.py` — fine-tuning script ready
- [x] `configs/weight_plate_colours.json` — HSV ranges configured
- [ ] Training images (50+ per colour in gym lighting)
- [ ] YOLO model trained → `weight_id_v1.onnx`
- [ ] Deployed and tested on barbell station

---

## Related

- [[Data/AlphaFit Plates]] — colour → kg mapping and full HSV ranges
- [[System/YOLO Pipeline]] — object detection backbone
- [[Hardware/Camera Placement]] — 45° mounting angle is critical
- [[Hardware/Machine Pi]] — runs on same Pi as rep tracking
- [[Hardware/Costs]] — 2 extra cameras per barbell station (£70)
- [[Data/Training Requirements]] — 50+ photos per colour needed
- [[Projects/E-Weight]] — alternative method for pin-loaded cable machines
- [[Decisions/Stack Choices]] — why YOLO for plate detection
