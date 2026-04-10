---
title: Project 2 — Weight ID
tags: [project, weight-id, yolo, colour, active]
status: built-needs-images
created: 2026-04-09
---

# Project 2 — Weight ID (Free Weights)

> Reads barbell plate colours → total weight in kg. No QR codes, no manual entry.

Part of [[Home]]. Applies to **free weight barbells** only. For cable/pin machines see [[Projects/E-Weight]].

---

## How It Works

The camera looks along the barbell sleeve from the side (~45°). Each plate appears as a coloured stripe ring in frame.

```
Camera → YOLO bounding box per plate → HSV colour → kg value → sum + 20kg bar = total
```

### AlphaFit Plate Colours

See [[Data/AlphaFit Plates]] for HSV ranges.

| Colour | Weight | Reliability |
|--------|--------|-------------|
| Red    | 25 kg  | 99% |
| Blue   | 20 kg  | 99% |
| Yellow | 15 kg  | 99% |
| Green  | 10 kg  | 98% |
| White  | 5 kg   | 95% |

Barbell bar = **20 kg** added automatically.

---

## Two Detection Modes

| Mode | How | When used |
|------|-----|-----------|
| **YOLO** | Bounding box per plate → colour classify each box | After training |
| **Colour scan** | HSV mask whole frame → count distinct bands | Works right now, no training needed |

Colour scan is the fallback — good enough to start logging approximate weights from day one.

---

## Camera Placement

- Mount on barbell frame, looking along sleeve from the side (~45°)
- **Two cameras per barbell station** — left and right sleeves
- At least one side always has a clear view of the plates

---

## Key Files

| File | Purpose |
|------|---------|
| `weight_id/detector.py` | `WeightDetector` — main class |
| `weight_id/colour_matcher.py` | `ColourMatcher` — HSV → kg |
| `weight_id/train.py` | YOLOv11 fine-tuning pipeline |
| `configs/weight_plate_colours.json` | HSV colour ranges |
| `data/weight_plates/` | Training images (not in git) |

---

## Training

```bash
# 1. Collect 50+ photos per colour → data/weight_plates/images/train/
# 2. Train:
make train-weight
# → models/weights/weight_id_v1.onnx

# Test colour scan immediately (no training):
make test-weight
```

---

## Status

- [x] Colour scan working (no training needed)
- [x] YOLO pipeline built
- [ ] Training images collected (50+ per colour)
- [ ] YOLO model trained and deployed

---

## Related

- [[Data/AlphaFit Plates]] — colour → kg mapping and HSV ranges
- [[System/YOLO Pipeline]] — object detection backbone
- [[Hardware/Camera Placement]] — how to mount for best plate view
- [[Hardware/Machine Pi]] — hardware this runs on
- [[Hardware/Costs]] — 2 extra cameras per barbell station
- [[Data/Training Requirements]] — 50+ photos per colour needed
- [[Projects/E-Weight]] — alternative for pin-loaded machines
