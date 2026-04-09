# weight_id — Weight Plate Identification (Project 2)

Identifies how much weight is loaded on a barbell by reading AlphaFit plate stripe colours.

| File | Purpose |
|------|---------|
| `detector.py` | `WeightDetector` — main class, YOLO + colour fallback |
| `colour_matcher.py` | HSV colour ranges → kg values for AlphaFit plates |
| `train.py` | Fine-tunes YOLOv11 on your gym plate images |

## Camera placement

Mount on the barbell frame, looking along the sleeve from the side (~45°).
Each plate appears as a coloured stripe ring. Two cameras per station (one each side) so at least one always has a clear view.

## AlphaFit plate colours

| Stripe | Weight |
|--------|--------|
| Red | 25 kg |
| Blue | 20 kg |
| Yellow | 15 kg |
| Green | 10 kg |
| White | 5 kg |

## Quick start (colour scan — no training needed)

```python
from weight_id.detector import WeightDetector
import cv2

detector = WeightDetector()   # uses colour scan fallback
frame    = cv2.imread("barbell.jpg")
reading  = detector.identify(frame)
print(f"{reading.total_kg} kg  [{reading.method}  {reading.confidence:.0%}]")
```

## Training the YOLO model

```bash
# 1. Collect 50+ photos per plate colour at your gym
# 2. Annotate in CVAT or Roboflow → export YOLO format
# 3. Place in data/weight_plates/images/train/ + labels/train/
make train-weight
make deploy PI=pi@IP
```

## Tuning HSV colours

If plates aren't detected reliably under your gym lighting, adjust the HSV ranges in `configs/weight_plate_colours.json`. Set `SHOW_PREVIEW=True` in `pi/config.py` and run `pi/main.py` to see live colour masks.
