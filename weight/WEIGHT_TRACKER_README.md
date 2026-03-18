# XL Fitness — Weight Plate Detector

**Demo v0.2** | Browser-based, no server required

## What it does

Opens the device camera and uses real-time colour detection to identify XL Fitness weight plates by their coloured bands, counting the total weight on screen.

| Plate Colour | Weight |
|---|---|
| White (low-saturation band) | 5 kg |
| Yellow / Green stripe | 10 kg |
| Blue | 20 kg |

## How to use

Open `weight_plate_detector.html` in a mobile browser (Chrome or Safari).  
Tap the camera icon, point at the plates, and the total weight is displayed live with bounding boxes drawn around each detected plate.

## Algorithm

- **HSV colour masking** — each frame is classified pixel-by-pixel into blue / yellow-green / white channels
- **Horizontal band scanning** — rows are scanned for continuous colour fills; each distinct run = one plate
- **Floor exclusion** — the bottom 28% of the frame is ignored to prevent the grey gym floor triggering white detections
- **Centre-zone filter** — only bands centred in the frame are counted (ignores bench bars and rack edges)
- **Median smoothing** — counts are smoothed over 5 frames to prevent flickering

## Validated

Tested against real XL Fitness plates:  
**1 × Blue (20 kg) + 2 × Yellow (10 kg) + 2 × White (5 kg) = 50 kg ✓**

See `detection_reference.jpg` for the annotated reference image.

## Files

| File | Description |
|---|---|
| `weight_plate_detector.html` | Self-contained detector — open directly in browser |
| `detection_reference.jpg` | Annotated reference showing correct 50 kg detection |
| `README.md` | This file |

## Next steps

- Add barbell weight (20 kg bar) as a toggle option
- Train a lightweight CNN for more robust detection under varied lighting
- Integrate with the tablet kiosk UI (`tablet.html`) to auto-populate the weight field
