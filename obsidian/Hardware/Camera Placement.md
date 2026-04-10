---
title: Camera Placement
tags: [hardware, camera, placement, mounting]
created: 2026-04-09
---

# Camera Placement

Camera angle is critical. Bad placement = bad keypoints = bad predictions.

---

## Machine Camera (Rep Tracking + User Tracking)

- **Position:** Side-on to the machine, ~1.5–2m away
- **Angle:** Slightly elevated, angled down ~15°
- **Frame:** Person fully visible — head to hips minimum, ideally full body
- **Goal:** Clear view of shoulder and elbow joints for angle measurement

### Machine Zone ROI

After mounting, set `MACHINE_ZONE_ROI` in `pi/config.py`:
```python
MACHINE_ZONE_ROI = (0.15, 0.05, 0.85, 0.95)  # (x1, y1, x2, y2) normalised
```

Enable `SHOW_PREVIEW = True` to see the ROI overlay while adjusting.
Tighten the box to just cover the seat + working space — prevents tracking people walking past.

---

## Barbell Weight Camera (Weight ID)

- **Position:** Mounted on barbell rack frame, looking along the sleeve
- **Angle:** ~45° from the side, looking inward along the sleeve
- **Goal:** Each plate visible as a distinct coloured stripe ring

**Two cameras per barbell station** — one per sleeve (left and right).
At least one side always has an unobstructed view.

---

## Door Camera (User Tracking — Entry)

- **Position:** Above door frame, angled down ~30°
- **Goal:** Face visible for ArcFace recognition as member enters
- **Lighting:** Ensure face is well-lit — avoid backlight from windows behind person

---

## Weight Stack ROI (Optical Flow)

For cable machines (pin-loaded), the weight stack movement validates reps:
```python
WEIGHT_STACK_ROI = (0.72, 0.05, 0.92, 0.88)  # right 20% of frame
```

Covers the weight stack in the camera frame — adjust per machine layout.

---

## Calibration Steps

1. Mount camera
2. Set `SHOW_PREVIEW = True` in `pi/config.py`
3. Run `python pi/main.py`
4. Adjust `MACHINE_ZONE_ROI` until box tightly wraps machine area
5. Adjust `WEIGHT_STACK_ROI` until box covers weight stack
6. Set `SHOW_PREVIEW = False` for production

---

## Related

- [[Hardware/Machine Pi]] — the hardware
- [[System/YOLO Pipeline]] — what the camera feeds
- [[Projects/Weight ID]] — barbell weight detection
