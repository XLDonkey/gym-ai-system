---
title: Training Requirements
tags: [data, training, blockers]
created: 2026-04-09
---

# Training Requirements

What data must be collected before the AI can go live. The code is built — these are the real-world steps.

---

## Project 1 — Rep Tracking (LSTM)

**Target: 300+ annotated segments (≥30 per class)**

| Class | Label | Min needed | Collected |
|-------|-------|-----------|-----------|
| 0 | no_person | 30 | 0 |
| 1 | user_present | 30 | 0 |
| 2 | on_machine | 30 | 0 |
| 3 | good_rep | 30 | 0 |
| 4 | bad_rep | 30 | 0 |
| 5 | false_rep | 30 | 0 |
| 6 | resting | 30 | 0 |
| 7 | half_rep | 30 | 0 |
| **Total** | | **240 min** | **0** |

### How to Collect

```bash
# 1. Pi records sessions automatically when person detected
#    (RECORD_SESSIONS = True in pi/config.py)

# 2. Sync recordings from Pi to Mac Mini
make sync PI=pi@192.168.1.x

# 3. Open annotation tool
make annotate
# Opens pose/label.html in browser
# Watch video → label each segment → saves to data/annotations/

# 4. Check progress
make stats
# Shows count per class

# 5. Train when ready
make train
# → models/weights/activity_v1.onnx
```

---

## Project 2 — Weight ID (YOLO)

**Target: 50+ photos per plate colour**

| Colour | Weight | Min needed | Collected |
|--------|--------|-----------|-----------|
| Red | 25 kg | 50 | 0 |
| Blue | 20 kg | 50 | 0 |
| Yellow | 15 kg | 50 | 0 |
| Green | 10 kg | 50 | 0 |
| White | 5 kg | 50 | 0 |

### How to Collect

- Mount camera at ~45° along barbell sleeve
- Load different plate combinations
- Photograph in varied lighting (overhead, natural, evening)
- Save to `data/weight_plates/images/train/`

**Note:** Colour scan fallback works immediately — no training needed to start logging approximate weights.

```bash
make test-weight   # test colour scan right now
make train-weight  # train YOLO after collecting images
```

---

## Project 3 — User Tracking (Face Enrolment)

**Enrol every member once.**

```bash
make enrol NAME="Matthew"
# Opens webcam → captures face → saves 512-dim embedding → Supabase members table
```

Repeat for each member. No minimum count — one enrolment per person is enough.

---

## Summary Checklist

- [ ] Install Pi at machine, verify 30fps YOLO keypoints
- [ ] Record 300+ rep segments across all 8 classes
- [ ] Annotate with `pose/label.html`
- [ ] Train LSTM: `make train`
- [ ] Deploy to Pi: `make deploy PI=pi@IP`
- [ ] Photograph 50+ images per plate colour in gym lighting
- [ ] Train weight detector: `make train-weight`
- [ ] Enrol all members: `make enrol NAME="..."`
- [ ] Set `SUPABASE_URL` + `SUPABASE_SERVICE_KEY` in `pi/config.py`
- [ ] Test end-to-end: person sits → rep counted → session logged to Supabase

---

## Related

- [[System/LSTM Model]] — what gets trained
- [[System/Review Loop]] — ongoing improvement after go-live
- [[Data/AlphaFit Plates]] — colour ranges for weight ID
- [[Projects/User Tracking]] — enrolment process
