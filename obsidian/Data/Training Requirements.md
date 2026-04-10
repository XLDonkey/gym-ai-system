---
title: Training Requirements
tags: [data, training, blockers]
created: 2026-04-09
updated: 2026-04-10
---

# Training Requirements

The code is built. These are the real-world steps before the AI goes live.

---

## Project 1 — Rep Tracking (LSTM)

**Target: 300+ annotated segments (≥30 per class)**

| Class | Label | Min | Notes |
|-------|-------|-----|-------|
| 0 | no_person | 30 | Easy — camera without anyone |
| 1 | user_present | 30 | Person standing nearby |
| 2 | on_machine | 30 | Seated, not lifting |
| 3 | good_rep | 30+ | Core class — most important |
| 4 | bad_rep | 30 | Exaggerate bad form deliberately |
| 5 | false_rep | 30 | Adjusting handles, stretching |
| 6 | resting | 30 | Seated between sets |
| 7 | half_rep | 30 | Stop halfway through ROM |
| **Total** | | **240 min / 300 target** | |

### Collection Workflow

```bash
# Pi records automatically when person detected (RECORD_SESSIONS = True)

# Step 1: Sync recordings from Pi to Mac Mini
make sync PI=pi@192.168.1.x

# Step 2: Annotate
make annotate
# Opens pose/label.html in browser
# Load video → draw time segments → pick class → export JSON
# Saves to data/annotations/{machine_id}_{date}.json

# Step 3: Check progress
make stats
# Shows: class 0: 12, class 1: 8, class 2: 31, ...

# Step 4: Extract sequences
make extract
# Converts video + JSON → numpy arrays in data/processed/

# Step 5: Train when ready
make train
# PyTorch LSTM (Apple Silicon MPS) → models/weights/activity_v1.onnx

# Step 6: Deploy
make deploy PI=pi@192.168.1.x
```

### Tips for Hard Classes

| Class | How to collect |
|-------|----------------|
| bad_rep (4) | Film yourself consciously bouncing, using momentum |
| half_rep (7) | Stop movement halfway — deliberate partial ROM |
| false_rep (5) | Adjust the cable pin, reach for water bottle while seated |
| resting (6) | Sit in machine between sets with hands in lap |

---

## Project 2 — Weight ID (YOLO)

**Target: 50+ photos per plate colour (250 total)**

| Colour | Weight | Min | Status |
|--------|--------|-----|--------|
| Red | 25 kg | 50 | 0 |
| Blue | 20 kg | 50 | 0 |
| Yellow | 15 kg | 50 | 0 |
| Green | 10 kg | 50 | 0 |
| White | 5 kg | 50 | 0 |

### Collection Steps

1. Mount camera at ~45° along barbell sleeve (see [[Hardware/Camera Placement]])
2. Load different plate combinations (1 plate, 2 plates, mixed)
3. Photograph in varied lighting: overhead fluorescent, natural, evening
4. Mix background plates with foreground plates
5. Save to `data/weight_plates/images/train/` (and `/val/`)
6. YOLO format: each image needs a `.txt` label file

```bash
make train-weight   # once images collected
# → models/weights/weight_id_v1.onnx

make test-weight    # test colour scan right now (no training needed)
```

---

## Project 3 — User Tracking (Face Enrolment)

**Enrol every member once.**

```bash
make enrol NAME="Matthew"
# Opens webcam
# Captures 5 frames, averages 512-dim ArcFace embeddings
# Saves to Supabase members table
# Immediately available (no restart needed)

# Alternative: use still photo
python face/enroll_member.py --name "Sarah" --image /path/to/photo.jpg

# Check who's enrolled
python face/enroll_member.py --list
```

Re-enrol in different lighting if recognition is unreliable (e.g. member always wears a hat).

---

## Infrastructure Checklist (Do Once)

- [ ] Set `SUPABASE_URL` in `pi/config.py`
- [ ] Set `SUPABASE_SERVICE_KEY` in `pi/config.py` (service_role key, not anon)
- [ ] Set `GITHUB_REVIEW_TOKEN` in `pi/config.py` (PAT with Contents:write)
- [ ] Configure `rclone` on Pi with Google Drive credentials
- [ ] Verify `GOOGLE_DRIVE_FOLDER_ID` = `1KNDC4wctZqVt8s41U4ALWHJ45OM5U9FA`
- [ ] Set `MACHINE_ZONE_ROI` per machine (use `SHOW_PREVIEW = True` to calibrate)
- [ ] Set `WEIGHT_STACK_ROI` per machine (covers the weight stack in frame)

---

## Go-Live Checklist

- [ ] 300+ annotated rep segments collected
- [ ] LSTM trained: `make train`
- [ ] LSTM deployed: `make deploy PI=pi@IP`
- [ ] 50+ weight plate photos per colour collected
- [ ] Weight YOLO trained: `make train-weight`
- [ ] All members enrolled: `make enrol NAME="..."`
- [ ] Supabase credentials set in `pi/config.py`
- [ ] End-to-end test: person sits → rep counted → session logged in Supabase
- [ ] Tablet showing live rep count (check `display/tablet.html`)
- [ ] Staff view showing machine cards (`display/staff.html`)

---

## Related

- [[System/LSTM Model]] — what gets trained
- [[System/Review Loop]] — ongoing improvement after go-live
- [[Data/AlphaFit Plates]] — colour ranges for weight plate collection
- [[Projects/Rep Tracking]] — needs 300+ segments
- [[Projects/Weight ID]] — needs 50+ photos per colour
- [[Projects/User Tracking]] — needs member enrolment
- [[Hardware/Machine Pi]] — Pi records the training footage
- [[Hardware/Camera Placement]] — correct angle = better training data
