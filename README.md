# XL Fitness AI Overseer

Real-time gym machine monitoring using computer vision and neural networks.
Runs on Raspberry Pi 5 + Hailo AI HAT+. Identifies members, counts reps, classifies form, and logs everything to Supabase.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  GITHUB  (data hub)                                                 │
│  data/annotations/  ← committed label JSONs from annotation tool   │
│  data/review/       ← Pi-flagged uncertain clips for human review   │
│  models/registry.json ← version table for all 4 models             │
│  models/weights/    ← NOT in git (use Google Drive)                 │
└────────────┬────────────────────────┬───────────────────────────────┘
             │                        │
       annotate                 review clips
             │                        │
     ┌───────▼────────┐    ┌──────────▼──────────┐
     │  MAC MINI      │    │  HUMAN REVIEWER     │
     │  PyTorch LSTM  │◄───│  sets true_class in │
     │  MPS backend   │    │  *_meta.json        │
     │  train_pytorch │    └─────────────────────┘
     │  export ONNX   │
     └───────┬────────┘
             │  scp activity_v*.onnx
             ▼
     ┌───────────────────────────────────────────────────────────┐
     │  RASPBERRY PI 5  (one per machine)                        │
     │  Hailo NPU → YOLOv11-pose → 17 keypoints                 │
     │  onnxruntime → LSTM → 8-class softmax (ONE state wins)   │
     │  InsightFace → face embeddings → member ID               │
     │  Optical flow → weight stack → phantom rep rejection     │
     │  confidence < 0.50 → clip_reporter → upload to GitHub   │
     └───────────────────────────────────────────────────────────┘
```

**The gate:** The LSTM's softmax output means exactly ONE of the 8 activity states is active at any time. "Resting" and "on_machine" cannot both be true — the probabilities must sum to 1.0 and argmax picks a single winner.

---

## What It Does

- **Detects engagement** — person must be seated on the machine (not just nearby) before a session starts
- **Counts reps** — rule-based now → LSTM activity classifier once training data is ready
- **Classifies form** — 8-class activity schema per machine
- **Identifies members** — InsightFace face recognition matched against member registry
- **Validates reps** — optical flow on weight stack confirms weight actually moved
- **Flags uncertain clips** — low-confidence ONNX inference → saved to GitHub → human review → retraining
- **Logs everything** — sessions, reps, ROM, duration → Supabase (PostgreSQL)

---

## Activity Classes

| ID | Label | Description |
|----|-------|-------------|
| 0 | `no_person` | Nobody at the machine |
| 1 | `user_present` | Person nearby, not seated |
| 2 | `on_machine` | Seated, engaged, not yet lifting |
| 3 | `good_rep` | Full ROM, controlled, weight moving |
| 4 | `bad_rep` | Uncontrolled, bouncing, momentum |
| 5 | `false_rep` | Stretching, adjusting handle/seat/pin |
| 6 | `resting` | Seated between sets |
| 7 | `half_rep` | Partial ROM or single arm only |

---

## Repo Structure

```
gym-ai-system/
│
├── pi/                         # Raspberry Pi edge node
│   ├── main.py                 # Main inference loop
│   ├── config.py               # Per-machine config  ← edit per Pi
│   ├── onnx_classifier.py      # ONNX activity model (8-class, softmax gate)
│   ├── clip_reporter.py        # Uploads low-confidence clips to GitHub
│   ├── engagement_detector.py  # Seated-on-machine check before session starts
│   ├── weight_tracker.py       # Optical flow — validates weight moved
│   └── session_recorder.py
│
├── face/                       # Face recognition
│   ├── face_recognizer.py      # InsightFace ArcFace wrapper + IdentityWindow
│   └── enroll_member.py        # CLI to register members
│
├── members/                    # Database layer
│   ├── db_client.py            # Supabase REST client (pure stdlib)
│   └── schema.sql              # Run in Supabase SQL editor
│
├── train/                      # Model training (run on Mac Mini)
│   ├── train_pytorch.py        # LSTM training — Mac Mini MPS backend → ONNX export
│   ├── extract_sequences.py    # Extract keypoint sequences from annotated videos
│   ├── auto_label_yt.py        # Claude Opus auto-labels YouTube videos
│   └── overseer_train.py       # Legacy training script
│
├── pose/
│   ├── index.html              # Start page
│   ├── label.html              # Annotation tool — draw zones, label segments, export JSON
│   └── alpha.html              # Live rep counter (browser-based)
│
├── data/
│   ├── annotations/            # Label JSONs — commit these
│   ├── review/                 # Pi-flagged clips for human review — commit reviewed JSONs
│   ├── raw/                    # Videos — NOT in git (Google Drive)
│   └── processed/              # Extracted sequences — NOT in git
│
├── models/
│   ├── registry.json           # All 4 model versions + review loop config
│   └── weights/                # .onnx files — NOT in git (Google Drive / scp)
│
├── configs/
│   └── lat_pulldown.json       # Per-machine config JSON
│
└── server/                     # Mac Mini proxy server
    └── index.js
```

---

## Quick Start

**1. Set up Supabase** — run `members/schema.sql` in the SQL editor

**2. Configure the Pi:**
```python
# pi/config.py
MACHINE_ID           = "xlf-pi-001"
SUPABASE_URL         = "https://xxxx.supabase.co"
SUPABASE_SERVICE_KEY = "eyJ..."
MACHINE_ZONE_ROI     = (0.15, 0.05, 0.85, 0.95)  # set via annotation tool

# Once ONNX model is trained and deployed:
ONNX_MODEL_PATH      = "/home/pi/xlf/models/activity_v1.onnx"
GITHUB_REVIEW_TOKEN  = "ghp_..."   # PAT with Contents:write
```

**3. Install and run:**
```bash
cd pi && bash setup.sh && python3 main.py
```

**4. Enrol members:**
```bash
python3 face/enroll_member.py --name "Matthew"
```

---

## Annotation Workflow

1. Open `pose/label.html` in a browser
2. Load a Pi recording
3. Press **B** → draw a box around the seated person → copies to `MACHINE_ZONE_ROI` in config.py
4. Press **S** to mark start → press **1–8** for the class → press **E** for end of segment
5. Export JSON to `data/annotations/`

**Target: 300+ segments across all 8 classes before first training run (≥30 per class).**

---

## Training Pipeline (Mac Mini)

```bash
cd train && pip install -r requirements.txt

# Option A: Train from pre-extracted sequences
python3 extract_sequences.py \
    --annotations ../data/annotations/ \
    --output ../data/processed/

python3 train_pytorch.py \
    --sequences ../data/processed/ \
    --output ../models/weights/activity_v1.onnx

# Option B: Train from raw videos + annotations (slower, auto-extracts)
python3 train_pytorch.py \
    --annotations ../data/annotations/ \
    --videos ../data/raw/ \
    --output ../models/weights/activity_v1.onnx

# Include human-reviewed Pi clips in training:
python3 train_pytorch.py \
    --sequences ../data/processed/ \
    --review ../data/review/ \
    --output ../models/weights/activity_v2.onnx

# Auto-label from YouTube (boosts dataset):
export ANTHROPIC_API_KEY=sk-ant-...
python3 auto_label_yt.py --url "https://youtube.com/watch?v=..." --machine lat_pulldown
```

**Deploy to Pi:**
```bash
scp models/weights/activity_v1.onnx pi@xlf-pi-001:/home/pi/xlf/models/
# Update ONNX_MODEL_PATH in pi/config.py and restart main.py
```

---

## Review Loop (Pi → GitHub → Mac Mini → Pi)

When the ONNX model is uncertain (confidence < 0.50 on any frame), the Pi automatically:
1. Saves the 30-frame keypoint window to `data/review/{machine_id}/{date}/`
2. Uploads it to GitHub via the API

**To process review clips:**
1. Check `data/review/` on GitHub — open `*_meta.json` files
2. Set `"true_class": 3` (or the correct class ID) in each reviewed JSON
3. Commit the changes
4. Retrain: `python3 train/train_pytorch.py --review data/review/`
5. Deploy new ONNX to Pi

---

## Model Registry

| Model | Type | Status |
|-------|------|--------|
| Activity classifier v0.1 | Rule-based angles | Live |
| Activity classifier v0.2 | MLP | Waiting for 60+ samples |
| Activity classifier v1.0 | LSTM (8-class, softmax gate) | Waiting for 300+ samples |
| Face ID | InsightFace buffalo_sc ArcFace | Live |
| Pose detection | YOLOv11n-pose | Live |
| Weight verifier | Optical flow (Farneback) | Live |

Full version details in `models/registry.json`.
Weights stored in Google Drive (not in git).

---

## Hardware

| Component | Purpose |
|-----------|---------|
| Raspberry Pi 5 (4GB) | Edge inference |
| Hailo-8 AI HAT+ (26 TOPS) | NPU — runs YOLOv11-pose at 30fps |
| Pi Camera Module 3 Wide | Side-on view of machine |
| PoE+ HAT | Single-cable power + network |
| Mac Mini M4 | PyTorch training (MPS backend) |
