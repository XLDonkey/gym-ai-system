# XL Fitness AI Overseer

Real-time gym machine monitoring using computer vision and neural networks.
Runs on Raspberry Pi 5 + Hailo AI HAT+. Identifies members, counts reps, classifies form, and logs everything to Supabase.

---

## What It Does

- **Detects engagement** — knows when someone is on the machine vs just nearby
- **Counts reps** — rule-based now → LSTM classifier once training data is ready
- **Classifies form** — 8-class activity schema per machine
- **Identifies members** — InsightFace face recognition matched against member registry
- **Validates reps** — optical flow on weight stack confirms weight actually moved
- **Logs everything** — sessions, reps, ROM, duration → Supabase (PostgreSQL)

---

## Repo Structure

```
gym-ai-system/
│
├── pi/                       # Raspberry Pi edge node
│   ├── main.py               # Main inference loop
│   ├── config.py             # Per-machine config  ← edit this per Pi
│   ├── engagement_detector.py
│   ├── weight_tracker.py
│   ├── session_recorder.py
│   └── setup.sh
│
├── face/                     # Face recognition
│   ├── face_recognizer.py
│   └── enroll_member.py      # CLI to register members
│
├── members/                  # Database layer
│   ├── db_client.py          # Supabase REST client (pure stdlib)
│   └── schema.sql            # Run in Supabase SQL editor
│
├── train/                    # Model training
│   ├── overseer_train.py     # LSTM on YOLO keypoint sequences
│   ├── extract_sequences.py
│   ├── train_model.py        # Fast MLP for < 500 samples
│   └── auto_label_yt.py      # Claude Opus auto-labels YouTube videos
│
├── pose/
│   └── label.html            # Annotation tool — open in browser
│
├── configs/                  # Machine configs
│   └── lat_pulldown.json
│
├── models/
│   ├── registry.json         # Version registry
│   └── weights/              # .onnx/.hef — NOT in git, use Google Drive
│
├── data/
│   ├── annotations/          # Label JSONs — commit these
│   ├── raw/                  # Videos — NOT in git
│   └── processed/            # Sequences — NOT in git
│
└── server/                   # Mac Mini proxy server
    └── index.js
```

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

## Quick Start

**1. Set up Supabase** — run `members/schema.sql` in the SQL editor

**2. Configure the Pi:**
```python
# pi/config.py
MACHINE_ID           = "xlf-pi-001"
SUPABASE_URL         = "https://xxxx.supabase.co"
SUPABASE_SERVICE_KEY = "eyJ..."
MACHINE_ZONE_ROI     = (0.15, 0.05, 0.85, 0.95)  # set via annotation tool
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
3. Press **B** → draw a box around the seated person → copies directly to `config.py`
4. Press **S** to mark start → **1–8** to label → export JSON to `data/annotations/`

**Target: 300+ segments across all 8 classes before first training run.**

---

## Training

```bash
cd train && pip install -r requirements.txt

# Extract sequences from annotated videos
python3 extract_sequences.py --annotations ../data/annotations/ --output ../data/processed/

# Train
python3 overseer_train.py --data ../data/processed/ --output ../models/weights/

# Auto-label from YouTube (boosts dataset)
export ANTHROPIC_API_KEY=sk-ant-...
python3 auto_label_yt.py --url "https://youtube.com/watch?v=..." --machine "lat_pulldown"
```

---

## Model Versions

| Version | Type | Status |
|---------|------|--------|
| v0.1 | Rule-based angle threshold | Live |
| v0.2 | MLP | Waiting for annotations |
| v1.0 | LSTM | Waiting for annotations |

Weights stored in Google Drive — see `models/registry.json`.

---

## Hardware

| Component | Purpose |
|-----------|---------|
| Raspberry Pi 5 (4GB) | Edge inference |
| Hailo-8 AI HAT+ (26 TOPS) | NPU acceleration |
| Pi Camera Module 3 Wide | Side-on view |
| PoE+ HAT | Single-cable power + network |
