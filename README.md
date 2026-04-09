# Gym Overseer AI

> Real-time AI that watches every machine, counts every rep, identifies every member — automatically.

**GitHub:** `Matt-xlfitness/Gym-Overseer-AI` &nbsp;·&nbsp; **Collaborators:** Matt (owner), XLDonkey (maintain)

---

## What This Is

A self-improving AI system for XL Fitness. One Raspberry Pi per machine. No phones. No QR codes. No manual input from staff or members.

A member sits down → the system sees them → starts a session → counts every rep → classifies form → logs to their profile. When the AI isn't sure what it's seeing, it flags the clip, sends it to GitHub, and Matt corrects it. The model retrains and gets smarter. Every week it improves.

---

## Two-Minute Overview

```
┌──────────────────────────────────────────────────────────────┐
│  RASPBERRY PI  (one per machine)                             │
│                                                              │
│  Camera → YOLO pose → 17 keypoints                          │
│         → Engagement check (is person seated?)              │
│         → Weight stack check (did plates move?)             │
│         → ONNX LSTM → 8-class output (ONE state wins)       │
│         → Supabase (log reps + sessions)                     │
│         → Low confidence? → flag clip to GitHub             │
└──────────────────────┬───────────────────────────────────────┘
                       │ SSH / GitHub
┌──────────────────────▼───────────────────────────────────────┐
│  MAC MINI  (training server)                                 │
│                                                              │
│  rsync ← Pi recordings                                       │
│  pose/label.html → annotate segments → data/annotations/     │
│  pose/review.html → review Pi-flagged clips                  │
│  make train → PyTorch LSTM → ONNX export                    │
│  make deploy → scp model to Pi                              │
└──────────────────────────────────────────────────────────────┘
```

---

## The 8 Activity Classes

| ID | Label | Description | Counts as rep? |
|----|-------|-------------|---------------|
| 0 | `no_person` | Nobody at the machine | — |
| 1 | `user_present` | Person nearby, not seated | — |
| 2 | `on_machine` | Seated, engaged, about to lift | Starts session |
| 3 | `good_rep` | Full ROM, controlled, weight moving | ✓ Yes |
| 4 | `bad_rep` | Bouncing, swinging, uncontrolled | ✓ Yes (flagged) |
| 5 | `false_rep` | Stretching, adjusting handle/seat/pin | ✗ No |
| 6 | `resting` | Seated between sets | — |
| 7 | `half_rep` | Partial ROM or single arm only | ✓ Yes (flagged) |

**The gate:** Softmax output means exactly ONE class is active at a time. The state machine also blocks rep classes (3–7) until class 2 (on_machine) is confirmed for 10 consecutive frames. A bad_rep cannot appear before engagement.

---

## Repository Structure

```
Gym-Overseer-AI/
│
├── pi/                          ← Runs on each Raspberry Pi
│   ├── main.py                  ← Main inference loop (start here)
│   ├── config.py                ← Per-machine config — edit for each Pi
│   ├── onnx_classifier.py       ← LSTM activity model inference
│   ├── activity_state_machine.py← Phase gating (IDLE → ENGAGED)
│   ├── clip_reporter.py         ← Flags uncertain clips to GitHub
│   ├── engagement_detector.py   ← Seated-on-machine check
│   ├── weight_tracker.py        ← Optical flow — validates weight moved
│   ├── session_recorder.py      ← Video recording to local disk
│   ├── requirements.txt         ← pip install -r requirements.txt
│   └── setup.sh                 ← One-shot Pi setup script
│
├── face/                        ← Member face recognition
│   ├── face_recognizer.py       ← InsightFace ArcFace wrapper
│   └── enroll_member.py         ← Register a member's face
│
├── members/                     ← Supabase database layer
│   ├── schema.sql               ← Run in Supabase SQL editor (once)
│   └── db_client.py             ← REST client — no extra packages needed
│
├── train/                       ← Run on Mac Mini
│   ├── train_pytorch.py         ← LSTM training (MPS backend)
│   ├── extract_sequences.py     ← Convert annotated videos → .npy sequences
│   └── auto_label_yt.py         ← Claude AI auto-labels YouTube videos
│
├── pose/
│   ├── label.html               ← Annotation tool — open in browser
│   ├── review.html              ← Review portal — classify Pi-flagged clips
│   ├── review_server.py         ← Local server for review portal
│   └── index.html               ← Start page (links to all tools)
│
├── data/
│   ├── annotations/             ← Label JSONs — COMMIT THESE
│   ├── review/                  ← Pi-flagged clips — COMMIT after reviewing
│   ├── raw/                     ← Raw videos — NOT in git (too large)
│   └── processed/               ← Extracted sequences — NOT in git
│
├── models/
│   ├── registry.json            ← Version log for all 4 models
│   └── weights/                 ← .onnx files — NOT in git (use scp/Drive)
│
├── configs/
│   ├── lat_pulldown.json        ← Live machine config
│   └── machine_template.json    ← Copy this for each new machine
│
├── scripts/
│   └── mac_mini_setup.sh        ← One-shot Mac Mini setup
│
├── Makefile                     ← Common commands (make train, make deploy...)
└── bible.html                   ← Full technical document — open in browser
```

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| YOLO pose detection | ✅ Live | Running on lat pulldown Pi |
| Rule-based rep counting | ✅ Live | Angle threshold fallback |
| Weight stack verification | ✅ Live | Optical flow, rejects phantom reps |
| Engagement detection | ✅ Live | Prevents bystander sessions |
| Face recognition | ⏳ Ready | Run `enroll_member.py` to add members |
| Supabase logging | ⏳ Ready | Set `SUPABASE_URL` + `KEY` in `config.py` |
| ONNX activity classifier | ❌ Needs data | Collect 300+ annotations first |
| Review portal | ✅ Built | `python3 pose/review_server.py` |
| Training pipeline | ✅ Built | `make train` on Mac Mini |
| Review loop | ✅ Built | Pi flags → GitHub → portal → retrain |

---

## Setup

### Mac Mini (training server) — run once

```bash
bash <(curl -s https://raw.githubusercontent.com/Matt-xlfitness/Gym-Overseer-AI/main/scripts/mac_mini_setup.sh)
```

Then open a new terminal tab and type `xlf` to jump into the project.

### Raspberry Pi — run once per Pi

```bash
git clone https://github.com/Matt-xlfitness/Gym-Overseer-AI.git
cd Gym-Overseer-AI/pi
bash setup.sh
```

Then edit `pi/config.py` for this machine (machine ID, zone ROI, weights).

### Supabase — run once

1. Create a free project at [supabase.com](https://supabase.com)
2. Open SQL Editor → paste + run `members/schema.sql`
3. Copy your project URL and service role key into `pi/config.py`

### Enrol a member

```bash
source ~/.xlf-env/bin/activate
python3 face/enroll_member.py --name "Matthew"
# Follow prompts — looks at webcam for 3 seconds
```

---

## Day-to-Day Workflow

### Collecting training data

```
1. Pi records sessions to /home/pi/xlf_recordings/
2. On Mac Mini: make sync PI=pi@192.168.1.XX
3. Open pose/label.html in browser
4. Load video → label segments → Export JSON → save to data/annotations/
5. git add data/annotations/ && git commit -m "annotations: session date"
```

### Reviewing Pi-flagged clips

```
1. On Mac Mini: git pull   (picks up clips the Pi uploaded overnight)
2. python3 pose/review_server.py
3. Open http://localhost:8787
4. Watch skeleton animation → click correct class → auto-saves
5. git add data/review/ && git commit -m "reviewed: batch date"
```

### Training a new model

```bash
make train
# Trains on all annotations + reviewed Pi clips
# Exports to models/weights/activity_v*.onnx
# Takes ~5 minutes on Mac Mini M4
```

### Deploying to a Pi

```bash
make deploy PI=pi@192.168.1.XX
# Copies .onnx to Pi, restarts the service
# Watch logs: ssh pi@192.168.1.XX "sudo journalctl -u xlf-overseer -f"
```

---

## Key Configuration (`pi/config.py`)

```python
MACHINE_ID        = "xlf-pi-001"          # unique per Pi
MACHINE_NAME      = "Nautilus Lat Pulldown"
MACHINE_ZONE_ROI  = (0.15, 0.05, 0.85, 0.95)  # set via label.html bounding box tool

# Supabase — get from supabase.com dashboard
SUPABASE_URL      = "https://xxxx.supabase.co"
SUPABASE_SERVICE_KEY = "eyJ..."

# ONNX model — set after first training run
ONNX_MODEL_PATH   = "/home/pi/xlf/models/activity_v1.onnx"

# GitHub review loop — PAT with Contents:write scope
GITHUB_REVIEW_TOKEN = "ghp_..."
GITHUB_REVIEW_REPO  = "Matt-xlfitness/Gym-Overseer-AI"
```

---

## Make Commands

```bash
make sync    PI=pi@192.168.1.XX   # pull recordings from Pi → data/raw/
make train                         # extract sequences + train LSTM + export ONNX
make deploy  PI=pi@192.168.1.XX   # push latest .onnx to Pi + restart service
make review                        # start review portal at localhost:8787
make stats                         # show annotation counts per class
make pending                       # show how many Pi clips need reviewing
make logs    PI=pi@192.168.1.XX   # tail Pi service logs live
```

---

## The Review Loop (Pi → GitHub → Mac Mini → Pi)

```
Pi uncertain (confidence < 0.50)
  ↓
clip_reporter.py saves:
  • 30-frame keypoint window embedded in _meta.json
  • Uploads to: data/review/{machine_id}/{date}/
  ↓
Mac Mini: git pull → sees new clips
  ↓
python3 pose/review_server.py → open http://localhost:8787
  ↓
Review portal: skeleton animation → click correct class → saved
  ↓
git commit data/review/ → git push
  ↓
make train  (includes reviewed clips automatically)
  ↓
make deploy PI=pi@192.168.1.XX
  ↓
Pi now smarter. Loop repeats.
```

---

## Hardware Per Machine

| Part | Purpose | ~Cost |
|------|---------|-------|
| Raspberry Pi 5 (4GB) | Edge inference | £80 |
| Hailo-8 AI HAT+ (26 TOPS) | NPU — YOLO at 30fps | £70 |
| Pi Camera Module 3 Wide | Side-on view | £35 |
| PoE+ HAT | Single cable: power + network | £25 |
| Mount + housing | Wall/ceiling bracket | £20 |
| **Total per machine** | | **£230** |

Shared: Mac Mini M4 (~£700) trains models for all machines.

---

## The Bible

Open `bible.html` in a browser for the complete technical reference — architecture, AI models, hardware decisions, training strategy, and enterprise scale plan.

```bash
open ~/Gym-Overseer-AI/bible.html    # Mac Mini
# or visit: pose/index.html → "The AI Bible"
```
