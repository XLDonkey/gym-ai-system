# XL Fitness AI Overseer

> One camera per machine. Every rep counted, every weight logged, every member tracked — automatically.

**Repo:** `xldonkey/gym-ai-system` &nbsp;·&nbsp; **Owner:** Matt (XL Fitness) &nbsp;·&nbsp; **Maintainer:** XLDonkey

---

## The Four Projects

The Overseer captures four pieces of data for every set: **who**, **what exercise**, **how many reps**, and **how much weight**. These map to four independent AI modules:

| # | Project | What it does | Status |
|---|---------|-------------|--------|
| 1 | **Rep Tracking** | Counts reps, classifies form (good/bad/half), gates rep classes until user is confirmed engaged | Live (rule-based) |
| 2 | **Weight ID** | Reads AlphaFit plate colours on barbells — Red=25kg, Blue=20kg, Yellow=15kg, Green=10kg, White=5kg | Built, needs training data |
| 3 | **User Tracking** | Face ID at the gym door assigns a member name; IoU tracker follows them across the floor | Built, needs enrolment |
| 4 | **E-Weight** | Electric brushless motor weight stacks on cable machines — weight read from motor API, not camera | Phase 2 (hardware pending) |

Each project has its own folder and training pipeline. They all feed into `pi/main.py` on each Raspberry Pi.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  ENTRY PI  (door-facing camera)                                  │
│                                                                  │
│  Camera → InsightFace ArcFace → member identity                 │
│         → PersonDB (shared across all machine Pis)              │
└───────────────────────────┬─────────────────────────────────────┘
                            │ PersonDB (local network)
┌───────────────────────────▼─────────────────────────────────────┐
│  MACHINE PI  (one per machine — lat pulldown, squat rack, etc.)  │
│                                                                  │
│  Camera → YOLO Pose → 17 keypoints                              │
│         → GymTracker → track_id → member identity              │
│         → ActivityStateMachine → phase gate (IDLE / ENGAGED)   │
│         → LSTM (30-frame window) → 8-class activity             │
│         → WeightDetector → AlphaFit plate colours → kg          │
│         → WeightStackTracker → optical flow → rep validated     │
│         → Supabase → session + rep + weight logged              │
│         → confidence < 50% → clip flagged → GitHub review      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  MAC MINI  (training server)                                     │
│                                                                  │
│  rsync ← Pi recordings                                          │
│  pose/label.html → annotate → data/annotations/                 │
│  pose/review_server.py → review Pi-flagged clips                │
│  make train → LSTM → ONNX → scp to Pi                          │
│  make train-weight → YOLO → ONNX → scp to Pi                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project 1 — Rep Tracking

**Goal:** Count every rep, classify form (good/bad/half), gate rep classes until the user is confirmed seated.

**How it works:**
- YOLO Pose extracts 17 COCO keypoints (17 joints × 3 values = 51 features per frame)
- A 30-frame rolling window feeds an LSTM classifier
- Softmax output — exactly one of the 8 classes wins at a time
- ActivityStateMachine gates rep classes (3–7) until `on_machine` (class 2) is confirmed for 10 consecutive frames
- Uncertain predictions (confidence < 50%) are saved and uploaded to GitHub for human review

**8 activity classes:**

| ID | Label | Description | Counts as rep? |
|----|-------|-------------|---------------|
| 0 | `no_person` | Nobody at the machine | — |
| 1 | `user_present` | Person nearby, not seated | — |
| 2 | `on_machine` | Seated, engaged, about to lift | Starts session |
| 3 | `good_rep` | Full ROM, controlled, weight moving | Yes |
| 4 | `bad_rep` | Bouncing, swinging, uncontrolled | Yes (flagged) |
| 5 | `false_rep` | Stretching, adjusting handle/seat | No |
| 6 | `resting` | Seated between sets | — |
| 7 | `half_rep` | Partial ROM or single arm only | Yes (flagged) |

**Key files:**
- `pi/main.py` — main inference loop
- `pi/onnx_classifier.py` — LSTM inference wrapper
- `pi/activity_state_machine.py` — phase gating (IDLE → ENGAGED)
- `train/train_pytorch.py` — LSTM training (Mac Mini, MPS)
- `train/extract_sequences.py` — convert annotated videos → .npy
- `pose/label.html` — annotation tool (open in browser)
- `pose/review_server.py` — review portal at localhost:8787

**To collect data and train:**
```bash
make sync PI=pi@192.168.1.XX      # pull recordings from Pi
# Open pose/label.html → annotate → save to data/annotations/
make train                         # train LSTM + export ONNX
make deploy PI=pi@192.168.1.XX    # push model to Pi
```

---

## Project 2 — Weight ID (Free Weights)

**Goal:** Identify how much weight is loaded on a barbell by reading AlphaFit plate stripe colours.

**How it works:**
- Camera mounts on the barbell frame, looking along the sleeve from the side (~45°)
- Each plate appears as a coloured stripe ring in the frame
- YOLOv11 finds bounding boxes around each plate stripe
- HSV colour classification within each box → colour → kg value
- All plates summed + 20 kg bar = total weight
- Fallback: whole-frame colour band scanning when no YOLO model is trained yet

**Plate colours (AlphaFit):**

| Stripe | Weight | Reliability |
|--------|--------|-------------|
| Red | 25 kg | 99% |
| Blue | 20 kg | 99% |
| Yellow | 15 kg | 99% |
| Green | 10 kg | 98% |
| White | 5 kg | 95% |

**Key files:**
- `weight_id/detector.py` — `WeightDetector` class (YOLO + colour fallback)
- `weight_id/colour_matcher.py` — HSV colour → kg mapping
- `weight_id/train.py` — YOLO fine-tuning pipeline
- `configs/weight_plate_colours.json` — HSV range overrides per gym lighting
- `data/weight_plates/` — training images (add your own gym photos here)
- `weight/weight_plate_detector.html` — browser prototype (proof-of-concept)

**To train the plate detector:**
```bash
# 1. Collect 50+ photos per plate colour at your gym
# 2. Annotate in CVAT or Roboflow → export YOLO format → data/weight_plates/
make train-weight
make deploy PI=pi@192.168.1.XX
```

---

## Project 3 — User Tracking

**Goal:** Know who every person is from the moment they walk in.

**How it works:**
1. **Entry camera** (door-facing Pi) runs InsightFace ArcFace face recognition
2. Over the first 10 seconds, it collects multiple face reads (`IdentityWindow`)
3. Best match → member identity registered in `PersonDB`
4. Each **machine Pi** runs a local `GymTracker` (IoU bounding box matching)
5. When a person sits at a machine, `GymTracker.closest_track()` returns their track ID
6. `PersonDB.get_member(track_id)` returns their name and Supabase member_id
7. Session is logged to the correct member automatically

**Key files:**
- `user_tracking/entry_camera.py` — background thread monitoring the door camera
- `user_tracking/gym_tracker.py` — IoU-based person tracker per camera zone
- `user_tracking/person_db.py` — thread-safe identity registry (shared across Pis)
- `face/face_recognizer.py` — InsightFace ArcFace wrapper
- `face/enroll_member.py` — enrol a new member (3-second webcam capture)

**To enrol a member:**
```bash
source ~/.xlf-env/bin/activate
python3 face/enroll_member.py --name "Matthew"
# Looks at webcam for 3 seconds, saves 512-dim embedding to Supabase
```

---

## Project 4 — E-Weight (Electric Motor Stacks)

**Goal:** Know the exact weight on cable machines — to the nearest 1 kg — without any camera detection.

**How it works:**
- Custom-built electric weight stacks replace traditional pin-loaded iron stacks
- Brushless motors drive the selector to the correct position and lock
- The motor controller exposes a local HTTP API: `GET /api/weight → {"weight_kg": 42.5}`
- The Pi calls this API when a session starts — weight is always exact
- No OCR, no camera detection, no pin position estimation needed

**Status:** Phase 2 — hardware design in progress. API spec defined, placeholder client ready.

**Key files:**
- `e_weight/stack_client.py` — `StackClient` HTTP API client (ready for when hardware ships)

---

## Repository Structure

```
gym-ai-system/
│
├── pi/                         ← Runs on each machine Raspberry Pi
│   ├── main.py                 ← Main inference loop (start here)
│   ├── config.py               ← Per-machine settings — edit for each Pi
│   ├── onnx_classifier.py      ← LSTM activity model inference
│   ├── activity_state_machine.py ← Phase gating (IDLE → ENGAGED)
│   ├── clip_reporter.py        ← Flags uncertain clips to GitHub
│   ├── engagement_detector.py  ← Seated-on-machine check
│   ├── weight_tracker.py       ← Optical flow — validates weight moved
│   ├── session_recorder.py     ← Video recording to local disk
│   └── setup.sh                ← One-shot Pi setup script
│
├── weight_id/                  ← Project 2: Free weight plate detection
│   ├── detector.py             ← WeightDetector (YOLO + colour fallback)
│   ├── colour_matcher.py       ← HSV colour → kg mapping
│   └── train.py                ← YOLOv11 fine-tuning pipeline
│
├── user_tracking/              ← Project 3: Person identification & tracking
│   ├── entry_camera.py         ← Door camera face recognition (background thread)
│   ├── gym_tracker.py          ← IoU person tracker per camera zone
│   └── person_db.py            ← Thread-safe identity registry
│
├── e_weight/                   ← Project 4: Electric motor stack [Phase 2]
│   └── stack_client.py         ← HTTP API client for motor controller
│
├── face/                       ← Face recognition (shared by Project 3)
│   ├── face_recognizer.py      ← InsightFace ArcFace wrapper
│   └── enroll_member.py        ← Register a member's face
│
├── members/                    ← Supabase database layer
│   ├── schema.sql              ← Run in Supabase SQL editor (once)
│   └── db_client.py            ← REST client — no extra packages needed
│
├── train/                      ← Project 1: Rep tracking training (Mac Mini)
│   ├── train_pytorch.py        ← LSTM training (MPS backend)
│   ├── extract_sequences.py    ← Annotated videos → .npy sequences
│   └── auto_label_yt.py        ← Claude AI auto-labels YouTube videos
│
├── pose/
│   ├── label.html              ← Annotation tool (open in browser)
│   ├── review.html             ← Review portal — classify Pi-flagged clips
│   └── review_server.py        ← Local server for review portal (localhost:8787)
│
├── data/
│   ├── annotations/            ← Activity label JSONs — COMMIT THESE
│   ├── review/                 ← Pi-flagged clips — COMMIT after reviewing
│   ├── weight_plates/          ← Weight plate training images — NOT in git
│   │   ├── images/train/
│   │   ├── images/val/
│   │   ├── labels/train/
│   │   └── labels/val/
│   ├── members/                ← Member face photos — NOT in git
│   └── raw/                    ← Pi recordings — NOT in git
│
├── models/
│   ├── registry.json           ← Version log for all models
│   └── weights/                ← .onnx files — NOT in git (use scp)
│
├── configs/
│   ├── lat_pulldown.json       ← Live machine config
│   ├── machine_template.json   ← Template for new machines
│   └── weight_plate_colours.json ← HSV colour ranges for AlphaFit plates
│
├── weight/                     ← Original browser prototype (reference only)
│   └── weight_plate_detector.html
│
├── scripts/
│   └── mac_mini_setup.sh       ← One-shot Mac Mini setup
│
├── Makefile                    ← Common commands
└── bible.html                  ← Full technical document (open in browser)
```

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| YOLO pose detection | Live | Running on lat pulldown Pi, Hailo AI HAT+ |
| Rule-based rep counting | Live | Angle threshold fallback (no ML needed) |
| ActivityStateMachine gating | Live | Blocks rep classes until user confirmed engaged |
| Weight stack verification | Live | Optical flow, rejects phantom reps |
| Clip reporter + review loop | Live | Pi flags → GitHub → review portal → retrain |
| Review portal | Built | `python3 pose/review_server.py` |
| LSTM rep classifier | Needs data | Collect 300+ annotations, then `make train` |
| Weight plate detector | Built | Colour scan works now; YOLO needs training images |
| Face recognition | Ready | Run `face/enroll_member.py` to add members |
| Entry camera tracking | Built | Run `python3 user_tracking/entry_camera.py` |
| Gym floor tracker | Built | Integrated in pi/main.py |
| Supabase logging | Ready | Set `SUPABASE_URL` + `KEY` in `pi/config.py` |
| Electric weight stack | Phase 2 | Hardware in development |

---

## Setup

### Mac Mini (training server)

```bash
bash <(curl -s https://raw.githubusercontent.com/xldonkey/gym-ai-system/main/scripts/mac_mini_setup.sh)
```

### Raspberry Pi (one per machine)

```bash
git clone https://github.com/xldonkey/gym-ai-system.git
cd gym-ai-system/pi
bash setup.sh
# Edit pi/config.py for this machine
```

### Supabase

1. Create a free project at supabase.com
2. SQL Editor → paste + run `members/schema.sql`
3. Copy project URL and service role key into `pi/config.py`

### Enrol a member

```bash
source ~/.xlf-env/bin/activate
python3 face/enroll_member.py --name "Matthew"
```

---

## Make Commands

```bash
# Project 1 — Rep Tracking
make sync     PI=pi@IP    # pull Pi recordings → data/raw/
make train                 # train LSTM + export ONNX
make deploy   PI=pi@IP    # push ONNX to Pi + restart service
make review                # review portal at localhost:8787
make stats                 # annotation counts per class
make pending               # how many Pi clips need reviewing
make annotate              # open annotation tool in browser

# Project 2 — Weight ID
make train-weight          # train YOLO plate detector + export ONNX
make test-weight           # quick colour-scan test (no training needed)

# Project 3 — User Tracking
make enrol NAME="Matthew"  # enrol a new member face
make entry-camera          # run entry camera standalone

# Deployment
make logs     PI=pi@IP    # tail Pi service logs
make ssh      PI=pi@IP    # SSH into Pi
```

---

## The Review Loop (Pi → GitHub → Mac Mini → Pi)

```
Pi uncertain (confidence < 0.50)
  ↓
clip_reporter.py saves 30-frame keypoint window → uploads to GitHub data/review/
  ↓
Mac Mini: git pull → new clips appear
  ↓
python3 pose/review_server.py → http://localhost:8787
  ↓
Review portal: skeleton animation → click correct class → auto-saved
  ↓
git commit data/review/ → git push
  ↓
make train  (includes reviewed clips)
  ↓
make deploy PI=pi@IP → Pi now smarter. Repeat.
```

---

## Hardware Per Machine

| Part | Purpose | Cost |
|------|---------|------|
| Raspberry Pi 5 (4GB) | Edge inference | £80 |
| Hailo-8 AI HAT+ (26 TOPS) | NPU for YOLO at 30fps | £70 |
| Pi Camera Module 3 Wide | Side-on machine view | £35 |
| PoE+ HAT | Single cable: power + network | £25 |
| Mount + housing | Wall/ceiling bracket | £20 |
| **Total per machine** | | **£230** |

Additional per barbell station: 2× Pi Camera Module 3 (£35 each) for weight ID.
Entry camera Pi: 1× Raspberry Pi 5 + camera (£115).

Shared: Mac Mini M4 (~£700) trains all models.

---

## The Bible

Full technical reference — architecture, all 4 AI models, hardware decisions, training strategy, enterprise scale plan.

```bash
open ~/gym-ai-system/bible.html
```
