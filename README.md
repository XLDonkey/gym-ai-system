# XL Fitness AI Overseer

> One camera per machine. Every rep counted, every weight logged, every member tracked — automatically.
> No phones. No QR codes. No staff input.

---

## The Four Projects

The Overseer captures four things for every set: **reps**, **weight**, **who**, and eventually **cable machine weight**. Each is an independent AI module.

| # | Project | What it does | MVP | Status |
|---|---------|-------------|-----|--------|
| **1** | [Rep Tracking (NN)](#1-rep-tracking) | Counts reps, classifies form — LSTM 8-class neural network | ✅ | Live (rule-based). LSTM training data being collected |
| **2** | [Weight Detection](#2-weight-detection) | Camera at barbell horn → reads AlphaFit plate colours → kg | ✅ | Built — needs training images |
| **3** | [Person Tracking](#3-person-tracking) | YOLO bounding box → IoU tracker → who is at the machine | ✅ no face ID | Built — MVP uses bounding box only |
| **4** | [E-Weight](#4-e-weight) | Electric motor weight stacks on cable machines → digital weight readout | ❌ Phase 2 | Hardware not yet built |

---

## Display Layer

Live data flows from each Pi to a **Samsung tablet mounted on the machine** via Supabase Realtime:

```
Pi (inference)
  → Supabase machine_state table (live update per frame change)
    → Samsung Tablet (website, Kiosk mode)
       → displays rep count, weight, form, session summary
```

Each tablet is assigned to a machine in Supabase. First-launch setup screen lets staff pick the machine — then it locks to that machine's live data feed.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  MACHINE PI  (Raspberry Pi 5 + Hailo-8 AI HAT+)          │
│  One per gym machine                                      │
│                                                           │
│  Camera → YOLO Pose (30fps, Hailo NPU)                   │
│         → 17 body keypoints per frame                     │
│                                                           │
│  ┌─ Project 1: Rep Tracking ──────────────────────────┐  │
│  │  30-frame window → LSTM → 8-class activity          │  │
│  │  Rule-based angle counting (fallback / MVP)         │  │
│  │  EngagementDetector → zone + pose gate              │  │
│  │  WeightStackTracker → optical flow validates rep    │  │
│  └────────────────────────────────────────────────────┘  │
│                                                           │
│  ┌─ Project 2: Weight Detection ──────────────────────┐  │
│  │  Camera at barbell horn → HSV colour scan           │  │
│  │  AlphaFit plates: Red=25 Blue=20 Yellow=15          │  │
│  │  Green=10 White=5 + 20kg bar = total                │  │
│  └────────────────────────────────────────────────────┘  │
│                                                           │
│  ┌─ Project 3: Person Tracking ───────────────────────┐  │
│  │  YOLO bounding box → IoU tracker → track_id        │  │
│  │  MVP: bounding box only (no face ID)               │  │
│  │  Post-MVP: ArcFace → member name                   │  │
│  └────────────────────────────────────────────────────┘  │
│                                                           │
│  ┌─ Project 4: E-Weight (Phase 2) ────────────────────┐  │
│  │  Motor controller API → GET /api/weight             │  │
│  │  100% accurate weight on cable machines             │  │
│  │  Hardware not yet built                             │  │
│  └────────────────────────────────────────────────────┘  │
│                                                           │
│  → Supabase (machine_state, sessions, sets, reps)        │
│  → Samsung Tablet (Supabase Realtime → live GUI)         │
│  → Low-confidence clips → GitHub data/review/ → retrain  │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  MAC MINI  (training server, stays off-site)             │
│  make sync → pull Pi recordings                          │
│  make annotate → label video segments                    │
│  make train → LSTM → ONNX → deploy to Pi                │
│  make review → annotate low-confidence clips             │
└──────────────────────────────────────────────────────────┘
```

---

## Project 1 — Rep Tracking

**Goal:** Count every rep. Classify form as good, bad, or partial. Gate sessions so only genuine machine use is counted.

### Two modes

| Mode | When | How |
|------|------|-----|
| **Rule-based** (live now) | `ONNX_MODEL_PATH` blank | Elbow angle threshold: crosses 90° → 130° = 1 rep |
| **LSTM 8-class** (post-training) | `ONNX_MODEL_PATH` set | 30-frame keypoint window → neural network |

### 8 Activity Classes

| ID | Label | Counts? | Valid phase |
|----|-------|---------|-------------|
| 0 | `no_person` | — | IDLE |
| 1 | `user_present` | — | IDLE |
| 2 | `on_machine` | Starts session | Both |
| 3 | `good_rep` | ✅ Yes | ENGAGED |
| 4 | `bad_rep` | ✅ Flagged | ENGAGED |
| 5 | `false_rep` | ❌ No | ENGAGED |
| 6 | `resting` | — | ENGAGED |
| 7 | `half_rep` | ✅ Flagged | ENGAGED |

10 consecutive `on_machine` frames required to enter ENGAGED phase.

### LSTM model

```
Input:  (batch, 30, 51)   30 frames × 17 keypoints × [x, y, conf]
LSTM:   51 → 128 hidden
Linear: 128 → 64 → ReLU → 8
Output: softmax — one class wins
```

### Commands

```bash
make sync PI=pi@IP       # pull Pi recordings to Mac Mini
make annotate            # open annotation tool
make train               # train LSTM + export ONNX
make deploy PI=pi@IP     # push ONNX to Pi + restart
make review              # review uncertain clips (localhost:8787)
make stats               # annotation count per class
```

**Data needed:** 300+ annotated segments, ≥30 per class.

### Key files

```
pi/main.py                    Main inference loop
pi/activity_state_machine.py  IDLE / ENGAGED phase gate
pi/onnx_classifier.py         LSTM ONNX inference, 30-frame buffer
pi/engagement_detector.py     Zone + pose gate
pi/clip_reporter.py           Flags uncertain clips to GitHub
train/train_pytorch.py        Mac Mini LSTM training script
pose/label.html               Video annotation tool
pose/review_server.py         Review portal (localhost:8787)
```

---

## Project 2 — Weight Detection

**Goal:** Camera at barbell horn → reads plate colours → total weight in kg. Zero manual entry.

### AlphaFit plate colours

| Colour | Weight |
|--------|--------|
| Red stripe | 25 kg |
| Blue stripe | 20 kg |
| Yellow stripe | 15 kg |
| Green stripe | 10 kg |
| White stripe | 5 kg |

Bar: **20 kg** added automatically.

### Camera placement

Mount on barbell rack looking along the horn/sleeve from the side (~45°).
Two cameras per barbell station — one per sleeve — at least one always clear.

### Detection modes

| Mode | Requires training | Accuracy |
|------|------------------|---------|
| Colour scan (HSV full-frame) | No — works immediately | ~90% |
| YOLO per-plate detection | 50+ photos per colour | ~99% |

### Commands

```bash
make test-weight         # test colour scan now (no training needed)
make train-weight        # train YOLO detector after collecting images
```

### Key files

```
weight_id/detector.py             WeightDetector — YOLO + colour fallback
weight_id/colour_matcher.py       HSV colour ranges → kg
weight_id/train.py                YOLOv11 fine-tuning
configs/weight_plate_colours.json HSV config (tune per gym lighting)
data/weight_plates/               Training images (not in git)
```

---

## Project 3 — Person Tracking

**Goal:** Know who is at each machine. MVP = bounding box only. Post-MVP adds face ID.

### MVP (bounding box — no face ID)

```
YOLO bounding box → IoU tracker → track_id
EngagementDetector → is this person seated and working?
→ Session attributed to anonymous "Member"
```

### Post-MVP (face ID)

```
Entry Pi (door) → ArcFace → PersonDB → member name + UUID
Floor tracker → closest_track(machine_roi) → named member
→ Session attributed in Supabase
```

### Commands

```bash
make enrol NAME="Matthew"    # enrol a member (post-MVP)
make entry-camera            # run door camera standalone
```

### Key files

```
user_tracking/gym_tracker.py    IoU bounding box tracker
user_tracking/person_db.py      track_id ↔ member registry
user_tracking/entry_camera.py   Door camera — ArcFace (post-MVP)
face/recognizer.py              InsightFace wrapper
```

---

## Project 4 — E-Weight

**Goal:** Electric brushless motor weight stacks on cable machines — weight is a software value. 100% accurate.

> **Phase 2 — hardware not yet built.** Code is ready, disabled by default (`enabled=False`).

```
Motor controller (per cable machine)
  → GET /api/weight → {"weight_kg": 42.5}
  → Pi reads at session start
  → Written to Supabase — no camera detection needed
```

| Machine type | Weight method | Accuracy |
|---|---|---|
| Free weight barbells | Project 2 plate colours | ~95–99% |
| Cable / pin machines (Phase 2) | E-Weight motor API | 100% |

### Key files

```
e_weight/stack_client.py    StackClient — HTTP API client (disabled)
```

---

## Repository Structure

```
Gym-Overseer-AI/
│
├── pi/               Main inference loop — integrates all 4 projects
│   ├── main.py       Orchestrates everything per frame
│   └── config.py     All settings for one deployed Pi
│
├── weight_id/        Project 2 — weight plate detection
├── user_tracking/    Project 3 — person tracking + face ID
├── e_weight/         Project 4 — electric motor weight stack
│
├── display/          Tablet + staff display layer
│   ├── tablet.html   Member kiosk (Samsung tablet)
│   ├── staff.html    Staff floor view — all machines
│   ├── ws_server.py  WebSocket broadcaster
│   └── set_reporter.py  Set complete → Supabase
│
├── train/            Project 1 — LSTM training pipeline (Mac Mini)
├── pose/             Annotation tool + review portal
├── face/             InsightFace ArcFace wrapper (post-MVP)
├── members/          Supabase REST client
│
├── configs/          Per-machine JSON configs
├── data/             Annotations, review clips (training images not in git)
├── models/           ONNX model registry (weights not in git)
│
├── obsidian/         Full knowledge graph — open as Obsidian vault
└── CLAUDE.md         AI context — read automatically by Claude Code
```

---

## Hardware (Per Machine)

| Part | Purpose | Cost |
|------|---------|------|
| Raspberry Pi 5 (4GB) | Edge inference | £80 |
| Hailo-8 AI HAT+ (26 TOPS) | YOLO at 30fps on NPU | £70 |
| Pi Camera Module 3 Wide | Machine side-on view | £35 |
| PoE+ HAT | One cable: power + network | £25 |
| Mount + housing | | £20 |
| **Total per machine** | | **£230** |

Barbell station: +£70 (2× sleeve cameras for Project 2).
Entry Pi (door, face ID post-MVP): £140.
Mac Mini M4 (training, one-off): ~£700.

---

## Current Status

| Task | Status |
|------|--------|
| Rule-based rep counting | ✅ Live |
| LSTM training data collection | 🔄 In progress |
| Tablet display GUI (Figma) | 🔄 Designing |
| Weight detection — colour scan | ✅ Built |
| Weight detection — YOLO trained | ⏳ Needs 50+ images per colour |
| Person tracking — bounding box | ✅ Built |
| Face ID / member enrolment | ⏳ Post-MVP |
| Supabase Realtime → tablet | ⏳ Building |
| E-Weight hardware | ❌ Phase 2 |
