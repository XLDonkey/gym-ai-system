# XL Fitness AI Overseer — Claude Context

This file is read automatically by Claude Code at the start of every session.
It contains the full project context so Claude never loses track.

---

## What This Project Is

A self-improving AI gym tracking system for **XL Fitness**.
One camera per machine. Every rep counted, every weight logged, every member tracked — automatically.
No phones. No QR codes. No staff input.

**Primary repo:** `Matt-xlfitness/Gym-Overseer-AI` (private — production)
**Public repo:** `xldonkey/gym-ai-system` (hosts GitHub Pages / Bible only)
**Bible (live):** https://xldonkey.github.io/gym-ai-system/bible.html
**Branch for all development:** `claude/gym-tracking-neural-network-sHldz`

---

## MVP v1 Scope (Current Focus)

Building in this order:
1. **Tablet interface** → `display/tablet.html` — member-facing kiosk display ← **IN PROGRESS**
2. **Weight detection Pi** → camera looks at barbell horns/sleeves → detects plate colours → kg
3. **Person tracking** → bounding box IoU tracker (no face detection in MVP)

**Explicitly OUT of MVP v1:**
- Face detection / member identification
- Supabase session logging (can be stubbed)
- LSTM classifier (rule-based angle counting only for now)

**Running in parallel:**
- Recording LSTM training data on the machine Pi

---

## The Four Projects

| # | Name | Status | Key file |
|---|------|--------|----------|
| 1 | **Rep Tracking** | Live (rule-based). LSTM training data being collected | `pi/activity_state_machine.py` |
| 2 | **Weight ID** | Building — Pi camera looks at barbell horns | `weight_id/detector.py` |
| 3 | **User Tracking** | MVP = bounding box only, no face ID | `user_tracking/gym_tracker.py` |
| 4 | **E-Weight** | Phase 2 — hardware pending | `e_weight/stack_client.py` |

---

## System Architecture

```
ENTRY PI (door camera)
  └─ InsightFace ArcFace → member identity → PersonDB [NOT IN MVP]

MACHINE PI (one per machine)
  ├─ YOLO Pose          → 17 keypoints / frame
  ├─ GymTracker         → bounding box IoU tracking [MVP: no face ID]
  ├─ ActivityStateMachine → IDLE / ENGAGED phase gate
  ├─ LSTM Classifier    → 8-class activity [post-MVP: needs training data]
  ├─ WeightDetector     → AlphaFit plate colour → kg [building now]
  ├─ WeightStackTracker → optical flow → validates rep
  ├─ ws_server.py       → WebSocket → tablet (100ms)
  ├─ set_reporter.py    → HTTP POST → Supabase / Power Automate
  └─ clip_reporter.py   → confidence < 50% → GitHub review

MAC MINI (training server)
  ├─ make sync          → rsync Pi recordings
  ├─ make train         → LSTM → ONNX
  └─ make deploy        → scp ONNX to Pi
```

---

## Activity Classes (8-class schema)

| ID | Label | Counts? |
|----|-------|---------|
| 0 | `no_person` | — |
| 1 | `user_present` | — |
| 2 | `on_machine` | Starts session |
| 3 | `good_rep` | Yes |
| 4 | `bad_rep` | Flagged |
| 5 | `false_rep` | No |
| 6 | `resting` | — |
| 7 | `half_rep` | Flagged |

**IDLE phase** → only classes 0,1,2 valid. **ENGAGED phase** → classes 2–7 valid.
Needs 10 consecutive `on_machine` frames to transition IDLE → ENGAGED.

---

## LSTM Model

```
Input:  (batch, 30, 51)   30 frames × 51 features (17 keypoints × xyz)
LSTM:   51 → 128 hidden
Dropout: 0.3
Linear: 128 → 64 → ReLU → 8
Output: softmax (one class wins)
```

- ONNX file deployed to Pi: `models/weights/activity_v1.onnx`
- Confidence < 0.65 → accept but log
- Confidence < 0.50 → flag clip to `data/review/` on GitHub

---

## AlphaFit Weight Plates (Project 2)

| Stripe | Weight |
|--------|--------|
| Red    | 25 kg  |
| Blue   | 20 kg  |
| Yellow | 15 kg  |
| Green  | 10 kg  |
| White  | 5 kg   |

Camera looks along barbell horn/sleeve (~45°). HSV colour classification.
Barbell bar = 20 kg added automatically.

---

## Key Config (`pi/config.py`)

- `ONNX_MODEL_PATH` — blank = rule-based counting (safe default)
- `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` — set per deployment
- `GOOGLE_DRIVE_FOLDER_ID` — `1KNDC4wctZqVt8s41U4ALWHJ45OM5U9FA`
- `TABLET_WS_PORT` — 8788 (WebSocket to tablet.html)
- `POWER_AUTOMATE_URL` — blank = console-only mode
- `GITHUB_REVIEW_TOKEN` — PAT for clip upload to data/review/

---

## Git Push

Push to both repos:
```bash
# Primary (private, production)
git push https://Matt-xlfitness:<PAT>@github.com/Matt-xlfitness/Gym-Overseer-AI.git claude/gym-tracking-neural-network-sHldz

# Public (GitHub Pages only)
git push https://Matt-xlfitness:<PAT>@github.com/XLDonkey/gym-ai-system.git claude/gym-tracking-neural-network-sHldz
```

---

## Key Decisions Made

- **Primary repo** is `Matt-xlfitness/Gym-Overseer-AI` (private) — not xldonkey
- **MVP v1**: tablet interface → weight detection → person tracking (bounding box). No face ID.
- **Supabase** (not Power Apps) for database — Postgres + Realtime, scales to multiple gyms
- **Power Automate** for set reporting (interim) — replace with Next.js `/api/set` endpoint later
- **ONNX** for Pi deployment (not PyTorch) — faster inference, no torch on Pi
- **Rule-based** angle counting now, LSTM later (training data being collected)
- **IoU tracker** (not DeepSORT) — lightweight, no GPU needed on Pi
- **buffalo_sc** ArcFace model (not buffalo_l) — faster, good enough on Pi [post-MVP]

---

## Obsidian Knowledge Graph

Full interlinked notes in `obsidian/` folder.
Open `Gym-Overseer-AI/obsidian/` as a vault in Obsidian for graph view.
Update by telling Claude what changed — both CLAUDE.md and Obsidian notes updated together.
