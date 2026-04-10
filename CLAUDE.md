# XL Fitness AI Overseer — Claude Context

This file is read automatically by Claude Code at the start of every session.
It contains the full project context so Claude never loses track.

---

## What This Project Is

A self-improving AI gym tracking system for **XL Fitness**.
One camera per machine. Every rep counted, every weight logged, every member tracked — automatically.
No phones. No QR codes. No staff input.

**Repo:** `xldonkey/gym-ai-system` (public — hosts GitHub Pages)
**Private repo:** `Matt-xlfitness/Gym-Overseer-AI` (production, private)
**Bible (live):** https://xldonkey.github.io/gym-ai-system/bible.html
**Branch for all development:** `claude/gym-tracking-neural-network-sHldz`

---

## The Four Projects

| # | Name | Status | Key file |
|---|------|--------|----------|
| 1 | **Rep Tracking** | Live (rule-based). LSTM needs training data | `pi/activity_state_machine.py` |
| 2 | **Weight ID** | Built, needs training images | `weight_id/detector.py` |
| 3 | **User Tracking** | Built, needs member enrolment | `user_tracking/gym_tracker.py` |
| 4 | **E-Weight** | Phase 2 — hardware pending | `e_weight/stack_client.py` |

---

## System Architecture

```
ENTRY PI (door camera)
  └─ InsightFace ArcFace → member identity → PersonDB

MACHINE PI (one per machine)
  ├─ YOLO Pose          → 17 keypoints / frame
  ├─ GymTracker         → track_id → member from PersonDB
  ├─ ActivityStateMachine → IDLE / ENGAGED phase gate
  ├─ LSTM Classifier    → 8-class activity (30-frame window)
  ├─ WeightDetector     → AlphaFit plate colour → kg
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

Camera looks along barbell sleeve (~45°). HSV colour classification.
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

## Git Authentication

The session proxy authenticates as `Matt-xlfitness`. Push to XLDonkey repo using:
```bash
git push https://Matt-xlfitness:<PAT>@github.com/XLDonkey/gym-ai-system.git claude/gym-tracking-neural-network-sHldz
git fetch https://Matt-xlfitness:<PAT>@github.com/XLDonkey/gym-ai-system.git claude/gym-tracking-neural-network-sHldz:refs/remotes/origin/claude/gym-tracking-neural-network-sHldz
```

---

## Current Blockers (Data Collection)

- [ ] Collect 300+ annotated rep segments → `make train`
- [ ] Collect 50+ weight plate photos per colour → `make train-weight`
- [ ] Enrol all members → `make enrol NAME="..."`
- [ ] Set Supabase credentials in `pi/config.py`
- [ ] Deploy models to Pi → `make deploy PI=pi@IP`

---

## Key Decisions Made

- **Supabase** (not Power Apps) for database — Postgres + Realtime, scales to multiple gyms
- **Power Automate** for set reporting (interim) — replace with Next.js `/api/set` endpoint later
- **ONNX** for Pi deployment (not PyTorch) — faster inference, no torch on Pi
- **Rule-based** angle counting now, LSTM later (needs training data first)
- **IoU tracker** (not DeepSORT) — lightweight, no GPU needed on Pi
- **buffalo_sc** ArcFace model (not buffalo_l) — faster, good enough accuracy on Pi
- **xldonkey/gym-ai-system** (public) hosts Bible/GitHub Pages; private repo stays private

---

## Obsidian Knowledge Graph

Full interlinked notes in `obsidian/` folder. Open in Obsidian → File → Open Vault → select repo folder.
The graph view shows connections between all concepts.
