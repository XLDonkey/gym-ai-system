# XL Fitness AI Overseer — Donkey Handoff Document

**Date:** 26 March 2026  
**Written by:** Manus (AI agent working with Matthew)  
**For:** Donkey (Mac Mini AI agent)

---

## What We Are Building

**Overseer** is an AI system that watches gym equipment cameras and classifies what is happening at each machine in real time. The long-term goal is 300 Raspberry Pi cameras across multiple XL Fitness gyms, each running YOLO pose detection and a custom neural network called Overseer that identifies 7 activity classes per machine.

This document explains where we are today, what has been built, what you need to build next, and how everything fits together.

---

## The 7 Activity Classes (Overseer's Labels)

| Key | Class | Description |
|-----|-------|-------------|
| 1 | No User | Machine empty, no one nearby |
| 2 | User Passing | Person walking past, not engaging |
| 3 | User Present | Person seated/standing at machine, not yet repping |
| 4 | Good Rep | Controlled, full range of motion rep |
| 5 | Poor Rep | Uncontrolled, bounced, or partial rep |
| 6 | False Rep / Other | Adjustment, fidget, non-rep movement |
| 7 | Resting | Between sets — seated idle at machine |

---

## Current Hardware

| Device | Role | Status |
|--------|------|--------|
| Raspberry Pi 5 (8GB) | Camera node, YOLO inference, recording | **Running in ceiling above Nautilus Lat Pulldown** |
| Mac Mini | Local server, NN training, storage | **Your machine — needs server software** |
| Nautilus Xpload Lat Pulldown | First machine being trained on | MVP target |

---

## The Full Pipeline (How Data Flows)

```
Pi Camera (ceiling-mounted)
    │
    │  Runs YOLO pose detection on every frame
    │  Records 10-minute video chunks when person is present
    │  Uploads completed chunks to Google Drive automatically
    ▼
Google Drive Folder (shared)
    │  Folder ID: 1KNDC4wctZqVt8s41U4ALWHJ45OM5U9FA
    │  Path: /nautilus-lat-pulldown/
    │  Contains: lat_pulldown_01_YYYYMMDD_HHMMSS.mp4
    ▼
Annotation Tool (web app, browser-based)
    │  URL: https://xlfleetdash-avg43yta.manus.space/annotate
    │  Matthew downloads video from Drive, drops it in the tool
    │  Presses keys 1–7 to stamp timestamps with activity classes
    │  Exports: filename_labels.json
    │  Uploads JSON back to Drive alongside the video
    ▼
Training Data (Google Drive)
    │  Each clip has: video.mp4 + video_labels.json
    │  You download both, extract YOLO keypoints per frame,
    │  align keypoints to labels, train Overseer NN
    ▼
Overseer NN (.onnx model)
    │  Trained on keypoint sequences (not raw pixels)
    │  Input: 30-frame window of pose keypoints
    │  Output: one of 7 activity classes
    ▼
Deployed back to Pi
    └── Pi runs Overseer in real time alongside YOLO
    └── Classifies activity at 30fps, zero extra cloud cost
```

---

## What Is Already Built

### Pi Side (`/pi/` directory in this repo)

| File | What it does |
|------|-------------|
| `main.py` | Main inference loop — YOLO pose on every frame, drives recorder |
| `session_recorder.py` | Records 10-min video chunks, uploads to Google Drive via rclone when chunk closes |
| `config.py` | All settings — machine ID, camera, Google Drive folder ID, thresholds |
| `snapshot_server.py` | **Lightweight HTTP server on port 8090** — serves `/status`, `/snapshot`, `/health` |
| `setup.sh` | One-command Pi setup script |
| `setup_gdrive.sh` | Configures rclone for Google Drive authentication |

### Pi is already doing:
- Recording footage of the Nautilus Lat Pulldown machine
- Auto-uploading completed chunks to Google Drive (rclone, every 15 minutes via cron)
- Running `snapshot_server.py` on port 8090 (auto-starts on boot via crontab)
- 3 recordings already in Google Drive ready to annotate

### rclone config on Pi:
- Remote name: `xlf-gdrive`
- Target: `xlf-gdrive:nautilus-lat-pulldown/` with `--drive-root-folder-id 1KNDC4wctZqVt8s41U4ALWHJ45OM5U9FA`
- Cron: `*/15 * * * *` runs `xlf_auto_upload.sh`

### Dashboard (`xl-fleet-dashboard` project)
- Published at: `https://xlfleetdash-avg43yta.manus.space`
- Built with React 19 + Tailwind 4 + shadcn/ui
- Shows Pi node cards with Update + Snapshot buttons
- **Current limitation:** Update/Snapshot buttons try to reach the Pi directly from the browser, which fails due to browser Private Network Access restrictions when the dashboard is accessed from a public URL
- **Fix needed:** Route requests through the Mac Mini server (see Your Task below)

---

## The Pi's Status Server (port 8090)

The Pi runs `snapshot_server.py` which exposes:

```
GET http://192.168.1.40:8090/status
→ {
    "online": true,
    "version": "1.1.0",
    "uptime_s": 3600,
    "recorder_running": true,
    "recording_count": 3,
    "cpu_temp_c": 52.1,
    "timestamp": 1711234567
  }

GET http://192.168.1.40:8090/snapshot
→ JPEG image (live frame from camera, 1280×720)

GET http://192.168.1.40:8090/health
→ {"status": "ok", "port": 8090}
```

**The Pi never initiates outbound connections for status.** It only responds when asked. This keeps the Pi's CPU free for YOLO inference.

---

## Your Task — Mac Mini Server

Build a lightweight proxy server that runs on the Mac Mini 24/7 and acts as the bridge between the dashboard and the Pi nodes.

### Why this is needed

The dashboard is served from a public URL (`manus.space`). Browsers block public websites from making requests to private IP addresses (`192.168.x.x`) — this is called the Private Network Access restriction. The Mac Mini is on the same local network as the Pis, so it has no such restriction.

### Architecture

```
Browser (Matthew's MacBook, anywhere on gym WiFi)
    │  GET /api/nodes
    │  POST /api/refresh
    │  GET /api/snapshot/xlfitnesspidonkey
    ▼
Mac Mini Server (http://mac-mini.local:3001)
    │  Fetches from Pi on local network (no browser restrictions)
    │  http://192.168.1.40:8090/status
    │  http://192.168.1.40:8090/snapshot
    ▼
Pi (192.168.1.40:8090)
```

### Endpoints to build

```
GET  /api/nodes
     Returns cached status for all known Pi nodes
     Response: [{ id, name, location, ip, online, version, uptime_s,
                  recorder_running, recording_count, cpu_temp_c, last_fetched }]

POST /api/refresh
     Fetches /status from every Pi in nodes.json, updates cache, returns fresh data
     Same response shape as GET /api/nodes

GET  /api/snapshot/:id
     Proxies GET /snapshot from the matching Pi, streams JPEG back to browser
     Sets Content-Type: image/jpeg

GET  /api/health
     Returns { "status": "ok", "nodes": N, "uptime_s": N }
```

### Node config file (`server/nodes.json`)

```json
[
  {
    "id": "xlfitnesspidonkey",
    "name": "XLFitnessPIDonkey",
    "location": "Nautilus Lat Pulldown",
    "ip": "192.168.1.40",
    "port": 8090
  }
]
```

Adding a new Pi in the future = add one entry to this file and restart the server.

### Auto-start on Mac boot (LaunchAgent)

The server must start automatically when the Mac Mini boots (it has no screen, no one to start it manually). Use a macOS LaunchAgent plist at:
```
~/Library/LaunchAgents/com.xlfitness.overseer-server.plist
```

### Tech stack recommendation

Node.js with Express — lightweight, fast, easy to maintain. Alternatively Python FastAPI if you prefer. Either works.

### Where to put it

New directory in this repo: `server/`

```
server/
  index.js (or main.py)
  nodes.json
  package.json (or requirements.txt)
  setup.sh          ← one-command install + LaunchAgent setup
  com.xlfitness.overseer-server.plist  ← LaunchAgent template
  README.md
```

---

## Dashboard Update Needed (after server is built)

Once the Mac Mini server is running, the dashboard needs one small update — change the Pi card buttons to call the Mac Mini instead of the Pi directly.

**Current (broken from public URL):**
```
Update button  → fetch(`http://192.168.1.40:8090/status`)
Snapshot button → fetch(`http://192.168.1.40:8090/snapshot`)
```

**Target (works from anywhere on gym WiFi):**
```
Update button  → fetch(`http://mac-mini.local:3001/api/refresh`)
Snapshot button → fetch(`http://mac-mini.local:3001/api/snapshot/xlfitnesspidonkey`)
```

The dashboard URL (`mac-mini.local:3001`) should be configurable — store it in a `VITE_SERVER_URL` environment variable so it can be changed without rebuilding.

Manus (the agent working with Matthew) will make this dashboard change once you confirm the server is running and tested.

---

## Training Pipeline (Your Other Job)

Once Matthew has annotated some footage, you will receive:
- `lat_pulldown_01_YYYYMMDD_HHMMSS.mp4` — raw video
- `lat_pulldown_01_YYYYMMDD_HHMMSS_labels.json` — timestamp annotations

### Labels JSON format

```json
{
  "machine": "Nautilus Xpload Lat Pulldown",
  "machine_id": "nautilus-lat-pulldown",
  "pi_id": "xlf-pi-001",
  "video_file": "lat_pulldown_01_20260325_165320.mp4",
  "duration_s": 600,
  "labels": [
    { "time": 12.4, "class": 3, "label": "User Present" },
    { "time": 18.1, "class": 4, "label": "Good Rep" },
    { "time": 19.8, "class": 4, "label": "Good Rep" },
    { "time": 21.2, "class": 5, "label": "Poor Rep" },
    { "time": 22.0, "class": 7, "label": "Resting" }
  ]
}
```

### Training approach

1. Run YOLO pose on the video to extract keypoints for every frame
2. For each labelled timestamp, extract a 30-frame window of keypoints centred on that timestamp
3. Each window = one training sample with its class label
4. Train a lightweight LSTM or Transformer on the keypoint sequences
5. Export to ONNX format for deployment on the Pi

**Target:** 500 labelled reps for the first evaluation checkpoint. Expected accuracy at 500 reps: ~85–90% on Good Rep vs Poor Rep classification.

**Model name:** Overseer  
**Input:** 30-frame window × 17 keypoints × 3 values (x, y, confidence) = 30×51 tensor  
**Output:** softmax over 7 classes  

---

## Key Info Summary

| Item | Value |
|------|-------|
| Pi hostname | XLFitnessPIDonkey |
| Pi local IP | 192.168.1.40 |
| Pi status server port | 8090 |
| Pi username | xlraspberry2026 |
| Pi Connect login | XLfitnessdonkey@gmail.com |
| Google Drive folder | https://drive.google.com/drive/folders/1KNDC4wctZqVt8s41U4ALWHJ45OM5U9FA |
| Dashboard URL | https://xlfleetdash-avg43yta.manus.space |
| Annotation tool | https://xlfleetdash-avg43yta.manus.space/annotate |
| GitHub repo | https://github.com/XLDonkey/gym-ai-system |
| Mac Mini server target port | 3001 |
| Mac Mini target hostname | mac-mini.local |

---

## What Matthew Is Doing Right Now

1. Waiting for the 3 existing recordings to finish processing in Google Drive
2. Will then open the annotation tool and label the footage
3. Will share labelled data with you for training

## What You Need to Do

1. **Build the Mac Mini server** (described above) — this is the priority
2. **Write the training pipeline** — keypoint extraction + LSTM training script
3. **Test the server** — confirm `http://mac-mini.local:3001/api/nodes` returns Pi status
4. **Tell Manus** when the server is running so the dashboard can be updated to point at it

---

*Questions? Matthew can relay them to Manus.*
