---
title: Machine Pi (Hardware)
tags: [hardware, pi, hailo, camera]
created: 2026-04-09
updated: 2026-04-10
---

# Machine Pi

One Raspberry Pi 5 per gym machine. Runs the full inference stack locally — no cloud needed during a session.

---

## Bill of Materials (Per Cable/Pin Machine)

| Part | Purpose | Cost |
|------|---------|------|
| Raspberry Pi 5 (4GB RAM) | Edge inference | £80 |
| Hailo-8 AI HAT+ (26 TOPS) | NPU — YOLO at 30fps | £70 |
| Pi Camera Module 3 Wide | Side-on machine view | £35 |
| PoE+ HAT | Single cable: power + network | £25 |
| Mount + housing | Bracket | £20 |
| **Total per machine** | | **£230** |

---

## Per Barbell Station (Add-on)

| Part | Purpose | Cost |
|------|---------|------|
| Pi Camera Module 3 × 2 | Weight plate detection (left + right sleeve) | £70 |
| **Total add-on** | | **£70** |

Barbell station total: **£300**

---

## Entry Camera Pi (Door)

| Part | Cost |
|------|------|
| Raspberry Pi 5 (4GB) | £80 |
| Pi Camera Module 3 | £35 |
| PoE+ HAT | £25 |
| **Total** | **£140** |

Runs `user_tracking/entry_camera.py` — ArcFace face recognition at the gym door.

---

## Mac Mini (Training Server — not at gym)

| Part | Cost |
|------|------|
| Mac Mini M4 (16GB) | ~£700 |

Runs PyTorch training on Apple Silicon MPS backend. Stays at office/home.

---

## Software Stack on Pi

```
Python 3.11
ultralytics >= 8.0.0     (YOLOv11)
onnxruntime >= 1.16.0    (LSTM inference)
opencv-python >= 4.8.0   (optical flow, video recording)
numpy >= 1.24.0
requests >= 2.31.0       (Supabase, GitHub, Power Automate)
insightface >= 0.7.3     (ArcFace — optional, entry Pi only)
websockets               (asyncio WebSocket server)
```

Install: `pip install -r pi/requirements.txt`

---

## Processes Running on Each Machine Pi

| Process | What |
|---------|------|
| `python pi/main.py` | Main inference loop (systemd service: `xlf-overseer`) |
| `MachineWSServer` (thread) | WebSocket broadcaster, port 8788 |
| `SessionRecorder` (thread) | Video write + Google Drive upload |
| `ClipReporter` | Uploads uncertain ONNX clips to GitHub |
| `GymTracker` | IoU person tracking |
| *(Entry Pi only)* `EntryCamera` (thread) | ArcFace face recognition |

---

## Network

- All Pis on same local network (WiFi or PoE switch)
- Pi IP addresses reserved in router (static DHCP by MAC address)
- No internet needed for inference — only for:
  - Supabase set/session logging (HTTP)
  - GitHub clip upload (HTTPS)
  - Google Drive video upload (rclone)
- PoE+ HAT: one cable per machine = power + Ethernet

---

## Crontab (Pi)

```bash
0 2 * * 1-5  python /home/pi/xlf/pi/uploader.py
# 2am Mon–Fri: upload any missed video chunks to Google Drive

0 3 * * *    cd /home/pi/xlf && git pull origin main
# 3am daily: pull latest ONNX model (if make deploy ran on Mac Mini)
```

---

## Related

- [[System/YOLO Pipeline]] — main workload on Hailo NPU
- [[System/LSTM Model]] — ONNX inference on Pi CPU (~5ms)
- [[System/WebSocket Layer]] — ws_server.py runs here
- [[System/Session Recorder]] — records video, uploads to Google Drive
- [[System/Review Loop]] — Pi flags clips; Mac Mini retrains; Pi re-deploys
- [[System/Engagement Detector]] — runs in main loop on Pi
- [[Hardware/Camera Placement]] — where to mount cameras
- [[Hardware/Costs]] — full cost breakdown
- [[Projects/Rep Tracking]] — primary project on Pi
- [[Projects/Weight ID]] — also runs on Pi (barbell stations)
- [[Projects/User Tracking]] — entry Pi + machine Pi
- [[Projects/E-Weight]] — Pi calls motor API (Phase 2)
