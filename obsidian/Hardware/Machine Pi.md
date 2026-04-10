---
title: Machine Pi (Hardware)
tags: [hardware, pi, hailo, camera]
created: 2026-04-09
---

# Machine Pi

One Raspberry Pi 5 per gym machine. Runs the full inference stack locally — no cloud needed during a session.

---

## Bill of Materials (Per Machine)

| Part | Purpose | Cost |
|------|---------|------|
| Raspberry Pi 5 (4GB) | Edge inference | £80 |
| Hailo-8 AI HAT+ (26 TOPS) | NPU — YOLO at 30fps | £70 |
| Pi Camera Module 3 Wide | Side-on machine view | £35 |
| PoE+ HAT | Single cable: power + network | £25 |
| Mount + housing | Bracket | £20 |
| **Total per machine** | | **£230** |

**Additional for barbell stations:** 2× Pi Camera Module 3 (£35 each) for [[Projects/Weight ID]].

---

## Entry Camera Pi

One additional Pi at the gym door for [[Projects/User Tracking]]:
- 1× Raspberry Pi 5 (4GB)
- 1× Pi Camera Module 3
- Total: ~£115

---

## Mac Mini (Training Server)

| Part | Cost |
|------|------|
| Mac Mini M4 (16GB) | ~£700 |

Runs PyTorch training on Apple Silicon MPS backend.
Not deployed at the gym — stays in office/home.

---

## Network

- All Pis on same local WiFi / PoE switch
- Pi IP addresses reserved in router (static DHCP)
- No internet required for inference — only for Supabase uploads and GitHub clip review
- PoE+ HAT means one cable per machine (power + data)

---

## Software Stack on Pi

```
Python 3.11
ultralytics (YOLO)
onnxruntime
opencv-python-headless
insightface (buffalo_sc)
websockets (asyncio)
requests
numpy
```

Install: `pip install -r pi/requirements.txt`

---

## Related

- [[System/YOLO Pipeline]] — main workload on Pi
- [[System/LSTM Model]] — ONNX inference on Pi
- [[Hardware/Camera Placement]] — where to mount
- [[Hardware/Costs]] — full cost breakdown
