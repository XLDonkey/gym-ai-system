---
title: Project 4 — E-Weight
tags: [project, e-weight, motor, phase-2, pending]
status: phase-2-hardware-pending
created: 2026-04-09
updated: 2026-04-10
---

# Project 4 — E-Weight (Electric Weight Stacks)

> Custom brushless motor stacks on cable machines — weight read digitally from motor API. 100% accurate.

Part of [[Home]]. Replaces camera-based weight detection for **pin-loaded cable machines**. Hardware not yet built — code is ready.

---

## The Problem It Solves

Pin-loaded cable machines (lat pulldowns, cable rows, tricep pushdowns) have traditional iron weight stacks. Camera-based detection is hard:
- Stack is side-on, pins obscure the weight number
- Lighting inside the frame varies
- Only partial plate visibility

Current solution: [[System/Architecture|optical flow]] detects that the stack *moved* but doesn't know *how much*.

E-Weight solution: **replace the stack with a motor-controlled equivalent** — weight is a software value. Read it directly. 100% accurate, 1 kg increments.

---

## How It Works

```
Custom brushless motor weight stack
  → Motor controller board (WiFi + HTTP server)
  → GET /api/weight → {"weight_kg": 42.5, "locked": true}
  → Pi reads at session start
  → Written to Supabase sets table
```

Zero camera detection needed for weight on cable machines.

---

## API Spec (Motor Controller)

```http
GET  /api/weight
→ {"weight_kg": 42.5, "locked": true, "motor_temp": 38}

POST /api/weight
body: {"weight_kg": 50.0}
→ {"ok": true, "weight_kg": 50.0}

GET  /api/status
→ {"locked": true, "motor_temp": 38, "firmware": "1.2.0"}
```

---

## StackClient (`e_weight/stack_client.py`)

```python
client = StackClient(
    stack_ip="192.168.1.200",
    port=80,
    timeout_s=2.0,
    enabled=False          # ← False until hardware ships
)

weight = client.get_weight()    # Returns None (disabled)
ok     = client.set_weight(50)  # Returns False (disabled)
locked = client.is_locked()     # Returns False (disabled)
```

To enable: set `enabled=True` and provide `stack_ip`.

---

## Weight Detection Comparison

| Machine type | Weight method | Accuracy |
|---|---|---|
| Free weight barbells | [[Projects/Weight ID]] — plate colours + YOLO | ~95–99% |
| Cable / pin machines (now) | Optical flow — confirms *movement*, not weight | N/A |
| Cable / pin machines (Phase 2) | E-Weight motor API | 100%, 1 kg increments |

---

## Hardware (Phase 2)

- Custom brushless motor weight stack (design TBD)
- Motor controller board with WiFi module + HTTP server
- Replaces traditional pin-loaded iron stacks per machine
- Motor manages weight stack position → software-defined weight

**No hardware designed or ordered yet.**

---

## Config (`pi/config.py`)

```python
# Phase 2 — disabled until hardware ships
E_WEIGHT_ENABLED = False
STACK_IP         = ""      # e.g. "192.168.1.200"
```

---

## Status

- [x] `StackClient` code written
- [x] HTTP API spec defined
- [x] Integrated into `pi/main.py` (disabled by default)
- [ ] Hardware designed
- [ ] Hardware prototyped
- [ ] Hardware tested on cable machine
- [ ] Production deployment

---

## Related

- [[Projects/Weight ID]] — alternative for free weight barbells
- [[System/Database Schema]] — `weight_kg` written to sets table regardless of source
- [[System/WebSocket Layer]] — weight broadcast to tablet
- [[Hardware/Machine Pi]] — Pi calls the motor controller API
- [[Hardware/Costs]] — no extra cameras needed (digital readout saves £70/station)
