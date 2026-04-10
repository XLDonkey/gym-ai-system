---
title: Project 4 — E-Weight
tags: [project, e-weight, motor, phase-2, pending]
status: phase-2-hardware-pending
created: 2026-04-09
---

# Project 4 — E-Weight (Electric Weight Stacks)

> Custom motor stacks on cable machines — weight read digitally from motor API. 100% accurate.

Part of [[Home]]. This replaces camera-based weight detection for **pin-loaded cable machines**. Hardware not yet built.

---

## The Concept

Traditional cable machines have iron pin-loaded weight stacks — hard to see, lighting-dependent, camera angle awkward.

The E-Weight replaces these with **brushless motor stacks**:
- Motor controller manages exact weight in software
- Exposes a local HTTP API: `GET /api/weight → {"weight_kg": 42.5}`
- Pi calls the API at session start — weight is always exact, 1 kg increments
- **Zero camera detection needed for weight** — digital readout

This is **100% accurate** vs ~95% for camera-based plate detection.

---

## API Spec

```
GET  /api/weight          → {"weight_kg": 42.5}
POST /api/weight          → {"weight_kg": 50.0}   (set weight remotely)
GET  /api/status          → {"locked": true, "motor_temp": 38}
```

---

## Code Status

Code is fully written and ready. Disabled until hardware ships.

```python
# e_weight/stack_client.py
client = StackClient(stack_ip="192.168.1.200", enabled=False)  # enabled=False for now
weight = client.get_weight()   # returns None until hardware connected
```

To enable: set `enabled=True` and provide the motor controller's IP.

---

## Key Files

| File | Purpose |
|------|---------|
| `e_weight/stack_client.py` | HTTP API client |
| `pi/config.py` | `E_WEIGHT_ENABLED`, `STACK_IP` |

---

## Relationship to Project 2

| Machine type | Weight method |
|---|---|
| Free weight barbells | [[Projects/Weight ID]] — camera + plate colours |
| Cable / pin machines | E-Weight — motor API (Phase 2) |
| Cable / pin machines (interim) | Weight stack optical flow tracker (in `pi/`) |

---

## Hardware (Phase 2)

- Custom brushless motor weight stack (design TBD)
- Motor controller board with WiFi + HTTP server
- Replaces traditional iron stacks per machine

---

## Status

- [x] Code written (`e_weight/stack_client.py`)
- [x] API spec defined
- [ ] Hardware designed
- [ ] Hardware built
- [ ] Tested on cable machine

---

## Related

- [[Projects/Weight ID]] — alternative for free weight barbells
- [[System/Database Schema]] — weight_kg written to sets table
- [[System/WebSocket Layer]] — weight broadcast to tablet
- [[Hardware/Machine Pi]] — Pi calls the motor API
- [[Hardware/Costs]] — no extra cameras needed (digital readout)
