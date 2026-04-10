---
title: Hardware Costs
tags: [hardware, costs, budget]
created: 2026-04-09
---

# Hardware Costs

---

## Per Machine (Cable / Pin-Loaded)

| Part | Cost |
|------|------|
| Raspberry Pi 5 (4GB) | £80 |
| Hailo-8 AI HAT+ (26 TOPS) | £70 |
| Pi Camera Module 3 Wide | £35 |
| PoE+ HAT | £25 |
| Mount + housing | £20 |
| **Total** | **£230** |

---

## Per Barbell Station (Add-on to above)

| Part | Cost |
|------|------|
| 2× Pi Camera Module 3 (weight plate cameras) | £70 |
| **Total add-on** | **£70** |

Barbell station total: **£300**

---

## Entry Camera

| Part | Cost |
|------|------|
| Raspberry Pi 5 (4GB) | £80 |
| Pi Camera Module 3 | £35 |
| PoE+ HAT | £25 |
| **Total** | **£140** |

---

## Training Server (One-Off)

| Part | Cost |
|------|------|
| Mac Mini M4 (16GB) | ~£700 |

Stays in office. Trains all models, serves review portal.

---

## Example Gym (10 machines)

| Item | Qty | Unit | Total |
|------|-----|------|-------|
| Machine Pis (cable) | 7 | £230 | £1,610 |
| Machine Pis (barbell) | 3 | £300 | £900 |
| Entry camera Pi | 1 | £140 | £140 |
| Mac Mini M4 | 1 | £700 | £700 |
| **Total** | | | **£3,350** |

---

## Notes

- PoE+ HAT means one cable per machine (power + Ethernet) — clean install
- Pis are passive — no moving parts, expected 5+ year lifespan
- Hailo HAT is optional for testing (YOLO runs on CPU at ~5fps, fine for development)

---

## Related

- [[Hardware/Machine Pi]] — full hardware spec
- [[Hardware/Camera Placement]] — how to mount
