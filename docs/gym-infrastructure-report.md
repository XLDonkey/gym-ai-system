# XL Fitness — Physical Infrastructure Report
## End-Game Camera & Network Setup

**Prepared by:** Donkey 🫏  
**Date:** March 2026  
**Scope:** Full gym deployment — 120+ machines

---

## The System Overview

Every machine in the gym has **3 cameras** doing 3 separate jobs:

```
                    📷 Camera 1 — Ceiling fisheye
                    Person tracking + Face ID
                    
         📷 Camera 2              📷 Camera 3
         Side-on to machine       Weight stack
         Rep counting             Weight detection
         Form scoring
         
         [MACHINE]               [Weight Stack]
              👤
           Member
```

---

## Camera Specifications

### Camera 1 — Person ID / Occupancy (Ceiling)
**Job:** Face recognition, person tracking, who is at which machine

| Spec | Requirement |
|------|------------|
| Resolution | 4K (8MP) minimum |
| Frame rate | 5–15 fps (face ID doesn't need high fps) |
| Field of view | Wide — 90–120° to cover machine area |
| Type | Ceiling dome |
| Power | PoE (single cable) |

**Recommended:** Hikvision DS-2CD2147G2H-LI (4MP AcuSense)  
**Est. price:** ~$180–250 AUD each

---

### Camera 2 — Rep Counter / Form Scoring (Side-on)
**Job:** Elbow angle tracking, rep counting, form scoring

| Spec | Requirement |
|------|------------|
| Resolution | 1080p (2MP) — MoveNet doesn't need more |
| Frame rate | 30 fps minimum |
| Field of view | Narrow — 60–80°, aimed at user profile |
| Type | Fixed bullet or mini dome, wall/ceiling mount |
| Power | PoE |
| Placement | 2–3m from machine, chest–shoulder height, 45° angle |

**Recommended:** Hikvision DS-2CD2T23G2-2I (2MP AcuSense, 2.8mm)  
**Est. price:** ~$120–160 AUD each

**Critical note:** NOT straight overhead — needs side-on profile view of elbow. Mount on wall or angled ceiling bracket at 45°.

---

### Camera 3 — Weight Detection (Stack-facing)
**Job:** Reads weight plate colours/pin position to detect loaded weight

| Spec | Requirement |
|------|------------|
| Resolution | 1080p |
| Frame rate | 1–5 fps (weight doesn't move fast) |
| Field of view | Narrow, tight shot of weight stack |
| Type | Fixed mini camera, close range |
| Power | PoE |
| Placement | Directly facing the weight stack, 0.5–1m away |

**Recommended:** Hikvision DS-2CD2T23G2-4I (2MP, 4mm lens for tight framing)  
**Est. price:** ~$120–150 AUD each

---

## Network Infrastructure

### PoE Switches

Single cable to each camera carries both power and data. This is the cleanest way to run a gym camera network.

**Recommended:** Ubiquiti UniFi USW-Pro-24-PoE  
- 24 PoE ports, gigabit, managed  
- Powers up to 24 cameras from one switch  
- ~$1,200–1,500 AUD each  
- Need approximately 15 switches for 120 machines (3 cameras × 120 = 360 cameras ÷ 24 ports)

**Simpler/cheaper option for Phase 1:** Ubiquiti UniFi USW-Lite-16-PoE (~$350 AUD, 16 ports)

### Network Layout
```
Mac Mini M4 (Edge Server)
    ↓ (10Gbps fibre or CAT6A)
Core Switch (Ubiquiti)
    ↓ (Gigabit PoE runs)
Zone Switches (one per gym zone)
    ↓ (PoE to each camera)
120 Machines × 3 cameras = 360 cameras
```

### Cabling
- **CAT6A** throughout (supports PoE++ and future-proofs to 10Gbps)
- Maximum cable run: 100m per PoE segment
- Estimated cable: 3–5km for full gym (depends on layout)
- Run cables through ceiling void — clean, hidden

---

## Edge Server

**Recommended:** Apple Mac Mini M4 Pro  
- Runs all AI inference locally (no cloud dependency)
- Neural Engine: 38 TOPS — handles 120 camera streams
- 32–64GB RAM recommended
- 2TB SSD for local storage
- ~$3,000–4,500 AUD

**Why local vs cloud:**
- Zero latency (cloud adds 100–300ms round-trip)
- No ongoing cloud compute costs
- Works if internet goes down
- Privacy — member data never leaves the gym

---

## Full Cost Estimate — 120 Machines

### Cameras
| Item | Qty | Unit price | Total |
|------|-----|-----------|-------|
| Camera 1 (Face ID, 4K dome) | 120 | $220 | $26,400 |
| Camera 2 (Rep counter, 2MP) | 120 | $140 | $16,800 |
| Camera 3 (Weight detector, 2MP) | 120 | $135 | $16,200 |
| **Camera subtotal** | | | **$59,400** |

### Network
| Item | Qty | Unit price | Total |
|------|-----|-----------|-------|
| UniFi Pro 24 PoE Switch | 15 | $1,350 | $20,250 |
| UniFi Core Switch | 1 | $2,500 | $2,500 |
| CAT6A cabling + installation | — | estimate | $15,000 |
| **Network subtotal** | | | **$37,750** |

### Compute
| Item | Qty | Unit price | Total |
|------|-----|-----------|-------|
| Mac Mini M4 Pro (32GB) | 2 | $3,500 | $7,000 |
| NAS storage (camera archive) | 1 | $2,000 | $2,000 |
| **Compute subtotal** | | | **$9,000** |

### Installation
| Item | Est. cost |
|------|----------|
| Electrician (power runs) | $5,000 |
| IT/network installer | $8,000 |
| Camera mounting | $4,000 |
| **Installation subtotal** | **$17,000** |

---

## Total End-Game Investment

| Category | Cost |
|----------|------|
| Cameras | $59,400 |
| Network | $37,750 |
| Compute | $9,000 |
| Installation | $17,000 |
| **TOTAL** | **~$123,000 AUD** |

---

## Phased Approach (Recommended)

Don't spend $123K upfront. Build it in phases:

| Phase | Machines | Investment | What you prove |
|-------|----------|-----------|---------------|
| Phase 1 (now) | 1 | ~$500 | Rep counting works |
| Phase 2 | 5 | ~$8,000 | Multi-machine, full flow |
| Phase 3 | 20 | ~$25,000 | Scale, face ID, weight |
| Phase 4 | 120 | ~$90,000 | Full gym deployment |

Each phase pays for itself before the next begins.

---

## Phase 1 Hardware (Right Now)

Before spending anything on the full system, the prototype needs:

| Item | Option | Cost |
|------|--------|------|
| Camera | DJI Osmo Action 4 + power bank | ~$530 |
| OR existing tablet | Use tablet camera | $0 |
| Mount | GorillaPod + machine clamp | $50 |
| Power | USB-C power bank (26,800mAh) | $150 |

**Recommendation:** Start with the tablet you already have. Mount it side-on at shoulder height, 2m from the machine. Once rep counting is reliable, upgrade to a dedicated camera.

---

## Key Decisions Before Full Deployment

1. **Ceiling height** — determines camera angle and lens selection
2. **Existing network** — can gym WiFi infrastructure be reused?
3. **Electrician scope** — power runs for cameras + Mac Mini placement
4. **Privacy policy** — face ID requires member consent (standard in gym T&Cs)
5. **GymMaster API** — NFC card tap triggers session start

---

*This document will be updated as Phase 1 results inform Phase 2 decisions.*  
*Last updated: March 2026 — Donkey 🫏*
