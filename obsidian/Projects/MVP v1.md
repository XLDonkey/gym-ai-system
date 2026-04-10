---
title: MVP v1 — Build Plan
tags: [project, mvp, active, priority]
status: in-progress
created: 2026-04-10
---

# MVP v1 — Build Plan

> Get something running in the gym as fast as possible. Prove the core loop works.

Part of [[Home]].

---

## MVP Scope

### In
- **Tablet interface** — member-facing kiosk display (`display/tablet.html`)
- **Weight detection** — camera at barbell horns → plate colours → kg
- **Person tracking** — bounding box IoU tracker (person present / not present)
- **Rep counting** — rule-based angle counting (no LSTM needed)

### Out (post-MVP)
- Face detection / member identification — too complex for MVP
- LSTM 8-class classifier — training data still being collected
- Supabase session logging — can stub/console-log for now
- E-Weight motor hardware — Phase 2

---

## Build Order

```
1. Tablet Interface        ← NOW
   display/tablet.html — the member-facing kiosk display

2. Weight Detection Pi     ← NEXT
   Camera at barbell horns/sleeves
   HSV colour scan → plate weights → total kg
   No training needed (colour scan works immediately)

3. Person Tracking         ← THEN
   YOLO bounding box
   IoU tracker — person present / at machine / not present
   No face ID
```

---

## Tablet Interface — What It Shows

The tablet sits on or near each machine. Members see:

| Element | Source |
|---------|--------|
| Rep counter (huge) | Rule-based angle count |
| Current set weight | Weight detection Pi |
| Set counter | Increments on rest detection |
| Form indicator | Good / Partial / Bad (rule-based) |
| Person status | Bounding box confirmed = active |
| Idle screen | XL Fitness branding when nobody present |

Connects to Pi via WebSocket: `ws://[pi-ip]:8788`
Served from Pi — works entirely on local network, no internet needed.

---

## Weight Detection Setup

- Camera mounted at barbell station looking along the horn/sleeve (~45°)
- Two cameras per station (left + right horn) — one always has clear view
- Colour scan works immediately — no training images needed yet
- YOLO fine-tuning comes later for higher accuracy

See [[Projects/Weight ID]] for full detail.

---

## Person Tracking (MVP)

MVP version — no names, no faces:
- YOLO detects person bounding box
- IoU tracker assigns `track_id`
- `closest_track(machine_zone_roi)` → is someone at this machine?
- Binary: **person present** / **person absent**
- Tablet shows generic "Member" instead of a name

See [[Projects/User Tracking]] for full detail.

---

## Running in Parallel

While MVP is being built:
- Pi is recording sessions → building LSTM training dataset
- Annotate with `make annotate` whenever time allows
- Target: 300+ segments before LSTM training

---

## Related

- [[Projects/Rep Tracking]] — core of what MVP demonstrates
- [[Projects/Weight ID]] — weight horn camera, step 2
- [[Projects/User Tracking]] — bounding box only for MVP, step 3
- [[System/WebSocket Layer]] — tablet connects via this
- [[Hardware/Camera Placement]] — weight horn camera mounting
- [[Data/Training Requirements]] — LSTM data being collected in parallel
