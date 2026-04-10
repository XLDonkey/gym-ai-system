---
title: WebSocket Layer
tags: [system, display, websocket, tablet, staff]
created: 2026-04-09
---

# WebSocket Layer — Live Display

The Pi broadcasts live state to the tablet on the machine. The staff view connects to all Pis simultaneously. See also [[Decisions/Display Layer]].

---

## Architecture

```
Pi (ws_server.py)  →  ws://[pi-ip]:8788  →  tablet.html  (member view)
Pi (ws_server.py)  →  ws://[pi-ip]:8788  →  staff.html   (staff view, connects to all Pis)
```

Broadcast rate: **10fps** (every 100ms).
Tablet reconnects automatically if Pi restarts (2s delay).

---

## Broadcast Payload

```json
{
  "machine_id":    "xlf-pi-001",
  "machine_name":  "Nautilus Lat Pulldown",
  "member_name":   "Matthew",
  "member_id":     "M1089",
  "phase":         "ENGAGED",
  "activity":      "good_rep",
  "activity_id":   3,
  "confidence":    0.87,
  "rep_count":     8,
  "weight_kg":     52.5,
  "form_score":    0.78,
  "session_reps":  24,
  "session_sets":  3,
  "timestamp":     "2026-04-09T14:32:11.542Z"
}
```

---

## tablet.html

- Mounts on machine in **Kiosk mode** (full screen, no browser UI)
- Opens at `http://[pi-ip]/` (Pi serves the file)
- Auto-detects Pi IP from `window.location.hostname`
- Shows: large rep counter (100–180px font), member name, weight, form pills, confidence bar
- **Idle overlay** covers screen when no member seated (shows XL Fitness branding)
- Rep counter **flashes** on each new rep

---

## staff.html

- Open on any browser on the local network
- Staff enters each Pi's IP + machine name → persisted to `localStorage`
- One card per machine, colour-coded:
  - 🟢 Active (rep in progress)
  - 🟣 Resting (seated, between sets)
  - ⚫ Idle (no person)
  - 🔴 Alert (bad form)
- Header bar: total members / active / resting / idle

---

## Key Files

| File | Purpose |
|------|---------|
| `display/ws_server.py` | `MachineWSServer` — asyncio WebSocket broadcaster |
| `display/tablet.html` | Member-facing kiosk display |
| `display/staff.html` | Staff floor view, multi-machine |
| `display/set_reporter.py` | Fires HTTP POST on set complete |

---

## Set Complete Flow

When a set ends (`set_reporter.py`):

```python
SetReporter.report_set(
    reps=10, form_good=7, form_partial=2, form_bad=1,
    weight_kg=52.5, member_id="M1089", member_name="Matthew"
)
```

Sends to Power Automate (interim) or direct to Supabase (future).
See [[Decisions/Display Layer]].
