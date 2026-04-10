---
title: WebSocket Layer
tags: [system, display, websocket, tablet, staff]
created: 2026-04-09
updated: 2026-04-10
---

# WebSocket Layer — Live Display

The Pi broadcasts live machine state to the tablet and staff view at 10fps via WebSocket.

---

## Architecture

```
Pi (MachineWSServer, port 8788)
    │
    ├──→ ws://[pi-ip]:8788  →  display/tablet.html  (member kiosk)
    └──→ ws://[pi-ip]:8788  →  display/staff.html   (staff floor view)
```

Broadcast rate: **10 Hz** (every 100ms).
All clients auto-reconnect if Pi restarts (2s delay for tablet, 3s for staff).

---

## Full Broadcast Payload

```json
{
  "machine_id":    "xlf-pi-001",
  "machine_name":  "Nautilus Lat Pulldown",
  "member_name":   "Matthew",
  "member_id":     "uuid-...",
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

`activity_id` maps to the [[System/Activity Classes|8-class schema]].

---

## MachineWSServer (`display/ws_server.py`)

```python
server = MachineWSServer(
    machine_id="xlf-pi-001",
    machine_name="Nautilus Lat Pulldown",
    port=8788,
    enabled=True
)
server.start()   # daemon thread with asyncio event loop

# Called every frame from pi/main.py:
server.update_state(
    member_name="Matthew",
    member_id="uuid-...",
    activity="good_rep",
    activity_id=3,
    phase="ENGAGED",
    rep_count=8,
    weight_kg=52.5,
    form_score=0.78,
    session_reps=24,
    session_sets=3,
    confidence=0.87
)
```

---

## tablet.html (Member Kiosk)

- Opens in **Kiosk mode** (full-screen browser, no UI chrome)
- Auto-detects Pi IP from `window.location.hostname`
- Connects to `ws://[pi-ip]:8788`

**What's shown:**
| Element | Source |
|---------|--------|
| Machine name (top bar) | `machine_name` |
| Member avatar + name | `member_name[0]` initial, `member_name` |
| Weight badge | `weight_kg` |
| Large rep counter | `rep_count` (font: 100–180px, clamp) |
| Rep counter colour | [[System/Activity Classes|Activity colour]] |
| Rep flash animation | Triggers on `rep_count` increment |
| Form pills (Good/Partial/Bad) | Incremented client-side on each rep |
| Activity label | `activity` string |
| Confidence bar | `confidence` 0–1 |
| Sets / Total Reps / Volume | `session_sets`, `session_reps`, `weight_kg × session_reps` |

**Idle overlay:** shown when `member_name` is null and `phase ≠ ENGAGED`. Displays XL Fitness branding and "Sit down to start your session."

---

## staff.html (Staff Floor View)

- Open on **any browser** on the local network
- Staff enters each Pi's IP + machine name once → persisted to `localStorage`
- Connects to each Pi's WebSocket directly (no server in the middle)

**Machine card states:**
| CSS class | Condition | Border |
|-----------|-----------|--------|
| `.active` | `activity_id ∈ {2,3,4,5,7}` | Green glow |
| `.resting` | `activity_id === 6` | Purple glow |
| `.idle` | `activity_id ∈ {0,1}` | 55% opacity |
| `.alert` | `activity_id === 4` (bad_rep) | Red glow |

**Header bar:** total members in gym / active sets / resting / idle machines.

---

## Set Reporter (`display/set_reporter.py`)

Fires when a set is detected as complete (member stands up or long rest):

```python
reporter = SetReporter(
    power_automate_url="https://prod-xx.logic.azure.com/...",  # or SERVER_URL
    machine_id="xlf-pi-001",
    machine_name="Nautilus Lat Pulldown",
    model_version="v1.0-lstm"
)

reporter.report_set(
    reps=10, form_good=7, form_partial=2, form_bad=1,
    weight_kg=52.5, member_id="M1089", member_name="Matthew"
)
```

Sends to Power Automate (current) or Next.js `/api/set` (planned). See [[Decisions/Display Layer]].

---

## Config (`pi/config.py`)

```python
TABLET_DISPLAY_ENABLED = True
TABLET_WS_PORT         = 8788
POWER_AUTOMATE_URL     = ""    # Power Automate HTTP trigger (interim)
SERVER_URL             = ""    # Custom web app endpoint (future)
```

---

## Related

- [[System/Activity Classes]] — `activity_id` drives all display colours
- [[System/Database Schema]] — set_reporter writes completed sets here
- [[Decisions/Display Layer]] — why WebSocket, why plain HTML, Power Apps vs Next.js
- [[Hardware/Machine Pi]] — server runs on Pi
