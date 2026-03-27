# Overseer GUI — Web App Brief
### For a new Manus chat session

---

## What We Are Building

A single web application that serves as the display layer for the **XL Fitness Overseer AI system**. The app has three distinct views served from the same codebase:

1. **Member Tablet View** — shown on a tablet mounted to each gym machine, in kiosk mode
2. **Staff Floor View** — shown on a staff iPad or front desk monitor, all machines at a glance
3. **Admin / Management View** — system health, Pi status, model versions, alerts

This is **not** a data entry app. It is a **real-time display app**. Data flows in from Raspberry Pis via WebSocket and HTTP. The app displays it. No forms, no complex user flows.

---

## Context — The Broader System

The Overseer system consists of:
- **Raspberry Pi 5 + Hailo-8 AI HAT+** mounted above each gym machine
- The Pi runs three neural networks locally (pose/rep counting, weight detection, person ID)
- The Pi pushes live data via **WebSocket** to the tablet on that machine (every 100ms)
- The Pi sends a **HTTP POST** to a backend when a set is complete (for logging)
- A **Mac Mini ("Donkey Runs")** on the local network handles training and model updates
- The gym currently uses **Power Apps in Kiosk mode** for tablets — this web app will eventually replace it

The web app does **not** need to connect to the Pis directly during the MVP phase. It should be built with **mock/simulated data** that mirrors the real WebSocket payload format, so it can be developed and tested without the hardware being present.

---

## Tech Stack

- **React 19 + TypeScript** (already scaffolded — use the existing project)
- **Tailwind CSS 4 + shadcn/ui** (already installed)
- **Wouter** for routing
- **WebSocket** (native browser API) for live data
- No backend required for MVP — use simulated data with realistic timing

---

## Design Direction

**Feel:** Clean, clinical, high-contrast. Think gym performance monitor meets medical dashboard. Not a consumer app — a professional tool that staff and members can read at a glance from 1–2 metres away.

**Colour palette:**
- Background: very dark navy/charcoal (`#07070f` or similar)
- Accent: electric green (`#22c55e`) for active/good states
- Warning: amber (`#fbbf24`) for partial reps / attention needed
- Danger: red (`#ef4444`) for bad form / alerts
- Primary accent: purple (`#8B5CF6`) for XL Fitness brand

**Typography:**
- Large numbers must be **very large and bold** — rep counters should be readable from across the room
- Use a condensed/display font for numbers (e.g. Barlow Condensed, DIN, or similar)
- Body text: clean sans-serif (not Inter — try Geist, DM Sans, or IBM Plex Sans)

**Layout:**
- Member tablet view: fullscreen, portrait or landscape, single machine focus, massive rep counter
- Staff view: dense grid of machine cards, all visible without scrolling on a standard monitor
- No sidebars on the member tablet view — it is a kiosk

---

## Routes

| Route | View | Description |
|---|---|---|
| `/` | Staff Floor View | All machines grid, real-time status |
| `/machine/:id` | Member Tablet View | Single machine, live rep counter, kiosk mode |
| `/admin` | Admin View | Pi health, model versions, system alerts |

---

## Member Tablet View (`/machine/:id`)

This is the most important view. It is what members see while they train.

### States to Design

**State 1 — Machine Idle**
- Machine name (large, centred)
- Last session summary (who used it, how many reps, what weight)
- Ambient clock
- Subtle animated "waiting" indicator

**State 2 — Member Seated, Not Yet Lifting**
- Member name + avatar/photo placeholder
- Weight detected on stack (large number, e.g. "52.5 kg")
- Exercise name detected (e.g. "Lat Pulldown")
- Green "Ready" pill indicator
- Set counter (e.g. "Set 3 of ?")

**State 3 — Active Set In Progress**
- **MASSIVE rep counter** — this is the hero element, should be ~200–300px font size
- Live form indicator below the counter: coloured pill — "GOOD" (green) / "PARTIAL" (amber) / "BAD" (red)
- Weight confirmed (smaller, top corner)
- Member name (smaller, top corner)
- Subtle pulsing animation on the rep counter when a rep is counted

**State 4 — Rest Between Sets**
- Set summary: "Set 2 complete — 10 reps"
- Form breakdown: small coloured bar showing ratio of good/partial/bad reps
- Rest timer counting up (or down if a target rest time is set)
- Total volume so far (weight × reps, cumulative)
- "Start next set" is automatic — no button needed, it detects movement

**State 5 — Session Complete**
- Full session summary
- All sets listed (set number, reps, weight, form score)
- Total volume, best set, form score percentage
- "Great session!" or similar positive message
- Auto-returns to idle after 30 seconds

**State 6 — Identity Confirmation Required**
- "Please tap your card" prompt
- Shows photo of who the system thinks they are
- Subtle pulsing border

### Mock Data for Development
Simulate a realistic session: idle → member detected → 3 sets of 10 reps with realistic timing → rest periods → session complete. Loop this simulation so the view can be demonstrated without hardware.

---

## Staff Floor View (`/`)

A dense grid showing all machines simultaneously.

### Machine Card
Each machine gets a card showing:
- Machine name (e.g. "Lat Pulldown 1")
- Status indicator dot: Green (active set), Amber (resting), Grey (idle), Red (alert)
- Current member name + small avatar (or "Empty" if idle)
- Current rep count / set number
- Weight on stack
- Session duration timer
- Form score (small coloured bar)

### Layout
- Cards should be compact enough to show **20–30 machines** on a single screen without scrolling on a 1080p monitor
- Cards should be slightly larger on a 4K display
- Clicking a card navigates to `/machine/:id` for that machine's full view

### Summary Bar (top of page)
- Total members currently in gym
- Machines in use vs idle
- Active alerts count
- Current time

### Alert Panel (right sidebar or bottom strip)
- List of flagged events: bad form detected, machine idle >30 min with member seated, Pi offline
- Each alert has a dismiss button
- Alerts older than 5 minutes auto-dismiss

### Mock Data
Simulate 12 machines with a mix of states: 4 active sets, 3 resting, 3 idle, 1 alert, 1 Pi offline.

---

## Admin View (`/admin`)

Simpler — a status dashboard for the technical team.

### Sections
1. **Pi Fleet Status** — table of all Pis: hostname, IP, last seen, driver version, model version loaded, uptime
2. **Model Registry** — list of trained model versions: v1, v2, v3 etc. with accuracy, training date, which Pis are running it
3. **Recent Alerts** — log of system events (Pi went offline, model updated, low confidence flags)
4. **Training Queue** — how many unannotated videos are waiting, how many annotated segments are ready for next training run

---

## Simulated WebSocket Payload Format

The app should expect data in this format from the Pi WebSocket:

```json
{
  "machine_id": "lat-pulldown-01",
  "timestamp": "2026-03-27T14:32:11.234Z",
  "state": "active_set",
  "member": {
    "id": "M1089",
    "name": "Sarah Chen",
    "confidence": 0.94
  },
  "current_set": {
    "set_number": 2,
    "reps": 7,
    "weight_kg": 52.5,
    "exercise": "Lat Pulldown",
    "form_current": "good",
    "form_breakdown": { "good": 6, "partial": 1, "bad": 0 }
  },
  "session": {
    "start_time": "2026-03-27T14:28:00Z",
    "total_sets": 1,
    "total_reps": 10,
    "total_volume_kg": 525
  }
}
```

States: `"idle"` | `"member_detected"` | `"active_set"` | `"resting"` | `"session_complete"` | `"confirm_identity"`

---

## What Is NOT In Scope

- User authentication / login (not needed for MVP)
- Data entry or form submission
- Connecting to real Pis (use mock data)
- Payment or membership management (that stays in the existing system)
- Mobile app (this is a web app for tablets and desktop monitors)

---

## Deliverable

A working React web app with:
- All three routes implemented
- Member tablet view cycling through all 6 states with simulated data
- Staff floor view showing 12 mock machines with mixed states
- Admin view with mock Pi fleet and model registry data
- Responsive enough to work on a 10" tablet (landscape) and a 24" monitor
- Dark theme throughout
- Smooth state transitions (fade/slide animations between states)

The app should feel like it could be shown to a potential investor or gym owner tomorrow as a working demo of what the system will look like.
