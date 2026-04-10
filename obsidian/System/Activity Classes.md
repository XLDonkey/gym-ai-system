---
title: Activity Classes
tags: [system, lstm, classes, schema]
created: 2026-04-09
updated: 2026-04-10
---

# Activity Classes (8-Class Schema)

The LSTM outputs exactly one class per 30-frame window. This is the core schema for [[Projects/Rep Tracking]].

---

## The 8 Classes

| ID | Label | Description | Counts rep? | Valid phase |
|----|-------|-------------|-------------|-------------|
| 0 | `no_person` | Empty frame, nobody detected | — | IDLE |
| 1 | `user_present` | Person visible, standing, not using machine | — | IDLE |
| 2 | `on_machine` | Seated and engaged, not lifting | Starts session | Both |
| 3 | `good_rep` | Full ROM, controlled, weight moving | ✅ Yes | ENGAGED |
| 4 | `bad_rep` | Bouncing, swinging, momentum-assisted | ✅ Flagged | ENGAGED |
| 5 | `false_rep` | Adjusting handles, stretching | ❌ No | ENGAGED |
| 6 | `resting` | Seated between sets, handles released | — | ENGAGED |
| 7 | `half_rep` | Partial ROM or single-arm | ✅ Flagged | ENGAGED |

**REP_CLASSES** = {3, 4, 7} — these increment `rep_count` and contribute to form score.
**ACTIVE_CLASSES** = {2, 3, 4, 5, 6, 7} — valid in ENGAGED phase.

---

## The Phase Gate

```
IDLE phase → valid: {0, 1, 2}
  Rep classes {3,4,5,6,7} → logits masked to -∞ before softmax
  They literally cannot be predicted in IDLE

ENGAGED phase → valid: {2, 3, 4, 5, 6, 7}
  All rep classes allowed

Transition IDLE → ENGAGED:
  10 consecutive frames classified as class 2 (on_machine)
  AND EngagementDetector confirms: in zone + seated + wrists up

Transition ENGAGED → IDLE:
  45 consecutive frames classified as class 0 or 1
  OR SESSION_TIMEOUT_S = 300 seconds (5 min hard limit)
```

The masking works by setting invalid class logits to `-∞` then re-running softmax. This re-normalises the probability distribution over only the valid classes — not just low probability, **impossible**.

---

## Display Colours

| Class | Colour | Hex | Used in |
|-------|--------|-----|---------|
| no_person | Grey | `#64748b` | tablet.html, staff.html |
| user_present | Grey | `#64748b` | tablet.html, staff.html |
| on_machine | Blue | `#3b82f6` | tablet.html, staff.html |
| good_rep | Green | `#22c55e` | tablet.html, staff.html |
| bad_rep | Red | `#ef4444` | tablet.html, staff.html |
| false_rep | Yellow | `#eab308` | tablet.html, staff.html |
| resting | Purple | `#a855f7` | tablet.html, staff.html |
| half_rep | Orange | `#f97316` | tablet.html, staff.html |

The `rep_count` number on the tablet is coloured by the current activity class.
The status dot on each machine card in `staff.html` uses the same colours.

---

## Form Score

Calculated at set complete:
```
form_score = good_reps / (good_reps + partial_reps + bad_reps)
```

Logged to Supabase `sets` table. Displayed on tablet as three pills: **Good / Partial / Bad**.

---

## Class Hierarchy

```
ActivityState
├─ IDLE phase (valid: 0, 1, 2)
│  ├─ 0: no_person           (empty frame)
│  ├─ 1: user_present        (standing, not seated)
│  └─ 2: on_machine          ← 10-frame threshold → ENGAGED
│
└─ ENGAGED phase (valid: 2, 3, 4, 5, 6, 7)
   ├─ 2: on_machine           (still valid, session underway)
   ├─ 3: good_rep    ─┐
   ├─ 4: bad_rep     ├─ REP_CLASSES → rep_count++
   └─ 7: half_rep    ─┘
   ├─ 5: false_rep             (ignored, no rep counted)
   └─ 6: resting               (between sets, session active)
   
   45 frames of {0,1} → back to IDLE
```

---

## Confidence Handling

| Level | Action |
|-------|--------|
| ≥ 0.65 | Accept, high confidence |
| 0.50–0.65 | Accept, log as uncertain |
| < 0.50 | Accept + upload 30-frame clip to GitHub for review |

See [[System/Review Loop]] for the full improvement loop.

---

## Training Data per Class

| Class | Min needed | Why |
|-------|-----------|-----|
| no_person (0) | 30 | Easy — camera without anyone |
| user_present (1) | 30 | Person nearby, standing |
| on_machine (2) | 30 | Seated, not moving |
| good_rep (3) | 30+ | Most important — core class |
| bad_rep (4) | 30 | Hard to collect naturally — exaggerate bad form |
| false_rep (5) | 30 | Adjusting handles, stretching |
| resting (6) | 30 | Seated, resting between sets |
| half_rep (7) | 30 | Stop halfway through ROM deliberately |

**Total minimum: 240. Target: 300+.**

---

## Related

- [[System/LSTM Model]] — the model predicting these classes
- [[System/Engagement Detector]] — gates the IDLE → ENGAGED transition
- [[System/Review Loop]] — uncertain predictions improve via human annotation
- [[System/WebSocket Layer]] — `activity_id` broadcast in WebSocket payload
- [[Projects/Rep Tracking]] — uses these classes to count reps
- [[Data/Training Requirements]] — data collection targets
