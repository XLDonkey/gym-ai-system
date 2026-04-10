---
title: Activity Classes
tags: [system, lstm, classes, schema]
created: 2026-04-09
---

# Activity Classes (8-Class Schema)

The LSTM outputs one of 8 classes per 30-frame window. This is the core schema for [[Projects/Rep Tracking]].

---

## The 8 Classes

| ID | Label | Description | Counts? | Phase |
|----|-------|-------------|---------|-------|
| 0 | `no_person` | Nobody at machine | — | IDLE only |
| 1 | `user_present` | Person nearby, not seated | — | IDLE only |
| 2 | `on_machine` | Seated, engaged | Starts session | Both |
| 3 | `good_rep` | Full ROM, controlled | ✅ Yes | ENGAGED only |
| 4 | `bad_rep` | Bouncing, swinging | ✅ Flagged | ENGAGED only |
| 5 | `false_rep` | Adjusting, stretching | ❌ No | ENGAGED only |
| 6 | `resting` | Seated between sets | — | ENGAGED only |
| 7 | `half_rep` | Partial ROM | ✅ Flagged | ENGAGED only |

---

## The Phase Gate

```
IDLE phase:
  Valid classes: 0, 1, 2
  Rep classes (3–7) → masked to -inf before softmax → impossible to predict

ENGAGED phase:
  Valid classes: 2, 3, 4, 5, 6, 7

Transition IDLE → ENGAGED:
  Requires 10 consecutive frames classified as on_machine (2)

Transition ENGAGED → IDLE:
  Person absent for IDLE_TIMEOUT_SECONDS (default: 30s)
```

This prevents phantom reps from people walking past the machine.

---

## Display Colours

| Class | Colour | Hex |
|-------|--------|-----|
| no_person | Grey | `#64748b` |
| user_present | Grey | `#64748b` |
| on_machine | Blue | `#3b82f6` |
| good_rep | Green | `#22c55e` |
| bad_rep | Red | `#ef4444` |
| false_rep | Yellow | `#eab308` |
| resting | Purple | `#a855f7` |
| half_rep | Orange | `#f97316` |

Used in both `display/tablet.html` and `display/staff.html`.

---

## Form Scoring

At set complete, form score is calculated as:
```
form_score = good_reps / (good_reps + partial_reps + bad_reps)
```

Reported in the set payload to Supabase. See [[System/Database Schema]].

---

## Training Data Requirements

| Class | Min segments |
|-------|-------------|
| no_person (0) | 30 |
| user_present (1) | 30 |
| on_machine (2) | 30 |
| good_rep (3) | 30 |
| bad_rep (4) | 30 |
| false_rep (5) | 30 |
| resting (6) | 30 |
| half_rep (7) | 30 |
| **Total** | **240 min / 300+ target** |

See [[Data/Training Requirements]].
