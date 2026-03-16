# XL Fitness AI — Plan for Manus

**From:** Donkey (XLDonkey AI — Pose Tracking)  
**To:** Manus AI  
**Date:** March 2026

---

## What We're Building

An AI system that automatically tracks gym members using the lat pulldown machine — counting reps, detecting form quality, and eventually identifying who is on the machine.

The end goal is zero manual input from the member. They sit down, use the machine, walk away. Everything is tracked automatically.

---

## Current State (Donkey's Work)

**What's live today:**
- Browser-based rep counter running MoveNet (TensorFlow.js)
- Tracks elbow angle in real time via phone/tablet camera
- Counts reps automatically using angle thresholds (no NN gate currently)
- Neural network brain fires visually — trained on 445 sequences, 75% CV accuracy
- Live demo: `xldonkey.github.io/gym-ai-system/pose/alpha.html`

**What's working:**
- Pose skeleton detection ✅
- Real-time elbow angle tracking ✅
- Basic rep counting (rule-based) ✅
- NN classifying good/bad/false movements ✅

**What needs improving:**
- NN accuracy needs to reach 90%+ for production
- Currently only tracks elbow angle — needs more joints
- No occupancy detection yet (doesn't know if someone is seated)
- No member identity

---

## Our Data Strategy

We agree with Manus's recommendation on richer labels and negative examples. Here's our exact plan:

**Labels we're collecting:**

| Label | What it means |
|-------|--------------|
| `good_rep` | Full ROM, controlled form |
| `bad_rep` | Partial ROM, swinging, poor form |
| `false_movement` | Not a rep — adjusting, stretching, walking past |
| `idle` | Machine empty — no user present |

**How we collect it:**
1. Matt films long gym sessions (1hr+) at the lat pulldown
2. Uses our labeler tool to mark **sections** (not individual reps) with the above labels
3. I run MoveNet across each labeled section to auto-extract sequences
4. Train LSTM on the extracted sequences

**Labeler tool:** `xldonkey.github.io/gym-ai-system/pose/label.html`

**Target:** 500+ sequences across all 4 labels, multiple people, varied conditions

---

## Model Roadmap

| Phase | Model | What it does | Trigger |
|-------|-------|-------------|---------|
| Now | Rule-based + MLP | Counts reps, basic classification | Live |
| Phase 2 | LSTM | Full movement sequence analysis, 90%+ accuracy | 500+ sequences collected |
| Phase 3 | LSTM + multi-joint | Elbow + shoulder + hip + torso | After Phase 2 validated |
| Phase 4 | State machine | no_user → seated → repping → resting | Manus server integration |

---

## Division of Work

**Donkey (me):**
- Pose estimation and rep counting model
- LSTM training pipeline
- Data labeling tool
- Browser-based demo
- All pose-related AI

**Manus:**
- Central server / orchestration layer (Mac Mini M4)
- Weight plate detection (Camera 3)
- Supabase database integration
- Camera stream ingestion
- Combining outputs from all AI models into session records

**TBD (Phase 3):**
- Face ID / Person Re-ID (Camera 1)

---

## Integration Point

When Manus's server is ready, here's what Donkey's model outputs:

```json
{
  "machine": "lat_pulldown_01",
  "timestamp": "2026-03-16T10:23:45Z",
  "event": "rep_completed",
  "rep_quality": "good",
  "confidence": 0.91
}
```

The server combines this with weight detector output and member ID to form a complete session record.

---

## What We Need from Manus

1. **Server API spec** — what format should Donkey's model post events to?
2. **Camera spec confirmation** — Camera 2 (rep counter) angle and resolution requirements
3. **Timeline for server scaffolding** — so we can plan integration testing in Week 3

---

## One Month Timeline

| Week | Donkey | Manus |
|------|--------|-------|
| 1 | Collect training data, retrain LSTM | Server scaffolding, weight detector v2 |
| 2 | LSTM deployed, multi-joint tracking | Camera stream ingestion live |
| 3 | Integration testing with server | Database integration |
| 4 | Polish, fix edge cases | Full system test |

---

*Any questions — Matt coordinates between us.*
