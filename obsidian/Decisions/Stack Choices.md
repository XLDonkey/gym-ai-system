---
title: Stack Choices
tags: [decisions, architecture, tech-stack]
created: 2026-04-09
---

# Stack Choices — Why We Chose What We Chose

Every major technical decision, and the reasoning. Saves relitigating these questions.

---

## Database: Supabase (not Power Apps / Dataverse)

**Chosen:** Supabase (PostgreSQL + Realtime + Auth)

| Factor | Supabase | Power Apps / Dataverse |
|--------|----------|----------------------|
| Price | Free tier → $25/mo | £6–9 per user per month |
| Realtime | Built-in (< 1s) | Polling, ~5s lag |
| Multi-gym | `gym_id` column + RLS | Expensive licensing |
| Custom UI | Full control | Rigid Power Apps canvas |
| Data ownership | Your Postgres database | Locked in Microsoft |
| SQL | Full PostgreSQL | Limited |

**Decision:** Supabase scales from 1 gym to 100 gyms at minimal cost. Power Apps is fine for a proof of concept but becomes expensive and inflexible.

---

## Pi Inference: ONNX (not PyTorch)

**Chosen:** ONNX Runtime on Pi

| Factor | ONNX Runtime | PyTorch |
|--------|-------------|---------|
| Install size | ~50MB | ~800MB |
| Pi 5 inference | ~5ms | ~15ms |
| Dependencies | Minimal | Heavy (torch, etc.) |
| Hailo support | Via ONNX export | Not directly |

**Decision:** Train on Mac Mini with PyTorch, export to ONNX, deploy ONNX to Pi. Best of both worlds.

---

## Tracking: IoU (not DeepSORT / ByteTrack)

**Chosen:** IoU greedy matching

| Factor | IoU | DeepSORT / ByteTrack |
|--------|-----|----------------------|
| CPU cost | Near zero | Requires Re-ID model |
| GPU needed | No | Ideally yes |
| Accuracy | Good enough for gym | Better in crowds |
| Complexity | ~50 lines | Large dependency |

**Decision:** A gym machine has 1–2 people in frame at most. IoU matching is completely adequate. DeepSORT adds weight for no benefit.

---

## Face Model: buffalo_sc (not buffalo_l)

**Chosen:** InsightFace `buffalo_sc`

| Factor | buffalo_sc | buffalo_l |
|--------|-----------|-----------|
| Speed (Pi) | ~80ms | ~300ms |
| Accuracy | Good | Excellent |
| Model size | Small | Large |

**Decision:** At a gym door, "good" accuracy is enough — members enrol once and are the only people expected. Speed matters more to avoid blocking the inference loop.

---

## Angle Counting: Rule-Based Now, LSTM Later

**Current:** Rule-based (elbow/shoulder angle thresholds)
**Future:** LSTM 8-class classifier

**Why rule-based first:** Zero training data needed. Live from day one. Gives time to collect real gym data for LSTM training.

**Why switch to LSTM:** Rules can't handle "bad rep" vs "adjusting", can't detect "resting", can't classify half reps. LSTM learns from real data and handles all 8 classes.

---

## Set Reporting: Power Automate Now, Next.js Later

**Current:** Pi → Power Automate HTTP webhook
**Planned:** Pi → Next.js `/api/set` on Vercel

Only `POWER_AUTOMATE_URL` (or `SERVER_URL`) in `pi/config.py` changes. Pi code is unchanged.

**Why Power Automate first:** Zero infrastructure to set up. Works today.
**Why replace it:** Power Automate has rate limits, costs money at scale, adds latency, and is opaque.

---

## Public vs Private Repos

| Repo | Visibility | Purpose |
|------|-----------|---------|
| `xldonkey/gym-ai-system` | **Public** | GitHub Pages (Bible), open development |
| `Matt-xlfitness/Gym-Overseer-AI` | **Private** | Production code, credentials |

GitHub Pages requires a public repo on the free plan. The Bible lives at `xldonkey/gym-ai-system`, production stays private.

---

## Related

- [[Decisions/Display Layer]] — tablet vs web app decisions
- [[System/Architecture]] — how all pieces connect
- [[System/Database Schema]] — Supabase schema detail
