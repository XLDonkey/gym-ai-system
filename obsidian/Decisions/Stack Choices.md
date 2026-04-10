---
title: Stack Choices
tags: [decisions, architecture, tech-stack]
created: 2026-04-09
updated: 2026-04-10
---

# Stack Choices — Why We Chose What We Chose

Every major technical decision with reasoning. Prevents relitigating these.

---

## Database: Supabase (not Power Apps / Dataverse)

**Chosen:** Supabase (PostgreSQL + Realtime + Auth + REST)

| Factor | Supabase | Power Apps / Dataverse |
|--------|----------|----------------------|
| Price (1 gym) | Free tier | £6–9 per user per month |
| Price (10 gyms) | $25/mo flat | Hundreds per month |
| Realtime | Built-in (< 1s) | Polling, 5–10s lag |
| Multi-gym | `gym_id` column + RLS | Expensive licences |
| Custom UI | Full control | Rigid Power Apps canvas |
| Data ownership | Your Postgres database | Locked in Microsoft |
| SQL | Full PostgreSQL | Limited Dataverse query |
| Face embeddings | `FLOAT[]` array column | Complex workaround |

**Decision:** Supabase scales from 1 gym to 100 gyms at minimal cost. Power Apps is a low-code tool — not appropriate for a production product.

---

## Pi Inference: ONNX Runtime (not PyTorch)

**Chosen:** Export PyTorch → ONNX, run ONNX Runtime on Pi.

| Factor | ONNX Runtime | PyTorch |
|--------|-------------|---------|
| Install size on Pi | ~50 MB | ~800 MB |
| Inference time | ~5ms | ~15ms |
| Dependencies | Minimal | Heavy |
| Hailo HAT support | Via ONNX export | Not directly |
| Pi SD card impact | Minimal | Large |

**Decision:** Train on Mac Mini with PyTorch (best DX), deploy ONNX to Pi (best performance). No PyTorch on Pi.

---

## Person Tracking: IoU Matching (not DeepSORT / ByteTrack)

**Chosen:** Greedy IoU matching (~50 lines of code)

| Factor | IoU | DeepSORT |
|--------|-----|----------|
| CPU cost | Near zero | Requires Re-ID model |
| GPU needed | No | Ideally yes |
| Accuracy at gym | Good enough | Overkill |
| Complexity | Simple | Large dependency |
| Code | `user_tracking/gym_tracker.py` | Separate library |

**Decision:** Gym machines have 1–2 people in frame at most. IoU works perfectly. DeepSORT solves problems we don't have.

---

## Face Model: buffalo_sc (not buffalo_l)

**Chosen:** InsightFace `buffalo_sc`

| Factor | buffalo_sc | buffalo_l |
|--------|-----------|-----------|
| Pi 5 speed | ~80ms | ~300ms |
| Accuracy | Good | Excellent |
| Model size | Small | Large |

**Decision:** At a gym door, "good" accuracy with fast speed beats "excellent" accuracy too slow for real-time. Members enrol once and are the only people expected.

---

## Rep Counting: Rule-Based Now, LSTM Later

**Current:** Angle-based rule-based counting (live from day one, zero training data)
**Future:** LSTM 8-class classifier (300+ training segments)

**Why rule-based first:** The system works from installation day. Rules work well for standard reps on a lat pulldown. They give us time to collect real data.

**Why LSTM eventually:** Rules can't distinguish "bad_rep" from "adjusting", can't detect "resting" vs "done", can't handle half-reps. LSTM handles all 8 classes and improves via the review loop.

**Switch:** set `ONNX_MODEL_PATH` in `pi/config.py`. Blank = rules. Path set = LSTM.

---

## Set Reporting: Power Automate Now, Custom Later

**Current:** Pi → HTTP POST → Power Automate webhook → Dataverse
**Planned:** Pi → HTTP POST → Next.js `/api/set` on Vercel → Supabase Realtime

Only `POWER_AUTOMATE_URL` (or `SERVER_URL`) changes in `pi/config.py`. Pi code unchanged.

**Why Power Automate first:** Zero infrastructure. Works today. Free up to ~750 runs/month.
**Why replace:** Rate-limited, opaque, adds latency, locks into Microsoft.

---

## Video Storage: Google Drive (not S3 / local only)

**Chosen:** Local Pi buffer → rclone → Google Drive

| Factor | Google Drive | S3 |
|--------|-------------|-----|
| Cost | 15GB free, £2/mo for 100GB | $0.023/GB/month |
| Mac Mini access | Drag to Finder, rclone sync | AWS CLI |
| Setup | rclone + service account | IAM + keys |
| Simplicity | Very simple | More config |

**Decision:** Team uses Google accounts already. Google Drive is accessible to non-technical users.

---

## GitHub Pages: xldonkey/gym-ai-system (public)

**Production code:** `Matt-xlfitness/Gym-Overseer-AI` — **private**
**Dev + Bible:** `xldonkey/gym-ai-system` — **public**

GitHub Pages requires a public repo on the free plan. Bible lives at `xldonkey.github.io/gym-ai-system/bible.html`.

---

## Edge Inference vs Cloud (Not Even Considered)

Every inference runs **on the Pi**. No frames sent to cloud for analysis.

Why:
- Privacy — member data never leaves the gym network
- Latency — cloud round-trip adds 100–300ms
- Reliability — works when internet is down
- Cost — no cloud GPU inference fees
- GDPR — biometric data stays on premises

---

## Related

- [[Decisions/Display Layer]] — tablet vs web app decisions
- [[System/Database Schema]] — Supabase schema
- [[System/LSTM Model]] — ONNX deployment
- [[Projects/User Tracking]] — IoU tracker + face model choices
