---
title: Database Schema
tags: [system, database, supabase, schema]
created: 2026-04-09
---

# Database Schema (Supabase)

Supabase (PostgreSQL) stores all members, sessions, and sets. See [[Decisions/Stack Choices]].

---

## Tables

### `members`

```sql
id          uuid primary key default gen_random_uuid()
created_at  timestamptz default now()
name        text not null
member_code text unique          -- e.g. "M1089"
face_embedding  vector(512)      -- ArcFace embedding (pgvector)
enrolled_at timestamptz
gym_id      text                 -- for multi-gym support
```

### `sessions`

```sql
id          uuid primary key
member_id   uuid references members(id)
machine_id  text                 -- e.g. "xlf-pi-001"
machine_name text
started_at  timestamptz
ended_at    timestamptz
total_sets  int
total_reps  int
```

### `sets`

```sql
id          uuid primary key
session_id  uuid references sessions(id)
member_id   uuid references members(id)
machine_id  text
timestamp   timestamptz
exercise    text
weight_kg   float
reps        int
form_good   int
form_partial int
form_bad    int
form_score  float                -- good / (good + partial + bad)
model_version text               -- e.g. "v1.0-lstm"
```

### `pi_status`

```sql
machine_id   text primary key
machine_name text
last_seen    timestamptz
state        jsonb               -- last WebSocket payload
ip_address   text
model_version text
```

---

## Set Payload (Pi → Supabase)

```json
{
  "machine_id":     "lat-pulldown-01",
  "machine_name":   "Nautilus Lat Pulldown",
  "member_id":      "M1089",
  "member_name":    "Matthew",
  "timestamp":      "2026-04-09T14:32:11Z",
  "exercise":       "Lat Pulldown",
  "weight_kg":      52.5,
  "reps":           10,
  "form_breakdown": {"good": 7, "partial": 2, "bad": 1},
  "form_score":     0.78,
  "model_version":  "v1.0-lstm"
}
```

---

## Config

Set in `pi/config.py`:
```python
SUPABASE_URL         = "https://xxxx.supabase.co"
SUPABASE_SERVICE_KEY = "eyJ..."   # service_role JWT (not anon key)
```

Use the **service_role** key on the Pi — it bypasses Row-Level Security and can write directly.

---

## Multi-Gym

Add `gym_id` column to members + machines tables, use Supabase Row-Level Security (RLS) to restrict each staff account to their gym's data.

---

## Related

- [[Decisions/Stack Choices]] — why Supabase
- [[System/WebSocket Layer]] — set_reporter sends to Supabase
- [[Projects/User Tracking]] — members table populated via enrolment
