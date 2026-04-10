---
title: Database Schema
tags: [system, database, supabase, schema]
created: 2026-04-09
updated: 2026-04-10
---

# Database Schema (Supabase)

PostgreSQL on Supabase. Full schema in `members/schema.sql`.

---

## Tables

### `members`

```sql
CREATE TABLE members (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name            text NOT NULL,
    face_embedding  float[] NOT NULL,   -- 512-dim ArcFace vector
    enrolled_at     timestamptz DEFAULT now(),
    active          boolean DEFAULT true
);
```

Populated by `make enrol NAME="..."` → `face/enroll_member.py`.
Loaded at Pi startup by `FaceRecognizer.load_members()`.

### `sessions`

```sql
CREATE TABLE sessions (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    member_id       uuid REFERENCES members(id),
    machine_id      text NOT NULL,     -- e.g. "xlf-pi-001"
    machine_name    text,
    started_at      timestamptz DEFAULT now(),
    ended_at        timestamptz,
    total_reps      int DEFAULT 0,
    avg_rom         float,             -- average range of motion (degrees)
    avg_duration_s  float              -- average rep duration
);
```

### `reps`

```sql
CREATE TABLE reps (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      uuid REFERENCES sessions(id),
    rep_number      int NOT NULL,
    rom_degrees     float,             -- range of motion this rep
    duration_s      float,             -- rep duration
    timestamp       timestamptz DEFAULT now()
);
```

### `sets` (from set_reporter)

```sql
CREATE TABLE sets (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      uuid REFERENCES sessions(id),
    member_id       uuid REFERENCES members(id),
    machine_id      text,
    timestamp       timestamptz,
    exercise        text,
    weight_kg       float,
    reps            int,
    form_good       int DEFAULT 0,
    form_partial    int DEFAULT 0,
    form_bad        int DEFAULT 0,
    form_score      float,             -- good / (good + partial + bad)
    model_version   text               -- e.g. "v1.0-lstm"
);
```

---

## Views

### `member_stats`
Lifetime aggregates per member:
```sql
-- total_sessions, total_reps, avg_form_score, last_session_at
-- Used in member dashboard (future)
```

### `daily_volume`
Per-day per-member volume:
```sql
-- date, member_id, total_reps, total_sets, total_weight_moved_kg
-- total_weight_moved_kg = SUM(weight_kg * reps) across all sets that day
```

---

## Row-Level Security (RLS)

```sql
-- Anonymous: read-only (for public leaderboards, future)
-- service_role: full read/write (Pi uses this key)
-- authenticated: own rows only (member app, future)
```

Pi uses the **service_role** key (bypasses RLS) — set in `pi/config.py`:
```python
SUPABASE_URL         = "https://xxxx.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

Never use the anon key on the Pi — it can't write.

---

## Supabase Client (`members/db_client.py`)

Thin REST wrapper — no Supabase SDK, just `requests`:

| Method | Purpose |
|--------|---------|
| `get_all_members()` | Load face embeddings at startup |
| `get_member(member_id)` | Fetch one member |
| `create_member(name, embedding)` | Enrol new member |
| `deactivate_member(member_id)` | Soft-delete |
| `create_session(machine_id, machine_name, member_id)` | Start session |
| `close_session(session_id, total_reps, avg_rom, avg_duration_s)` | End session |
| `log_rep(session_id, rep_number, rom_degrees, duration_s)` | Per-rep log |

---

## Set Payload (Pi → Supabase via set_reporter)

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

## Multi-Gym (Future)

Add `gym_id` column to `members`, `sessions`, `sets`.
RLS policy: `gym_id = auth.jwt() -> 'gym_id'`.
Each gym's staff sees only their data.

---

## Related

- [[Projects/User Tracking]] — members table populated here
- [[System/WebSocket Layer]] — set_reporter sends to Supabase
- [[System/Architecture]] — db_client.py position in main loop
- [[Decisions/Stack Choices]] — why Supabase over Power Apps
