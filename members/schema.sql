-- ═══════════════════════════════════════════════════════════════════════════
--  XL Fitness AI Overseer — Supabase Schema
--  Run this in the Supabase SQL editor to set up the full database.
-- ═══════════════════════════════════════════════════════════════════════════

-- ── Extensions ────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── Members ───────────────────────────────────────────────────────────────────
-- One row per registered gym member.
-- face_embedding stores the 512-dim InsightFace vector as a float array.
-- Matching is done in Python with cosine similarity (no pgvector required).

CREATE TABLE IF NOT EXISTS members (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT        NOT NULL,
    face_embedding  FLOAT[]     NOT NULL,   -- 512-dim InsightFace ArcFace embedding
    enrolled_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    active          BOOLEAN     NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_members_active ON members(active);

-- ── Sessions ──────────────────────────────────────────────────────────────────
-- One row per machine use. member_id is NULL when the person is not recognised.

CREATE TABLE IF NOT EXISTS sessions (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    member_id       UUID        REFERENCES members(id) ON DELETE SET NULL,
    machine_id      TEXT        NOT NULL,   -- matches MACHINE_ID in pi/config.py
    machine_name    TEXT,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at        TIMESTAMPTZ,
    total_reps      INT         NOT NULL DEFAULT 0,
    avg_rom         FLOAT,                  -- average range-of-motion across reps (degrees)
    avg_duration_s  FLOAT                   -- average rep duration (seconds)
);

CREATE INDEX IF NOT EXISTS idx_sessions_member_id  ON sessions(member_id);
CREATE INDEX IF NOT EXISTS idx_sessions_machine_id ON sessions(machine_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON sessions(started_at DESC);

-- ── Reps ──────────────────────────────────────────────────────────────────────
-- One row per completed rep. Linked to sessions via session_id.

CREATE TABLE IF NOT EXISTS reps (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID        NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    rep_number      INT         NOT NULL,
    rom_degrees     FLOAT,                  -- range of motion for this rep (degrees)
    duration_s      FLOAT,                  -- how long the rep took (seconds)
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reps_session_id ON reps(session_id);
CREATE INDEX IF NOT EXISTS idx_reps_timestamp  ON reps(timestamp DESC);

-- ── View: member_stats ────────────────────────────────────────────────────────
-- Aggregated lifetime stats per member. Used by the dashboard.

CREATE OR REPLACE VIEW member_stats AS
SELECT
    m.id,
    m.name,
    m.enrolled_at,
    COUNT(DISTINCT s.id)                        AS total_sessions,
    COALESCE(SUM(s.total_reps), 0)              AS total_reps,
    ROUND(AVG(s.total_reps)::NUMERIC, 1)        AS avg_reps_per_session,
    ROUND(AVG(s.avg_rom)::NUMERIC, 1)           AS avg_rom,
    MAX(s.started_at)                           AS last_session_at,
    (
        SELECT machine_name
        FROM sessions s2
        WHERE s2.member_id = m.id AND s2.ended_at IS NOT NULL
        GROUP BY machine_name
        ORDER BY COUNT(*) DESC
        LIMIT 1
    )                                           AS favourite_machine
FROM members m
LEFT JOIN sessions s
    ON s.member_id = m.id AND s.ended_at IS NOT NULL
WHERE m.active = TRUE
GROUP BY m.id, m.name, m.enrolled_at;

-- ── View: daily_volume ────────────────────────────────────────────────────────
-- Per-member, per-day rep totals. Used for progress charts.

CREATE OR REPLACE VIEW daily_volume AS
SELECT
    s.member_id,
    m.name                              AS member_name,
    DATE(s.started_at)                  AS workout_date,
    s.machine_id,
    s.machine_name,
    COUNT(DISTINCT s.id)                AS sessions,
    SUM(s.total_reps)                   AS total_reps,
    ROUND(AVG(s.avg_rom)::NUMERIC, 1)   AS avg_rom
FROM sessions s
JOIN members m ON m.id = s.member_id
WHERE s.ended_at IS NOT NULL
GROUP BY s.member_id, m.name, DATE(s.started_at), s.machine_id, s.machine_name;

-- ── Row Level Security ────────────────────────────────────────────────────────
-- Lock down tables so only the service-role key can write.
-- Anonymous (anon) key can read — safe for the dashboard.

ALTER TABLE members  ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE reps     ENABLE ROW LEVEL SECURITY;

-- Read-only for anon (dashboard)
CREATE POLICY "anon read members"  ON members  FOR SELECT USING (true);
CREATE POLICY "anon read sessions" ON sessions FOR SELECT USING (true);
CREATE POLICY "anon read reps"     ON reps     FOR SELECT USING (true);

-- Full access for service role (Pi + server)
CREATE POLICY "service all members"  ON members  FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "service all sessions" ON sessions FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "service all reps"     ON reps     FOR ALL USING (auth.role() = 'service_role');
