"""
XL Fitness AI Overseer — Supabase Database Client
Wraps all DB operations used by the Pi and server.

Usage:
    from members.db_client import SupabaseClient
    db = SupabaseClient()
    session_id = db.create_session(member_id="...", machine_id="xlf-pi-001", ...)
    db.log_rep(session_id, rep_number=1, rom_degrees=65.2, duration_s=1.4)
    db.close_session(session_id, total_reps=10)
"""

import os
import json
import math
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timezone
from typing import Optional

# ── Supabase REST helpers ──────────────────────────────────────────────────────

class SupabaseClient:
    """Thin REST wrapper around Supabase — no SDK dependency needed on Pi."""

    def __init__(
        self,
        url: str  = None,
        key: str  = None,
    ):
        self.url = (url or os.environ.get("SUPABASE_URL", "")).rstrip("/")
        self.key = key or os.environ.get("SUPABASE_SERVICE_KEY", "")

        if not self.url or not self.key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set "
                "(env vars or constructor args)."
            )

        self._base = f"{self.url}/rest/v1"
        self._headers = {
            "apikey":        self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type":  "application/json",
            "Prefer":        "return=representation",
        }

    # ── Low-level REST ─────────────────────────────────────────────────────────

    def _request(self, method: str, path: str, body: dict = None, params: dict = None) -> dict | list:
        url = f"{self._base}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)

        data = json.dumps(body).encode() if body else None
        req  = urllib.request.Request(url, data=data, headers=self._headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Supabase {method} {path} → HTTP {e.code}: {e.read().decode()}") from e

    def _get(self, path: str, params: dict = None):
        return self._request("GET", path, params=params)

    def _post(self, path: str, body: dict):
        return self._request("POST", path, body=body)

    def _patch(self, path: str, body: dict, params: dict = None):
        return self._request("PATCH", path, body=body, params=params)

    # ── Members ────────────────────────────────────────────────────────────────

    def get_all_members(self) -> list[dict]:
        """Return all active members with their face embeddings."""
        return self._get("/members", params={"active": "eq.true"})

    def get_member(self, member_id: str) -> Optional[dict]:
        rows = self._get("/members", params={"id": f"eq.{member_id}", "limit": "1"})
        return rows[0] if rows else None

    def create_member(self, name: str, face_embedding: list[float]) -> dict:
        """Enroll a new member. Returns the created row."""
        rows = self._post("/members", {
            "name":           name,
            "face_embedding": face_embedding,
        })
        return rows[0] if isinstance(rows, list) else rows

    def deactivate_member(self, member_id: str):
        """Soft-delete a member (sets active=false)."""
        self._patch("/members", {"active": False}, params={"id": f"eq.{member_id}"})

    # ── Sessions ───────────────────────────────────────────────────────────────

    def create_session(
        self,
        machine_id:   str,
        machine_name: str  = None,
        member_id:    str  = None,
    ) -> str:
        """
        Open a new session. Returns the new session UUID.
        member_id may be None for unrecognised users.
        """
        body = {
            "machine_id":   machine_id,
            "machine_name": machine_name,
            "started_at":   _utcnow(),
        }
        if member_id:
            body["member_id"] = member_id

        rows = self._post("/sessions", body)
        row  = rows[0] if isinstance(rows, list) else rows
        return row["id"]

    def assign_member_to_session(self, session_id: str, member_id: str):
        """Update session with identified member (called once face is recognised)."""
        self._patch("/sessions", {"member_id": member_id}, params={"id": f"eq.{session_id}"})

    def close_session(
        self,
        session_id:  str,
        total_reps:  int,
        avg_rom:     float = None,
        avg_duration_s: float = None,
    ):
        """Mark session as ended and record summary stats."""
        body = {
            "ended_at":   _utcnow(),
            "total_reps": total_reps,
        }
        if avg_rom is not None:
            body["avg_rom"] = round(avg_rom, 1)
        if avg_duration_s is not None:
            body["avg_duration_s"] = round(avg_duration_s, 2)

        self._patch("/sessions", body, params={"id": f"eq.{session_id}"})

    def get_recent_sessions(self, limit: int = 20) -> list[dict]:
        return self._get("/sessions", params={
            "order":  "started_at.desc",
            "limit":  str(limit),
        })

    def get_member_sessions(self, member_id: str, limit: int = 50) -> list[dict]:
        return self._get("/sessions", params={
            "member_id": f"eq.{member_id}",
            "order":     "started_at.desc",
            "limit":     str(limit),
        })

    # ── Reps ───────────────────────────────────────────────────────────────────

    def log_rep(
        self,
        session_id:  str,
        rep_number:  int,
        rom_degrees: float = None,
        duration_s:  float = None,
    ):
        """Insert a single rep record."""
        body = {
            "session_id": session_id,
            "rep_number": rep_number,
            "timestamp":  _utcnow(),
        }
        if rom_degrees is not None:
            body["rom_degrees"] = round(rom_degrees, 1)
        if duration_s is not None:
            body["duration_s"] = round(duration_s, 2)

        self._post("/reps", body)

    def get_session_reps(self, session_id: str) -> list[dict]:
        return self._get("/reps", params={
            "session_id": f"eq.{session_id}",
            "order":      "rep_number.asc",
        })

    # ── Stats views ────────────────────────────────────────────────────────────

    def get_member_stats(self, member_id: str = None) -> list[dict]:
        """Query the member_stats view. Pass member_id to filter to one member."""
        params = {}
        if member_id:
            params["id"] = f"eq.{member_id}"
        return self._get("/member_stats", params=params or None)

    def get_daily_volume(self, member_id: str = None, days: int = 30) -> list[dict]:
        """Query daily_volume view for the last N days."""
        params = {"order": "workout_date.desc", "limit": str(days * 10)}
        if member_id:
            params["member_id"] = f"eq.{member_id}"
        return self._get("/daily_volume", params=params)


# ── Face matching ──────────────────────────────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two 512-dim vectors."""
    dot  = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return dot / (norm_a * norm_b)


def find_best_match(
    query_embedding:   list[float],
    members:           list[dict],
    threshold:         float = 0.40,
) -> tuple[Optional[str], Optional[str], float]:
    """
    Match a query embedding against all members.

    Returns (member_id, member_name, confidence) where confidence ∈ [0, 1].
    Returns (None, None, 0.0) when no match exceeds the threshold.

    threshold=0.40 is a reasonable default for InsightFace ArcFace — raise to
    0.45+ for stricter matching in well-lit environments.
    """
    best_id    = None
    best_name  = None
    best_score = 0.0

    for m in members:
        emb = m.get("face_embedding")
        if not emb:
            continue
        score = cosine_similarity(query_embedding, emb)
        if score > best_score:
            best_score = score
            best_id    = m["id"]
            best_name  = m["name"]

    if best_score >= threshold:
        return best_id, best_name, round(best_score, 3)
    return None, None, round(best_score, 3)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
