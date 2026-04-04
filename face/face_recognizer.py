"""
XL Fitness AI Overseer — Face Recognition Module
Uses InsightFace (buffalo_sc model) to extract 512-dim ArcFace embeddings
and match them against the Supabase member registry.

Designed to run on Raspberry Pi 5 — the buffalo_sc model is compact (~170MB)
and runs at ~100ms per face on Pi 5 CPU (no accelerator needed).

Usage:
    recognizer = FaceRecognizer()
    recognizer.load_members(db.get_all_members())

    # In main loop — call every N frames when a person is present
    result = recognizer.identify_from_frame(frame, bbox_xyxy)
    if result.member_id:
        print(f"Recognised: {result.member_name} ({result.confidence:.2f})")
"""

import os
import sys
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional

# InsightFace import — graceful failure if not installed
try:
    import insightface
    from insightface.app import FaceAnalysis
    _INSIGHTFACE_AVAILABLE = True
except ImportError:
    _INSIGHTFACE_AVAILABLE = False
    print("[face] WARNING: insightface not installed — face recognition disabled.")
    print("[face]   Install: pip install insightface onnxruntime")


@dataclass
class RecognitionResult:
    member_id:   Optional[str]   # UUID or None
    member_name: Optional[str]   # e.g. "John Smith" or None
    confidence:  float           # cosine similarity score [0, 1]
    latency_ms:  float           # how long recognition took
    face_found:  bool            # was a face detected in the crop?


class FaceRecognizer:
    """
    Wraps InsightFace for gym member identification.

    Workflow:
      1.  __init__()             — load the InsightFace model
      2.  load_members(rows)     — cache all member embeddings from Supabase
      3.  identify_from_frame()  — call every ~30 frames when a person is present
    """

    # Cosine similarity threshold for a positive match.
    # 0.40 works well for controlled indoor lighting.
    # Raise to 0.45 in bright, consistent lighting environments.
    DEFAULT_THRESHOLD = 0.40

    def __init__(
        self,
        model_name:  str   = "buffalo_sc",   # small but accurate; use buffalo_l for best accuracy
        providers:   list  = None,            # ONNX providers; None = auto-detect
        threshold:   float = DEFAULT_THRESHOLD,
    ):
        self.threshold   = threshold
        self._members    = []   # list of {"id": ..., "name": ..., "embedding": np.ndarray}
        self._model      = None
        self._ready      = False

        if not _INSIGHTFACE_AVAILABLE:
            return

        providers = providers or ["CPUExecutionProvider"]

        try:
            t0 = time.time()
            app = FaceAnalysis(
                name       = model_name,
                providers  = providers,
                allowed_modules = ["detection", "recognition"],
            )
            # det_size: smaller = faster on Pi; 320×320 is enough for gym cameras
            app.prepare(ctx_id=0, det_size=(320, 320))
            self._model = app
            self._ready = True
            print(f"[face] Model '{model_name}' loaded in {(time.time()-t0)*1000:.0f}ms")
        except Exception as e:
            print(f"[face] ERROR loading model '{model_name}': {e}")

    @property
    def ready(self) -> bool:
        return self._ready

    # ── Member cache ───────────────────────────────────────────────────────────

    def load_members(self, rows: list[dict]):
        """
        Cache member embeddings from Supabase rows.
        rows is the output of db_client.get_all_members().
        Call this once at startup and again after new enrolments.
        """
        self._members.clear()
        loaded = 0
        for row in rows:
            emb = row.get("face_embedding")
            if not emb:
                continue
            self._members.append({
                "id":        row["id"],
                "name":      row["name"],
                "embedding": np.array(emb, dtype=np.float32),
            })
            loaded += 1
        print(f"[face] {loaded} member embeddings cached")

    def add_member(self, member_id: str, name: str, embedding: list[float]):
        """Add a single new member to the in-memory cache (no DB call)."""
        self._members.append({
            "id":        member_id,
            "name":      name,
            "embedding": np.array(embedding, dtype=np.float32),
        })

    # ── Identification ─────────────────────────────────────────────────────────

    def identify_from_frame(
        self,
        frame:     np.ndarray,  # full BGR frame from OpenCV
        bbox_xyxy: tuple = None, # (x1, y1, x2, y2) bounding box from YOLO, optional
        padding:   float = 0.3, # fraction of bbox to expand for face region
    ) -> RecognitionResult:
        """
        Detect and identify a face within a frame.

        If bbox_xyxy is provided, only the head region (top portion of the bbox)
        is searched — this is faster and avoids false detections.

        Returns a RecognitionResult. Check result.face_found before using
        result.member_id (face_found=False means InsightFace found no face).
        """
        if not self._ready:
            return RecognitionResult(None, None, 0.0, 0.0, False)

        t0 = time.time()

        # Crop to head region for speed
        crop = _crop_head(frame, bbox_xyxy, padding) if bbox_xyxy else frame

        try:
            faces = self._model.get(crop)
        except Exception as e:
            print(f"[face] inference error: {e}")
            return RecognitionResult(None, None, 0.0, (time.time()-t0)*1000, False)

        if not faces:
            return RecognitionResult(None, None, 0.0, (time.time()-t0)*1000, False)

        # Use the largest detected face
        face       = max(faces, key=lambda f: _face_area(f))
        query_emb  = face.normed_embedding  # 512-dim unit vector

        member_id, member_name, score = self._match(query_emb)
        latency_ms = (time.time() - t0) * 1000

        return RecognitionResult(
            member_id   = member_id,
            member_name = member_name,
            confidence  = score,
            latency_ms  = round(latency_ms, 1),
            face_found  = True,
        )

    def extract_embedding(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract the raw 512-dim embedding from a face in the frame.
        Returns None if no face found. Used during member enrolment.
        """
        if not self._ready:
            return None
        faces = self._model.get(frame)
        if not faces:
            return None
        face = max(faces, key=lambda f: _face_area(f))
        return face.normed_embedding

    # ── Internal matching ──────────────────────────────────────────────────────

    def _match(
        self,
        query: np.ndarray,
    ) -> tuple[Optional[str], Optional[str], float]:
        """
        Cosine similarity match against all cached member embeddings.
        Returns (member_id, member_name, score). Returns (None, None, score)
        if best score is below threshold.
        """
        if not self._members:
            return None, None, 0.0

        # Stack all embeddings for fast batch dot product
        matrix = np.stack([m["embedding"] for m in self._members])   # (N, 512)
        scores = matrix @ query                                         # (N,) — already unit vectors

        best_i = int(np.argmax(scores))
        best_score = float(scores[best_i])

        if best_score >= self.threshold:
            m = self._members[best_i]
            return m["id"], m["name"], round(best_score, 3)

        return None, None, round(best_score, 3)


# ── Identity window ────────────────────────────────────────────────────────────

class IdentityWindow:
    """
    Accumulates recognition results over the first N seconds of a session
    and returns the most-confident identification once the window closes.

    Prevents a single bad frame from wrongly identifying a member.

    Usage:
        window = IdentityWindow(duration_s=10)
        window.add(result)
        ...
        final = window.best()  # RecognitionResult or None
    """

    def __init__(self, duration_s: float = 10.0):
        self.duration_s = duration_s
        self._start     = time.time()
        self._results   = []
        self._locked    = False
        self._best      = None

    @property
    def open(self) -> bool:
        """True while still collecting results."""
        return not self._locked and (time.time() - self._start) < self.duration_s

    @property
    def expired(self) -> bool:
        """True once the window has closed."""
        return (time.time() - self._start) >= self.duration_s

    def add(self, result: RecognitionResult):
        if self._locked or not result.face_found:
            return
        self._results.append(result)

        # Lock early if we get a high-confidence match
        if result.member_id and result.confidence >= 0.50:
            self._locked = True
            self._best   = result

    def best(self) -> Optional[RecognitionResult]:
        """
        Return the best result collected so far.
        Returns None if no faces were found at all.
        """
        if self._best:
            return self._best
        if not self._results:
            return None
        # Pick highest-confidence result that has a member_id; fallback to highest overall
        identified = [r for r in self._results if r.member_id]
        pool       = identified if identified else self._results
        return max(pool, key=lambda r: r.confidence)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _crop_head(
    frame:     np.ndarray,
    bbox_xyxy: tuple,
    padding:   float,
) -> np.ndarray:
    """
    Crop the head region from a YOLO person bounding box.
    We take the top 35% of the box (where the face should be)
    and expand it by `padding` fraction in all directions.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    box_h = y2 - y1
    face_y2 = y1 + int(box_h * 0.35)   # top 35% = head

    pad_x = int((x2 - x1) * padding)
    pad_y = int((face_y2 - y1) * padding)

    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, face_y2 + pad_y)

    return frame[cy1:cy2, cx1:cx2]


def _face_area(face) -> float:
    """Return bounding box area for a detected InsightFace face."""
    b = face.bbox  # [x1, y1, x2, y2]
    return (b[2] - b[0]) * (b[3] - b[1])
