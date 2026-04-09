"""
XL Fitness AI — Clip Reporter
Uploads low-confidence ONNX inference clips to GitHub for human review.

Flow:
    Pi runs ONNX → confidence < 0.50 → ClipReporter.maybe_report()
        → saves 30-frame keypoint window (.npy) + metadata (.json) locally
        → uploads both to GitHub data/review/{machine_id}/{date}/ via REST API
        → human annotates in label.html → commits label JSON
        → Mac Mini retrains → new ONNX pushed to models/weights/
        → Pi downloads new model → improved accuracy

The GitHub PAT needs:   Contents: write   (Issues: write — optional, for auto-issue)
Set GITHUB_REVIEW_TOKEN in config.py.  Leave blank to save locally only.

Rate-limiting: a 30-second cooldown between uploads so a confused model
doesn't spam GitHub with hundreds of clips per minute.
"""

import base64
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import requests


class ClipReporter:
    """
    Saves low-confidence ONNX clips to disk and uploads to GitHub.

    Args:
        github_token:         Personal Access Token with Contents:write scope.
        repo:                 "Matt-xlfitness/gym-ai-system"
        machine_id:           e.g. "xlf-pi-001"
        confidence_threshold: Upload when confidence is below this (default 0.50).
        cooldown_s:           Min seconds between uploads (prevents spam, default 30).
        local_backup_dir:     Where to save clips locally (survives upload failure).
        enabled:              Set False to disable entirely (skips even local save).
    """

    def __init__(
        self,
        github_token: str,
        repo: str,
        machine_id: str,
        confidence_threshold: float = 0.50,
        cooldown_s: float = 30.0,
        local_backup_dir: str = "/tmp/xlf_review_clips",
        enabled: bool = True,
    ):
        self.token                = github_token
        self.repo                 = repo
        self.machine_id           = machine_id
        self.confidence_threshold = confidence_threshold
        self.cooldown_s           = cooldown_s
        self.local_backup_dir     = local_backup_dir
        self.enabled              = enabled
        self._can_upload          = enabled and bool(github_token)
        self._last_upload_time    = 0.0
        self._total_reported      = 0
        self._failed_uploads: list = []  # retry queue

        os.makedirs(local_backup_dir, exist_ok=True)

        if self._can_upload:
            print(f"[ClipReporter] Active  repo={repo}  threshold={confidence_threshold}")
        elif enabled:
            print("[ClipReporter] Local-only mode (no GitHub token configured)")
        else:
            print("[ClipReporter] Disabled")

    # ── Public API ────────────────────────────────────────────────────────────

    def maybe_report(self, result) -> bool:
        """
        Check if the classifier result warrants flagging.
        Call this every frame — it handles rate-limiting internally.

        Returns True if the clip was reported (saved/uploaded).
        """
        if not self.enabled:
            return False

        # Only flag when the model is genuinely uncertain
        if not result.low_confidence:
            return False

        # Rate-limit to avoid spamming GitHub
        now = time.time()
        if now - self._last_upload_time < self.cooldown_s:
            return False

        self._last_upload_time = now
        return self._report(result)

    def flush_retry_queue(self):
        """
        Try to upload any previously failed clips.
        Call this once at startup and occasionally during idle time.
        """
        if not self._failed_uploads or not self._can_upload:
            return

        retry = list(self._failed_uploads)
        self._failed_uploads.clear()
        print(f"[ClipReporter] Retrying {len(retry)} failed uploads...")

        for npy_path, meta_path, gh_base in retry:
            try:
                window = np.load(npy_path)
                with open(meta_path) as f:
                    meta = json.load(f)
                ok_npy  = self._github_upload(f"{gh_base}_kps.npy",  window.tobytes())
                ok_meta = self._github_upload(f"{gh_base}_meta.json", json.dumps(meta, indent=2).encode())
                if ok_npy and ok_meta:
                    print(f"[ClipReporter] Retry OK: {gh_base}")
                else:
                    self._failed_uploads.append((npy_path, meta_path, gh_base))
            except Exception as e:
                print(f"[ClipReporter] Retry error: {e}")
                self._failed_uploads.append((npy_path, meta_path, gh_base))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _report(self, result) -> bool:
        ts        = datetime.now(timezone.utc)
        uid       = ts.strftime("%Y%m%d_%H%M%S")
        date_path = ts.strftime("%Y-%m-%d")

        # Build keypoint array from buffer
        window = np.array(result.buffer, dtype=np.float32)  # (30, 51)

        # Metadata stored alongside the .npy for annotators
        meta = {
            "machine_id":      self.machine_id,
            "timestamp_utc":   ts.isoformat(),
            "predicted_class": result.class_id,
            "class_name":      result.class_name,
            "confidence":      round(result.confidence, 4),
            "probabilities":   (
                [round(float(p), 4) for p in result.probs]
                if result.probs is not None else []
            ),
            "window_frames":   len(result.buffer),
            "feature_dim":     51,
            "review_status":   "pending",
            "true_class":      None,  # annotator fills this in
            # Keypoints embedded for the review portal skeleton animation
            # Each entry is a flat list of 51 floats [x0,y0,c0, x1,y1,c1, ...]
            "keypoints_array": [
                [round(float(v), 4) for v in frame]
                for frame in result.buffer
            ],
        }

        # ── Save locally first ────────────────────────────────────────────────
        npy_local  = os.path.join(self.local_backup_dir, f"{uid}_kps.npy")
        meta_local = os.path.join(self.local_backup_dir, f"{uid}_meta.json")
        np.save(npy_local, window)
        with open(meta_local, "w") as f:
            json.dump(meta, f, indent=2)

        self._total_reported += 1
        print(
            f"[ClipReporter] #{self._total_reported} flagged — "
            f"class={result.class_name}  conf={result.confidence:.2f}  "
            f"saved={uid}"
        )

        if not self._can_upload:
            return True  # saved locally, that's all we can do

        # ── Upload to GitHub ──────────────────────────────────────────────────
        gh_base = f"data/review/{self.machine_id}/{date_path}/{uid}"

        ok_npy  = self._github_upload(f"{gh_base}_kps.npy",  window.tobytes())
        ok_meta = self._github_upload(f"{gh_base}_meta.json", json.dumps(meta, indent=2).encode())

        if ok_npy and ok_meta:
            print(f"[ClipReporter] Uploaded → {gh_base}")
        else:
            print(f"[ClipReporter] Upload failed — queued for retry: {uid}")
            self._failed_uploads.append((npy_local, meta_local, gh_base))

        return True

    def _github_upload(self, path: str, content: bytes) -> bool:
        """Upload a single file to GitHub via REST API (create or update)."""
        url = f"https://api.github.com/repos/{self.repo}/contents/{path}"
        headers = {
            "Authorization":        f"Bearer {self.token}",
            "Accept":               "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        # Check if file already exists so we can include its SHA (required for updates)
        sha = None
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                sha = r.json().get("sha")
        except Exception:
            pass  # no SHA = creating a new file, that's fine

        body: dict = {
            "message": f"[pi-review] Low-confidence clip — {self.machine_id}",
            "content": base64.b64encode(content).decode("ascii"),
        }
        if sha:
            body["sha"] = sha

        try:
            r = requests.put(url, headers=headers, json=body, timeout=30)
            if r.status_code in (200, 201):
                return True
            print(f"[ClipReporter] GitHub {r.status_code}: {r.text[:200]}")
            return False
        except Exception as e:
            print(f"[ClipReporter] Upload error: {e}")
            return False
