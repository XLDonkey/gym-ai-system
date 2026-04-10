---
title: Review Loop
tags: [system, review, training, self-improving]
created: 2026-04-09
updated: 2026-04-10
---

# Review Loop — How the Pi Gets Smarter

The system self-improves. Every week, uncertain predictions become labelled training data. The model retrains. The Pi gets smarter. Over months, accuracy converges toward 95%+.

---

## The Loop

```
Pi uncertain (LSTM confidence < 0.50)
    │
    ▼
ClipReporter saves locally:
    {uid}_kps.npy      30×51 float32 keypoints
    {uid}_meta.json    machine_id, timestamp, predicted_class,
                       confidence, probabilities (all 8 classes)
    │
    ▼
GitHub API upload:
    data/review/{machine_id}/{YYYY-MM-DD}/{uid}_kps.npy
    data/review/{machine_id}/{YYYY-MM-DD}/{uid}_meta.json
    (rate-limited: CLIP_COOLDOWN_S = 30s between uploads)
    │
Mac Mini: git pull
    ▼
make review  →  localhost:8787
    ─ Lists all clips with review_status="pending"
    ─ Shows skeleton animation (drawn from 30-frame keypoints)
    ─ Shows predicted class + confidence
    ─ 8 buttons: click correct class
    ─ Saves true_class to meta.json, review_status="reviewed"
    │
    ▼
git commit -m "reviewed N clips"
    │
    ▼
make train  →  train/train_pytorch.py
    ─ Loads data/annotations/ + data/review/ (reviewed only)
    ─ Trains LSTM on Apple Silicon MPS
    ─ Exports models/weights/activity_v1.onnx
    │
    ▼
make deploy PI=pi@IP
    ─ scp ONNX to Pi
    ─ restart xlf-overseer systemd service
    │
    ▼
Pi now smarter — uncertain clips will be rarer
```

---

## ClipReporter (`pi/clip_reporter.py`)

```python
reporter = ClipReporter(
    machine_id="xlf-pi-001",
    github_token="ghp_...",        # PAT with Contents:write
    github_repo="Matt-xlfitness/gym-ai-system",
    enabled=True,
    cooldown_s=30.0
)

reporter.maybe_report(keypoints_window, classifier_result)
# Triggers if: result.confidence < ONNX_REVIEW_THRESH (0.50)
# AND: cooldown has elapsed (30s between uploads)
```

**Leave `GITHUB_REVIEW_TOKEN` blank** to save clips locally only (no upload). Useful for testing.

---

## Clip Format

### `{uid}_kps.npy`
```python
np.ndarray shape (30, 51)  # float32
# 30 frames × 17 keypoints × 3 (x, y, conf)
```

### `{uid}_meta.json`
```json
{
  "machine_id":      "xlf-pi-001",
  "timestamp":       "2026-04-09T14:32:11Z",
  "predicted_class": 3,
  "confidence":      0.43,
  "probabilities":   [0.01, 0.02, 0.08, 0.43, 0.31, 0.05, 0.06, 0.04],
  "phase":           "ENGAGED",
  "review_status":   "pending",
  "true_class":      null
}
```

After review: `review_status` → `"reviewed"`, `true_class` → correct class ID.

---

## Review Portal (`pose/review_server.py`)

```bash
make review
# Starts HTTP server at localhost:8787
```

Features:
- Skeleton animation rendered from 30-frame keypoints (canvas element)
- Predicted class shown with confidence
- All 8 class probabilities shown as bar chart
- Click the correct class button → auto-advances to next clip
- Skip button for ambiguous clips
- Progress counter: "14 / 47 pending"

---

## Rate Limiting

`CLIP_COOLDOWN_S = 30.0` — minimum 30 seconds between uploads.

Without this, a confused model at a busy machine could upload thousands of clips per hour and hit GitHub API rate limits.

---

## Config (`pi/config.py`)

```python
ONNX_REVIEW_THRESH    = 0.50   # Below this → flag for review
CLIP_REPORTER_ENABLED = True
CLIP_COOLDOWN_S       = 30.0
GITHUB_REVIEW_TOKEN   = ""     # ghp_... PAT with Contents:write
GITHUB_REVIEW_REPO    = "Matt-xlfitness/gym-ai-system"
```

---

## Related

- [[System/LSTM Model]] — the model being improved
- [[System/Activity Classes]] — the 8 classes being labelled
- [[System/Session Recorder]] — raw video for initial training
- [[Projects/Rep Tracking]] — the project that benefits
- [[Data/Training Requirements]] — initial data collection (before review loop)
- [[People/Repos and Access]] — GitHub PAT needed for upload
