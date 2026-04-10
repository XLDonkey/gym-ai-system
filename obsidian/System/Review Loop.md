---
title: Review Loop
tags: [system, review, training, self-improving]
created: 2026-04-09
---

# Review Loop — How the Pi Gets Smarter

The system self-improves. When the model is uncertain, it flags the clip, a human corrects it, the model retrains, and the Pi gets smarter. Over weeks, accuracy improves continuously.

---

## The Loop

```
Pi uncertain (conf < 50%)
  ↓
clip_reporter.py saves 30-frame keypoint window + metadata
  ↓
Uploads to GitHub: data/review/{machine_id}/{date}/clip_{n}.json
  ↓
Mac Mini: git pull → make review
  ↓
http://localhost:8787
  → Shows skeleton animation from 30-frame keypoints
  → Human clicks the correct class
  → Saves label to data/annotations/
  ↓
git commit
  ↓
make train → models/weights/activity_v1.onnx
  ↓
make deploy PI=pi@IP
  ↓
Pi is now smarter
```

---

## Clip Format

Each clip saved to `data/review/` is a JSON file:

```json
{
  "machine_id": "xlf-pi-001",
  "timestamp": "2026-04-09T14:32:11Z",
  "keypoints": [[...30 frames of 17×3 keypoints...]],
  "predicted_class": 3,
  "confidence": 0.43,
  "phase": "ENGAGED"
}
```

---

## Review Portal (`pose/review_server.py`)

```bash
make review   # starts server at localhost:8787
```

Shows each clip as:
- Skeleton animation (drawn from keypoints)
- Predicted class + confidence
- 8 buttons: click the correct class
- Auto-advances to next clip

---

## Configuration

In `pi/config.py`:
```python
ONNX_REVIEW_THRESH    = 0.50   # clips below this are flagged
CLIP_REPORTER_ENABLED = True
CLIP_COOLDOWN_S       = 30.0   # min seconds between uploads
GITHUB_REVIEW_TOKEN   = ""     # PAT for GitHub upload
GITHUB_REVIEW_REPO    = "Matt-xlfitness/gym-ai-system"
```

Leave `GITHUB_REVIEW_TOKEN` blank to save clips locally only (no upload).

---

## Related

- [[System/LSTM Model]] — the model being improved
- [[System/Activity Classes]] — the 8 classes being labelled
- [[Projects/Rep Tracking]] — the project that benefits
