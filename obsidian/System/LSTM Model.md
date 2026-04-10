---
title: LSTM Model
tags: [system, lstm, model, pytorch, onnx]
created: 2026-04-09
---

# LSTM Activity Classifier

The core neural network for [[Projects/Rep Tracking]]. Classifies 30 frames of YOLO keypoints into one of 8 [[System/Activity Classes]].

---

## Architecture

```
Input:  (batch, 30, 51)
        30 frames × 51 features (17 keypoints × [x, y, confidence])

LSTM:   input_size=51, hidden_size=128, num_layers=1, batch_first=True
Dropout: 0.3 (applied between LSTM and Linear)
Linear:  128 → 64
ReLU
Linear:  64 → 8
Softmax  → one class wins
```

The LSTM sees 1 second of movement (30 frames at 30fps) as a single window.

---

## Phase Gate (Softmax Masking)

In IDLE phase, classes 3–7 are masked to `-inf` before softmax.
This means they literally cannot be predicted — not just low probability, **impossible**.

```python
if phase == "IDLE":
    logits[:, 3:] = float('-inf')
output = softmax(logits, dim=-1)
```

---

## Confidence Thresholds

| Threshold | Behaviour |
|-----------|-----------|
| ≥ 0.65 | Accept, high confidence |
| 0.50–0.65 | Accept, log as uncertain |
| < 0.50 | Accept, **flag clip for review** |

Config keys in `pi/config.py`:
- `ONNX_CONFIDENCE_THRESH = 0.65`
- `ONNX_REVIEW_THRESH = 0.50`

---

## Training

Run on **Mac Mini M4** (Apple Silicon, MPS backend).

```bash
make train
# → train/train_pytorch.py
# → models/weights/activity_v1.onnx
```

Training uses MPS → CUDA → CPU (best available device selected automatically).

Input normalisation: keypoints normalised to `[0, 1]` by frame width/height before training.

---

## ONNX Deployment to Pi

```bash
make deploy PI=pi@192.168.1.x
# scp models/weights/activity_v1.onnx → Pi
# restarts xlf-overseer systemd service
```

Pi runs inference via `onnxruntime` — no PyTorch needed on Pi.
Inference latency: ~5ms on Pi 5.

---

## Fallback

If `ONNX_MODEL_PATH` is blank in `pi/config.py`, the system uses **rule-based angle counting** instead.
Rule-based measures elbow/shoulder angle from keypoints directly.
Safe to use until LSTM is trained.

---

## Version Registry

All deployed model versions tracked in `models/registry.json`.

---

## Related

- [[System/Activity Classes]] — the 8 output classes
- [[System/Review Loop]] — how uncertain predictions improve the model
- [[Data/Training Requirements]] — data needed before training
- [[System/YOLO Pipeline]] — produces the input keypoints
