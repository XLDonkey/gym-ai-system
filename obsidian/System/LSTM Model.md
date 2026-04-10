---
title: LSTM Model
tags: [system, lstm, model, pytorch, onnx]
created: 2026-04-09
updated: 2026-04-10
---

# LSTM Activity Classifier

The neural network for [[Projects/Rep Tracking]]. Classifies 30 frames of YOLO keypoints into one of 8 [[System/Activity Classes]].

---

## Architecture

```
Input:  (batch, 30, 51)
        30 frames × 51 features
        17 keypoints × [x, y, confidence] per frame
        x, y normalised 0–1 by frame width/height

LSTM:   input_size=51
        hidden_size=128
        num_layers=1
        batch_first=True

Dropout: 0.3 (between LSTM and Linear)

Linear:  128 → 64
ReLU
Linear:  64 → 8

Phase gate: invalid class logits → -∞
Softmax → one class wins
```

One second of movement (30 frames at 30fps) = one prediction.

---

## Input Features (51 per frame)

17 COCO keypoints × 3 values each:

| Feature | Value |
|---------|-------|
| x | Normalised horizontal position (0–1 of frame width) |
| y | Normalised vertical position (0–1 of frame height) |
| conf | YOLO keypoint confidence (0–1) |

Zero-pad missing keypoints (YOLO didn't detect them) with `[0, 0, 0]`.

---

## Phase Masking (Softmax Gate)

```python
# In ONNXActivityClassifier / ActivityStateMachine:
if phase == Phase.IDLE:
    logits[3:]  = float('-inf')   # mask rep classes
    logits       = softmax(logits)
    # Classes 3–7 literally cannot be predicted

if phase == Phase.ENGAGED:
    logits[0:2] = float('-inf')   # could optionally mask idle classes
    logits       = softmax(logits)
```

This isn't just "low probability" — the masked classes are **impossible** in that phase.

---

## Confidence Thresholds

| Constant | Value | Behaviour |
|----------|-------|-----------|
| `ONNX_CONFIDENCE_THRESH` | 0.65 | Below = uncertain, accept anyway |
| `ONNX_REVIEW_THRESH` | 0.50 | Below = flag 30-frame clip to GitHub |

---

## Training Pipeline (Mac Mini M4)

```bash
# 1. Pull recordings from Pi
make sync PI=pi@192.168.1.x

# 2. Annotate videos
make annotate
# Opens pose/label.html → draw time segments → pick class → export JSON

# 3. Extract sequences
make extract
# train/extract_sequences.py → data/processed/*.npy

# 4. Train on Apple Silicon (MPS backend)
make train
# train/train_pytorch.py → models/weights/activity_v1.onnx

# 5. Deploy to Pi
make deploy PI=pi@192.168.1.x
# scp ONNX + restart xlf-overseer systemd service
```

Device priority: **MPS (Apple Silicon) → CUDA → CPU**

---

## ONNX Export

```python
torch.onnx.export(
    model,
    dummy_input,               # (1, 30, 51)
    "models/weights/activity_v1.onnx",
    input_names=["keypoints"],
    output_names=["logits"],
    dynamic_axes={"keypoints": {0: "batch_size"}}
)
```

Pi runs inference with `onnxruntime`:
```python
session = ort.InferenceSession("activity_v1.onnx")
logits = session.run(["logits"], {"keypoints": window})[0]
# window shape: (1, 30, 51) float32
```

Inference latency: **~5ms** on Pi 5 (CPU). Negligible.

---

## Model Registry (`models/registry.json`)

All deployed model versions tracked here. Key fields per entry:
```json
{
  "version":       "v1.0-lstm",
  "onnx_file":     "models/weights/activity_v1.onnx",
  "trained_on":    "2026-04-09",
  "num_segments":  342,
  "accuracy":      0.91,
  "deployed_to":   ["xlf-pi-001"],
  "notes":         "First LSTM — 8-class, lat pulldown only"
}
```

`model_version` from registry is embedded in every set payload sent to Supabase.

---

## Rule-Based Fallback

If `ONNX_MODEL_PATH = ""` in `pi/config.py`, the Pi uses angle-based counting instead:
- Measures elbow → shoulder → wrist angle every frame
- Counts rep when: crosses ANGLE_BOTTOM (90°) then ANGLE_TOP (130°)
- No 8-class output — just rep counts

This is the **safe default** until training data is collected.

---

## Auto-Update on Pi

Pi crontab (`3am daily`):
```bash
git pull https://github.com/Matt-xlfitness/gym-ai-system.git main
# Picks up new ONNX file if make deploy was run from Mac Mini
# New model active on next session start
```

---

## Related

- [[System/Activity Classes]] — the 8 output classes
- [[System/YOLO Pipeline]] — produces the 51-feature input
- [[System/Engagement Detector]] — supplies phase state for masking
- [[System/Review Loop]] — feeds uncertain clips back for retraining
- [[Projects/Rep Tracking]] — the project this model serves
- [[Data/Training Requirements]] — 300+ segments needed
- [[Hardware/Machine Pi]] — ONNX runs here at ~5ms
