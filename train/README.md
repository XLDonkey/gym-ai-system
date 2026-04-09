# train — Rep Tracking Training Pipeline (Mac Mini)

Runs on the Mac Mini M4 (Apple Silicon MPS backend). Never runs on the Pi.

| File | Purpose |
|------|---------|
| `train_pytorch.py` | Train the LSTM activity classifier → export ONNX |
| `extract_sequences.py` | Convert annotated videos + JSON labels → `.npy` keypoint sequences |
| `auto_label_yt.py` | Auto-label a YouTube video using Claude Opus (saves annotation JSON) |
| `test_synthetic.py` | Synthetic data tests — run with `make test` |

## Workflow

```
1. Film sessions at a machine → sync to Mac Mini (make sync PI=pi@IP)
2. Open pose/label.html → load video → draw segments → Export JSON → data/annotations/
3. make train   (extracts sequences + trains LSTM + exports ONNX)
4. make deploy PI=pi@IP
```

## Model architecture

```
Input:  (batch, 30, 51)  — 30 frames × 51 features (17 keypoints × 3)
LSTM:   51 → 128 hidden units
Dropout: 0.3
Linear: 128 → 64 → ReLU
Linear: 64 → 8
Output: (batch, 8) logits → softmax via ActivityStateMachine
```

## Data requirements

| Class | Min samples |
|-------|-------------|
| no_person | 30 |
| user_present | 30 |
| on_machine | 30 |
| good_rep | 30 |
| bad_rep | 30 |
| false_rep | 30 |
| resting | 30 |
| half_rep | 30 |
| **Total** | **300+** |

Run `make stats` to see current annotation counts per class.
