#!/usr/bin/env python3
"""
Synthetic pipeline test — no real video needed.
Creates dummy keypoint data and labels, trains LSTM, exports ONNX, runs inference.
"""
import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
NUM_CLASSES = 7
WINDOW_FRAMES = 30
FEATURE_DIM = 51   # 17 kps × 3
EPOCHS = 20
BATCH = 8
NUM_SAMPLES_PER_CLASS = 10

print("XL Fitness Overseer — Synthetic Pipeline Test")
print("=" * 50)

# ── 1. Generate synthetic keypoints ──────────────────────────────────────────
print("\n[1] Generating synthetic keypoints...")
rng = np.random.default_rng(42)

# 1000 frames of random "video" keypoints
total_frames = 1000
fake_keypoints = rng.random((total_frames, FEATURE_DIM)).astype(np.float32)
fps = 30.0
print(f"  Generated {total_frames} frames @ {fps} fps")

# ── 2. Generate synthetic labels JSON ────────────────────────────────────────
print("\n[2] Building synthetic labels...")
labels = []
for cls in range(1, NUM_CLASSES + 1):
    for i in range(NUM_SAMPLES_PER_CLASS):
        # Spread samples across the video, avoiding edges
        t = float(rng.integers(15, total_frames - 15)) / fps
        labels.append({"time": round(t, 2), "class": cls, "label": f"Class {cls}"})

label_data = {"labels": labels}
print(f"  Generated {len(labels)} label events ({NUM_SAMPLES_PER_CLASS} per class)")

# ── 3. Build training windows ─────────────────────────────────────────────────
print("\n[3] Building training windows...")
# Import from overseer_train
from overseer_train import build_training_samples, WINDOW_FRAMES as WF, FEATURE_DIM as FD

with tempfile.NamedTemporaryFile(mode='w', suffix='_labels.json', delete=False) as f:
    json.dump(label_data, f)
    tmp_labels = f.name

X, y = build_training_samples(fake_keypoints, fps, tmp_labels)
os.unlink(tmp_labels)
print(f"  X shape: {X.shape}  y shape: {y.shape}")
assert X.shape == (len(labels), WINDOW_FRAMES, FEATURE_DIM), f"Unexpected shape: {X.shape}"
assert y.min() == 0 and y.max() == NUM_CLASSES - 1, "Class range error"
print("  ✓ Window shapes correct")

# ── 4. Train ──────────────────────────────────────────────────────────────────
print("\n[4] Training LSTM...")
from overseer_train import train

model, accuracy = train(X, y, epochs=EPOCHS, batch_size=BATCH)
print(f"  Final accuracy: {accuracy*100:.1f}%")

# ── 5. Export ONNX ────────────────────────────────────────────────────────────
print("\n[5] Exporting ONNX...")
output_path = "/tmp/overseer_synthetic_test.onnx"
from overseer_train import export_onnx
export_onnx(model, output_path)
assert os.path.exists(output_path), "ONNX file not created"
print(f"  ✓ File size: {os.path.getsize(output_path)/1024:.1f} KB")

# ── 6. Inference test ─────────────────────────────────────────────────────────
print("\n[6] Running ONNX inference...")
import onnxruntime as ort
sess = ort.InferenceSession(output_path)
dummy = rng.random((1, WINDOW_FRAMES, FEATURE_DIM)).astype(np.float32)
out = sess.run(None, {"keypoints": dummy})
logits = out[0]  # shape (1, 7)
pred_class = logits.argmax(axis=1)[0]
print(f"  Input shape:  {dummy.shape}")
print(f"  Output shape: {logits.shape}")
print(f"  Predicted class (0-indexed): {pred_class}")
print("  ✓ Inference works")

print("\n" + "=" * 50)
print("ALL TESTS PASSED ✓")
print("The training pipeline is fully functional.")
print("Ready for real annotated footage from Google Drive.")
print("=" * 50)
