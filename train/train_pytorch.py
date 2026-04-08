#!/usr/bin/env python3
"""
XL Fitness AI — Mac Mini Training Pipeline
Trains the LSTM activity classifier from annotation JSONs + keypoint sequences,
then exports to ONNX for Pi deployment.

Run this on the Mac Mini (Apple Silicon — uses MPS backend automatically).

Usage:
    # Train from pre-extracted sequences (fastest):
    python3 train/train_pytorch.py \
        --sequences data/processed/ \
        --output models/weights/activity_v1.onnx

    # Train from raw videos + annotation JSONs (auto-extracts keypoints):
    python3 train/train_pytorch.py \
        --annotations data/annotations/ \
        --videos data/raw/ \
        --output models/weights/activity_v1.onnx

    # Review flagged clips from Pi before training:
    python3 train/train_pytorch.py \
        --review data/review/ \
        --sequences data/processed/ \
        --output models/weights/activity_v2.onnx

The script updates models/registry.json automatically after training.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Schema ────────────────────────────────────────────────────────────────────
# Must match pi/onnx_classifier.py, pose/label.html, and data/annotations/ JSONs

CLASS_NAMES = [
    "no_person",    # 0
    "user_present", # 1
    "on_machine",   # 2
    "good_rep",     # 3
    "bad_rep",      # 4
    "false_rep",    # 5
    "resting",      # 6
    "half_rep",     # 7
]
NUM_CLASSES     = len(CLASS_NAMES)
WINDOW_FRAMES   = 30   # LSTM sequence length
KEYPOINTS       = 17   # COCO YOLOv11-pose
FEATURES_PER_KP = 3    # x, y, confidence
FEATURE_DIM     = KEYPOINTS * FEATURES_PER_KP  # 51

REPO_ROOT = Path(__file__).parent.parent


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(hidden: int = 128, dense: int = 64, dropout: float = 0.3):
    """
    Lightweight LSTM for on-device gym activity classification.

    Architecture:
        Input  →  LSTM(51→128)  →  Dropout(0.3)  →  Linear(128→64)
               →  ReLU  →  Linear(64→8)  →  [softmax at inference]

    The final linear outputs raw logits. CrossEntropyLoss applies softmax
    during training. At inference, we apply softmax manually — this IS
    the gate: probabilities sum to 1, argmax picks exactly one class.

    Input shape:  (batch, 30, 51)
    Output shape: (batch, 8)    ← 8 mutually exclusive activity classes
    """
    import torch.nn as nn

    class OverseerLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm    = nn.LSTM(FEATURE_DIM, hidden, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.fc1     = nn.Linear(hidden, dense)
            self.relu    = nn.ReLU()
            self.fc2     = nn.Linear(dense, NUM_CLASSES)

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            h = h.squeeze(0)      # (batch, hidden)
            h = self.dropout(h)
            h = self.relu(self.fc1(h))
            return self.fc2(h)    # logits (batch, 8)

    return OverseerLSTM()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sequences(seq_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load pre-extracted .npy sequence files from data/processed/.
    Each file: {stem}_X.npy  (N, 30, 51) and {stem}_y.npy  (N,)
    """
    seq_path = Path(seq_dir)
    X_all, y_all = [], []

    for x_file in sorted(seq_path.glob("*_X.npy")):
        y_file = x_file.parent / x_file.name.replace("_X.npy", "_y.npy")
        if not y_file.exists():
            print(f"  WARNING: No matching y file for {x_file.name}, skipping")
            continue
        X = np.load(str(x_file))
        y = np.load(str(y_file))
        X_all.append(X)
        y_all.append(y)
        print(f"  Loaded: {x_file.name}  {X.shape[0]} samples")

    if not X_all:
        print(f"ERROR: No sequence files found in {seq_dir}")
        print("  Run train/extract_sequences.py first, or use --annotations + --videos")
        sys.exit(1)

    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)


def load_review_clips(review_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load human-reviewed clips from data/review/.
    Only includes clips where true_class has been set by an annotator.
    """
    review_path = Path(review_dir)
    X_all, y_all = [], []

    for meta_file in sorted(review_path.rglob("*_meta.json")):
        with open(meta_file) as f:
            meta = json.load(f)

        if meta.get("true_class") is None:
            continue  # skip unreviewed clips

        npy_file = meta_file.parent / meta_file.name.replace("_meta.json", "_kps.npy")
        if not npy_file.exists():
            continue

        window = np.load(str(npy_file))  # (30, 51)
        X_all.append(window[np.newaxis])
        y_all.append(meta["true_class"])

    if X_all:
        X = np.concatenate(X_all, axis=0)
        y = np.array(y_all, dtype=np.int64)
        print(f"  Review clips: {len(X)} annotated samples from {review_dir}")
        return X, y

    print(f"  No reviewed clips found in {review_dir} (set true_class in *_meta.json to include them)")
    return np.empty((0, WINDOW_FRAMES, FEATURE_DIM), dtype=np.float32), np.empty(0, dtype=np.int64)


def load_annotations(annotations_dir: str, videos_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract keypoints from raw videos + annotation JSONs on-the-fly.
    Slower than loading pre-extracted sequences but works without extract_sequences.py.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print("Loading YOLOv11-pose for keypoint extraction...")
    yolo = YOLO("yolo11n-pose.pt")

    ann_path = Path(annotations_dir)
    vid_path = Path(videos_dir)
    X_all, y_all = [], []

    for ann_file in sorted(ann_path.glob("*.json")):
        with open(ann_file) as f:
            ann = json.load(f)

        video_name = Path(ann.get("video", "")).name
        video_file = vid_path / video_name
        if not video_file.exists():
            print(f"  WARNING: Video not found: {video_file}, skipping")
            continue

        print(f"\n  Processing: {ann_file.name} + {video_name}")
        X, y = _extract_from_pair(yolo, str(video_file), ann)
        X_all.append(X)
        y_all.append(y)

    if not X_all:
        print("ERROR: No video+annotation pairs found.")
        sys.exit(1)

    return np.concatenate(X_all), np.concatenate(y_all)


def _extract_from_pair(yolo, video_path: str, ann: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract keypoints from one video and build samples from its annotations."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Extract keypoints for every frame
    all_kps = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo(frame, verbose=False)
        kp_flat = np.zeros(FEATURE_DIM, dtype=np.float32)
        if results and results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            kp = results[0].keypoints.data[0].cpu().numpy()  # (17, 3)
            h, w = frame.shape[:2]
            kp[:, 0] /= w
            kp[:, 1] /= h
            kp_flat = kp.flatten()
        all_kps.append(kp_flat)
    cap.release()

    all_kps = np.array(all_kps, dtype=np.float32)  # (num_frames, 51)
    half    = WINDOW_FRAMES // 2
    X, y    = [], []

    for seg in ann.get("annotations", []):
        class_id = seg["class_id"]       # 0-indexed
        start_f  = int(seg["start_s"] * fps)
        end_f    = int(seg["end_s"]   * fps)
        center_f = (start_f + end_f) // 2

        window = np.zeros((WINDOW_FRAMES, FEATURE_DIM), dtype=np.float32)
        for i, fi in enumerate(range(center_f - half, center_f - half + WINDOW_FRAMES)):
            if 0 <= fi < len(all_kps):
                window[i] = all_kps[fi]

        X.append(window)
        y.append(class_id)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 1e-3,
    label_smoothing: float = 0.1,
) -> tuple:
    """
    Train the LSTM.

    label_smoothing reduces overconfidence on noisy gym annotations.
    Returns (model, best_val_accuracy).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
        sys.exit(1)

    print(f"\nDataset: {len(X)} samples  |  {NUM_CLASSES} classes")
    counts = np.bincount(y, minlength=NUM_CLASSES)
    for i, name in enumerate(CLASS_NAMES):
        bar = "█" * counts[i]
        print(f"  [{i}] {name:<14} {counts[i]:4d}  {bar}")

    if len(X) < 8:
        print("\nERROR: Too few samples. Need at least 8 (ideally 300+).")
        print("       Record more footage, annotate in pose/label.html, then retrain.")
        sys.exit(1)

    # Stratify only if every class has at least 2 samples
    min_count = min(counts[counts > 0])
    stratify  = y if min_count >= 2 else None

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    # Select device: MPS for Mac Mini Apple Silicon, CUDA for Linux, CPU fallback
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\nDevice: {device}")

    model     = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    X_tr_t  = torch.FloatTensor(X_tr)
    y_tr_t  = torch.LongTensor(y_tr)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)

    loader   = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)

    best_val_acc = 0.0
    best_state   = None

    print(f"\nTraining {epochs} epochs (batch={batch_size}, lr={lr}, label_smooth={label_smoothing})")
    print("─" * 55)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            preds   = model(X_val_t).argmax(dim=1)
            val_acc = (preds == y_val_t).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = total_loss / len(loader)
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.3f}"
                  f"  best={best_val_acc:.3f}")

    print("─" * 55)
    print(f"Best validation accuracy: {best_val_acc*100:.1f}%")

    if best_state:
        model.load_state_dict(best_state)

    return model, best_val_acc


# ── ONNX export ───────────────────────────────────────────────────────────────

def export_onnx(model, output_path: str) -> None:
    """Export the trained PyTorch model to ONNX for Pi deployment."""
    import torch

    model = model.cpu().eval()
    os.makedirs(Path(output_path).parent, exist_ok=True)

    dummy = torch.zeros(1, WINDOW_FRAMES, FEATURE_DIM)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names  = ["keypoints"],
        output_names = ["class_logits"],
        dynamic_axes = {
            "keypoints":    {0: "batch"},
            "class_logits": {0: "batch"},
        },
        opset_version = 17,
        do_constant_folding = True,
    )
    print(f"\nONNX model saved: {output_path}")

    # Verify the exported model
    try:
        import onnx
        import onnxruntime as ort
        m = onnx.load(output_path)
        onnx.checker.check_model(m)

        # Quick inference test
        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        test_in = np.zeros((1, WINDOW_FRAMES, FEATURE_DIM), dtype=np.float32)
        out = sess.run(None, {"keypoints": test_in})[0]
        assert out.shape == (1, NUM_CLASSES), f"Bad output shape: {out.shape}"
        print("ONNX verified ✓  inference test passed")
    except Exception as e:
        print(f"WARNING: ONNX verification failed: {e}")

    # Metadata file (deployed alongside the .onnx)
    meta_path = output_path.replace(".onnx", "_meta.json")
    meta = {
        "class_names":   CLASS_NAMES,
        "num_classes":   NUM_CLASSES,
        "window_frames": WINDOW_FRAMES,
        "feature_dim":   FEATURE_DIM,
        "input_shape":   [1, WINDOW_FRAMES, FEATURE_DIM],
        "model":         "OverseerLSTM",
        "exported":      datetime.utcnow().isoformat() + "Z",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata: {meta_path}")


# ── Registry update ───────────────────────────────────────────────────────────

def update_registry(output_path: str, accuracy: float, num_samples: int) -> None:
    """Add the new model to models/registry.json."""
    registry_path = REPO_ROOT / "models" / "registry.json"

    try:
        with open(registry_path) as f:
            registry = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        registry = {"models": [], "active_version": None}

    version_str = f"v{len(registry['models'])}.0-lstm"
    model_name  = Path(output_path).name

    new_entry = {
        "version":          version_str,
        "type":             "lstm",
        "description":      f"LSTM activity classifier — {NUM_CLASSES} classes",
        "deployed":         False,
        "accuracy":         round(accuracy, 4),
        "training_samples": num_samples,
        "trained_at":       datetime.utcnow().isoformat() + "Z",
        "onnx_file":        model_name,
        "meta_file":        model_name.replace(".onnx", "_meta.json"),
        "class_names":      CLASS_NAMES,
        "drive_folder":     None,  # update when weights are uploaded to Drive
        "notes":            "",
    }

    registry["models"].append(new_entry)
    print(f"\nRegistry updated: {version_str}  acc={accuracy*100:.1f}%")
    print(f"  → Set 'deployed: true' and 'drive_folder' before pushing to Pi")

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="XL Fitness — Mac Mini LSTM Training Pipeline"
    )
    parser.add_argument(
        "--sequences", default=str(REPO_ROOT / "data" / "processed"),
        help="Directory of pre-extracted *_X.npy + *_y.npy files (default: data/processed/)",
    )
    parser.add_argument(
        "--annotations", default=None,
        help="Directory of annotation JSONs from pose/label.html (overrides --sequences)",
    )
    parser.add_argument(
        "--videos", default=str(REPO_ROOT / "data" / "raw"),
        help="Directory of raw .mp4 files (used with --annotations)",
    )
    parser.add_argument(
        "--review", default=str(REPO_ROOT / "data" / "review"),
        help="Directory of Pi-flagged review clips to include in training",
    )
    parser.add_argument(
        "--output", default=str(REPO_ROOT / "models" / "weights" / "activity_v1.onnx"),
        help="Output ONNX path",
    )
    parser.add_argument("--epochs",           type=int,   default=60)
    parser.add_argument("--batch-size",       type=int,   default=16)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--label-smoothing",  type=float, default=0.1)
    parser.add_argument("--no-review",        action="store_true",
                        help="Skip loading review clips from --review dir")
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  XL Fitness AI — Mac Mini Training Pipeline")
    print("═" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.annotations:
        print(f"\nLoading from annotations: {args.annotations}")
        X, y = load_annotations(args.annotations, args.videos)
    else:
        print(f"\nLoading pre-extracted sequences: {args.sequences}")
        X, y = load_sequences(args.sequences)

    # Augment with human-reviewed Pi review clips
    if not args.no_review and Path(args.review).exists():
        print(f"\nChecking review clips: {args.review}")
        X_rev, y_rev = load_review_clips(args.review)
        if len(X_rev):
            X = np.concatenate([X, X_rev], axis=0)
            y = np.concatenate([y, y_rev], axis=0)
            print(f"  Dataset after review augmentation: {len(X)} samples")

    # ── Train ─────────────────────────────────────────────────────────────────
    model, accuracy = train(
        X, y,
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        lr              = args.lr,
        label_smoothing = args.label_smoothing,
    )

    # ── Export ────────────────────────────────────────────────────────────────
    export_onnx(model, args.output)
    update_registry(args.output, accuracy, len(X))

    print("\n" + "═" * 60)
    print(f"  Training complete!")
    print(f"  Samples:  {len(X)}")
    print(f"  Accuracy: {accuracy*100:.1f}%")
    print(f"  ONNX:     {args.output}")
    print("═" * 60)
    print("\nNext steps:")
    print("  1. Upload the .onnx to Google Drive and update models/registry.json")
    print("  2. scp the .onnx to the Pi:  models/weights/activity_v1.onnx")
    print("  3. Set ONNX_MODEL_PATH in pi/config.py")
    print("  4. Restart pi/main.py — it will load the ONNX automatically")
    print()


if __name__ == "__main__":
    main()
