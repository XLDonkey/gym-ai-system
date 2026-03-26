#!/usr/bin/env python3
"""
XL Fitness Overseer — Training Pipeline
Trains a lightweight LSTM on YOLOv11-pose keypoint sequences extracted
from annotated gym footage, then exports to ONNX for Pi deployment.

Usage:
    # From local files:
    python3 train/overseer_train.py \
        --video path/to/video.mp4 \
        --labels path/to/labels.json \
        --output models/overseer_v1.onnx

    # From Google Drive folder (auto-downloads all pairs):
    python3 train/overseer_train.py \
        --drive-folder 1KNDC4wctZqVt8s41U4ALWHJ45OM5U9FA \
        --output models/overseer_v1.onnx
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

# ── Class definitions ─────────────────────────────────────────────────────────
# Classes are 1-indexed in labels JSON to match human-readable labels.
# Internally we use 0-indexed for training.

CLASS_NAMES = [
    "No User",        # class 1 → index 0
    "User Passing",   # class 2 → index 1
    "User Present",   # class 3 → index 2
    "Good Rep",       # class 4 → index 3
    "Poor Rep",       # class 5 → index 4
    "False Rep / Other",  # class 6 → index 5
    "Resting",        # class 7 → index 6
]
NUM_CLASSES = len(CLASS_NAMES)
WINDOW_FRAMES = 30       # frames in each training window
KEYPOINTS = 17           # YOLOv11-pose COCO keypoints
FEATURES_PER_KP = 3      # x, y, confidence
FEATURE_DIM = KEYPOINTS * FEATURES_PER_KP  # 51

# ── Google Drive downloader ───────────────────────────────────────────────────

def download_from_drive(folder_id: str, dest_dir: str) -> list[tuple[str, str]]:
    """
    Downloads all video+label pairs from a Google Drive folder.
    Returns list of (video_path, labels_path) tuples.
    """
    try:
        import gdown
    except ImportError:
        print("ERROR: gdown not installed. Run: pip3 install gdown")
        sys.exit(1)

    os.makedirs(dest_dir, exist_ok=True)
    print(f"Fetching file list from Google Drive folder: {folder_id}")

    # List all files in the folder
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    try:
        gdown.download_folder(url, output=dest_dir, quiet=False, use_cookies=False)
    except Exception as e:
        print(f"ERROR downloading from Drive: {e}")
        sys.exit(1)

    # Find and pair video/label files
    pairs = _pair_files(dest_dir)
    if not pairs:
        print(f"ERROR: No video+label pairs found in {dest_dir}")
        sys.exit(1)
    return pairs


def _pair_files(directory: str) -> list[tuple[str, str]]:
    """
    Pairs .mp4 files with their _labels.json counterpart by stem.
    e.g. lat_pulldown_01_20240101_120000.mp4
      ↔  lat_pulldown_01_20240101_120000_labels.json
    """
    dir_path = Path(directory)
    videos = {f.stem: f for f in dir_path.rglob("*.mp4")}
    labels = {f.stem: f for f in dir_path.rglob("*_labels.json")}

    pairs = []
    for video_stem, video_path in videos.items():
        label_stem = f"{video_stem}_labels"
        if label_stem in labels:
            pairs.append((str(video_path), str(labels[label_stem])))
            print(f"  Paired: {video_path.name} ↔ {labels[label_stem].name}")
        else:
            print(f"  WARNING: No labels found for {video_path.name}, skipping")

    return pairs


# ── Keypoint extraction ───────────────────────────────────────────────────────

def extract_keypoints_from_video(video_path: str) -> tuple[np.ndarray, float]:
    """
    Runs YOLOv11-pose on every frame and extracts 17 keypoints per frame.
    Returns:
        keypoints: float32 array of shape (num_frames, FEATURE_DIM)
        fps: video frame rate
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip3 install ultralytics")
        sys.exit(1)

    print(f"\nLoading YOLOv11-pose model...")
    model = YOLO("yolo11n-pose.pt")  # auto-downloads on first run

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames} frames @ {fps:.2f} fps — {total_frames/fps:.1f}s")

    all_keypoints = []
    frame_idx = 0

    print("Extracting keypoints (this may take a few minutes)...")
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        # Take the highest-confidence person detection
        kp_flat = np.zeros(FEATURE_DIM, dtype=np.float32)
        if results and results[0].keypoints is not None:
            kps = results[0].keypoints
            if kps.data is not None and len(kps.data) > 0:
                # kps.data shape: (num_people, 17, 3) — x, y, conf
                # Pick person with highest mean keypoint confidence
                best_person = 0
                if len(kps.data) > 1:
                    confs = kps.data[:, :, 2].mean(dim=1)
                    best_person = int(confs.argmax())
                kp_data = kps.data[best_person].cpu().numpy()  # (17, 3)
                # Normalise x,y by frame dimensions
                h, w = frame.shape[:2]
                kp_data[:, 0] /= w
                kp_data[:, 1] /= h
                kp_flat = kp_data.flatten()

        all_keypoints.append(kp_flat)
        frame_idx += 1

        if frame_idx % 100 == 0:
            elapsed = time.time() - start
            rate = frame_idx / elapsed
            eta = (total_frames - frame_idx) / rate if rate > 0 else 0
            print(f"  Frame {frame_idx}/{total_frames} — {rate:.1f} fps — ETA {eta:.0f}s")

    cap.release()
    elapsed = time.time() - start
    print(f"Done: {frame_idx} frames in {elapsed:.1f}s")

    return np.array(all_keypoints, dtype=np.float32), fps


# ── Window extraction ─────────────────────────────────────────────────────────

def build_training_samples(
    keypoints: np.ndarray,
    fps: float,
    labels_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each labelled timestamp, extract a 30-frame window centred on it.
    Returns:
        X: float32 array of shape (num_samples, WINDOW_FRAMES, FEATURE_DIM)
        y: int array of shape (num_samples,) with 0-indexed class labels
    """
    with open(labels_path) as f:
        label_data = json.load(f)

    labels = label_data["labels"]
    num_frames = len(keypoints)
    half = WINDOW_FRAMES // 2

    X, y = [], []

    for entry in labels:
        t = entry["time"]
        cls = entry["class"]  # 1-indexed
        label_name = entry.get("label", "")

        # Convert timestamp → frame index
        center_frame = int(round(t * fps))

        # Extract window, pad with zeros at boundaries
        start_frame = center_frame - half
        end_frame = start_frame + WINDOW_FRAMES

        window = np.zeros((WINDOW_FRAMES, FEATURE_DIM), dtype=np.float32)
        for i, fi in enumerate(range(start_frame, end_frame)):
            if 0 <= fi < num_frames:
                window[i] = keypoints[fi]

        X.append(window)
        y.append(cls - 1)  # convert to 0-indexed
        print(f"  Sample: t={t:.1f}s frame={center_frame} class={cls} ({label_name})")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model():
    """Build the LSTM overseer model in PyTorch."""
    import torch
    import torch.nn as nn

    class OverseerLSTM(nn.Module):
        def __init__(self, input_dim=FEATURE_DIM, hidden=128, dropout=0.3, dense=64, num_classes=NUM_CLASSES):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden, dense)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(dense, num_classes)

        def forward(self, x):
            # x: (batch, seq_len, input_dim)
            out, (h, _) = self.lstm(x)
            # Use last hidden state
            h = h.squeeze(0)  # (batch, hidden)
            h = self.dropout(h)
            h = self.relu(self.fc1(h))
            return self.fc2(h)  # logits

    return OverseerLSTM()


# ── Training ──────────────────────────────────────────────────────────────────

def train(X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 16) -> tuple:
    """
    Train the LSTM model.
    Returns: (model, final_accuracy)
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split

    print(f"\nTraining dataset: {len(X)} samples, {NUM_CLASSES} classes")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Train/val split
    if len(X) >= 4:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
    else:
        print("WARNING: Very few samples — using all data for training (no val split)")
        X_train, X_val, y_train, y_val = X, X, y, y

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t.to(device))
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_acc = (val_preds == y_val).mean()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch:3d}/{epochs} — loss={avg_loss:.4f}  val_acc={val_acc:.3f}")

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    print(f"\nBest validation accuracy: {best_val_acc:.3f} ({best_val_acc*100:.1f}%)")
    return model, best_val_acc


# ── ONNX export ───────────────────────────────────────────────────────────────

def export_onnx(model, output_path: str):
    """Export the trained model to ONNX format."""
    import torch

    # ONNX export must run on CPU
    model = model.cpu()
    model.eval()
    os.makedirs(Path(output_path).parent, exist_ok=True)

    # Dummy input: batch=1, seq=30, features=51
    dummy = torch.zeros(1, WINDOW_FRAMES, FEATURE_DIM)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["keypoints"],
        output_names=["class_logits"],
        dynamic_axes={"keypoints": {0: "batch"}, "class_logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"\nModel exported to ONNX: {output_path}")

    # Verify
    try:
        import onnx
        m = onnx.load(output_path)
        onnx.checker.check_model(m)
        print("ONNX model verified ✓")
    except Exception as e:
        print(f"WARNING: ONNX verification failed: {e}")

    # Save metadata alongside the model
    meta_path = output_path.replace(".onnx", "_meta.json")
    meta = {
        "class_names": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "window_frames": WINDOW_FRAMES,
        "keypoints": KEYPOINTS,
        "feature_dim": FEATURE_DIM,
        "input_shape": [1, WINDOW_FRAMES, FEATURE_DIM],
        "model": "OverseerLSTM",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved: {meta_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="XL Fitness Overseer — Training Pipeline")
    parser.add_argument("--video", help="Path to input video file (.mp4)")
    parser.add_argument("--labels", help="Path to labels JSON file")
    parser.add_argument("--drive-folder", help="Google Drive folder ID to auto-download training data")
    parser.add_argument("--drive-dest", default="/Users/donkeybot01/Desktop/Training Data/sessions/",
                        help="Local directory to download Drive files into")
    parser.add_argument("--output", default="models/overseer_v1.onnx", help="Output ONNX model path")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    if not args.drive_folder and (not args.video or not args.labels):
        parser.error("Provide either --drive-folder OR both --video and --labels")

    # ── Collect training pairs ──
    pairs = []
    if args.drive_folder:
        print(f"Downloading from Google Drive folder: {args.drive_folder}")
        pairs = download_from_drive(args.drive_folder, args.drive_dest)
    else:
        pairs = [(args.video, args.labels)]

    # ── Build dataset across all pairs ──
    all_X, all_y = [], []

    for video_path, labels_path in pairs:
        print(f"\n{'='*60}")
        print(f"Processing: {Path(video_path).name}")
        print(f"Labels:     {Path(labels_path).name}")
        print(f"{'='*60}")

        keypoints, fps = extract_keypoints_from_video(video_path)
        print(f"\nBuilding training windows from labels...")
        X, y = build_training_samples(keypoints, fps, labels_path)
        all_X.append(X)
        all_y.append(y)

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    print(f"\nTotal training samples: {len(X_all)}")

    # ── Train ──
    model, accuracy = train(X_all, y_all, epochs=args.epochs, batch_size=args.batch_size)

    # ── Export ──
    export_onnx(model, args.output)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Samples:  {len(X_all)}")
    print(f"  Accuracy: {accuracy*100:.1f}%")
    print(f"  Model:    {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
