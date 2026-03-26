#!/usr/bin/env python3
"""
XL Fitness Overseer — Google Drive Training Data Downloader

Downloads all video+label pairs from the XL Fitness Google Drive folder.
Pairs are matched by filename stem:
  lat_pulldown_01_20240101_120000.mp4
  lat_pulldown_01_20240101_120000_labels.json

Usage:
    python3 train/download_training_data.py
    python3 train/download_training_data.py --folder-id 1KNDC4wctZqVt8s41U4ALWHJ45OM5U9FA
    python3 train/download_training_data.py --dest /custom/path/
"""

import argparse
import os
import sys
from pathlib import Path

DRIVE_FOLDER_ID = "1KNDC4wctZqVt8s41U4ALWHJ45OM5U9FA"
DEFAULT_DEST = "/Users/donkeybot01/Desktop/Training Data/sessions/"


def download_folder(folder_id: str, dest: str) -> list[tuple[str, str]]:
    """Download Google Drive folder contents and return paired (video, labels) paths."""
    try:
        import gdown
    except ImportError:
        print("ERROR: gdown is not installed.")
        print("Run: pip3 install gdown")
        sys.exit(1)

    os.makedirs(dest, exist_ok=True)
    print(f"Destination: {dest}")
    print(f"Drive folder: https://drive.google.com/drive/folders/{folder_id}")
    print()

    url = f"https://drive.google.com/drive/folders/{folder_id}"
    try:
        gdown.download_folder(url, output=dest, quiet=False, use_cookies=False)
    except Exception as e:
        print(f"\nERROR: Failed to download from Google Drive: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure the folder is shared publicly (Anyone with link → Viewer)")
        print("  2. Try: gdown --folder https://drive.google.com/drive/folders/<id>")
        print("  3. If hitting rate limits, wait and retry")
        sys.exit(1)

    # Find and pair files
    dest_path = Path(dest)
    videos = sorted(dest_path.rglob("*.mp4"))
    label_files = {f.stem: f for f in dest_path.rglob("*_labels.json")}

    pairs = []
    print("\nPairing files:")
    for video in videos:
        label_stem = f"{video.stem}_labels"
        if label_stem in label_files:
            pairs.append((str(video), str(label_files[label_stem])))
            print(f"  ✓ {video.name}")
            print(f"    ↔ {label_files[label_stem].name}")
        else:
            print(f"  ✗ {video.name} — no matching labels file, skipping")

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Download XL Fitness training data from Google Drive"
    )
    parser.add_argument(
        "--folder-id",
        default=DRIVE_FOLDER_ID,
        help=f"Google Drive folder ID (default: {DRIVE_FOLDER_ID})",
    )
    parser.add_argument(
        "--dest",
        default=DEFAULT_DEST,
        help=f"Local destination directory (default: {DEFAULT_DEST})",
    )
    args = parser.parse_args()

    print("XL Fitness — Training Data Downloader")
    print("=" * 50)

    pairs = download_folder(args.folder_id, args.dest)

    print(f"\n{'='*50}")
    if pairs:
        print(f"Downloaded {len(pairs)} session(s):")
        for video, labels in pairs:
            print(f"  Video:  {video}")
            print(f"  Labels: {labels}")
        print(f"\nReady to train! Run:")
        print(f"  python3 train/overseer_train.py \\")
        print(f"    --video '{pairs[0][0]}' \\")
        print(f"    --labels '{pairs[0][1]}' \\")
        print(f"    --output models/overseer_v1.onnx")
        print(f"\nOr train on all sessions from Drive at once:")
        print(f"  python3 train/overseer_train.py \\")
        print(f"    --drive-folder {args.folder_id} \\")
        print(f"    --output models/overseer_v1.onnx")
    else:
        print("No complete video+label pairs found.")
        print("Make sure both .mp4 and _labels.json files are in the Drive folder.")


if __name__ == "__main__":
    main()
