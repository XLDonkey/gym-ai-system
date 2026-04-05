"""
XL Fitness AI Overseer — YouTube Auto-Labeller
───────────────────────────────────────────────
A *separate* AI (Claude Opus via Anthropic API) watches exercise videos from
YouTube and generates rep labels in the same format used by the training
pipeline — no human annotation required.

How it works
────────────
1.  Download the YouTube video with yt-dlp
2.  Extract frames at 2 fps with OpenCV
3.  Send batches of frames (sliding 5-second windows) to Claude Opus vision
4.  Claude identifies exercise phases in each frame:
        3 = User Present / Setting Up
        4 = Good Rep  (controlled, full ROM)
        5 = Poor Rep  (bounced, partial, rushed)
        7 = Resting   (between sets)
5.  Aggregate Claude's responses into a labels.json file
6.  Optionally upload the video + labels to Google Drive for training

Usage
─────
    # Label a single YouTube video
    python3 train/auto_label_yt.py \\
        --url "https://www.youtube.com/watch?v=..." \\
        --machine "Nautilus Xpload Lat Pulldown"

    # Label multiple URLs from a text file (one URL per line)
    python3 train/auto_label_yt.py \\
        --url-file urls.txt \\
        --machine "Lat Pulldown" \\
        --upload-drive

    # Preview mode — just print Claude's narration, don't write files
    python3 train/auto_label_yt.py \\
        --url "https://..." \\
        --machine "Lat Pulldown" \\
        --dry-run

Requirements
────────────
    pip install anthropic yt-dlp opencv-python numpy

Environment
───────────
    ANTHROPIC_API_KEY   — your Anthropic API key (required)
    SUPABASE_URL        — optional, to write labels straight to the DB
    SUPABASE_SERVICE_KEY
"""

import os
import sys
import json
import base64
import argparse
import tempfile
import subprocess
import math
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic SDK not installed. Run: pip install anthropic")
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL          = "claude-opus-4-6"       # vision + adaptive thinking
SAMPLE_FPS     = 2                       # frames extracted per second from video
WINDOW_FRAMES  = 10                      # frames per API call  (~5 s of video)
WINDOW_STRIDE  = 6                       # step between windows (3 s overlap)
JPEG_QUALITY   = 75                      # lower = smaller payload, still readable
MAX_FRAME_DIM  = 640                     # resize frames to fit within this box

# Activity classes (must match train/overseer_train.py)
CLASS_LABELS = {
    1: "No User",
    2: "User Passing",
    3: "User Present",
    4: "Good Rep",
    5: "Poor Rep",
    6: "False Rep / Other",
    7: "Resting",
}

# ── System prompt Claude uses to understand the task ──────────────────────────

SYSTEM_PROMPT = """You are a gym exercise analyst specialising in machine-based
strength training. Your job is to watch short clips of exercise footage and
label each moment with the correct activity class.

Activity classes:
  1 = No User           — nobody at the machine
  2 = User Passing      — person walking past, not using machine
  3 = User Present      — person at machine but not actively exercising
                          (adjusting seat, reading the display, talking)
  4 = Good Rep          — one complete rep: full range-of-motion, controlled
                          tempo, clean form (not rushed, not bounced)
  5 = Poor Rep          — rep with partial ROM, bouncing the weight,
                          using momentum, or otherwise bad form
  6 = False Rep / Other — fidgeting, adjusting, half-started movement
  7 = Resting           — person sitting idle at the machine between sets

You will receive a numbered sequence of video frames. For each frame respond
with a JSON array of objects, one per frame:

[
  {"frame": 1, "time_s": 0.0, "class": 3, "label": "User Present",
   "note": "user adjusting seat"},
  {"frame": 2, "time_s": 0.5, "class": 4, "label": "Good Rep",
   "note": "pulling bar down, elbows flared"},
  ...
]

Rules:
- Output ONLY the JSON array — no preamble, no markdown fences.
- Use the exact class numbers above.
- "time_s" is the wall-clock timestamp I give you for each frame.
- "note" is a brief description (≤10 words) of what you see.
- If you cannot see a person clearly, use class 1 or 2.
- When a single rep spans multiple frames, label each frame with class 4 or 5.
  The first frame where the concentric phase starts = start of the rep.
"""


# ── Download ──────────────────────────────────────────────────────────────────

def download_video(url: str, out_dir: Path) -> Path:
    """Download a YouTube video with yt-dlp. Returns path to downloaded file."""
    print(f"[yt-dlp] Downloading: {url}")
    out_template = str(out_dir / "%(title)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]",
        "--merge-output-format", "mp4",
        "--output", out_template,
        "--quiet",
        "--no-playlist",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr}")

    mp4_files = list(out_dir.glob("*.mp4"))
    if not mp4_files:
        raise RuntimeError("yt-dlp ran but no .mp4 found in output directory.")
    return mp4_files[0]


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(video_path: Path, fps: int = SAMPLE_FPS) -> list[dict]:
    """
    Extract frames from video at `fps` frames per second.
    Returns list of {"index": int, "time_s": float, "jpeg_b64": str}.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(round(video_fps / fps)))
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[extract] {video_path.name}  "
          f"fps={video_fps:.1f}  total_frames={total}  "
          f"sampling every {frame_step} frames")

    frames = []
    frame_idx = 0
    sample_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            time_s = frame_idx / video_fps
            jpeg   = _encode_frame(frame)
            frames.append({
                "index":   sample_idx,
                "time_s":  round(time_s, 2),
                "jpeg_b64": jpeg,
            })
            sample_idx += 1

        frame_idx += 1

    cap.release()
    print(f"[extract] {len(frames)} frames extracted  "
          f"({len(frames)/fps:.1f}s of video at {fps}fps)")
    return frames


def _encode_frame(frame: np.ndarray) -> str:
    """Resize frame to fit MAX_FRAME_DIM and return base64 JPEG."""
    h, w = frame.shape[:2]
    scale = min(MAX_FRAME_DIM / max(h, w), 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return base64.b64encode(buf.tobytes()).decode()


# ── Claude vision API ─────────────────────────────────────────────────────────

def label_window(
    client:      anthropic.Anthropic,
    frames:      list[dict],
    machine_name: str,
    window_idx:  int,
) -> list[dict]:
    """
    Send a window of frames to Claude Opus and get per-frame labels back.
    Returns a list of label dicts matching the training data format.
    """
    # Build the user message: header text + one image block per frame
    content = [
        {
            "type": "text",
            "text": (
                f"Machine: {machine_name}\n"
                f"Window {window_idx + 1} — "
                f"{len(frames)} frames, "
                f"{frames[0]['time_s']:.1f}s – {frames[-1]['time_s']:.1f}s\n\n"
                "Label each frame:\n"
            ),
        }
    ]

    for f in frames:
        content.append({
            "type": "text",
            "text": f"Frame {f['index'] + 1}  (t={f['time_s']:.2f}s)",
        })
        content.append({
            "type": "image",
            "source": {
                "type":       "base64",
                "media_type": "image/jpeg",
                "data":       f["jpeg_b64"],
            },
        })

    # Use streaming + adaptive thinking for reliability on complex visual tasks
    labels = []
    raw_text = ""

    with client.messages.stream(
        model      = MODEL,
        max_tokens = 4096,
        thinking   = {"type": "adaptive"},
        system     = SYSTEM_PROMPT,
        messages   = [{"role": "user", "content": content}],
    ) as stream:
        final = stream.get_final_message()

    # Extract text blocks (skip thinking blocks)
    for block in final.content:
        if block.type == "text":
            raw_text += block.text

    # Parse JSON
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = "\n".join(raw_text.split("\n")[1:-1])

    try:
        parsed = json.loads(raw_text)
        if not isinstance(parsed, list):
            raise ValueError("Expected a JSON array")
        labels = parsed
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  [WARN] JSON parse failed for window {window_idx}: {e}")
        print(f"  Raw: {raw_text[:200]}")
        # Fallback: mark all frames as Unknown
        for f in frames:
            labels.append({
                "frame":   f["index"] + 1,
                "time_s":  f["time_s"],
                "class":   3,
                "label":   "User Present",
                "note":    "parse error — review manually",
            })

    return labels


# ── Label aggregation ─────────────────────────────────────────────────────────

def deduplicate_labels(all_labels: list[dict]) -> list[dict]:
    """
    Merge overlapping windows. When the same timestamp appears in multiple
    windows, take the most specific class (4 or 5 over 3 over 7 over 1).
    """
    CLASS_SPECIFICITY = {4: 5, 5: 5, 6: 4, 3: 3, 7: 2, 2: 1, 1: 0}

    seen: dict[float, dict] = {}
    for entry in all_labels:
        t = entry["time_s"]
        if t not in seen:
            seen[t] = entry
        else:
            existing_spec = CLASS_SPECIFICITY.get(seen[t]["class"], 0)
            new_spec      = CLASS_SPECIFICITY.get(entry["class"], 0)
            if new_spec > existing_spec:
                seen[t] = entry

    return sorted(seen.values(), key=lambda x: x["time_s"])


def to_training_labels(deduped: list[dict], machine_name: str) -> dict:
    """Convert deduplicated label list to the training data JSON format."""
    return {
        "machine":    machine_name,
        "source":     "auto_label_yt",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "labels": [
            {
                "time":  entry["time_s"],
                "class": entry["class"],
                "label": CLASS_LABELS.get(entry["class"], entry.get("label", "Unknown")),
                "note":  entry.get("note", ""),
            }
            for entry in deduped
        ],
    }


# ── Google Drive upload ────────────────────────────────────────────────────────

def upload_to_drive(video_path: Path, labels_path: Path, folder_id: str):
    """Upload video + labels JSON to Google Drive using rclone."""
    if not folder_id:
        print("[drive] No folder ID — skipping upload")
        return

    for path in [video_path, labels_path]:
        cmd = ["rclone", "copy", str(path), f"gdrive:{folder_id}/",
               "--progress", "--transfers", "1"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[drive] WARNING: upload of {path.name} failed:\n{result.stderr}")
        else:
            print(f"[drive] Uploaded: {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def process_video(
    url:          str,
    machine_name: str,
    output_dir:   Path,
    dry_run:      bool = False,
    drive_folder: str  = "",
    keep_video:   bool = False,
) -> Path | None:
    """
    Full pipeline for one YouTube URL.
    Returns path to the generated labels.json, or None on failure.
    """
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # 1. Download
        try:
            video_path = download_video(url, tmp_path)
        except RuntimeError as e:
            print(f"[ERROR] Download failed: {e}")
            return None

        # 2. Extract frames
        try:
            frames = extract_frames(video_path, fps=SAMPLE_FPS)
        except RuntimeError as e:
            print(f"[ERROR] Frame extraction failed: {e}")
            return None

        if len(frames) < WINDOW_FRAMES:
            print(f"[WARN] Video too short ({len(frames)} frames) — skipping")
            return None

        # 3. Slide window across frames, call Claude on each
        all_labels   = []
        window_count = math.ceil((len(frames) - WINDOW_FRAMES) / WINDOW_STRIDE) + 1

        print(f"\n[claude] Analysing {len(frames)} frames in "
              f"{window_count} windows via {MODEL}...")
        print(f"[claude] Machine: {machine_name}\n")

        for w_idx in range(0, len(frames), WINDOW_STRIDE):
            window = frames[w_idx : w_idx + WINDOW_FRAMES]
            if not window:
                break

            t_start = window[0]["time_s"]
            t_end   = window[-1]["time_s"]
            print(f"  Window {w_idx // WINDOW_STRIDE + 1}/{window_count}  "
                  f"[{t_start:.1f}s – {t_end:.1f}s]  "
                  f"({len(window)} frames)", end="", flush=True)

            if dry_run:
                print("  [DRY RUN — skipping API call]")
                continue

            try:
                labels = label_window(client, window, machine_name,
                                      w_idx // WINDOW_STRIDE)
                all_labels.extend(labels)
                # Print summary of what Claude said
                classes = [l["class"] for l in labels if isinstance(l.get("class"), int)]
                rep_count = classes.count(4) + classes.count(5)
                print(f"  → {rep_count} rep frames detected")
            except Exception as e:
                print(f"\n  [ERROR] API call failed: {e}")
                continue

        if dry_run or not all_labels:
            print("\n[done] Dry run complete — no files written.")
            return None

        # 4. Deduplicate + format
        deduped        = deduplicate_labels(all_labels)
        training_data  = to_training_labels(deduped, machine_name)

        good_reps  = sum(1 for l in deduped if l["class"] == 4)
        poor_reps  = sum(1 for l in deduped if l["class"] == 5)
        print(f"\n[result] {len(deduped)} unique timestamps  "
              f"| good reps: {good_reps}  poor reps: {poor_reps}")

        # 5. Write labels.json
        output_dir.mkdir(parents=True, exist_ok=True)
        stem         = video_path.stem[:60].replace(" ", "_")
        labels_path  = output_dir / f"{stem}_labels.json"

        labels_path.write_text(json.dumps(training_data, indent=2))
        print(f"[saved] {labels_path}")

        # 6. Optionally copy video to output dir and upload both to Drive
        if keep_video or drive_folder:
            dest_video = output_dir / video_path.name
            import shutil
            shutil.copy2(video_path, dest_video)
            print(f"[saved] {dest_video}")

            if drive_folder:
                upload_to_drive(dest_video, labels_path, drive_folder)

        return labels_path


def main():
    parser = argparse.ArgumentParser(
        description="XL Fitness — YouTube auto-labeller (Claude Opus vision)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--url",      help="YouTube URL to process")
    parser.add_argument("--url-file", help="Text file with one URL per line")
    parser.add_argument(
        "--machine", required=True,
        help='Machine name, e.g. "Nautilus Xpload Lat Pulldown"',
    )
    parser.add_argument(
        "--output-dir", default="train/yt_labels",
        help="Directory to write labels.json files (default: train/yt_labels)",
    )
    parser.add_argument(
        "--drive-folder", default="",
        help="Google Drive folder ID to upload video + labels (requires rclone)",
    )
    parser.add_argument(
        "--keep-video", action="store_true",
        help="Copy downloaded video to output-dir (always true when --drive-folder is set)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Download and extract frames but skip API calls",
    )
    parser.add_argument(
        "--sample-fps", type=int, default=SAMPLE_FPS,
        help=f"Frames per second to sample (default: {SAMPLE_FPS})",
    )
    args = parser.parse_args()

    if not args.url and not args.url_file:
        parser.error("Provide --url or --url-file")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key and not args.dry_run:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    # Collect URLs
    urls = []
    if args.url:
        urls.append(args.url)
    if args.url_file:
        with open(args.url_file) as f:
            urls.extend(line.strip() for line in f if line.strip() and not line.startswith("#"))

    output_dir = Path(args.output_dir)

    print(f"\n{'═'*56}")
    print(f"  XL Fitness — YouTube Auto-Labeller")
    print(f"  Model  : {MODEL}")
    print(f"  Machine: {args.machine}")
    print(f"  URLs   : {len(urls)}")
    print(f"  Output : {output_dir}")
    print(f"  Dry run: {args.dry_run}")
    print(f"{'═'*56}\n")

    results = []
    for i, url in enumerate(urls, 1):
        print(f"\n── URL {i}/{len(urls)} ──────────────────────────────────")
        path = process_video(
            url          = url,
            machine_name = args.machine,
            output_dir   = output_dir,
            dry_run      = args.dry_run,
            drive_folder = args.drive_folder,
            keep_video   = args.keep_video,
        )
        results.append((url, path))

    # Summary
    print(f"\n{'═'*56}")
    print("Summary:")
    ok  = sum(1 for _, p in results if p)
    err = len(results) - ok
    print(f"  Processed: {len(results)}  |  OK: {ok}  |  Failed: {err}")
    for url, path in results:
        status = str(path) if path else "FAILED"
        print(f"  {url[:60]}  →  {status}")
    print(f"{'═'*56}\n")

    if ok:
        print(f"Labels written to: {output_dir}")
        print(f"Next step: copy labels to Google Drive and run /api/train")


if __name__ == "__main__":
    main()
