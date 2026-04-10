---
title: Session Recorder
tags: [system, recording, google-drive, video]
created: 2026-04-10
---

# Session Recorder

> Automatically records every session to MP4, uploads to Google Drive, and manages local storage.

Part of [[System/Architecture]]. Code: `pi/session_recorder.py`.

---

## What It Does

The Pi records video whenever a person is detected. Each recording is:
1. Split into 10-minute MP4 chunks
2. Uploaded to Google Drive in the background
3. Deleted locally after successful upload

This creates the raw footage needed to train the [[System/LSTM Model]].

---

## Recording Config (`pi/config.py`)

| Setting | Value | Meaning |
|---------|-------|---------|
| `RECORD_SESSIONS` | True | Enable/disable recording |
| `RECORDINGS_DIR` | `/home/pi/xlf_recordings` | Local buffer directory |
| `MAX_LOCAL_STORAGE_GB` | 10 | Auto-delete oldest if exceeded |
| `RECORD_ONLY_WHEN_ENGAGED` | False | False = record whenever person is present |
| `CHUNK_DURATION_SECONDS` | 600 | 10-minute chunks |
| `GOOGLE_DRIVE_FOLDER_ID` | `1KNDC4wctZqVt8s41U4ALWHJ45OM5U9FA` | Upload destination |

---

## File Naming

```
{machine_id}_{YYYYMMDD}_{HHMM}_chunk{N}.mp4

Examples:
  xlf-pi-001_20260409_1432_chunk1.mp4
  xlf-pi-001_20260409_1432_chunk2.mp4
```

---

## Upload Pipeline

```
Write frame → ring buffer → OpenCV VideoWriter → local .mp4
    │
    (chunk full: 600s or size limit)
    │
    ▼
Background thread:
    rclone copy {chunk}.mp4 gdrive:{FOLDER_ID}/
    → Success: delete local file
    → Failure: add to retry_queue
    
Retry queue:
    flush_retry_failed_uploads() called at startup
    (uploads anything left over from previous session)
```

**rclone** must be configured on the Pi with Google Drive credentials.

---

## Local Storage Safety Net

```python
if local_storage_used_gb > MAX_LOCAL_STORAGE_GB:
    # Delete oldest local chunks (those already uploaded)
    # Prevents Pi SD card filling up
```

---

## SessionRecorder API

```python
recorder = SessionRecorder(
    machine_id="xlf-pi-001",
    recordings_dir="/home/pi/xlf_recordings",
    chunk_duration_s=600,
    drive_folder_id="1KNDC4wctZqVt8s41U4ALWHJ45OM5U9FA"
)

recorder.start_session()           # opens new video file
recorder.write_frame(frame)        # writes frame, rotates chunk if needed
recorder.end_session()             # closes current chunk, queues upload
recorder.tick_idle()               # called when no person (may close idle chunks)
recorder.retry_failed_uploads()    # called at startup
```

---

## Automation (Pi Crontab)

```bash
# 2am Mon–Fri: run uploader for any chunks missed
0 2 * * 1-5  python /home/pi/xlf/pi/uploader.py

# 3am daily: pull latest ONNX model
0 3 * * *    cd /home/pi/xlf && git pull origin main
             # new ONNX active next session start
```

---

## Why Google Drive?

- Easy access from Mac Mini for training — drag to Finder or `rclone sync`
- Folder ID: `1KNDC4wctZqVt8s41U4ALWHJ45OM5U9FA`
- Shared between Matt-xlfitness and XLDonkey accounts
- `make sync PI=pi@IP` pulls recordings from Pi → Mac Mini for annotation

---

## Related

- [[System/Architecture]] — recording fits in main loop
- [[System/Review Loop]] — recordings are the raw training data
- [[Projects/Rep Tracking]] — sessions being recorded
- [[Data/Training Requirements]] — recordings → annotate → train
- [[Hardware/Machine Pi]] — Pi SD card storage + upload
