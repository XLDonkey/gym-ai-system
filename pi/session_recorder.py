"""
XL Fitness AI Overseer — Session Recorder
Records video in 10-minute chunks. After each chunk closes, a background
thread immediately uploads it to Google Drive via rclone, then deletes the
local copy to keep the SD card / USB drive clear.

Chunk naming convention:
  {MACHINE_ID}_{YYYYMMDD}_{HHMM}_chunk{N}.mp4
  e.g.  xlf-pi-001_20260325_1430_chunk1.mp4
"""

import cv2
import os
import threading
import subprocess
import time
from datetime import datetime

CHUNK_DURATION_SECONDS = 600   # 10 minutes per chunk
IDLE_TIMEOUT_SECONDS   = 30    # Stop recording after 30s with no person detected


class SessionRecorder:
    def __init__(self, recordings_dir, machine_id, gdrive_folder_id="",
                 fps=30, width=1280, height=720):
        self.recordings_dir   = recordings_dir
        self.machine_id       = machine_id
        self.gdrive_folder_id = gdrive_folder_id
        self.fps              = fps
        self.width            = width
        self.height           = height
        self.writer           = None
        self.is_recording     = False
        self.current_file     = None
        self.chunk_number     = 0
        self.chunk_start_time = None
        self.session_tag      = None
        self._last_seen       = None

        os.makedirs(recordings_dir, exist_ok=True)
        print(f"[REC] Storage: {recordings_dir}")
        print(f"[REC] Chunk duration: {CHUNK_DURATION_SECONDS // 60} minutes")

    def start_session(self):
        """Call when a person first arrives at the machine."""
        if self.is_recording:
            return
        self.session_tag  = datetime.now().strftime('%Y%m%d_%H%M')
        self.chunk_number = 0
        self._last_seen   = time.time()
        self._start_chunk()

    def write_frame(self, frame):
        """Call every frame while a person is present."""
        if not self.is_recording:
            return
        self.writer.write(frame)
        elapsed = time.time() - self.chunk_start_time
        if elapsed >= CHUNK_DURATION_SECONDS:
            self._rotate_chunk()

    def end_session(self):
        """Call when the person leaves (idle timeout reached)."""
        if self.is_recording:
            self._close_chunk(final=True)

    def tick_idle(self, person_present: bool):
        """
        Call every frame. Returns True if session ended due to idle timeout.
        """
        if not self.is_recording:
            return False
        if person_present:
            self._last_seen = time.time()
            return False
        if self._last_seen is None:
            self._last_seen = time.time()
        if time.time() - self._last_seen >= IDLE_TIMEOUT_SECONDS:
            self.end_session()
            return True
        return False

    def _chunk_filename(self):
        return os.path.join(
            self.recordings_dir,
            f"{self.machine_id}_{self.session_tag}_chunk{self.chunk_number}.mp4"
        )

    def _start_chunk(self):
        self.chunk_number    += 1
        self.current_file     = self._chunk_filename()
        self.chunk_start_time = time.time()
        fourcc      = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.current_file, fourcc, self.fps, (self.width, self.height)
        )
        self.is_recording = True
        print(f"[REC] ▶ Chunk {self.chunk_number} started → {os.path.basename(self.current_file)}")

    def _rotate_chunk(self):
        finished_file = self.current_file
        self._release_writer()
        self._log_chunk(finished_file)
        self._upload_async(finished_file)
        self._start_chunk()

    def _close_chunk(self, final=False):
        finished_file = self.current_file
        self._release_writer()
        self.is_recording = False
        self._log_chunk(finished_file)
        self._upload_async(finished_file)
        if final:
            print(f"[REC] ■ Session ended — {self.chunk_number} chunk(s) recorded")

    def _release_writer(self):
        if self.writer:
            self.writer.release()
            self.writer = None

    def _log_chunk(self, file_path):
        if file_path and os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / 1024 / 1024
            print(f"[REC] ✓ Chunk saved: {os.path.basename(file_path)} ({size_mb:.1f} MB)")

    def _upload_async(self, file_path):
        if not self.gdrive_folder_id:
            print(f"[UPLOAD] No Google Drive folder set — keeping local: {os.path.basename(file_path)}")
            return
        t = threading.Thread(target=self._upload_and_delete, args=(file_path,), daemon=True)
        t.start()

    def _upload_and_delete(self, file_path):
        filename = os.path.basename(file_path)
        dest     = f"xlf-gdrive:{self.machine_id}/recordings/"
        print(f"[UPLOAD] ↑ Uploading {filename} ...")
        try:
            result = subprocess.run(
                ['rclone', 'copy', file_path, dest,
                 '--drive-root-folder-id', self.gdrive_folder_id],
                capture_output=True, text=True, timeout=600
            )
            if result.returncode == 0:
                print(f"[UPLOAD] ✓ {filename} → Google Drive")
                os.remove(file_path)
                print(f"[UPLOAD] 🗑  Local copy deleted: {filename}")
            else:
                print(f"[UPLOAD] ✗ Failed: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print(f"[UPLOAD] ✗ Timeout uploading {filename}")
        except Exception as e:
            print(f"[UPLOAD] ✗ Error: {e}")

    def get_local_recordings(self):
        return sorted([
            os.path.join(self.recordings_dir, f)
            for f in os.listdir(self.recordings_dir)
            if f.endswith('.mp4')
        ])

    def retry_failed_uploads(self):
        pending = self.get_local_recordings()
        if pending:
            print(f"[UPLOAD] {len(pending)} pending uploads from previous sessions")
            for f in pending:
                self._upload_async(f)
