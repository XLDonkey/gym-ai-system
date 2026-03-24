"""
XL Fitness AI Overseer — Session Recorder
Records video clips of active sessions for training data.
"""

import cv2
import os
import shutil
from datetime import datetime


class SessionRecorder:
    def __init__(self, recordings_dir):
        self.recordings_dir = recordings_dir
        self.writer = None
        self.is_recording = False
        self.current_file = None
        os.makedirs(recordings_dir, exist_ok=True)
        print(f"[REC] Recordings directory: {recordings_dir}")
    
    def start_recording(self, machine_id, fps=30, width=1280, height=720):
        if self.is_recording:
            return
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{machine_id}_{timestamp}.mp4"
        self.current_file = os.path.join(self.recordings_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.current_file, fourcc, fps, (width, height))
        self.is_recording = True
        print(f"[REC] Recording started: {filename}")
    
    def write_frame(self, frame):
        if self.is_recording and self.writer:
            self.writer.write(frame)
    
    def stop_recording(self):
        if self.is_recording and self.writer:
            self.writer.release()
            self.writer = None
            self.is_recording = False
            size_mb = os.path.getsize(self.current_file) / 1024 / 1024
            print(f"[REC] Recording saved: {os.path.basename(self.current_file)} ({size_mb:.1f}MB)")
            self._check_storage()
    
    def _check_storage(self):
        """Delete oldest recordings if storage exceeds limit."""
        from config import MAX_LOCAL_STORAGE_GB
        total = sum(
            os.path.getsize(os.path.join(self.recordings_dir, f))
            for f in os.listdir(self.recordings_dir)
            if f.endswith('.mp4')
        ) / (1024**3)
        
        if total > MAX_LOCAL_STORAGE_GB:
            files = sorted([
                f for f in os.listdir(self.recordings_dir) if f.endswith('.mp4')
            ])
            while total > MAX_LOCAL_STORAGE_GB * 0.8 and files:
                oldest = files.pop(0)
                path = os.path.join(self.recordings_dir, oldest)
                size = os.path.getsize(path) / (1024**3)
                os.remove(path)
                total -= size
                print(f"[REC] Deleted old recording: {oldest} ({size:.2f}GB freed)")
    
    def get_recordings(self):
        """Return list of all recordings not yet uploaded."""
        return [
            os.path.join(self.recordings_dir, f)
            for f in sorted(os.listdir(self.recordings_dir))
            if f.endswith('.mp4')
        ]
