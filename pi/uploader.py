"""
XL Fitness AI Overseer — Nightly Uploader
Uploads session recordings to Google Drive at 2am Mon-Fri.
Run via cron: 0 2 * * 1-5 /home/pi/xlf/pi/uploader.py
"""

import os
import sys
import json
import time
from datetime import datetime
from session_recorder import SessionRecorder
from config import RECORDINGS_DIR, GOOGLE_DRIVE_FOLDER_ID, MACHINE_ID

def upload_to_drive(file_path, folder_id):
    """Upload a file to Google Drive using service account or OAuth."""
    try:
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive
        
        gauth = GoogleAuth()
        # Uses credentials.json in the pi/ directory
        gauth.LocalWebserverAuth()
        drive = GoogleDrive(gauth)
        
        filename = os.path.basename(file_path)
        file_obj = drive.CreateFile({
            'title': filename,
            'parents': [{'id': folder_id}]
        })
        file_obj.SetContentFile(file_path)
        file_obj.Upload()
        print(f"[UPLOAD] ✓ {filename}")
        return True
    except Exception as e:
        print(f"[UPLOAD] ✗ {file_path}: {e}")
        return False

def upload_via_rclone(file_path, folder_id):
    """Upload using rclone (simpler setup, recommended)."""
    import subprocess
    dest = f"gdrive:{folder_id}/{MACHINE_ID}/{os.path.basename(file_path)}"
    result = subprocess.run(
        ['rclone', 'copy', file_path, f"gdrive:{MACHINE_ID}/recordings/"],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode == 0:
        print(f"[UPLOAD] ✓ {os.path.basename(file_path)}")
        return True
    else:
        print(f"[UPLOAD] ✗ {result.stderr}")
        return False

def main():
    print(f"[UPLOADER] Starting nightly upload — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"[UPLOADER] Machine: {MACHINE_ID}")
    
    recorder = SessionRecorder(RECORDINGS_DIR)
    recordings = recorder.get_recordings()
    
    if not recordings:
        print("[UPLOADER] No recordings to upload.")
        return
    
    print(f"[UPLOADER] Found {len(recordings)} recordings to upload")
    
    uploaded = []
    failed = []
    
    for file_path in recordings:
        size_mb = os.path.getsize(file_path) / 1024 / 1024
        print(f"[UPLOADER] Uploading {os.path.basename(file_path)} ({size_mb:.1f}MB)...")
        
        # Try rclone first (easier setup), fall back to pydrive2
        success = upload_via_rclone(file_path, GOOGLE_DRIVE_FOLDER_ID)
        
        if success:
            uploaded.append(file_path)
            # Delete local copy after successful upload
            os.remove(file_path)
            print(f"[UPLOADER] Deleted local copy: {os.path.basename(file_path)}")
        else:
            failed.append(file_path)
    
    print(f"\n[UPLOADER] Done: {len(uploaded)} uploaded, {len(failed)} failed")
    
    if failed:
        print("[UPLOADER] Failed uploads will retry tomorrow.")

if __name__ == '__main__':
    main()
