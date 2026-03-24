"""
XL Fitness AI Overseer — Pi Configuration
Edit this file for your specific machine setup.
"""

# ── Machine Identity ──────────────────────────────────────────────────────────
MACHINE_ID = "lat_pulldown_01"
MACHINE_NAME = "Nautilus Nitro Lat Pulldown"

# ── Server ────────────────────────────────────────────────────────────────────
# Paste Manus's server IP here when ready
SERVER_URL = ""  # e.g. "http://192.168.1.50:8000/event"

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0          # 0 = Pi Camera (CSI), or USB camera index
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_RATE = 30

# ── Rep Detection Thresholds ──────────────────────────────────────────────────
ANGLE_TOP = 130           # Arms extended (top of pull) — rep complete above this
ANGLE_BOTTOM = 90         # Bar pulled down — rep starts below this
MIN_REP_DURATION_MS = 300 # Minimum time for a valid rep
MAX_REP_DURATION_MS = 12000  # Maximum time — safety reset
MIN_ROM_DEGREES = 40      # Minimum range of motion

# ── Engagement Detection ──────────────────────────────────────────────────────
# Wrist must be above shoulder to count as engaged (gripping overhead bar)
WRIST_ABOVE_SHOULDER_THRESHOLD = 0.10  # normalised y-offset

# ── Session Recording ─────────────────────────────────────────────────────────
RECORD_SESSIONS = True
RECORDINGS_DIR = "/home/pi/xlf_recordings"
MAX_LOCAL_STORAGE_GB = 50   # Auto-delete oldest clips when exceeded
RECORD_ONLY_WHEN_ENGAGED = True  # Don't record empty machine

# ── Upload Schedule ───────────────────────────────────────────────────────────
GOOGLE_DRIVE_FOLDER_ID = ""  # Paste Google Drive folder ID here
UPLOAD_HOUR = 2              # 2am upload
UPLOAD_DAYS = [0,1,2,3,4]   # Mon-Fri (0=Monday)

# ── YOLO Model ────────────────────────────────────────────────────────────────
YOLO_MODEL = "yolo11n-pose.pt"  # nano — fastest on Pi
YOLO_CONFIDENCE = 0.3
YOLO_DEVICE = "hailo"  # "hailo" for AI HAT+, "cpu" for Pi 5 alone

# ── Display ───────────────────────────────────────────────────────────────────
SHOW_PREVIEW = False    # Set True for debugging with a monitor attached
SHOW_SKELETON = True    # Draw keypoints on preview
