"""
XL Fitness AI Overseer — Configuration
Edit this file for each Pi deployment.
"""

# ── Machine Identity ──────────────────────────────────────────────────────────
MACHINE_ID   = "xlf-pi-001"                      # Unique ID for this Pi/camera
MACHINE_NAME = "Nautilus Xpload Lat Pulldown"     # Human-readable name

# ── Google Drive ──────────────────────────────────────────────────────────────
# Folder ID from the shared Google Drive link
# https://drive.google.com/drive/folders/1KNDC4wctZqVt8s41U4ALWHJ45OM5U9FA
GOOGLE_DRIVE_FOLDER_ID = "1KNDC4wctZqVt8s41U4ALWHJ45OM5U9FA"

# ── Fleet Dashboard ───────────────────────────────────────────────────────────
# URL of the fleet dashboard heartbeat endpoint (set once dashboard is deployed)
SERVER_URL = ""   # e.g. "https://your-dashboard.manus.space/api/heartbeat"

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX  = 0       # 0 = Pi Camera (CSI), or USB camera index
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
FRAME_RATE    = 30

# ── Recording ─────────────────────────────────────────────────────────────────
RECORD_SESSIONS          = True
RECORDINGS_DIR           = "/home/pi/xlf_recordings"   # Local buffer before upload
MAX_LOCAL_STORAGE_GB     = 10    # Auto-delete oldest if exceeded (safety net)
RECORD_ONLY_WHEN_ENGAGED = False  # False = record whenever person is present

# ── YOLO Model ────────────────────────────────────────────────────────────────
YOLO_MODEL      = "yolo11n-pose.pt"   # nano — fastest on Pi
YOLO_CONFIDENCE = 0.3
YOLO_DEVICE     = "cpu"   # "hailo" for AI HAT+, "cpu" for Pi 5 alone

# ── Rep Detection Thresholds ──────────────────────────────────────────────────
ANGLE_TOP             = 130    # Arms extended (top of pull) — rep complete above this
ANGLE_BOTTOM          = 90     # Bar pulled down — rep starts below this
MIN_REP_DURATION_MS   = 300
MAX_REP_DURATION_MS   = 12000
MIN_ROM_DEGREES       = 40

# ── Session Timeouts ──────────────────────────────────────────────────────────
IDLE_TIMEOUT_SECONDS  = 30     # Stop recording after 30s with no person
SESSION_TIMEOUT_S     = 300    # End session after 5 min away

# ── Display ───────────────────────────────────────────────────────────────────
SHOW_PREVIEW  = False   # Set True only when a monitor is attached
SHOW_SKELETON = True
