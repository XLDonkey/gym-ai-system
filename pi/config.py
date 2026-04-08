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

# ── Supabase Database ─────────────────────────────────────────────────────────
# Set these to enable rep + session logging to Supabase.
# Use the service-role key (not the anon key) so the Pi can write.
SUPABASE_URL         = ""   # e.g. "https://xxxx.supabase.co"
SUPABASE_SERVICE_KEY = ""   # service_role JWT from Supabase dashboard

# ── Face Recognition ──────────────────────────────────────────────────────────
FACE_RECOGNITION_ENABLED = True   # Set False to skip face ID (anonymous sessions)
FACE_MODEL               = "buffalo_sc"   # buffalo_sc = fast; buffalo_l = most accurate
FACE_THRESHOLD           = 0.40   # Cosine similarity threshold for positive match
FACE_CHECK_INTERVAL      = 30     # Run face recognition every N frames (30 = ~1s at 30fps)
FACE_IDENTITY_WINDOW_S   = 10     # Collect evidence for N seconds before locking identity

# ── Machine Engagement Detection ──────────────────────────────────────────────
# Prevents sessions starting when someone walks past or stands next to the machine.
# The person must be inside the zone AND in a seated/engaged pose before tracking begins.
#
# MACHINE_ZONE_ROI: normalised (x1, y1, x2, y2) covering the seat + working space.
# Set SHOW_PREVIEW=True and adjust until the box tightly wraps the machine area.
# Default: centre 70% of frame width, full height — adjust per camera placement.
#
# ENGAGEMENT_MIN_OVERLAP: how much of the person's bounding box must be inside
# the zone (0.0–1.0). 0.40 = at least 40% of their body inside the zone.
ENGAGEMENT_DETECTION_ENABLED = True
MACHINE_ZONE_ROI             = (0.15, 0.05, 0.85, 0.95)  # (x1,y1,x2,y2) normalised
ENGAGEMENT_MIN_OVERLAP       = 0.40
EXERCISE_TYPE                = "lat_pulldown"   # used to pick pose logic

# ── Weight Stack Tracker ───────────────────────────────────────────────────────
# Prevents phantom reps — verifies the weight actually moved during a rep.
# Set WEIGHT_TRACKING_ENABLED = False to disable (reps counted by angle alone).
#
# ROI = Region of Interest covering the weight stack in the camera frame.
# Values are normalised (0.0–1.0): (x1, y1, x2, y2)
# Default covers the right 20% of a side-on camera view — adjust for your setup.
# Use SHOW_PREVIEW=True to see the ROI overlay while calibrating.
WEIGHT_TRACKING_ENABLED   = True
WEIGHT_STACK_ROI          = (0.72, 0.05, 0.92, 0.88)  # (x1,y1,x2,y2) normalised
WEIGHT_MOVE_PX_THRESHOLD  = 1.5   # optical flow magnitude to count as "moving"
WEIGHT_MOVE_FRAME_RATIO   = 0.30  # fraction of rep frames that must show movement

# ── ONNX Activity Classifier ──────────────────────────────────────────────────
# Once the LSTM is trained (train/train_pytorch.py) and deployed here, the Pi
# switches from rule-based angle counting to NN-based 8-class activity detection.
# Leave ONNX_MODEL_PATH blank to keep using rule-based counting (safe default).
#
# The model outputs exactly ONE class at a time (softmax gate) — so "resting"
# and "on_machine" can never both be true simultaneously.
ONNX_MODEL_PATH        = ""     # e.g. "/home/pi/xlf/models/activity_v1.onnx"
ONNX_CONFIDENCE_THRESH = 0.65   # below this: uncertain but accept the prediction
ONNX_REVIEW_THRESH     = 0.50   # below this: flag clip to GitHub for human review

# ── Clip Reporter — GitHub Review Loop ────────────────────────────────────────
# When the ONNX model is uncertain (confidence < ONNX_REVIEW_THRESH), the Pi
# saves the 30-frame keypoint window and uploads it to GitHub data/review/.
# A human annotates it in pose/label.html → commits → Mac Mini retrains →
# new ONNX pushed to Pi → accuracy improves over time.
#
# GITHUB_REVIEW_TOKEN: PAT with Contents:write on the repo.
# Leave blank to save clips locally only (no GitHub upload).
GITHUB_REVIEW_TOKEN  = ""   # ghp_... Personal Access Token
GITHUB_REVIEW_REPO   = "Matt-xlfitness/gym-ai-system"
CLIP_REPORTER_ENABLED = True
CLIP_COOLDOWN_S       = 30.0   # minimum seconds between uploads
