"""
XL Fitness AI Overseer — Main Entry Point
Runs YOLO pose detection, counts reps, records sessions in 10-min chunks,
identifies members via InsightFace, and logs all stats to Supabase.

Start:  python3 main.py
Stop:   Ctrl+C
"""

import cv2
import sys
import time
import math
import requests
from datetime import datetime
from collections import deque

# Resolve repo root so we can import face + members modules
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from config import (
    MACHINE_ID, MACHINE_NAME,
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FRAME_RATE,
    RECORD_SESSIONS, RECORDINGS_DIR, GOOGLE_DRIVE_FOLDER_ID,
    YOLO_MODEL, YOLO_CONFIDENCE, YOLO_DEVICE,
    ANGLE_TOP, ANGLE_BOTTOM, MIN_REP_DURATION_MS,
    MAX_REP_DURATION_MS, MIN_ROM_DEGREES,
    SESSION_TIMEOUT_S, SHOW_PREVIEW, SHOW_SKELETON,
    SERVER_URL,
    SUPABASE_URL, SUPABASE_SERVICE_KEY,
    FACE_RECOGNITION_ENABLED, FACE_MODEL, FACE_THRESHOLD,
    FACE_CHECK_INTERVAL, FACE_IDENTITY_WINDOW_S,
    WEIGHT_TRACKING_ENABLED, WEIGHT_STACK_ROI,
    WEIGHT_MOVE_PX_THRESHOLD, WEIGHT_MOVE_FRAME_RATIO,
    ENGAGEMENT_DETECTION_ENABLED, MACHINE_ZONE_ROI,
    ENGAGEMENT_MIN_OVERLAP, EXERCISE_TYPE,
)
from weight_tracker import WeightStackTracker
from engagement_detector import EngagementDetector
from session_recorder import SessionRecorder

# ── Optional integrations (fail gracefully if not configured) ─────────────────

db         = None
recognizer = None

if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    try:
        from members.db_client import SupabaseClient
        db = SupabaseClient(url=SUPABASE_URL, key=SUPABASE_SERVICE_KEY)
        print("[DB] Supabase connected")
    except Exception as e:
        print(f"[DB] WARNING: Supabase init failed — {e}  (continuing without DB)")
else:
    print("[DB] Supabase not configured — set SUPABASE_URL + SUPABASE_SERVICE_KEY in config.py")

if FACE_RECOGNITION_ENABLED:
    try:
        from face.face_recognizer import FaceRecognizer, IdentityWindow
        recognizer = FaceRecognizer(model_name=FACE_MODEL, threshold=FACE_THRESHOLD)
        if recognizer.ready and db:
            members = db.get_all_members()
            recognizer.load_members(members)
    except Exception as e:
        print(f"[face] WARNING: Face recognizer init failed — {e}  (anonymous sessions only)")
        recognizer = None

# ── YOLO ──────────────────────────────────────────────────────────────────────

try:
    from ultralytics import YOLO
    model = YOLO(YOLO_MODEL)
    print(f"[YOLO] Model loaded: {YOLO_MODEL}")
except Exception as e:
    print(f"[YOLO] ERROR: {e}")
    exit(1)

# COCO keypoint indices
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW,    KP_R_ELBOW    = 7, 8
KP_L_WRIST,    KP_R_WRIST    = 9, 10
WRIST_ABOVE_SHOULDER_THRESHOLD = 0.10


def angle_between(a, b, c):
    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc)
    if mag < 1e-6:
        return 0.0
    return math.degrees(math.acos(max(-1, min(1, dot/mag))))


def get_best_arm(kps, conf_thresh=0.3):
    arms = [
        (kps[KP_L_SHOULDER], kps[KP_L_ELBOW], kps[KP_L_WRIST]),
        (kps[KP_R_SHOULDER], kps[KP_R_ELBOW], kps[KP_R_WRIST]),
    ]
    best, best_conf = None, 0.0
    for s, e, w in arms:
        conf = min(s[2], e[2], w[2])
        if conf > conf_thresh and conf > best_conf:
            wrist_above = w[1] < s[1] + WRIST_ABOVE_SHOULDER_THRESHOLD
            best = {
                'shoulder': s[:2], 'elbow': e[:2], 'wrist': w[:2],
                'angle': angle_between(s[:2], e[:2], w[:2]),
                'wrist_above_shoulder': wrist_above,
                'confidence': conf,
            }
            best_conf = conf
    return best


offline_queue = []

def post_event(event_type, payload):
    if not SERVER_URL:
        return
    body = {
        "machine_id":    MACHINE_ID,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "event_type":    event_type,
        "payload":       payload,
    }
    try:
        r = requests.post(SERVER_URL, json=body, timeout=3)
        if r.status_code == 200:
            while offline_queue:
                q = offline_queue.pop(0)
                try:
                    requests.post(SERVER_URL, json=q, timeout=3)
                except Exception:
                    offline_queue.insert(0, q)
                    break
    except Exception:
        offline_queue.append(body)
        if len(offline_queue) > 500:
            offline_queue.pop(0)


# ── Rep Counter ───────────────────────────────────────────────────────────────

class RepCounter:
    def __init__(self):
        self.phase          = 'waiting'
        self.rep_count      = 0
        self.session_active = False
        self.ema            = None
        self._rep_start_ms  = None
        self._min_angle     = 180.0
        self._max_angle     = 0.0
        self.angle_buffer   = deque(maxlen=90)

        # DB session state
        self._session_id    = None
        self._member_id     = None
        self._member_name   = None
        self._rom_history   = []
        self._dur_history   = []

        # Weight tracking — injected from main loop
        self.weight_tracker: WeightStackTracker = None

    def start_session(self, member_id=None, member_name=None):
        self.session_active = True
        self.rep_count      = 0
        self._rom_history   = []
        self._dur_history   = []
        self._member_id     = member_id
        self._member_name   = member_name
        self._session_id    = None

        if db:
            try:
                self._session_id = db.create_session(
                    machine_id   = MACHINE_ID,
                    machine_name = MACHINE_NAME,
                    member_id    = member_id,
                )
                print(f"[DB] Session created: {self._session_id}")
            except Exception as e:
                print(f"[DB] create_session failed: {e}")

        who = member_name or "unknown member"
        post_event('session_start', {'machine_id': MACHINE_ID, 'member': who})
        print(f"[SESSION] Started — {MACHINE_NAME}  member={who}")

    def assign_member(self, member_id: str, member_name: str):
        """Called once face recognition identifies the member mid-session."""
        if self._member_id or not self.session_active:
            return
        self._member_id   = member_id
        self._member_name = member_name
        print(f"[face] Identified: {member_name} (confidence lock)")
        if db and self._session_id:
            try:
                db.assign_member_to_session(self._session_id, member_id)
            except Exception as e:
                print(f"[DB] assign_member failed: {e}")

    def end_session(self):
        if not self.session_active:
            return
        self.session_active = False

        avg_rom  = sum(self._rom_history) / len(self._rom_history) if self._rom_history else None
        avg_dur  = sum(self._dur_history) / len(self._dur_history) if self._dur_history else None

        if db and self._session_id:
            try:
                db.close_session(
                    self._session_id,
                    total_reps     = self.rep_count,
                    avg_rom        = avg_rom,
                    avg_duration_s = avg_dur,
                )
                print(f"[DB] Session closed: {self._session_id}  reps={self.rep_count}")
            except Exception as e:
                print(f"[DB] close_session failed: {e}")

        post_event('session_end', {
            'machine_id': MACHINE_ID,
            'total_reps': self.rep_count,
            'member':     self._member_name or "unknown",
        })
        print(f"[SESSION] Ended — {self.rep_count} reps  member={self._member_name or 'unknown'}")

    def update(self, arm_data):
        if arm_data is None:
            return None

        raw = arm_data['angle']
        self.ema = raw if self.ema is None else 0.3*raw + 0.7*self.ema
        angle = self.ema
        self.angle_buffer.append(angle)

        if not self.session_active and arm_data['wrist_above_shoulder']:
            self.start_session()
        if not self.session_active:
            return None

        if self.phase == 'waiting':
            self._max_angle = max(self._max_angle, angle)
            if angle < ANGLE_BOTTOM and arm_data['wrist_above_shoulder']:
                self.phase         = 'down'
                self._min_angle    = angle
                self._rep_start_ms = time.time() * 1000
                # Tell weight tracker a rep has begun
                if self.weight_tracker:
                    self.weight_tracker.start_rep()

        elif self.phase == 'down':
            self._min_angle = min(self._min_angle, angle)
            dur_ms = time.time()*1000 - self._rep_start_ms
            if dur_ms > MAX_REP_DURATION_MS:
                self.phase = 'waiting'
                return None
            if angle > ANGLE_TOP and dur_ms >= MIN_REP_DURATION_MS:
                rom = self._max_angle - self._min_angle
                # Gate: require weight to have actually moved
                weight_confirmed = (
                    self.weight_tracker is None
                    or self.weight_tracker.weight_moved_during_rep()
                )
                if rom >= MIN_ROM_DEGREES and weight_confirmed:
                    self.rep_count += 1
                    dur_s = round(dur_ms / 1000, 2)
                    print(f"[REP] #{self.rep_count}  ROM={rom:.0f}°  dur={dur_ms:.0f}ms")

                    self._rom_history.append(rom)
                    self._dur_history.append(dur_s)

                    # Log rep to Supabase
                    if db and self._session_id:
                        try:
                            db.log_rep(
                                session_id  = self._session_id,
                                rep_number  = self.rep_count,
                                rom_degrees = round(rom, 1),
                                duration_s  = dur_s,
                            )
                        except Exception as e:
                            print(f"[DB] log_rep failed: {e}")

                    post_event('rep_completed', {
                        'rep_number': self.rep_count,
                        'rom':        round(rom, 1),
                        'duration_s': dur_s,
                        'member':     self._member_name or "unknown",
                    })

                self.phase      = 'waiting'
                self._min_angle = 180.0
                self._max_angle = angle

        return None


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print(f"\n[XLF] ═══════════════════════════════════════")
    print(f"[XLF]  AI Overseer — {MACHINE_NAME}")
    print(f"[XLF]  Machine ID : {MACHINE_ID}")
    print(f"[XLF]  Recording  : {RECORD_SESSIONS}")
    print(f"[XLF]  Drive ID   : {GOOGLE_DRIVE_FOLDER_ID or 'NOT SET'}")
    print(f"[XLF]  Supabase   : {'connected' if db else 'not configured'}")
    print(f"[XLF]  Face ID    : {'ready' if (recognizer and recognizer.ready) else 'disabled'}")
    print(f"[XLF] ═══════════════════════════════════════\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FRAME_RATE)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera — check CAMERA_INDEX in config.py")
        return

    print(f"[CAM] {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FRAME_RATE}fps")

    recorder = SessionRecorder(
        recordings_dir   = RECORDINGS_DIR,
        machine_id       = MACHINE_ID,
        gdrive_folder_id = GOOGLE_DRIVE_FOLDER_ID,
        fps              = FRAME_RATE,
        width            = FRAME_WIDTH,
        height           = FRAME_HEIGHT,
    ) if RECORD_SESSIONS else None

    if recorder:
        recorder.retry_failed_uploads()

    counter         = RepCounter()
    session_active  = False
    frame_count     = 0
    fps_timer       = time.time()

    # Weight stack tracker
    weight_tracker = WeightStackTracker(
        roi_norm         = WEIGHT_STACK_ROI,
        frame_w          = FRAME_WIDTH,
        frame_h          = FRAME_HEIGHT,
        move_threshold   = WEIGHT_MOVE_PX_THRESHOLD,
        move_frame_ratio = WEIGHT_MOVE_FRAME_RATIO,
        enabled          = WEIGHT_TRACKING_ENABLED,
    )
    counter.weight_tracker = weight_tracker
    print(f"[weight] Tracker {'enabled' if WEIGHT_TRACKING_ENABLED else 'disabled'}  "
          f"ROI={WEIGHT_STACK_ROI}")

    # Engagement detector — must be ON the machine before session starts
    engagement = EngagementDetector(
        zone_norm   = MACHINE_ZONE_ROI,
        exercise    = EXERCISE_TYPE,
        min_overlap = ENGAGEMENT_MIN_OVERLAP,
        enabled     = ENGAGEMENT_DETECTION_ENABLED,
    )
    print(f"[engage] Detector {'enabled' if ENGAGEMENT_DETECTION_ENABLED else 'disabled'}  "
          f"zone={MACHINE_ZONE_ROI}  exercise={EXERCISE_TYPE}")

    # Face recognition state
    identity_window = None   # IdentityWindow | None
    last_bbox       = None   # most recent YOLO bounding box

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame_count += 1

            results = model(
                frame,
                conf    = YOLO_CONFIDENCE,
                device  = None if YOLO_DEVICE == 'cpu' else YOLO_DEVICE,
                verbose = False,
            )

            person_detected = False
            arm_data        = None
            current_bbox    = None

            if results and results[0].keypoints is not None:
                kp_data = results[0].keypoints.data
                if len(kp_data) > 0:
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        areas      = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes.xyxy]
                        best_i     = areas.index(max(areas))
                        kps        = kp_data[best_i].cpu().numpy()
                        current_bbox = boxes.xyxy[best_i].cpu().numpy()  # [x1,y1,x2,y2]
                        kps[:, 0] /= frame.shape[1]
                        kps[:, 1] /= frame.shape[0]
                        arm_data        = get_best_arm(kps)
                        person_detected = True
                        last_bbox       = current_bbox

                        # ── Engagement check ──────────────────────────────────
                        # Only track people physically on the machine,
                        # not bystanders, PTs, or people walking past.
                        person_engaged = engagement.update(
                            kps     = kps,
                            bbox_px = current_bbox,
                            frame_w = frame.shape[1],
                            frame_h = frame.shape[0],
                        )

            if not person_detected:
                person_engaged = False

            # ── Session management ────────────────────────────────────────────
            if person_engaged and not session_active:
                session_active  = True
                identity_window = None

                # If face recognition available, start identity window
                if recognizer and recognizer.ready:
                    from face.face_recognizer import IdentityWindow
                    identity_window = IdentityWindow(duration_s=FACE_IDENTITY_WINDOW_S)

                # Start session (member_id assigned later once face is confirmed)
                counter.start_session()

                if recorder:
                    recorder.start_session()

            # ── Face recognition window ───────────────────────────────────────
            if (
                session_active
                and recognizer and recognizer.ready
                and identity_window
                and identity_window.open
                and person_detected
                and frame_count % FACE_CHECK_INTERVAL == 0
            ):
                result = recognizer.identify_from_frame(
                    frame,
                    bbox_xyxy = tuple(last_bbox) if last_bbox is not None else None,
                )
                if result.face_found:
                    identity_window.add(result)
                    if result.member_id:
                        print(f"[face] {result.member_name}  conf={result.confidence:.2f}  "
                              f"lat={result.latency_ms:.0f}ms")

            # Lock identity once window expires
            if (
                identity_window
                and identity_window.expired
                and not counter._member_id
            ):
                best = identity_window.best()
                if best and best.member_id:
                    counter.assign_member(best.member_id, best.member_name)
                else:
                    print("[face] Could not identify member — session logged as anonymous")
                identity_window = None  # done

            # ── Recording ─────────────────────────────────────────────────────
            if recorder:
                if person_engaged:
                    recorder.write_frame(frame)
                if recorder.tick_idle(person_engaged):
                    session_active = False
                    engagement.reset()
                    counter.end_session()
                    identity_window = None

            # ── Weight stack tracking ─────────────────────────────────────────
            weight_tracker.update(frame)

            counter.update(arm_data)

            # ── FPS logging ───────────────────────────────────────────────────
            if frame_count % 100 == 0:
                fps = 100 / (time.time() - fps_timer)
                fps_timer = time.time()
                print(f"[FPS] {fps:.1f}  Reps:{counter.rep_count}  "
                      f"Person:{person_detected}  "
                      f"Member:{counter._member_name or 'unknown'}  "
                      f"Rec:{recorder.is_recording if recorder else False}")

            if SHOW_PREVIEW:
                weight_tracker.draw_overlay(frame)
                engagement.draw_overlay(frame)
                cv2.imshow('XL Fitness AI Overseer', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n[XLF] Shutting down...")
    finally:
        if counter.session_active:
            counter.end_session()
        if recorder and recorder.is_recording:
            recorder.end_session()
        cap.release()
        cv2.destroyAllWindows()
        print("[XLF] Done.")


if __name__ == '__main__':
    main()
