"""
XL Fitness AI Overseer — Main Entry Point
Runs YOLO pose detection, counts reps, records sessions in 10-min chunks,
and uploads each chunk to Google Drive immediately after it closes.

Start:  python3 main.py
Stop:   Ctrl+C
"""

import cv2
import time
import math
import requests
from datetime import datetime
from collections import deque

from config import (
    MACHINE_ID, MACHINE_NAME,
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FRAME_RATE,
    RECORD_SESSIONS, RECORDINGS_DIR, GOOGLE_DRIVE_FOLDER_ID,
    YOLO_MODEL, YOLO_CONFIDENCE, YOLO_DEVICE,
    ANGLE_TOP, ANGLE_BOTTOM, MIN_REP_DURATION_MS,
    MAX_REP_DURATION_MS, MIN_ROM_DEGREES,
    SESSION_TIMEOUT_S, SHOW_PREVIEW, SHOW_SKELETON,
    SERVER_URL,
)
from session_recorder import SessionRecorder

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
        "machine_id": MACHINE_ID,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "payload": payload,
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

    def start_session(self):
        self.session_active = True
        self.rep_count      = 0
        post_event('session_start', {'machine_id': MACHINE_ID})
        print(f"[SESSION] Started — {MACHINE_NAME}")

    def end_session(self):
        if self.session_active:
            self.session_active = False
            post_event('session_end', {'machine_id': MACHINE_ID, 'total_reps': self.rep_count})
            print(f"[SESSION] Ended — {self.rep_count} reps")

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

        elif self.phase == 'down':
            self._min_angle = min(self._min_angle, angle)
            dur_ms = time.time()*1000 - self._rep_start_ms
            if dur_ms > MAX_REP_DURATION_MS:
                self.phase = 'waiting'
                return None
            if angle > ANGLE_TOP and dur_ms >= MIN_REP_DURATION_MS:
                rom = self._max_angle - self._min_angle
                if rom >= MIN_ROM_DEGREES:
                    self.rep_count += 1
                    print(f"[REP] #{self.rep_count}  ROM={rom:.0f}°  dur={dur_ms:.0f}ms")
                    post_event('rep_completed', {
                        'rep_number': self.rep_count,
                        'rom': round(rom, 1),
                        'duration_s': round(dur_ms/1000, 2),
                    })
                self.phase      = 'waiting'
                self._min_angle = 180.0
                self._max_angle = angle

        return None


def main():
    print(f"\n[XLF] ═══════════════════════════════════════")
    print(f"[XLF]  AI Overseer — {MACHINE_NAME}")
    print(f"[XLF]  Machine ID : {MACHINE_ID}")
    print(f"[XLF]  Recording  : {RECORD_SESSIONS}")
    print(f"[XLF]  Drive ID   : {GOOGLE_DRIVE_FOLDER_ID or 'NOT SET'}")
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

    counter        = RepCounter()
    session_active = False
    frame_count    = 0
    fps_timer      = time.time()

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

            if results and results[0].keypoints is not None:
                kp_data = results[0].keypoints.data
                if len(kp_data) > 0:
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        areas  = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes.xyxy]
                        best_i = areas.index(max(areas))
                        kps    = kp_data[best_i].cpu().numpy()
                        kps[:, 0] /= frame.shape[1]
                        kps[:, 1] /= frame.shape[0]
                        arm_data        = get_best_arm(kps)
                        person_detected = True

            # Session + recording management
            if person_detected and not session_active:
                session_active = True
                if recorder:
                    recorder.start_session()

            if recorder:
                if person_detected:
                    recorder.write_frame(frame)
                if recorder.tick_idle(person_detected):
                    session_active = False
                    counter.end_session()

            counter.update(arm_data)

            if frame_count % 100 == 0:
                fps = 100 / (time.time() - fps_timer)
                fps_timer = time.time()
                print(f"[FPS] {fps:.1f}  Reps:{counter.rep_count}  "
                      f"Person:{person_detected}  "
                      f"Rec:{recorder.is_recording if recorder else False}")

            if SHOW_PREVIEW:
                cv2.imshow('XL Fitness AI Overseer', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n[XLF] Shutting down...")
    finally:
        if recorder and recorder.is_recording:
            recorder.end_session()
        cap.release()
        cv2.destroyAllWindows()
        print("[XLF] Done.")


if __name__ == '__main__':
    main()
