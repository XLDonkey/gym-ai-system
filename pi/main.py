#!/usr/bin/env python3
"""
XL Fitness AI Overseer — Raspberry Pi Main Loop
Runs on Pi 5 + AI HAT+ (Hailo) + Pi AI Camera

Pipeline:
  Camera → YOLOv11-pose → keypoints → rep detection → session events → server
                                                      ↓
                                               session recording
"""

import cv2
import time
import math
import json
import requests
import threading
import os
from datetime import datetime
from collections import deque

from config import *
from session_recorder import SessionRecorder

# ── YOLO setup ────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    model = YOLO(YOLO_MODEL)
    print(f"[YOLO] Model loaded: {YOLO_MODEL}")
except Exception as e:
    print(f"[YOLO] ERROR loading model: {e}")
    print("[YOLO] Run: pip install ultralytics")
    exit(1)

# COCO keypoint indices
KP = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2,
    'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16,
}

# ── Angle calculation ─────────────────────────────────────────────────────────
def calc_angle(a, b, c):
    """Calculate angle at point b between points a, b, c."""
    ab = [a[0]-b[0], a[1]-b[1]]
    cb = [c[0]-b[0], c[1]-b[1]]
    dot = ab[0]*cb[0] + ab[1]*cb[1]
    mag = math.sqrt((ab[0]**2+ab[1]**2) * (cb[0]**2+cb[1]**2))
    if mag == 0:
        return None
    return math.degrees(math.acos(max(-1, min(1, dot/mag))))

def get_best_arm(keypoints, confidence_threshold=0.3):
    """Get the best visible arm's elbow angle and engagement state."""
    best = None
    best_conf = 0
    
    for side in [('left_shoulder','left_elbow','left_wrist'),
                 ('right_shoulder','right_elbow','right_wrist')]:
        si, ei, wi = [KP[k] for k in side]
        s, e, w = keypoints[si], keypoints[ei], keypoints[wi]
        
        if s[2] < confidence_threshold or e[2] < confidence_threshold or w[2] < confidence_threshold:
            continue
            
        conf = (s[2] + e[2] + w[2]) / 3
        if conf > best_conf:
            angle = calc_angle((s[0],s[1]), (e[0],e[1]), (w[0],w[1]))
            if angle is not None:
                # Check wrist above shoulder (engagement signal)
                wrist_above = w[1] < s[1] + WRIST_ABOVE_SHOULDER_THRESHOLD
                best_conf = conf
                best = {
                    'angle': angle,
                    'confidence': conf,
                    'wrist_above_shoulder': wrist_above,
                    'shoulder': (s[0], s[1]),
                    'elbow': (e[0], e[1]),
                    'wrist': (w[0], w[1]),
                }
    return best

# ── Event posting ─────────────────────────────────────────────────────────────
offline_queue = []

def post_event(event_type, payload):
    """Post event to Manus's server. Queues if offline."""
    if not SERVER_URL:
        return  # No server configured yet
    
    body = {
        "machine_id": MACHINE_ID,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "event_type": "pose_estimation",
        "payload": {"event": event_type, **payload}
    }
    
    try:
        r = requests.post(SERVER_URL, json=body, timeout=3)
        if r.status_code == 200:
            # Flush queue if connected
            while offline_queue:
                q = offline_queue.pop(0)
                try:
                    requests.post(SERVER_URL, json=q, timeout=3)
                except:
                    offline_queue.insert(0, q)
                    break
    except Exception:
        offline_queue.append(body)
        if len(offline_queue) > 500:
            offline_queue.pop(0)  # Prevent unlimited growth

# ── Rep state machine ─────────────────────────────────────────────────────────
class RepCounter:
    def __init__(self):
        self.phase = 'waiting'  # waiting → down → up (rep complete)
        self.rep_count = 0
        self.rep_start_time = None
        self.rep_max_angle = 0
        self.rep_min_angle = 999
        self.angle_buffer = deque(maxlen=90)  # 3s at 30fps
        self.ema = None  # Exponential moving average
        self.session_active = False
        self.rest_start = None
        
    def update(self, arm_data):
        """Process one frame's arm data. Returns rep event if rep completed."""
        if arm_data is None:
            self._handle_no_detection()
            return None
            
        raw_angle = arm_data['angle']
        
        # EMA smoothing (α=0.3)
        if self.ema is None:
            self.ema = raw_angle
        self.ema = 0.3 * raw_angle + 0.7 * self.ema
        angle = self.ema
        
        self.angle_buffer.append(angle)
        
        # Engagement detection
        engaged = arm_data['wrist_above_shoulder']
        
        if not self.session_active and engaged:
            self._start_session()
            
        if not self.session_active:
            return None
            
        # Rest detection (no movement for 5+ seconds)
        if not engaged and self.phase == 'waiting':
            if self.rest_start is None:
                self.rest_start = time.time()
            elif time.time() - self.rest_start > 5:
                # Resting between sets
                post_event('rest_period', {'rep_count': self.rep_count})
        else:
            self.rest_start = None
        
        # Rep state machine
        if self.phase == 'waiting':
            self.rep_max_angle = max(self.rep_max_angle, angle)
            if angle < ANGLE_BOTTOM and engaged:
                # Pull started
                self.phase = 'down'
                self.rep_min_angle = angle
                self.rep_start_time = time.time()
                
        elif self.phase == 'down':
            self.rep_min_angle = min(self.rep_min_angle, angle)
            dur_ms = (time.time() - self.rep_start_time) * 1000
            
            # Safety reset
            if dur_ms > MAX_REP_DURATION_MS:
                self.phase = 'waiting'
                self.rep_max_angle = angle
                self.rep_min_angle = 999
                return None
                
            if angle > ANGLE_TOP and dur_ms >= MIN_REP_DURATION_MS:
                # Rep completed
                rom = self.rep_max_angle - self.rep_min_angle
                if rom >= MIN_ROM_DEGREES:
                    self.rep_count += 1
                    self.phase = 'waiting'
                    self.rep_max_angle = angle
                    
                    # Calculate quality metrics
                    buf = list(self.angle_buffer)
                    avg = sum(buf)/len(buf) if buf else angle
                    std = math.sqrt(sum((x-avg)**2 for x in buf)/len(buf)) if buf else 0
                    
                    rep_event = {
                        'rep_number': self.rep_count,
                        'rep_quality': 'good',  # Placeholder — LSTM will classify
                        'elbow_angle_bottom': round(self.rep_min_angle, 1),
                        'rom': round(rom, 1),
                        'duration_s': round(dur_ms/1000, 2),
                        'avg_angle': round(avg, 1),
                        'std_angle': round(std, 1),
                        'confidence': round(arm_data['confidence'], 3),
                    }
                    
                    self.rep_min_angle = 999
                    post_event('rep_completed', rep_event)
                    print(f"[REP] #{self.rep_count} | ROM: {rom:.0f}° | bottom: {self.rep_min_angle:.0f}°")
                    return rep_event
                    
        return None
    
    def _start_session(self):
        if not self.session_active:
            self.session_active = True
            self.rep_count = 0
            post_event('session_start', {'machine_id': MACHINE_ID})
            post_event('user_seated_engaged', {'machine_id': MACHINE_ID})
            print(f"[SESSION] Started at {MACHINE_NAME}")
    
    def end_session(self):
        if self.session_active:
            self.session_active = False
            post_event('session_end', {
                'machine_id': MACHINE_ID,
                'total_reps': self.rep_count,
            })
            print(f"[SESSION] Ended — {self.rep_count} reps")
            self.rep_count = 0
    
    def _handle_no_detection(self):
        """Handle frames with no person detected."""
        pass  # Could trigger session end after timeout


# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_skeleton(frame, keypoints, arm_data, rep_count, phase):
    """Draw skeleton and rep info on frame."""
    # Draw keypoints
    for kp in keypoints:
        if kp[2] > 0.3:
            x, y = int(kp[0] * frame.shape[1]), int(kp[1] * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 120), -1)
    
    # Draw arm angle
    if arm_data:
        ex = int(arm_data['elbow'][0] * frame.shape[1])
        ey = int(arm_data['elbow'][1] * frame.shape[0])
        cv2.putText(frame, f"{arm_data['angle']:.0f}°",
                    (ex+10, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    # Draw rep count
    cv2.putText(frame, f"REPS: {rep_count}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
    
    # Draw phase
    colour = (0,255,0) if phase == 'down' else (150,150,150)
    cv2.putText(frame, f"Phase: {phase}",
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    
    return frame


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print(f"[XLF] Starting AI Overseer — {MACHINE_NAME}")
    print(f"[XLF] Server: {SERVER_URL or 'NOT CONFIGURED'}")
    print(f"[XLF] Recording: {RECORD_SESSIONS}")
    
    # Camera setup
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        exit(1)
    
    print(f"[CAM] {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FRAME_RATE}fps")
    
    # Session recording
    recorder = SessionRecorder(RECORDINGS_DIR) if RECORD_SESSIONS else None
    
    # Rep counter
    counter = RepCounter()
    
    # Session inactivity timeout (no person for 30s = end session)
    last_detection_time = time.time()
    SESSION_TIMEOUT_S = 300  # 5 minutes away = session ends
    no_person_frames = 0
    
    frame_count = 0
    fps_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[CAM] Frame read failed")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Run YOLO pose estimation
            results = model(
                frame,
                conf=YOLO_CONFIDENCE,
                device=YOLO_DEVICE if YOLO_DEVICE != 'cpu' else None,
                verbose=False,
                stream=False
            )
            
            arm_data = None
            person_detected = False
            
            if results and len(results[0].keypoints.data) > 0:
                # Get the most prominent person (largest bounding box)
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    # Pick person with largest box area
                    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes.xyxy]
                    best_idx = areas.index(max(areas))
                    
                    keypoints = results[0].keypoints.data[best_idx].cpu().numpy()
                    # Normalise to 0-1
                    keypoints[:, 0] /= frame.shape[1]
                    keypoints[:, 1] /= frame.shape[0]
                    
                    arm_data = get_best_arm(keypoints)
                    person_detected = True
                    last_detection_time = time.time()
                    no_person_frames = 0
            else:
                no_person_frames += 1
                
            # Session inactivity check
            if person_detected is False and counter.session_active:
                if time.time() - last_detection_time > SESSION_TIMEOUT_S:
                    counter.end_session()
                    if recorder:
                        recorder.stop_recording()
            
            # Update rep counter
            rep_event = counter.update(arm_data)
            
            # Session recording
            if recorder and RECORD_SESSIONS:
                if counter.session_active and RECORD_ONLY_WHEN_ENGAGED:
                    if not recorder.is_recording:
                        recorder.start_recording(MACHINE_ID)
                    recorder.write_frame(frame)
                elif not RECORD_ONLY_WHEN_ENGAGED:
                    if not recorder.is_recording:
                        recorder.start_recording(MACHINE_ID)
                    recorder.write_frame(frame)
                elif recorder.is_recording and not counter.session_active:
                    recorder.stop_recording()
            
            # Display preview (only if monitor connected)
            if SHOW_PREVIEW:
                display = frame.copy()
                if person_detected and arm_data:
                    display = draw_skeleton(display, keypoints, arm_data,
                                          counter.rep_count, counter.phase)
                cv2.imshow('XL Fitness AI Overseer', display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # FPS logging every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - fps_time
                fps = 100 / elapsed
                fps_time = time.time()
                print(f"[FPS] {fps:.1f} | Reps: {counter.rep_count} | Phase: {counter.phase} | Person: {person_detected}")
    
    except KeyboardInterrupt:
        print("\n[XLF] Shutting down...")
    finally:
        if counter.session_active:
            counter.end_session()
        if recorder and recorder.is_recording:
            recorder.stop_recording()
        cap.release()
        cv2.destroyAllWindows()
        print("[XLF] Done.")


if __name__ == '__main__':
    main()
