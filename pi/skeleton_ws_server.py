#!/usr/bin/env python3
"""
XL Fitness — Multi-Person Skeleton WebSocket Server
Runs on Raspberry Pi 5 + Camera Module 3

Streams all detected people's keypoints to connected browsers via WebSocket.
The phone viewer (skeleton_viewer.html) connects and renders the skeleton.

Usage:
    python3 skeleton_ws_server.py

Serves:
    WebSocket:  ws://192.168.1.40:8765   — keypoint stream
    HTTP:       http://192.168.1.40:8080  — serves skeleton_viewer.html
"""

import asyncio
import json
import time
import threading
import os
import cv2
import numpy as np
from http.server import HTTPServer, SimpleHTTPRequestHandler
from datetime import datetime

# ── Try to import websockets ──────────────────────────────────────────────────
try:
    import websockets
except ImportError:
    print("[ERROR] websockets not installed. Run: pip3 install websockets")
    exit(1)

# ── Try to import YOLO ────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics not installed. Run: pip3 install ultralytics")
    exit(1)

# ── Try to import picamera2 ───────────────────────────────────────────────────
try:
    from picamera2 import Picamera2
    USE_PICAMERA = True
except ImportError:
    USE_PICAMERA = False
    print("[WARN] picamera2 not found — falling back to OpenCV VideoCapture")

# ── Config ────────────────────────────────────────────────────────────────────
WS_HOST       = "0.0.0.0"
WS_PORT       = 8765
HTTP_PORT     = 8080
FRAME_W       = 640
FRAME_H       = 480
YOLO_MODEL    = os.path.expanduser("~/yolo11n-pose.pt")
CONF_THRESH   = 0.35
KP_CONF_MIN   = 0.2      # Minimum keypoint confidence to include
TARGET_FPS    = 20
SKIP_FRAMES   = 1        # Process every Nth frame (1 = every frame)

# Assign a distinct colour per person (cycles if >6 people)
PERSON_COLOURS = [
    "#00ff88",  # green
    "#00aaff",  # blue
    "#ff6600",  # orange
    "#ff00aa",  # pink
    "#ffff00",  # yellow
    "#aa00ff",  # purple
]

# COCO 17-keypoint skeleton connections
SKELETON_PAIRS = [
    [0, 1], [0, 2],           # nose → eyes
    [1, 3], [2, 4],           # eyes → ears
    [5, 6],                   # shoulders
    [5, 7], [7, 9],           # left arm
    [6, 8], [8, 10],          # right arm
    [5, 11], [6, 12],         # torso sides
    [11, 12],                 # hips
    [11, 13], [13, 15],       # left leg
    [12, 14], [14, 16],       # right leg
]

# ── Global state ──────────────────────────────────────────────────────────────
latest_frame_data = None
frame_lock = threading.Lock()
connected_clients = set()
clients_lock = threading.Lock()

# ── Model loading ─────────────────────────────────────────────────────────────
print(f"[YOLO] Loading model: {YOLO_MODEL}")
model = YOLO(YOLO_MODEL)
print("[YOLO] Model loaded ✓")

# ── Camera capture thread ─────────────────────────────────────────────────────
def camera_loop():
    global latest_frame_data

    if USE_PICAMERA:
        picam2 = Picamera2()
        cfg = picam2.create_preview_configuration(
            main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}
        )
        picam2.configure(cfg)
        picam2.start()
        print(f"[CAM] picamera2 started at {FRAME_W}x{FRAME_H}")
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        print(f"[CAM] OpenCV VideoCapture started at {FRAME_W}x{FRAME_H}")

    frame_interval = 1.0 / TARGET_FPS
    frame_count = 0

    while True:
        t0 = time.time()

        # Capture frame
        if USE_PICAMERA:
            rgb = picam2.capture_array("main")
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            ret, bgr = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

        frame_count += 1
        if frame_count % SKIP_FRAMES != 0:
            continue

        # Run YOLO pose on the frame
        results = model(bgr, verbose=False, conf=CONF_THRESH, imgsz=FRAME_W)

        people = []
        result = results[0]

        if result.keypoints is not None and len(result.keypoints.data) > 0:
            kps_all = result.keypoints.data.cpu().numpy()  # shape: (N, 17, 3)
            boxes_all = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else None

            for person_idx, kps in enumerate(kps_all):
                colour = PERSON_COLOURS[person_idx % len(PERSON_COLOURS)]

                # Build keypoints list: [x_norm, y_norm, confidence]
                keypoints = []
                for kp in kps:
                    x_norm = float(kp[0]) / FRAME_W
                    y_norm = float(kp[1]) / FRAME_H
                    conf   = float(kp[2])
                    keypoints.append([
                        round(x_norm, 4),
                        round(y_norm, 4),
                        round(conf, 3)
                    ])

                # Bounding box (normalised)
                bbox = None
                if boxes_all is not None and person_idx < len(boxes_all):
                    b = boxes_all[person_idx]
                    bbox = [
                        round(float(b[0]) / FRAME_W, 4),
                        round(float(b[1]) / FRAME_H, 4),
                        round(float(b[2]) / FRAME_W, 4),
                        round(float(b[3]) / FRAME_H, 4),
                    ]

                people.append({
                    "id":     person_idx,
                    "colour": colour,
                    "kps":    keypoints,
                    "bbox":   bbox,
                })

        # Build the message
        msg = {
            "t":       round(time.time() * 1000),  # ms timestamp
            "fw":      FRAME_W,
            "fh":      FRAME_H,
            "n":       len(people),
            "people":  people,
            "pairs":   SKELETON_PAIRS,
        }

        with frame_lock:
            latest_frame_data = json.dumps(msg)

        # Throttle to target FPS
        elapsed = time.time() - t0
        sleep_t = frame_interval - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)


# ── WebSocket handler ─────────────────────────────────────────────────────────
async def ws_handler(websocket):
    client_addr = websocket.remote_address
    print(f"[WS] Client connected: {client_addr}")

    with clients_lock:
        connected_clients.add(websocket)

    try:
        while True:
            with frame_lock:
                data = latest_frame_data

            if data:
                try:
                    await websocket.send(data)
                except websockets.exceptions.ConnectionClosed:
                    break

            await asyncio.sleep(1.0 / TARGET_FPS)

    except Exception as e:
        print(f"[WS] Client {client_addr} error: {e}")
    finally:
        with clients_lock:
            connected_clients.discard(websocket)
        print(f"[WS] Client disconnected: {client_addr}")


# ── HTTP server (serves the viewer HTML) ─────────────────────────────────────
class ViewerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Serve files from the same directory as this script
        super().__init__(*args, directory=os.path.dirname(os.path.abspath(__file__)), **kwargs)

    def log_message(self, format, *args):
        pass  # Suppress HTTP access logs


def run_http_server():
    server = HTTPServer(("0.0.0.0", HTTP_PORT), ViewerHandler)
    print(f"[HTTP] Serving on http://0.0.0.0:{HTTP_PORT}")
    server.serve_forever()


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    # Start camera thread
    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()

    # Start HTTP server thread
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()

    # Wait for first frame
    print("[MAIN] Waiting for first frame...")
    while latest_frame_data is None:
        await asyncio.sleep(0.1)
    print("[MAIN] First frame received ✓")

    # Start WebSocket server
    print(f"[WS] Starting WebSocket server on ws://0.0.0.0:{WS_PORT}")
    async with websockets.serve(ws_handler, WS_HOST, WS_PORT):
        print(f"\n{'='*50}")
        print(f"  XL Fitness Multi-Person Skeleton Server")
        print(f"  Viewer: http://<PI_IP>:{HTTP_PORT}/skeleton_viewer.html")
        print(f"  WS:     ws://<PI_IP>:{WS_PORT}")
        print(f"{'='*50}\n")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
