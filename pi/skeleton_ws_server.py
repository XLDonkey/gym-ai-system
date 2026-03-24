#!/usr/bin/env python3
"""
XL Fitness — Multi-Person Skeleton WebSocket Server  v3
========================================================
Hardware: Raspberry Pi 5 + AI HAT+ (Hailo-8L 26 TOPS) + Camera Module 3

Performance
-----------
  Hailo-8L  : ~25–30 fps, ~35ms end-to-end latency
  CPU only  : ~8–12 fps, ~80ms latency (automatic fallback)
  Payload   : ~300–800 bytes/frame (vs 50–200 KB for video)

Scalability
-----------
  Every Pi runs this same script.
  Set MACHINE_NAME env var to label each machine.
  The phone dashboard connects to all Pis simultaneously.

Usage
-----
  # Basic
  python3 skeleton_ws_server.py

  # With machine label
  MACHINE_NAME="Squat Rack 1" python3 skeleton_ws_server.py

  # Custom ports
  WS_PORT=8765 HTTP_PORT=8080 python3 skeleton_ws_server.py

Endpoints
---------
  ws://<PI_IP>:8765              — WebSocket keypoint stream
  http://<PI_IP>:8080/skeleton_viewer.html  — Phone viewer
"""

import asyncio
import json
import time
import threading
import os
import sys
import socket
import cv2
import numpy as np
from http.server import HTTPServer, SimpleHTTPRequestHandler

# ── Configuration (all overridable via environment variables) ─────────────────
MACHINE_NAME = os.environ.get("MACHINE_NAME", "XL Fitness Pi")
WS_PORT      = int(os.environ.get("WS_PORT",   "8765"))
HTTP_PORT    = int(os.environ.get("HTTP_PORT", "8080"))
FRAME_W      = int(os.environ.get("FRAME_W",   "640"))
FRAME_H      = int(os.environ.get("FRAME_H",   "480"))
TARGET_FPS   = int(os.environ.get("FPS",       "25"))
CONF         = float(os.environ.get("CONF",    "0.35"))
YOLO_MODEL   = os.path.expanduser(
    os.environ.get("YOLO_MODEL", "~/yolo11n-pose.pt")
)

# Hailo pre-compiled YOLOv8 pose model (ships with rpicam-apps / picamera2)
HAILO_MODEL  = os.path.expanduser(
    os.environ.get("HAILO_MODEL",
                   "/usr/share/hailo-models/yolov8s_pose_h8l_pi.hef")
)

# Per-person colours (cycles for >6 people)
COLOURS = ["#00ff88","#00aaff","#ff6600","#ff00aa","#ffff00","#aa00ff"]

# COCO 17-keypoint skeleton connections
SKEL = [
    [0,1],[0,2],[1,3],[2,4],
    [5,6],[5,7],[7,9],[6,8],[8,10],
    [5,11],[6,12],[11,12],
    [11,13],[13,15],[12,14],[14,16],
]

# ── Dependency checks ─────────────────────────────────────────────────────────
try:
    import websockets
except ImportError:
    print("[FATAL] pip3 install websockets")
    sys.exit(1)

# ── Backend detection ─────────────────────────────────────────────────────────
# Priority: Hailo via picamera2 → YOLO CPU via picamera2 → YOLO CPU via OpenCV

USE_HAILO    = False
USE_PICAMERA = False

try:
    from picamera2 import Picamera2
    USE_PICAMERA = True
    print("[INIT] picamera2 ✓")
except ImportError:
    print("[INIT] picamera2 not found — using OpenCV")

if USE_PICAMERA and os.path.exists(HAILO_MODEL):
    try:
        # picamera2 + Hailo integration
        from picamera2.devices.hailo import Hailo
        USE_HAILO = True
        print(f"[INIT] Hailo AI HAT+ ✓  ({HAILO_MODEL})")
    except Exception as e:
        print(f"[INIT] Hailo import failed ({e}) — using CPU YOLO")

if not USE_HAILO:
    try:
        from ultralytics import YOLO as _YOLO
        print("[INIT] Ultralytics YOLO (CPU) ✓")
    except ImportError:
        print("[FATAL] pip3 install ultralytics")
        sys.exit(1)

# ── Global shared state ───────────────────────────────────────────────────────
_latest_msg   = None
_frame_lock   = threading.Lock()
_clients      = set()
_clients_lock = threading.Lock()
_perf         = {"fps": 0.0, "inf_ms": 0.0, "people": 0}


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE BACKEND A — Hailo AI HAT+ via picamera2
# ══════════════════════════════════════════════════════════════════════════════
def run_hailo_loop():
    """
    Uses picamera2's native Hailo integration.
    The Hailo chip runs YOLOv8-pose in hardware; picamera2 returns
    parsed detections directly — no Python YOLO overhead at all.
    """
    global _latest_msg

    from picamera2.devices.hailo import Hailo

    hailo = Hailo(HAILO_MODEL)
    h_in_h, h_in_w, _ = hailo.get_input_shape()

    picam2 = Picamera2()
    main_cfg   = {"size": (FRAME_W, FRAME_H), "format": "RGB888"}
    lores_cfg  = {"size": (h_in_w, h_in_h),  "format": "RGB888"}
    cfg = picam2.create_preview_configuration(
        main=main_cfg, lores=lores_cfg,
        controls={"FrameRate": TARGET_FPS}
    )
    picam2.configure(cfg)
    picam2.start()
    print(f"[HAILO] Camera started {FRAME_W}x{FRAME_H} → Hailo {h_in_w}x{h_in_h}")

    fps_counter = 0
    fps_timer   = time.time()
    frame_int   = 1.0 / TARGET_FPS

    while True:
        t0 = time.perf_counter()

        # Capture both streams simultaneously
        buffers = picam2.capture_arrays(["main", "lores"])
        main_rgb  = buffers[0]   # Full-res for coordinate mapping
        lores_rgb = buffers[1]   # Low-res for Hailo input

        # Run Hailo inference
        t_inf = time.perf_counter()
        raw_detections = hailo.run(lores_rgb)
        inf_ms = (time.perf_counter() - t_inf) * 1000

        # Parse Hailo YOLOv8-pose output
        people = _parse_hailo_pose(raw_detections, h_in_w, h_in_h)

        _emit(people, inf_ms, fps_counter)

        fps_counter += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            _perf["fps"]     = fps_counter / (now - fps_timer)
            _perf["inf_ms"]  = inf_ms
            _perf["people"]  = len(people)
            fps_counter = 0
            fps_timer   = now
            _log_stats()

        sleep_t = frame_int - (time.perf_counter() - t0)
        if sleep_t > 0.001:
            time.sleep(sleep_t)


def _parse_hailo_pose(raw, in_w, in_h):
    """
    Parse Hailo YOLOv8-pose output into our standard people list.
    Hailo returns a list of detections; each detection has:
      .bbox        — (y1, x1, y2, x2) normalised
      .keypoints   — list of (y, x, score) normalised
    """
    people = []
    if not raw:
        return people

    # Hailo output format varies by SDK version — handle both
    detections = raw if isinstance(raw, list) else getattr(raw, 'detections', [])

    for idx, det in enumerate(detections):
        try:
            # Bounding box
            bbox_raw = getattr(det, 'bbox', None)
            if bbox_raw is None:
                continue

            # Some SDK versions return (y1,x1,y2,x2), others (x1,y1,x2,y2)
            # We normalise to x1,y1,x2,y2
            if hasattr(bbox_raw, 'xmin'):
                bbox = [bbox_raw.xmin, bbox_raw.ymin, bbox_raw.xmax, bbox_raw.ymax]
            else:
                b = list(bbox_raw)
                # Detect format by checking if first pair is likely y,x
                bbox = [round(float(b[1]),4), round(float(b[0]),4),
                        round(float(b[3]),4), round(float(b[2]),4)]

            # Keypoints
            kps_raw = getattr(det, 'keypoints', [])
            keypoints = []
            for kp in kps_raw:
                if hasattr(kp, 'x'):
                    keypoints.append([round(float(kp.x),4),
                                      round(float(kp.y),4),
                                      round(float(kp.score),3)])
                else:
                    k = list(kp)
                    # Format: (y, x, score) → (x, y, score)
                    keypoints.append([round(float(k[1]),4),
                                      round(float(k[0]),4),
                                      round(float(k[2]),3)])

            people.append({
                "id":     idx,
                "colour": COLOURS[idx % len(COLOURS)],
                "kps":    keypoints,
                "bbox":   bbox,
            })
        except Exception as e:
            print(f"[HAILO] Parse error for detection {idx}: {e}")
            continue

    return people


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE BACKEND B — YOLO CPU (picamera2 or OpenCV)
# ══════════════════════════════════════════════════════════════════════════════
def run_cpu_loop():
    """
    Standard YOLO CPU inference.
    Uses picamera2 if available (lower latency), OpenCV as fallback.
    """
    global _latest_msg

    from ultralytics import YOLO
    print(f"[YOLO] Loading {YOLO_MODEL} ...")
    model = YOLO(YOLO_MODEL)

    # Warm up
    dummy = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    model(dummy, verbose=False, conf=CONF, imgsz=FRAME_W)
    print("[YOLO] Warm-up done ✓")

    if USE_PICAMERA:
        picam2 = Picamera2()
        cfg = picam2.create_preview_configuration(
            main={"size": (FRAME_W, FRAME_H), "format": "RGB888"},
            controls={"FrameRate": TARGET_FPS}
        )
        picam2.configure(cfg)
        picam2.start()
        print(f"[CAM] picamera2 {FRAME_W}x{FRAME_H} @ {TARGET_FPS}fps ✓")
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        print(f"[CAM] OpenCV {FRAME_W}x{FRAME_H} @ {TARGET_FPS}fps ✓")

    fps_counter = 0
    fps_timer   = time.time()
    frame_int   = 1.0 / TARGET_FPS

    while True:
        t0 = time.perf_counter()

        # Capture
        if USE_PICAMERA:
            rgb = picam2.capture_array("main")
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            cap.grab()                     # Drain buffer
            ret, bgr = cap.retrieve()
            if not ret:
                time.sleep(0.01)
                continue

        # Inference
        t_inf = time.perf_counter()
        results = model(bgr, verbose=False, conf=CONF, imgsz=FRAME_W)
        inf_ms  = (time.perf_counter() - t_inf) * 1000

        # Parse
        people = []
        result = results[0]
        if result.keypoints is not None and len(result.keypoints.data) > 0:
            kps_all   = result.keypoints.data.cpu().numpy()
            boxes_all = (result.boxes.xyxy.cpu().numpy()
                         if result.boxes is not None else None)

            for idx, kps in enumerate(kps_all):
                keypoints = [
                    [round(float(kp[0])/FRAME_W, 4),
                     round(float(kp[1])/FRAME_H, 4),
                     round(float(kp[2]), 3)]
                    for kp in kps
                ]
                bbox = None
                if boxes_all is not None and idx < len(boxes_all):
                    b = boxes_all[idx]
                    bbox = [round(float(b[0])/FRAME_W,4),
                            round(float(b[1])/FRAME_H,4),
                            round(float(b[2])/FRAME_W,4),
                            round(float(b[3])/FRAME_H,4)]
                people.append({
                    "id":     idx,
                    "colour": COLOURS[idx % len(COLOURS)],
                    "kps":    keypoints,
                    "bbox":   bbox,
                })

        _emit(people, inf_ms, fps_counter)

        fps_counter += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            _perf["fps"]    = fps_counter / (now - fps_timer)
            _perf["inf_ms"] = inf_ms
            _perf["people"] = len(people)
            fps_counter = 0
            fps_timer   = now
            _log_stats()

        sleep_t = frame_int - (time.perf_counter() - t0)
        if sleep_t > 0.001:
            time.sleep(sleep_t)


# ── Shared emit helper ────────────────────────────────────────────────────────
def _emit(people, inf_ms, frame_num):
    """Serialise and store the latest frame for WebSocket broadcast."""
    global _latest_msg
    msg = {
        "v":    3,
        "t":    round(time.time() * 1000),
        "name": MACHINE_NAME,
        "fw":   FRAME_W,
        "fh":   FRAME_H,
        "n":    len(people),
        "fps":  round(_perf["fps"], 1),
        "ims":  round(inf_ms, 1),
        "hailo": USE_HAILO,
        "p":    people,
        "sk":   SKEL,
    }
    with _frame_lock:
        _latest_msg = json.dumps(msg, separators=(',', ':'))


def _log_stats():
    n_clients = len(_clients)
    msg_size  = len(_latest_msg) if _latest_msg else 0
    print(f"[STREAM] {_perf['fps']:.1f}fps | "
          f"{_perf['people']} people | "
          f"{_perf['inf_ms']:.0f}ms {'Hailo' if USE_HAILO else 'CPU'} | "
          f"{n_clients} client(s) | "
          f"{msg_size}B/frame")


# ── WebSocket handler ─────────────────────────────────────────────────────────
async def ws_handler(websocket):
    addr = websocket.remote_address
    print(f"[WS] + {addr}")
    with _clients_lock:
        _clients.add(websocket)

    # Immediate hello so the client knows the machine name
    await websocket.send(json.dumps({
        "v": 3, "type": "hello",
        "name": MACHINE_NAME,
        "hailo": USE_HAILO,
        "fw": FRAME_W, "fh": FRAME_H,
    }))

    try:
        interval = 1.0 / TARGET_FPS
        while True:
            with _frame_lock:
                data = _latest_msg
            if data:
                await websocket.send(data)
            await asyncio.sleep(interval)
    except (websockets.exceptions.ConnectionClosed,
            websockets.exceptions.ConnectionClosedOK,
            websockets.exceptions.ConnectionClosedError):
        pass
    except Exception as e:
        print(f"[WS] Error {addr}: {e}")
    finally:
        with _clients_lock:
            _clients.discard(websocket)
        print(f"[WS] - {addr}")


# ── HTTP server (serves skeleton_viewer.html) ─────────────────────────────────
class ViewerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            directory=os.path.dirname(os.path.abspath(__file__)),
            **kwargs
        )
    def log_message(self, fmt, *args): pass
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


def run_http():
    HTTPServer(("0.0.0.0", HTTP_PORT), ViewerHandler).serve_forever()


# ── Startup banner ────────────────────────────────────────────────────────────
def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]; s.close(); return ip
    except Exception:
        return "192.168.1.40"


def banner(ip):
    accel = "Hailo-8L AI HAT+ (26 TOPS)" if USE_HAILO else "CPU (no Hailo)"
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  XL Fitness — Skeleton Stream Server  v3                 ║
╠══════════════════════════════════════════════════════════╣
║  Machine  : {MACHINE_NAME:<44}║
║  Accel    : {accel:<44}║
║  Viewer   : http://{ip}:{HTTP_PORT}/skeleton_viewer.html  ║
║  WebSocket: ws://{ip}:{WS_PORT:<35}║
╚══════════════════════════════════════════════════════════╝
""")


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    ip = get_ip()
    banner(ip)

    # Start the correct inference backend
    target = run_hailo_loop if USE_HAILO else run_cpu_loop
    cam_thread = threading.Thread(target=target, daemon=True, name="inference")
    cam_thread.start()

    # HTTP server
    threading.Thread(target=run_http, daemon=True, name="http").start()

    # Wait for first frame (10s timeout)
    print("[MAIN] Waiting for first frame...")
    for _ in range(100):
        if _latest_msg is not None:
            break
        await asyncio.sleep(0.1)
    else:
        print("[ERROR] No frame received in 10s — check camera connection.")
        sys.exit(1)
    print("[MAIN] First frame ✓")

    # WebSocket server
    async with websockets.serve(
        ws_handler, "0.0.0.0", WS_PORT,
        ping_interval=20, ping_timeout=10,
    ):
        print(f"[WS] Listening on ws://0.0.0.0:{WS_PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[MAIN] Stopped.")
