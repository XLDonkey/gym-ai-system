#!/usr/bin/env python3
"""
XL Fitness — Pi Snapshot Server
Runs a tiny HTTP server on port 8090.
GET /snapshot  → captures a JPEG from the Pi camera and returns it.
GET /health    → returns {"status":"ok"}

The dashboard calls this when you click the "Snapshot" button on a Pi card.
CORS is open so the browser can fetch it from any origin.

Usage:
    python3 snapshot_server.py

Auto-start (add to crontab):
    @reboot python3 /home/xlraspberry2026/gym-ai-system/pi/snapshot_server.py &
"""

import io
import json
import logging
import subprocess
import tempfile
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PORT = 8090
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("snapshot")


def capture_jpeg() -> bytes:
    """Capture a single JPEG frame from the Pi camera.
    Tries rpicam-still first (Pi OS Bookworm/Trixie), falls back to libcamera-jpeg.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = f.name

    # Determine which camera tool is available
    import shutil
    if shutil.which("rpicam-still"):
        cmd = ["rpicam-still", "--output", tmp_path, "--timeout", "2000",
               "--nopreview", "--width", "1280", "--height", "720", "--quality", "85"]
    elif shutil.which("libcamera-jpeg"):
        cmd = ["libcamera-jpeg", "--output", tmp_path, "--timeout", "2000",
               "--nopreview", "--width", "1280", "--height", "720", "--quality", "85"]
    elif shutil.which("libcamera-still"):
        cmd = ["libcamera-still", "--output", tmp_path, "--timeout", "2000",
               "--nopreview", "--width", "1280", "--height", "720", "--quality", "85"]
    else:
        raise RuntimeError("No camera capture tool found (rpicam-still / libcamera-jpeg / libcamera-still)")

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=12)
        if result.returncode != 0:
            raise RuntimeError(f"{cmd[0]} failed (exit {result.returncode}): {result.stderr.decode()[:200]}")

        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


class SnapshotHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress default access log noise; use our own
        log.info(f"{self.address_string()} — {format % args}")

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]  # strip query string

        if path == "/snapshot":
            try:
                log.info("Capturing snapshot...")
                jpeg = capture_jpeg()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Cache-Control", "no-store")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(jpeg)
                log.info(f"Snapshot served ({len(jpeg):,} bytes)")
            except Exception as e:
                log.error(f"Snapshot error: {e}")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        elif path == "/health":
            body = json.dumps({"status": "ok", "port": PORT}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_response(404)
            self.send_cors_headers()
            self.end_headers()


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), SnapshotHandler)
    log.info(f"XL Fitness Snapshot Server listening on port {PORT}")
    log.info(f"  GET http://localhost:{PORT}/snapshot  → live JPEG from camera")
    log.info(f"  GET http://localhost:{PORT}/health    → status check")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Stopped.")
