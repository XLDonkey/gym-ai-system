#!/usr/bin/env python3
"""
XL Fitness — Review Portal Server
Serves the review portal locally on the Mac Mini.

Usage:
    python3 pose/review_server.py
    # Then open: http://localhost:8787

The portal lists all pending clips from data/review/, shows a skeleton
animation, and lets you click the correct class. Saves true_class back
to the JSON file automatically.

No dependencies beyond Python stdlib + numpy.
"""
import http.server
import json
import os
import sys
import urllib.parse
from pathlib import Path

import numpy as np

REPO_ROOT   = Path(__file__).parent.parent
REVIEW_DIR  = REPO_ROOT / "data" / "review"
PORTAL_HTML = Path(__file__).parent / "review.html"
PORT        = 8787

CLASS_NAMES = [
    "no_person", "user_present", "on_machine",
    "good_rep",  "bad_rep",      "false_rep",
    "resting",   "half_rep",
]


def load_pending_clips():
    """Find all _meta.json files where review_status == 'pending'."""
    clips = []
    for meta_file in sorted(REVIEW_DIR.rglob("*_meta.json")):
        try:
            with open(meta_file) as f:
                meta = json.load(f)
        except Exception:
            continue

        if meta.get("review_status") != "pending":
            continue

        # Load keypoints from companion .npy if available,
        # otherwise use embedded array from meta JSON
        kps_list = meta.get("keypoints_array")
        if kps_list is None:
            npy_file = meta_file.parent / meta_file.name.replace("_meta.json", "_kps.npy")
            if npy_file.exists():
                arr = np.load(str(npy_file))  # (30, 51)
                kps_list = arr.tolist()

        clips.append({
            "id":          meta_file.stem.replace("_meta", ""),
            "path":        str(meta_file.relative_to(REPO_ROOT)),
            "meta":        meta,
            "keypoints":   kps_list,
        })

    return clips


def save_annotation(meta_path_rel: str, true_class: int):
    """Save the annotator's decision back to the meta JSON."""
    meta_file = REPO_ROOT / meta_path_rel
    with open(meta_file) as f:
        meta = json.load(f)

    meta["true_class"]    = true_class
    meta["review_status"] = "reviewed"
    meta["reviewed_label"] = CLASS_NAMES[true_class] if true_class < len(CLASS_NAMES) else "unknown"

    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    return meta


class ReviewHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # suppress default access log noise

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path

        if path == "/" or path == "/review":
            # Serve the portal HTML
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(PORTAL_HTML, "rb") as f:
                self.wfile.write(f.read())

        elif path == "/api/clips":
            # Return pending clips as JSON
            clips = load_pending_clips()
            self._json({"clips": clips, "total": len(clips)})

        elif path == "/api/stats":
            # Quick count stats
            all_meta = list(REVIEW_DIR.rglob("*_meta.json"))
            pending  = sum(1 for p in all_meta if json.load(open(p)).get("review_status") == "pending")
            reviewed = len(all_meta) - pending
            self._json({"total": len(all_meta), "pending": pending, "reviewed": reviewed})

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/annotate":
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length))

            meta_path  = body.get("path")
            true_class = int(body.get("true_class"))

            if not meta_path or true_class < 0 or true_class > 7:
                self.send_response(400)
                self.end_headers()
                return

            updated = save_annotation(meta_path, true_class)
            print(f"  ✓ Annotated: {Path(meta_path).stem}  class={CLASS_NAMES[true_class]}")
            self._json({"ok": True, "updated": updated})

        else:
            self.send_response(404)
            self.end_headers()

    def _json(self, data):
        payload = json.dumps(data, indent=2).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    if not PORTAL_HTML.exists():
        print(f"ERROR: review.html not found at {PORTAL_HTML}")
        sys.exit(1)

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    clips = load_pending_clips()
    print(f"\n  XL Fitness — Review Portal")
    print(f"  Pending clips:  {len(clips)}")
    print(f"  Review dir:     {REVIEW_DIR}")
    print(f"  Open browser:   http://localhost:{PORT}")
    print(f"  Stop:           Ctrl+C\n")

    server = http.server.HTTPServer(("localhost", PORT), ReviewHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Portal stopped.")
