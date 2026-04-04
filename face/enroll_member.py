"""
XL Fitness AI Overseer — Member Enrollment CLI
Captures face samples from a webcam or image file, extracts an InsightFace
embedding, and registers the member in Supabase.

Usage:
    # Enroll from webcam (press SPACE to capture, Q to quit)
    python3 face/enroll_member.py --name "John Smith"

    # Enroll from an image file
    python3 face/enroll_member.py --name "John Smith" --image /path/to/photo.jpg

    # List all enrolled members
    python3 face/enroll_member.py --list

    # Remove a member
    python3 face/enroll_member.py --remove <member_id>

Environment:
    SUPABASE_URL          — your project URL  (e.g. https://xxx.supabase.co)
    SUPABASE_SERVICE_KEY  — service-role key (not the anon key)
"""

import sys
import os
import argparse
import numpy as np

# Resolve repo root so we can import siblings
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from face.face_recognizer import FaceRecognizer
from members.db_client import SupabaseClient

# Number of face samples to average during webcam enrolment.
# More samples = more stable embedding, but takes longer.
NUM_SAMPLES = 5


def enroll_from_webcam(db: SupabaseClient, recognizer: FaceRecognizer, name: str):
    """Capture NUM_SAMPLES face crops and average their embeddings."""
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python not installed. Run: pip install opencv-python")
        sys.exit(1)

    print(f"\n[enroll] Enrolling: {name}")
    print(f"[enroll] Position face in frame. Press SPACE to capture ({NUM_SAMPLES} samples needed). Press Q to cancel.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam (index 0).")
        sys.exit(1)

    samples    = []
    last_msg   = ""

    while len(samples) < NUM_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()

        # Show capture count
        msg = f"Captured: {len(samples)}/{NUM_SAMPLES}  [SPACE=capture  Q=cancel]"
        if msg != last_msg:
            last_msg = msg
        cv2.putText(display, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Enroll Member — XL Fitness", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[enroll] Cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return

        if key == ord(' '):
            emb = recognizer.extract_embedding(frame)
            if emb is None:
                print("[enroll] No face detected in frame — try again.")
            else:
                samples.append(emb)
                print(f"[enroll] Sample {len(samples)}/{NUM_SAMPLES} captured.")

    cap.release()
    cv2.destroyAllWindows()

    # Average the embeddings and re-normalise
    avg_emb = np.mean(np.stack(samples), axis=0)
    norm    = np.linalg.norm(avg_emb)
    if norm > 1e-9:
        avg_emb /= norm

    _save_member(db, recognizer, name, avg_emb)


def enroll_from_image(db: SupabaseClient, recognizer: FaceRecognizer, name: str, image_path: str):
    """Extract embedding from a still photo."""
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python not installed.")
        sys.exit(1)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: Cannot read image: {image_path}")
        sys.exit(1)

    emb = recognizer.extract_embedding(frame)
    if emb is None:
        print(f"ERROR: No face detected in {image_path}")
        sys.exit(1)

    print(f"[enroll] Face detected in image.")
    _save_member(db, recognizer, name, emb)


def _save_member(db: SupabaseClient, recognizer: FaceRecognizer, name: str, embedding: np.ndarray):
    """Persist embedding to Supabase and update the local cache."""
    emb_list = embedding.tolist()

    print(f"[enroll] Saving '{name}' to Supabase...")
    try:
        row = db.create_member(name=name, face_embedding=emb_list)
        member_id = row["id"]
        print(f"[enroll] SUCCESS — {name} enrolled with ID: {member_id}")
        recognizer.add_member(member_id, name, emb_list)
    except Exception as e:
        print(f"[enroll] ERROR saving to Supabase: {e}")
        sys.exit(1)


def list_members(db: SupabaseClient):
    members = db.get_all_members()
    if not members:
        print("No members enrolled.")
        return

    print(f"\n{'ID':<38}  {'Name':<30}  {'Enrolled At'}")
    print("-" * 85)
    for m in members:
        print(f"{m['id']:<38}  {m['name']:<30}  {m.get('enrolled_at', 'N/A')}")
    print(f"\nTotal: {len(members)} member(s)")


def remove_member(db: SupabaseClient, member_id: str):
    confirm = input(f"Remove member {member_id}? [y/N] ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return
    try:
        db.deactivate_member(member_id)
        print(f"[enroll] Member {member_id} deactivated.")
    except Exception as e:
        print(f"ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="XL Fitness — Member Face Enrollment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--name",   help="Member full name (required for enroll)")
    parser.add_argument("--image",  help="Path to image file (skips webcam capture)")
    parser.add_argument("--list",   action="store_true", help="List all enrolled members")
    parser.add_argument("--remove", metavar="MEMBER_ID", help="Deactivate a member by UUID")
    parser.add_argument(
        "--threshold", type=float, default=0.40,
        help="Cosine similarity threshold for matching (default: 0.40)",
    )
    args = parser.parse_args()

    # Validate env vars before loading heavy models
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    if not supabase_url or not supabase_key:
        print("ERROR: Set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables.")
        print("  export SUPABASE_URL=https://xxxx.supabase.co")
        print("  export SUPABASE_SERVICE_KEY=eyJ...")
        sys.exit(1)

    db = SupabaseClient()

    if args.list:
        list_members(db)
        return

    if args.remove:
        remove_member(db, args.remove)
        return

    # Enroll
    if not args.name:
        parser.error("--name is required for enrollment")

    recognizer = FaceRecognizer(threshold=args.threshold)
    if not recognizer.ready:
        print("ERROR: InsightFace model failed to load — check installation.")
        sys.exit(1)

    if args.image:
        enroll_from_image(db, recognizer, args.name, args.image)
    else:
        enroll_from_webcam(db, recognizer, args.name)


if __name__ == "__main__":
    main()
