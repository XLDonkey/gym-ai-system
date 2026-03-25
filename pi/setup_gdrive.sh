#!/bin/bash
# ============================================================
# XL Fitness AI Overseer — Google Drive Setup via rclone
# Run this ONCE on each Pi to authenticate with Google Drive.
#
# Usage:
#   bash setup_gdrive.sh
#
# What it does:
#   1. Installs rclone if not present
#   2. Configures an rclone remote called "xlf-gdrive"
#      pointing to your shared Google Drive folder
#   3. Tests the connection by listing the folder
# ============================================================

set -e

REMOTE_NAME="xlf-gdrive"
GDRIVE_FOLDER_ID="1KNDC4wctZqVt8s41U4ALWHJ45OM5U9FA"

echo ""
echo "========================================"
echo " XL Fitness — Google Drive Setup"
echo "========================================"
echo ""

# ── 1. Install rclone ─────────────────────────────────────────────────────────
if ! command -v rclone &> /dev/null; then
    echo "[1/3] Installing rclone..."
    sudo apt-get install -y rclone
else
    echo "[1/3] rclone already installed: $(rclone version | head -1)"
fi

# ── 2. Configure rclone remote ────────────────────────────────────────────────
echo ""
echo "[2/3] Configuring Google Drive remote..."
echo ""
echo "  This will open a browser window on your Mac/phone to authorise"
echo "  the Pi to access your Google Drive."
echo ""
echo "  IMPORTANT: Run this command on your Mac first to create an"
echo "  authorisation token, then paste it back here:"
echo ""
echo "  ┌─────────────────────────────────────────────────────────────┐"
echo "  │  rclone authorize \"drive\"                                   │"
echo "  └─────────────────────────────────────────────────────────────┘"
echo ""
echo "  (If rclone is not on your Mac: brew install rclone)"
echo ""

# Write the rclone config non-interactively once we have the token
# We use a heredoc approach — user pastes token when prompted

rclone config create "$REMOTE_NAME" drive \
    scope drive \
    root_folder_id "$GDRIVE_FOLDER_ID" \
    --non-interactive 2>/dev/null || true

echo ""
echo "  If the above failed, run interactively:"
echo "  rclone config"
echo "  → New remote → name: xlf-gdrive → type: drive → follow prompts"
echo ""

# ── 3. Test the connection ────────────────────────────────────────────────────
echo "[3/3] Testing connection to Google Drive..."
echo ""

if rclone lsd "$REMOTE_NAME:" --drive-root-folder-id "$GDRIVE_FOLDER_ID" 2>/dev/null; then
    echo ""
    echo "  ✓ Google Drive connected successfully!"
    echo "  Recordings will upload to:"
    echo "  https://drive.google.com/drive/folders/$GDRIVE_FOLDER_ID"
else
    echo ""
    echo "  ✗ Connection test failed."
    echo "  Run 'rclone config' manually to complete setup."
    echo "  See: https://rclone.org/drive/"
fi

echo ""
echo "========================================"
echo " Setup complete — rclone remote: $REMOTE_NAME"
echo "========================================"
echo ""
echo "Test manually with:"
echo "  rclone ls xlf-gdrive: --drive-root-folder-id $GDRIVE_FOLDER_ID"
echo ""
