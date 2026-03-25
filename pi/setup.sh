#!/bin/bash
# ============================================================
# XL Fitness AI Overseer — Pi Setup Script
# Run ONCE after flashing Raspberry Pi OS on a new Pi 5.
# Usage: bash setup.sh
# ============================================================

set -e

PI_USER="${USER:-pi}"
HOME_DIR="/home/$PI_USER"
REPO_DIR="$HOME_DIR/gym-ai-system"
VENV_DIR="$HOME_DIR/xlf-env"
RECORDINGS_DIR="$HOME_DIR/xlf_recordings"
GDRIVE_FOLDER_ID="1KNDC4wctZqVt8s41U4ALWHJ45OM5U9FA"

echo ""
echo "========================================"
echo " XL Fitness AI Overseer — Pi Setup"
echo " User: $PI_USER"
echo "========================================"
echo ""

# 1. System packages
echo "[1/8] Installing system packages..."
sudo apt-get update -q
sudo apt-get install -y -q \
    python3-pip python3-venv python3-opencv \
    git rclone libcamera-dev libcap-dev ffmpeg v4l-utils

# 2. Recordings directory
echo "[2/8] Creating recordings directory..."
mkdir -p "$RECORDINGS_DIR"

# 3. Python venv
echo "[3/8] Creating Python virtual environment..."
[ ! -d "$VENV_DIR" ] && python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# 4. Python packages
echo "[4/8] Installing Python packages..."
pip install --quiet --upgrade pip
pip install --quiet ultralytics opencv-python requests numpy

# 5. Clone / update repo
echo "[5/8] Cloning gym-ai-system repo..."
cd "$HOME_DIR"
if [ -d "$REPO_DIR" ]; then
    cd "$REPO_DIR" && git pull origin main
else
    git clone https://github.com/XLDonkey/gym-ai-system.git
fi

# 6. Download YOLO model
echo "[6/8] Downloading YOLOv11 pose model..."
cd "$REPO_DIR/pi"
"$VENV_DIR/bin/python3" -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"
echo "  Model ready."

# 7. systemd service (replaces cron — auto-restart on crash)
echo "[7/8] Installing systemd service..."
sudo tee /etc/systemd/system/xlf-overseer.service > /dev/null << SERVICE
[Unit]
Description=XL Fitness AI Overseer
After=network.target
Wants=network.target

[Service]
Type=simple
User=$PI_USER
WorkingDirectory=$REPO_DIR/pi
ExecStart=$VENV_DIR/bin/python3 $REPO_DIR/pi/main.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=xlf-overseer
WatchdogSec=60

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable xlf-overseer.service
echo "  Service installed and enabled on boot."

# 8. Google Drive instructions
echo "[8/8] Google Drive setup..."
echo ""
echo "  MANUAL STEP — On your Mac:"
echo "    brew install rclone"
echo "    rclone authorize \"drive\""
echo "  Copy the token, then on this Pi:"
echo "    bash $REPO_DIR/pi/setup_gdrive.sh"
echo ""

echo "========================================"
echo " Setup complete!"
echo "========================================"
echo ""
echo "NEXT STEPS:"
echo "1. Set up Google Drive:  bash $REPO_DIR/pi/setup_gdrive.sh"
echo "2. Check config:         nano $REPO_DIR/pi/config.py"
echo "3. Test camera:          libcamera-still -o ~/test.jpg"
echo "4. Start service:        sudo systemctl start xlf-overseer"
echo "5. Watch logs:           sudo journalctl -u xlf-overseer -f"
echo "6. Reboot test:          sudo reboot"
echo ""
