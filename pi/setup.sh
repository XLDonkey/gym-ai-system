#!/bin/bash
# XL Fitness AI Overseer — Pi Setup Script
# Run once after flashing Raspberry Pi OS to your Pi 5
# Usage: bash setup.sh

set -e

echo "======================================"
echo " XL Fitness AI Overseer — Pi Setup"
echo "======================================"
echo ""

# ── System update ──────────────────────────────────────────────────────────
echo "[1/7] Updating system..."
sudo apt-get update -q
sudo apt-get upgrade -y -q

# ── Install system dependencies ────────────────────────────────────────────
echo "[2/7] Installing system packages..."
sudo apt-get install -y -q \
    python3-pip \
    python3-venv \
    python3-opencv \
    git \
    rclone \
    libcamera-dev \
    libcap-dev

# ── Create virtual environment ─────────────────────────────────────────────
echo "[3/7] Creating Python virtual environment..."
python3 -m venv /home/pi/xlf-env
source /home/pi/xlf-env/bin/activate

# ── Install Python packages ────────────────────────────────────────────────
echo "[4/7] Installing Python packages..."
pip install --quiet --upgrade pip
pip install --quiet \
    ultralytics \
    opencv-python \
    requests \
    numpy

# ── Clone the repo ─────────────────────────────────────────────────────────
echo "[5/7] Cloning gym-ai-system repo..."
cd /home/pi
if [ -d "gym-ai-system" ]; then
    echo "  Repo already exists, pulling latest..."
    cd gym-ai-system && git pull origin main
else
    git clone https://github.com/XLDonkey/gym-ai-system.git
    cd gym-ai-system
fi

# ── Download YOLO model ────────────────────────────────────────────────────
echo "[6/7] Downloading YOLOv11 pose model..."
cd /home/pi/gym-ai-system/pi
python3 -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"
echo "  Model downloaded."

# ── Set up cron jobs ───────────────────────────────────────────────────────
echo "[7/7] Setting up cron jobs..."

# Run main.py on boot
MAIN_CRON="@reboot sleep 30 && /home/pi/xlf-env/bin/python3 /home/pi/gym-ai-system/pi/main.py >> /home/pi/xlf.log 2>&1"

# Nightly upload Mon-Fri at 2am
UPLOAD_CRON="0 2 * * 1-5 /home/pi/xlf-env/bin/python3 /home/pi/gym-ai-system/pi/uploader.py >> /home/pi/upload.log 2>&1"

# Auto-update model from GitHub Mon-Fri at 3am
UPDATE_CRON="0 3 * * 1-5 cd /home/pi/gym-ai-system && git pull origin main >> /home/pi/update.log 2>&1"

(crontab -l 2>/dev/null; echo "$MAIN_CRON") | crontab -
(crontab -l 2>/dev/null; echo "$UPLOAD_CRON") | crontab -
(crontab -l 2>/dev/null; echo "$UPDATE_CRON") | crontab -

echo ""
echo "======================================"
echo " Setup complete!"
echo "======================================"
echo ""
echo "NEXT STEPS:"
echo "1. Edit /home/pi/gym-ai-system/pi/config.py"
echo "   - Set MACHINE_ID to your machine name"
echo "   - Set SERVER_URL when Manus provides the IP"
echo "   - Set GOOGLE_DRIVE_FOLDER_ID for uploads"
echo ""
echo "2. Set up rclone for Google Drive uploads:"
echo "   rclone config"
echo "   (follow prompts to connect your Google Drive)"
echo ""
echo "3. Test the camera:"
echo "   libcamera-still -o test.jpg"
echo ""
echo "4. Run manually to test:"
echo "   source /home/pi/xlf-env/bin/activate"
echo "   python3 /home/pi/gym-ai-system/pi/main.py"
echo ""
echo "5. Reboot to start automatically:"
echo "   sudo reboot"
echo ""
