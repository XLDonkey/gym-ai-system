# XL Fitness AI Overseer — Raspberry Pi

Rep counting, session recording, and training data collection on Raspberry Pi 5 + AI HAT+.

---

## Hardware Required

| Item | Where to buy | Cost |
|------|-------------|------|
| Raspberry Pi 5 4GB | core-electronics.com.au | $146 |
| Raspberry Pi AI HAT+ (26 TOPS) | core-electronics.com.au | ~$140 |
| Raspberry Pi AI Camera (IMX500) | core-electronics.com.au | $124 |
| Official 27W USB-C Power Supply | core-electronics.com.au | ~$20 |
| Active Cooler | core-electronics.com.au | ~$15 |
| 128GB microSD card | JB HiFi / Core Electronics | ~$25 |
| Pi 5 Case | core-electronics.com.au | ~$15 |
| **Total** | | **~$485** |

---

## Quick Start

### 1. Flash the Pi
- Download Raspberry Pi Imager: raspberrypi.com/software
- Flash **Raspberry Pi OS (64-bit)** to the SD card
- In settings: set hostname, enable SSH, set WiFi credentials

### 2. SSH into the Pi
```bash
ssh pi@raspberrypi.local
```

### 3. Run the setup script
```bash
curl -s https://raw.githubusercontent.com/XLDonkey/gym-ai-system/main/pi/setup.sh | bash
```

This installs everything automatically.

### 4. Configure
Edit `/home/pi/gym-ai-system/pi/config.py`:
```python
MACHINE_ID = "lat_pulldown_01"   # Your machine name
SERVER_URL = ""                   # Manus's server IP (when ready)
GOOGLE_DRIVE_FOLDER_ID = ""      # Your Drive folder ID
```

### 5. Test
```bash
source /home/pi/xlf-env/bin/activate
python3 /home/pi/gym-ai-system/pi/main.py
```

You should see FPS and rep counting in the terminal.

### 6. Reboot to go live
```bash
sudo reboot
```

The Pi will automatically start counting reps on boot.

---

## What It Does

**During gym hours:**
- Runs YOLOv11-pose continuously
- Detects when someone sits at the machine (wrist-above-shoulder gate)
- Counts reps automatically
- Records sessions to `/home/pi/xlf_recordings/`
- Posts events to Manus's server (when configured)

**Every night at 2am (Mon-Fri):**
- Uploads recordings to Google Drive
- Deletes local copies after successful upload

**Every night at 3am:**
- Pulls latest model from GitHub (`git pull`)
- New model active from next session

---

## Files

| File | Purpose |
|------|---------|
| `main.py` | Main loop — camera → YOLO → rep counting |
| `config.py` | All settings — edit this for your machine |
| `session_recorder.py` | Records video clips of sessions |
| `uploader.py` | Uploads recordings to Google Drive |
| `setup.sh` | One-command setup script |

---

## Troubleshooting

**No camera detected:**
```bash
libcamera-still -o test.jpg  # Test camera
```

**YOLO not found:**
```bash
source /home/pi/xlf-env/bin/activate
pip install ultralytics
```

**Low FPS:**
- Confirm AI HAT+ is seated properly
- Set `YOLO_DEVICE = "hailo"` in config.py
- Reduce resolution: `FRAME_WIDTH = 640, FRAME_HEIGHT = 480`

**Check logs:**
```bash
tail -f /home/pi/xlf.log
```
