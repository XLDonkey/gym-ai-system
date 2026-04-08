#!/bin/bash
# ============================================================
# XL Fitness Gym Overseer AI — Mac Mini Training Server Setup
# Run once on a fresh Mac Mini.
# Usage: bash setup_mac_mini.sh
# ============================================================

set -e

REPO_URL="https://github.com/Matt-xlfitness/Gym-Overseer-AI.git"
REPO_DIR="$HOME/Gym-Overseer-AI"
VENV_DIR="$HOME/.xlf-env"

echo ""
echo "=================================================="
echo "  XL Fitness AI — Mac Mini Training Server Setup"
echo "=================================================="
echo ""

# ── 1. Clone repo ─────────────────────────────────────────
echo "[1/6] Cloning Gym-Overseer-AI repo..."
if [ -d "$REPO_DIR" ]; then
    echo "  Repo already exists — pulling latest..."
    cd "$REPO_DIR" && git pull origin main
else
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi
echo "  Repo ready at $REPO_DIR"

# ── 2. Python virtual environment ─────────────────────────
echo ""
echo "[2/6] Setting up Python virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  Created: $VENV_DIR"
else
    echo "  Already exists: $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# ── 3. Install Python packages ─────────────────────────────
echo ""
echo "[3/6] Installing Python packages (this takes a few minutes)..."
pip install --quiet --upgrade pip
pip install torch torchvision
pip install ultralytics          # YOLOv11 pose extraction
pip install onnx onnxruntime     # ONNX export + verify
pip install numpy opencv-python scikit-learn
pip install anthropic yt-dlp     # auto-labeller
pip install gdown                # Google Drive downloads
echo "  All packages installed."

# ── 4. Move existing YOLO models into repo ────────────────
echo ""
echo "[4/6] Checking for existing YOLO models..."
mkdir -p "$REPO_DIR/models/weights"
for MODEL in yolov8n-pose.pt yolov8n.pt yolo11n-pose.pt yolo11n.pt; do
    if [ -f "$HOME/$MODEL" ]; then
        cp "$HOME/$MODEL" "$REPO_DIR/models/weights/$MODEL"
        echo "  Copied $MODEL → models/weights/"
    fi
done
# Pre-download YOLOv11 pose if not present
if [ ! -f "$REPO_DIR/models/weights/yolo11n-pose.pt" ]; then
    echo "  Downloading yolo11n-pose.pt..."
    cd "$REPO_DIR/models/weights"
    "$VENV_DIR/bin/python3" -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"
    cd "$REPO_DIR"
fi
echo "  YOLO models ready."

# ── 5. Create data directories ────────────────────────────
echo ""
echo "[5/6] Creating data directories..."
mkdir -p "$REPO_DIR/data/raw"
mkdir -p "$REPO_DIR/data/annotations"
mkdir -p "$REPO_DIR/data/processed"
mkdir -p "$REPO_DIR/data/review"
mkdir -p "$REPO_DIR/models/weights"
echo "  Directories ready."

# ── 6. Shell shortcuts ────────────────────────────────────
echo ""
echo "[6/6] Adding shell shortcuts to ~/.zshrc..."

SHORTCUTS='
# ── XL Fitness AI Overseer ────────────────────────────────
alias xlf="cd ~/Gym-Overseer-AI && source ~/.xlf-env/bin/activate"
alias xlf-train="cd ~/Gym-Overseer-AI && source ~/.xlf-env/bin/activate && make train"
alias xlf-stats="cd ~/Gym-Overseer-AI && source ~/.xlf-env/bin/activate && make stats"
alias xlf-pending="cd ~/Gym-Overseer-AI && source ~/.xlf-env/bin/activate && make pending"
alias xlf-logs="ssh pi@\$PI_IP sudo journalctl -u xlf-overseer -f"
# Set your Pi IP here:
export PI_IP="192.168.1.XX"
# ─────────────────────────────────────────────────────────
'

# Only add once
if ! grep -q "XL Fitness AI Overseer" "$HOME/.zshrc" 2>/dev/null; then
    echo "$SHORTCUTS" >> "$HOME/.zshrc"
    echo "  Shortcuts added."
else
    echo "  Shortcuts already present."
fi

# ── Done ──────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  Setup complete!"
echo "=================================================="
echo ""
echo "NEXT STEPS:"
echo ""
echo "  1. Set your Pi's IP address:"
echo "     nano ~/.zshrc   # update PI_IP=192.168.1.XX"
echo ""
echo "  2. Start a new terminal tab and type:"
echo "     xlf             # → takes you into the project"
echo ""
echo "  3. Check annotation progress:"
echo "     xlf-stats"
echo ""
echo "  4. When you have 300+ annotations, train the model:"
echo "     xlf-train"
echo ""
echo "  5. Deploy to a Pi:"
echo "     make deploy PI=pi@\$PI_IP"
echo ""
echo "Repo:  $REPO_DIR"
echo "Env:   $VENV_DIR"
echo ""
