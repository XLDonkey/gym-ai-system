#!/bin/bash
# XL Fitness Overseer — Mac Mini Server Setup
# Run once: bash setup.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== XL Fitness Overseer Server Setup ==="

# Install Node.js if needed
if ! command -v node &> /dev/null; then
  echo "Installing Node.js via Homebrew..."
  brew install node
fi
echo "Node.js: $(node --version)"

# Install dependencies
echo "Installing npm packages..."
cd "$SCRIPT_DIR"
npm install

# Install LaunchAgent
PLIST_SRC="$SCRIPT_DIR/com.xlfitness.overseer-server.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/com.xlfitness.overseer-server.plist"

cp "$PLIST_SRC" "$PLIST_DEST"
launchctl unload "$PLIST_DEST" 2>/dev/null || true
launchctl load "$PLIST_DEST"

echo ""
echo "=== Done ==="
echo "Server running at http://localhost:3001"
echo "Test: curl http://localhost:3001/api/health"
echo "Logs: ~/Library/Logs/xlf-overseer-server.log"
