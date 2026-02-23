#!/bin/bash
# ============================================
# VIRON Local Setup — Ubuntu Desktop
# Run: sudo bash setup-local.sh
# ============================================

set -e
echo "🤖 VIRON Local Setup — Ubuntu Desktop"
echo "======================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run with sudo: sudo bash setup-local.sh${NC}"
    exit 1
fi

ACTUAL_USER=${SUDO_USER:-$USER}
VIRON_DIR=$(cd "$(dirname "$0")" && pwd)

echo -e "${GREEN}[1/5] Installing system packages...${NC}"
apt-get update -qq
apt-get install -y -qq python3-pip python3-opencv libopencv-dev \
    pulseaudio alsa-utils chromium-browser curl git > /dev/null 2>&1
echo "  ✓ System packages installed"

echo -e "${GREEN}[2/5] Installing Python packages...${NC}"
pip3 install flask flask-cors flask-socketio requests --break-system-packages -q 2>/dev/null || \
pip3 install flask flask-cors flask-socketio requests -q
echo "  ✓ Python packages installed"

echo -e "${GREEN}[3/5] Setting up config...${NC}"
CONFIG_FILE="$VIRON_DIR/backend/config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo ""
    echo -e "${YELLOW}╔══════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║  Anthropic API Key Required              ║${NC}"
    echo -e "${YELLOW}║  Get one at: console.anthropic.com       ║${NC}"
    echo -e "${YELLOW}╚══════════════════════════════════════════╝${NC}"
    echo ""
    read -p "  Enter your Anthropic API key (sk-ant-...): " API_KEY
    if [ -z "$API_KEY" ]; then
        echo -e "${RED}  No API key entered. You can add it later in backend/config.json${NC}"
        API_KEY="YOUR_API_KEY_HERE"
    fi
    cat > "$CONFIG_FILE" << EOCFG
{
    "anthropic_api_key": "$API_KEY",
    "model": "claude-sonnet-4-20250514",
    "camera_index": 0,
    "language": "auto",
    "volume": 75,
    "brightness": 80,
    "emotion_detection": true,
    "proactive_care": true,
    "port": 5000
}
EOCFG
    chown $ACTUAL_USER:$ACTUAL_USER "$CONFIG_FILE"
    chmod 600 "$CONFIG_FILE"
    echo "  ✓ Config created at backend/config.json"
else
    echo "  ✓ Config already exists"
fi

echo -e "${GREEN}[4/5] Setting permissions...${NC}"
chown -R $ACTUAL_USER:$ACTUAL_USER "$VIRON_DIR"
chmod +x "$VIRON_DIR/run.sh" 2>/dev/null || true
echo "  ✓ Permissions set"

echo -e "${GREEN}[5/5] Testing camera...${NC}"
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    cap.release()
    if ret:
        print('  ✓ Camera working (index 0)')
    else:
        print('  ⚠ Camera opened but no frame — check connection')
else:
    print('  ⚠ No camera found at index 0 — emotion detection will be simulated')
" 2>/dev/null || echo "  ⚠ Camera test skipped"

echo ""
echo -e "${GREEN}════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ VIRON is ready!${NC}"
echo -e "${GREEN}════════════════════════════════════════════${NC}"
echo ""
echo "  To start VIRON:"
echo "    cd $VIRON_DIR"
echo "    ./run.sh"
echo ""
echo "  Or manually:"
echo "    python3 backend/server.py"
echo "    Then open: http://localhost:5000"
echo ""
echo "  To edit API key:"
echo "    nano backend/config.json"
echo ""
