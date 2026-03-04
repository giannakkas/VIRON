#!/bin/bash
# VIRON Wake Word + Hardware Setup — Jetson Orin Nano
# ReSpeaker XVF3800 4-Mic Array + Brio 4K Camera
# Run: bash wakeword/setup.sh

set -e
echo "🎯 VIRON Hardware & Wake Word Setup"
echo "===================================="

# Check Python version
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python: $PYVER"

# Install system deps
echo "📦 Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq libspeexdsp-dev alsa-utils fswebcam v4l-utils

# Install Python packages (uses arecord for mic — no pyaudio needed)
echo "📦 Installing Python packages..."
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MINOR" -ge 11 ]; then
  pip3 install --break-system-packages openwakeword flask flask-cors
else
  pip3 install openwakeword flask flask-cors
fi

# Download pre-trained models
echo "📥 Downloading openWakeWord models..."
python3 -c "
import openwakeword
openwakeword.utils.download_models()
print('✅ Models downloaded')
"

# Create models directory
mkdir -p ~/VIRON/wakeword/models

# Test openWakeWord
echo ""
echo "🧪 Testing openWakeWord..."
python3 -c "
from openwakeword.model import Model
m = Model(wakeword_models=['hey_jarvis_v0.1'], vad_threshold=0.5)
print(f'✅ openWakeWord working! Models: {list(m.models.keys())}')
"

# Test ReSpeaker mic
echo ""
echo "🎤 Testing ReSpeaker mic..."
echo "   ALSA cards:"
cat /proc/asound/cards 2>/dev/null || echo "   (no cards found)"
echo ""
if arecord -D plughw:2,0 -f S16_LE -r 16000 -c 1 -d 1 /tmp/test_respeaker.wav 2>/dev/null; then
  SIZE=$(stat -c%s /tmp/test_respeaker.wav 2>/dev/null || echo 0)
  echo "✅ ReSpeaker mic working! (recorded ${SIZE} bytes)"
  rm -f /tmp/test_respeaker.wav
else
  echo "⚠ ReSpeaker not on card 2 — check 'arecord -l' and set VIRON_MIC_DEVICE"
fi

# Test Brio camera
echo ""
echo "📷 Testing Brio camera..."
if [ -e /dev/video0 ]; then
  echo "   /dev/video0 exists"
  v4l2-ctl -d /dev/video0 --list-formats-ext 2>/dev/null | head -10 || true
  if fswebcam -d /dev/video0 -r 640x480 --no-banner /tmp/test_camera.jpg 2>/dev/null; then
    SIZE=$(stat -c%s /tmp/test_camera.jpg 2>/dev/null || echo 0)
    echo "✅ Brio camera working! (snapshot ${SIZE} bytes)"
    rm -f /tmp/test_camera.jpg
  else
    echo "⚠ fswebcam failed — try: sudo apt install fswebcam"
  fi
else
  echo "⚠ /dev/video0 not found"
fi

echo ""
echo "===================================="
echo "✅ Setup complete!"
echo ""
echo "Start services:"
echo "  cd ~/VIRON"
echo "  python3 wakeword/service.py > /tmp/viron_wakeword.log 2>&1 &"
echo "  python3 backend/server.py > /tmp/viron_flask.log 2>&1 &"
echo "  cd gateway && python3 main.py > /tmp/viron_gateway.log 2>&1 &"
echo ""
echo "Environment variables:"
echo "  VIRON_MIC_DEVICE=plughw:2,0     ReSpeaker ALSA device"
echo "  VIRON_CAMERA_DEVICE=/dev/video0  Brio camera"
echo "  VIRON_WAKEWORD_THRESHOLD=0.5     Detection sensitivity"
echo "  VIRON_WAKEWORD_PORT=8085         Wake word service port"
echo "===================================="
