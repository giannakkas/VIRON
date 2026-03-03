#!/bin/bash
# VIRON Wake Word Setup — openWakeWord on Jetson Orin Nano
# Run: bash wakeword/setup.sh

set -e
echo "🎯 VIRON Wake Word Service Setup"
echo "================================"

# Check Python version
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python: $PYVER"

# Install system deps
echo "📦 Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq libspeexdsp-dev

# Install Python packages (no pyaudio needed — mic is in browser)
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

# Create models directory for custom models
mkdir -p ~/VIRON/wakeword/models

# Test the installation
echo ""
echo "🧪 Testing openWakeWord..."
python3 -c "
from openwakeword.model import Model
m = Model(wakeword_models=['hey_jarvis_v0.1'], vad_threshold=0.5)
print(f'✅ openWakeWord working! Models: {list(m.models.keys())}')
"

echo ""
echo "================================"
echo "✅ Setup complete!"
echo ""
echo "To start the wake word service:"
echo "  cd ~/VIRON && python3 wakeword/service.py &"
echo ""
echo "To train a custom 'Hey VIRON' model (recommended):"
echo "  python3 wakeword/train_hey_viron.py"
echo ""
echo "Environment variables:"
echo "  VIRON_WAKEWORD_THRESHOLD=0.65  Detection sensitivity (0-1)"
echo "  VIRON_WAKEWORD_PORT=8085       Service port"
echo "================================"
echo "✅ Setup complete!"
echo ""
echo "To start the wake word service:"
echo "  cd ~/VIRON && python3 wakeword/service.py &"
echo ""
echo "To train a custom 'Hey VIRON' model (recommended):"
echo "  python3 wakeword/train_hey_viron.py"
echo ""
echo "Environment variables:"
echo "  VIRON_WAKEWORD_THRESHOLD=0.65  Detection sensitivity (0-1)"
echo "  VIRON_WAKEWORD_PORT=8085       Service port"
echo "  VIRON_MIC_DEVICE=              Mic device index (from list above)"
echo "================================"
