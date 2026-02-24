#!/bin/bash
# VIRON Wake Word Setup
echo "╔══════════════════════════════════════╗"
echo "║   VIRON Wake Word Setup              ║"
echo "╚══════════════════════════════════════╝"

echo "[1/3] Installing openWakeWord..."
pip3 install openwakeword websockets resampy

echo "[2/3] Downloading pre-trained models..."
python3 -c "import openwakeword; openwakeword.utils.download_models()" 2>/dev/null

echo "[3/3] Testing..."
python3 -c "
from openwakeword.model import Model
m = Model(inference_framework='onnx')
print('✅ Models loaded:', list(m.models.keys()))
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Wake word system ready!"
    echo "   Start with: python3 wake-word/wake_server.py"
    echo ""
    echo "   Pre-trained models include 'hey jarvis' which works"
    echo "   until we train a custom 'hey viron' model."
else
    echo "❌ Setup failed. Check errors above."
fi
