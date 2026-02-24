#!/bin/bash
# VIRON — Start all services
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

echo "🤖 Starting VIRON..."
echo ""

# Check Ollama
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  ✓ Ollama running"
else
    echo "  ⚠ Starting Ollama..."
    ollama serve &>/dev/null &
    sleep 3
    echo "  ✓ Ollama started"
fi

# Check AI Router config
if [ ! -f "$SCRIPT_DIR/ai-router/.env" ]; then
    echo "  ❌ AI Router not configured. Run: bash ai-router/setup.sh"
    exit 1
fi

# Check Flask config
if [ ! -f "$SCRIPT_DIR/backend/config.json" ]; then
    echo "  ❌ Backend not configured. Run: sudo bash setup-local.sh"
    exit 1
fi

# Start AI Router (port 8000)
echo "  🧠 Starting AI Router (port 8000)..."
cd "$SCRIPT_DIR/ai-router"
python3 main.py &
AI_PID=$!
sleep 2

# Check it started
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "  ✓ AI Router running"
else
    echo "  ⚠ AI Router may still be starting..."
fi

# Start Flask backend (port 5000)
echo "  🖥️  Starting Face Server (port 5000)..."
cd "$SCRIPT_DIR"
python3 backend/server.py &
FLASK_PID=$!
sleep 2

# Start Wake Word Server (port 9000)
WAKE_PID=""
if python3 -c "import openwakeword" 2>/dev/null; then
    echo "  🎤 Starting Wake Word Server (port 9000)..."
    cd "$SCRIPT_DIR"
    python3 wake-word/wake_server.py &
    WAKE_PID=$!
    sleep 2
    echo "  ✓ Wake word server running"
else
    echo "  ⚠ openWakeWord not installed — wake word disabled"
    echo "    Install: bash wake-word/setup.sh"
fi

echo ""
echo "  ═══════════════════════════════════════"
echo "  🤖 VIRON is running!"
echo "  ═══════════════════════════════════════"
echo "  🖥️  Face:      http://localhost:5000"
echo "  🧠 AI API:    http://localhost:8000/docs"
if [ -n "$WAKE_PID" ]; then
echo "  🎤 Wake Word: ws://localhost:9000"
fi
echo "  ═══════════════════════════════════════"
echo ""
echo "  Press Ctrl+C to stop everything"
echo ""

# Handle shutdown
trap "echo ''; echo '🛑 Stopping VIRON...'; kill $AI_PID $FLASK_PID $WAKE_PID 2>/dev/null; exit" INT TERM

# Wait for either to exit
wait
