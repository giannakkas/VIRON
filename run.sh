#!/bin/bash
# VIRON — Start all services
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

echo "🤖 Starting VIRON..."
echo ""

# Check/start Ollama (local LLM for simple questions)
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    MODEL=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys,json;[print(m['name']) for m in json.load(sys.stdin).get('models',[])]" 2>/dev/null | head -1)
    echo "  ✓ Ollama running (model: ${MODEL:-unknown})"
else
    echo "  ⚠ Starting Ollama..."
    ollama serve &>/dev/null &
    sleep 3
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "  ✓ Ollama started"
    else
        echo "  ⚠ Ollama failed to start — simple questions will route to cloud"
    fi
fi

# Check if phi3 model is available
if curl -s http://localhost:11434/api/tags 2>/dev/null | grep -q "phi3"; then
    echo "  ✓ phi3 model ready"
else
    echo "  ⚠ phi3 model not found. Pulling..."
    ollama pull phi3 &
    echo "    (downloading in background — simple questions use cloud until done)"
fi

# Check Flask config
if [ ! -f "$SCRIPT_DIR/backend/config.json" ]; then
    echo ""
    echo "  ❌ Backend not configured!"
    echo "     Run: sudo bash setup-local.sh"
    echo "     Or create backend/config.json with your Anthropic API key"
    exit 1
fi

# Load .env if it exists (for GOOGLE_API_KEY, OPENAI_API_KEY etc)
if [ -f "$SCRIPT_DIR/backend/.env" ]; then
    echo "  ✓ Loading backend/.env"
    set -a
    source "$SCRIPT_DIR/backend/.env"
    set +a
fi

# Start Flask backend (port 5000) — now includes AI routing!
echo "  🖥️  Starting VIRON server (port 5000)..."
cd "$SCRIPT_DIR"
python3 backend/server.py &
FLASK_PID=$!
sleep 2

# Check it started
if curl -s http://localhost:5000/api/ping > /dev/null 2>&1; then
    echo "  ✓ VIRON server running"
else
    echo "  ⚠ Server may still be starting..."
fi

# Check router status
ROUTER_STATUS=$(curl -s http://localhost:5000/api/chat/status 2>/dev/null)
if echo "$ROUTER_STATUS" | python3 -c "import sys,json;d=json.load(sys.stdin);print('  ✓ AI Router: Ollama=%s, Claude=%s, Gemini=%s, ChatGPT=%s' % (d['providers']['ollama']['configured'], d['providers']['claude']['configured'], d['providers']['gemini']['configured'], d['providers']['chatgpt']['configured']))" 2>/dev/null; then
    true
else
    echo "  ⚠ AI Router status unknown"
fi

# Start Wake Word Server (port 9000) — optional
WAKE_PID=""
if python3 -c "import openwakeword" 2>/dev/null; then
    echo "  🎤 Starting Wake Word Server (port 9000)..."
    cd "$SCRIPT_DIR"
    python3 wake-word/wake_server.py &
    WAKE_PID=$!
    sleep 2
    echo "  ✓ Wake word server running"
else
    echo "  ℹ  openWakeWord not installed — using browser wake word"
fi

echo ""
echo "  ═══════════════════════════════════════════════"
echo "  🤖 VIRON is running!"
echo "  ═══════════════════════════════════════════════"
echo "  🖥️  UI:       http://localhost:5000"
echo "  📊 Status:   http://localhost:5000/api/chat/status"
if [ -n "$WAKE_PID" ]; then
echo "  🎤 Wake:     ws://localhost:9000"
fi
echo ""
echo "  Routing: Simple → Ollama | Complex → Claude Opus"
echo "  Fallback: Claude → Gemini → ChatGPT → Ollama"
echo "  ═══════════════════════════════════════════════"
echo ""
echo "  Press Ctrl+C to stop everything"
echo ""

# Handle shutdown
cleanup() {
    echo ""
    echo "🛑 Stopping VIRON..."
    kill $FLASK_PID $WAKE_PID 2>/dev/null
    exit
}
trap cleanup INT TERM

# Wait
wait
