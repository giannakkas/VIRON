#!/bin/bash
# ═══════════════════════════════════════════════════════════
# VIRON — Gemini Live Pipeline Startup
# ═══════════════════════════════════════════════════════════
# Starts: Voice Pipeline (Gemini Live) → Flask Backend → Gateway → Face
# Usage:  cd ~/VIRON && bash start_live.sh
# Stop:   bash start_live.sh stop
# ═══════════════════════════════════════════════════════════

cd "$(dirname "$0")"

# Stop mode
if [ "$1" = "stop" ]; then
    echo "🛑 Stopping VIRON..."
    pkill -f "gemini_live_pipeline.py" 2>/dev/null
    pkill -f "voice_pipeline.py" 2>/dev/null
    pkill -f "llama-server" 2>/dev/null
    pkill -f "backend/server.py" 2>/dev/null
    pkill -f "gateway/main.py" 2>/dev/null
    pkill -f "viron_kiosk.py" 2>/dev/null
    pkill -f "aplay.*plughw" 2>/dev/null
    sleep 2
    echo "✓ Stopped"
    exit 0
fi

# Load API keys
if [ -f .env ]; then
    set -a; source .env; set +a
    echo "✓ Loaded .env"
fi
export GEMINI_API_KEY PICOVOICE_ACCESS_KEY

# Kill any existing processes
echo "🧹 Cleaning up old processes..."
pkill -f "gemini_live_pipeline.py" 2>/dev/null
pkill -f "voice_pipeline.py" 2>/dev/null
pkill -f "llama-server" 2>/dev/null
pkill -f "backend/server.py" 2>/dev/null
pkill -f "gateway/main.py" 2>/dev/null
pkill -f "viron_kiosk.py" 2>/dev/null
pkill -f "aplay.*plughw" 2>/dev/null
sleep 2

echo ""
echo "🤖 ═══════════════════════════════════════════"
echo "   VIRON — Gemini Live Native Audio"
echo "   ═══════════════════════════════════════════"

# Validate required keys
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY not set in .env!"
    exit 1
fi
if [ -z "$PICOVOICE_ACCESS_KEY" ]; then
    echo "❌ PICOVOICE_ACCESS_KEY not set in .env!"
    exit 1
fi

# ─── Configure microphone ───
echo ""
echo "🎤 Configuring microphone..."
amixer -c 0 sset 'Capture' 60% 2>/dev/null || true
amixer -c 0 sset 'Mic' 60% 2>/dev/null || true
amixer -c 0 sset 'Auto Gain Control' off 2>/dev/null || true
echo "   ✅ Mic configured (60% gain — reduces TV pickup)"

# ─── 1. Start Flask Backend (port 5000) — serves face UI ───
echo ""
echo "🖥️  Starting Flask backend (port 5000)..."
python3 backend/server.py > /tmp/viron_flask.log 2>&1 &
echo "   PID: $! (log: /tmp/viron_flask.log)"
sleep 3

# ─── 2. Start Gemini Live Pipeline (port 8085) ───
echo ""
echo "🧠 Starting Gemini Live Pipeline (port 8085)..."
python3 gemini_live_pipeline.py > /tmp/viron_pipeline.log 2>&1 &
PIPELINE_PID=$!
echo "   PID: $PIPELINE_PID (log: /tmp/viron_pipeline.log)"
sleep 5

# ─── Summary ───
echo ""
echo "═══════════════════════════════════════════"
echo "🤖 VIRON is running! (Gemini Live Mode)"
echo "═══════════════════════════════════════════"
echo "   🖥️  Face:     http://$(hostname -I | awk '{print $1}'):5000"
echo "   🧠 Pipeline:  http://localhost:8085 (Gemini Live)"
echo "   🤖 Model:     ${GEMINI_LIVE_MODEL:-gemini-2.5-flash-native-audio-preview-12-2025}"
echo "═══════════════════════════════════════════"
echo ""
echo "   Logs: tail -f /tmp/viron_pipeline.log"
echo "   Stop: bash start_live.sh stop"
echo ""

# Show status
sleep 1
echo "Status:"
curl -sf http://localhost:5000/api/ping >/dev/null 2>&1 && echo "   ✅ Flask (5000)" || echo "   ❌ Flask (5000)"
curl -sf http://localhost:8085/health >/dev/null 2>&1 && echo "   ✅ Pipeline (8085)" || echo "   ❌ Pipeline (8085)"
echo ""

# ─── Restart face ───
echo "🔄 Restarting face (viron_kiosk.py)..."
export DISPLAY=:0
export XAUTHORITY=/home/test/.Xauthority
pkill -f "viron_kiosk.py" 2>/dev/null
sleep 2
python3 /home/test/viron_kiosk.py &>/dev/null &
echo "   ✅ Face restarted (PID: $!)"

# Hide cursor
pkill -f "unclutter" 2>/dev/null
if command -v unclutter &>/dev/null; then
    unclutter -idle 0 -root &>/dev/null &
fi
echo ""

# ─── Tail logs ───
echo "═══════════════════════════════════════════"
echo "📋 LIVE LOGS (Ctrl+C to stop watching)"
echo "═══════════════════════════════════════════"
echo ""
tail -f /tmp/viron_pipeline.log
