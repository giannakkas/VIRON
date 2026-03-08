#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# VIRON — Start Everything on Jetson Orin Nano
# ═══════════════════════════════════════════════════════════════
# Starts: llama.cpp (Gemma 2B) → wakeword → Flask → Gateway
# Usage:  cd ~/VIRON && bash start.sh
# Stop:   bash start.sh stop
# ═══════════════════════════════════════════════════════════════

cd "$(dirname "$0")"

# Stop mode
if [ "$1" = "stop" ]; then
    echo "🛑 Stopping VIRON..."
    pkill -f "llama-server" 2>/dev/null
    pkill -f "voice_pipeline.py" 2>/dev/null
    pkill -f "wakeword/service.py" 2>/dev/null
    pkill -f "backend/server.py" 2>/dev/null
    pkill -f "gateway/main.py" 2>/dev/null
    pkill -f "viron_kiosk.py" 2>/dev/null
    sleep 2
    echo "✓ Stopped"
    exit 0
fi

# Load API keys
if [ -f .env ]; then
    set -a; source .env; set +a
    echo "✓ Loaded .env"
fi
export OPENAI_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY

# Kill any existing VIRON processes
echo "🧹 Cleaning up old processes..."
pkill -f "llama-server" 2>/dev/null
pkill -f "voice_pipeline.py" 2>/dev/null
pkill -f "wakeword/service.py" 2>/dev/null
pkill -f "backend/server.py" 2>/dev/null
pkill -f "gateway/main.py" 2>/dev/null
pkill -f "viron_kiosk.py" 2>/dev/null
sleep 2

# ─── Find llama-server ───
LLAMA_SERVER=""
for p in \
    "$PWD/llama.cpp/build/bin/llama-server" \
    "/opt/llama.cpp/build/bin/llama-server" \
    "$(which llama-server 2>/dev/null)"; do
    if [ -n "$p" ] && [ -f "$p" ]; then
        LLAMA_SERVER="$p"
        break
    fi
done

# ─── Find Gemma model ───
GEMMA_MODEL=""
for p in \
    "$PWD/models/gemma-2-2b-it-Q4_K_M.gguf" \
    "$PWD/models/gemma-2b.gguf" \
    "$HOME/models/gemma-2-2b-it-Q4_K_M.gguf"; do
    if [ -f "$p" ]; then
        GEMMA_MODEL="$p"
        break
    fi
done

echo ""
echo "🤖 ═══════════════════════════════════════════"
echo "   VIRON — Starting All Services"
echo "   ═══════════════════════════════════════════"

# ─── 1. Start llama.cpp (Gemma 2B on GPU) ───
if [ -n "$LLAMA_SERVER" ] && [ -n "$GEMMA_MODEL" ]; then
    echo "🧠 Starting Gemma 2B on GPU (port 8081)..."
    $LLAMA_SERVER \
        --model "$GEMMA_MODEL" \
        --port 8081 \
        --host 0.0.0.0 \
        --n-gpu-layers 99 \
        --ctx-size 2048 \
        --threads 4 \
        --parallel 2 \
        > /tmp/viron_llama.log 2>&1 &
    echo "   PID: $! (log: /tmp/viron_llama.log)"
    
    # Wait for model to load
    echo "   ⏳ Loading model..."
    for i in $(seq 1 45); do
        if curl -sf http://localhost:8081/health >/dev/null 2>&1; then
            echo "   ✅ Gemma 2B ready!"
            break
        fi
        [ "$i" -eq 45 ] && echo "   ⚠ Timeout — check /tmp/viron_llama.log"
        sleep 2
    done
else
    echo "⚠ llama-server or Gemma model not found"
    [ -z "$LLAMA_SERVER" ] && echo "   Run: bash scripts/build_llamacpp.sh"
    [ -z "$GEMMA_MODEL" ] && echo "   Run: bash scripts/download_models.sh"
    echo "   Gateway will use cloud-only mode"
fi

# ─── 2. Start Voice Pipeline (wake word + STT + AI + TTS — port 8085) ───
echo ""
echo "🧠 Starting Voice Pipeline (port 8085)..."
python3 voice_pipeline.py > /tmp/viron_pipeline.log 2>&1 &
echo "   PID: $! (log: /tmp/viron_pipeline.log)"
sleep 5

# ─── 3. Start Flask backend ───
echo ""
echo "🖥️  Starting Flask backend (port 5000)..."
python3 backend/server.py > /tmp/viron_flask.log 2>&1 &
echo "   PID: $! (log: /tmp/viron_flask.log)"
sleep 3

# ─── 4. Start Gateway ───
echo ""
echo "🌐 Starting Gateway (port 8080)..."
# Point both router and tutor to same Gemma 2B instance
# (Mistral 7B doesn't fit on 8GB Jetson)
export ROUTER_URL=http://127.0.0.1:8081
export TUTOR_URL=http://127.0.0.1:8081
export DB_PATH="$PWD/gateway/data/viron.db"
mkdir -p gateway/data
cd gateway && python3 main.py > /tmp/viron_gateway.log 2>&1 &
cd ..
echo "   PID: $! (log: /tmp/viron_gateway.log)"
sleep 2

# ─── Summary ───
echo ""
echo "═══════════════════════════════════════════"
echo "🤖 VIRON is running!"
echo "═══════════════════════════════════════════"
echo "   🖥️  Face: http://$(hostname -I | awk '{print $1}'):5000"
echo "   🧠 LLM:  http://localhost:8081 (Gemma 2B)"
echo "   🌐 API:  http://localhost:8080/v1/chat"
echo "   🎯 Pipeline: http://localhost:8085 (wake+STT+AI+TTS)"
echo "═══════════════════════════════════════════"
echo ""
echo "   Logs: tail -f /tmp/viron_*.log"
echo "   Stop: bash start.sh stop"
echo ""

# Show running status
sleep 1
echo "Status:"
curl -sf http://localhost:8081/health >/dev/null 2>&1 && echo "   ✅ Gemma 2B (8081)" || echo "   ❌ Gemma 2B (8081)"
curl -sf http://localhost:8085/health >/dev/null 2>&1 && echo "   ✅ Voice Pipeline (8085)" || echo "   ❌ Voice Pipeline (8085)"
curl -sf http://localhost:5000/api/ping >/dev/null 2>&1 && echo "   ✅ Flask (5000)" || echo "   ❌ Flask (5000)"
curl -sf http://localhost:8080/health >/dev/null 2>&1 && echo "   ✅ Gateway (8080)" || echo "   ❌ Gateway (8080)"
echo ""

# ─── 5. Restart the face (viron_kiosk.py) ───
echo "🔄 Restarting face (viron_kiosk.py)..."
export DISPLAY=:0
export XAUTHORITY=/home/test/.Xauthority
pkill -f "viron_kiosk.py" 2>/dev/null
sleep 2
python3 /home/test/viron_kiosk.py &>/dev/null &
echo "   ✅ Face restarted (PID: $!)"
echo ""

# ─── 6. Tail logs (ALL services) ───
echo "═══════════════════════════════════════════"
echo "📋 LIVE LOGS — ALL SERVICES (Ctrl+C to stop watching)"
echo "═══════════════════════════════════════════"
echo ""
tail -f /tmp/viron_gateway.log /tmp/viron_pipeline.log /tmp/viron_flask.log
