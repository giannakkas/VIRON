#!/bin/bash
# VIRON — Start/Restart All Services
# Usage: ./scripts/start_all.sh [stop|restart|status]

VIRON_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$VIRON_DIR"

# Load environment
for envfile in "$VIRON_DIR/.env" "$HOME/VIRON/.env" "$HOME/.env"; do
    if [ -f "$envfile" ]; then
        echo "   Loading $envfile"
        set -a
        source "$envfile"
        set +a
        break
    fi
done

if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  WARNING: OPENAI_API_KEY not set! STT and Chat won't work."
    echo "   Create ~/VIRON/.env with your API keys (see .env.example)"
fi

BACKEND_PORT=${VIRON_BACKEND_PORT:-5000}
GATEWAY_PORT=${VIRON_GATEWAY_PORT:-8080}
WAKEWORD_PORT=${VIRON_WAKEWORD_PORT:-8085}

stop_all() {
    echo "🛑 Stopping all VIRON services..."
    pkill -f "backend/server.py" 2>/dev/null
    pkill -f "gateway/main.py" 2>/dev/null
    pkill -f "wakeword/service.py" 2>/dev/null
    pkill -f "voice_pipeline.py" 2>/dev/null
    # Free ports
    fuser -k $BACKEND_PORT/tcp 2>/dev/null
    fuser -k $GATEWAY_PORT/tcp 2>/dev/null
    fuser -k $WAKEWORD_PORT/tcp 2>/dev/null
    # Kill PulseAudio (interferes with ALSA)
    pkill -9 pulseaudio 2>/dev/null
    sleep 2
    echo "   Done."
}

start_all() {
    echo "🚀 Starting VIRON services..."
    echo "   Dir: $VIRON_DIR"
    echo ""
    
    # 1. Backend (Flask, port 5000)
    echo "   [1/3] Backend (port $BACKEND_PORT)..."
    cd "$VIRON_DIR"
    python3 backend/server.py > /tmp/viron_flask.log 2>&1 &
    BACKEND_PID=$!
    sleep 2
    if kill -0 $BACKEND_PID 2>/dev/null; then
        echo "   ✅ Backend started (PID $BACKEND_PID)"
    else
        echo "   ❌ Backend failed! Check: tail /tmp/viron_flask.log"
    fi
    
    # 2. Gateway (FastAPI, port 8080)
    echo "   [2/3] Gateway (port $GATEWAY_PORT)..."
    cd "$VIRON_DIR/gateway"
    python3 main.py > /tmp/viron_gateway.log 2>&1 &
    GATEWAY_PID=$!
    cd "$VIRON_DIR"
    sleep 2
    if kill -0 $GATEWAY_PID 2>/dev/null; then
        echo "   ✅ Gateway started (PID $GATEWAY_PID)"
    else
        echo "   ❌ Gateway failed! Check: tail /tmp/viron_gateway.log"
    fi
    
    # 3. Voice Pipeline — Porcupine + Silero VAD + Faster-Whisper (port 8085)
    echo "   [3/3] Voice Pipeline (port $WAKEWORD_PORT)..."
    cd "$VIRON_DIR"
    python3 voice_pipeline.py > /tmp/viron_wakeword.log 2>&1 &
    WAKEWORD_PID=$!
    sleep 2
    if kill -0 $WAKEWORD_PID 2>/dev/null; then
        echo "   ✅ Wakeword started (PID $WAKEWORD_PID)"
    else
        echo "   ❌ Wakeword failed! Check: tail /tmp/viron_wakeword.log"
    fi
    
    echo ""
    echo "═══════════════════════════════════════"
    show_status
}

show_status() {
    echo "📊 VIRON Service Status"
    echo "═══════════════════════════════════════"
    
    # Backend
    if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$BACKEND_PORT/ | grep -q "200\|302\|404"; then
        echo "   ✅ Backend     http://127.0.0.1:$BACKEND_PORT"
    else
        echo "   ❌ Backend     NOT RUNNING (port $BACKEND_PORT)"
    fi
    
    # Gateway
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$GATEWAY_PORT/health 2>/dev/null)
    if [ "$STATUS" = "200" ]; then
        echo "   ✅ Gateway     http://127.0.0.1:$GATEWAY_PORT"
    else
        echo "   ❌ Gateway     NOT RUNNING (port $GATEWAY_PORT)"
    fi
    
    # Wakeword
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$WAKEWORD_PORT/wakeword/status 2>/dev/null)
    if [ "$STATUS" = "200" ]; then
        MODE=$(curl -s http://127.0.0.1:$WAKEWORD_PORT/wakeword/status 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'mode={d.get(\"mode\",\"?\")}, listening={d.get(\"listening\",\"?\")}, paused={d.get(\"paused\",\"?\")}')" 2>/dev/null)
        echo "   ✅ Wakeword    http://127.0.0.1:$WAKEWORD_PORT ($MODE)"
    else
        echo "   ❌ Wakeword    NOT RUNNING (port $WAKEWORD_PORT)"
    fi
    
    echo "═══════════════════════════════════════"
    
    # Get local IP
    LOCAL_IP=$(hostname -I | awk '{print $1}')
    echo ""
    echo "   🌐 Open VIRON: http://${LOCAL_IP}:${BACKEND_PORT}/viron-complete.html"
    echo ""
}

case "${1:-start}" in
    stop)
        stop_all
        ;;
    restart)
        stop_all
        start_all
        ;;
    status)
        show_status
        ;;
    start)
        # Check if already running
        if curl -s -o /dev/null http://127.0.0.1:$BACKEND_PORT/ 2>/dev/null; then
            echo "⚠️  Services appear to be running. Use 'restart' to restart."
            show_status
            exit 0
        fi
        start_all
        ;;
    *)
        echo "Usage: $0 [start|stop|restart|status]"
        exit 1
        ;;
esac
