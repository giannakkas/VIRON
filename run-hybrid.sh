#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# VIRON — Start Hybrid AI Architecture (bare-metal, no Docker)
# ═══════════════════════════════════════════════════════════════
# Starts:
#   1. llama.cpp server for Gemma 2B router (port 8081)
#   2. llama.cpp server for Mistral 7B tutor (port 8082)
#   3. VIRON Hybrid Gateway (port 8080)
#   4. Original Flask backend (port 5000)
#
# Prerequisites:
#   - bash scripts/download_models.sh
#   - bash scripts/build_llamacpp.sh
#   - cp .env.example .env && edit .env
# ═══════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

# Load env
if [ -f .env ]; then
    set -a; source .env; set +a
    echo "✓ Loaded .env"
fi

# Auto-detect llama-server: local build → /opt → env override
if [ -z "$LLAMA_SERVER" ]; then
    if [ -f "$SCRIPT_DIR/llama.cpp/build/bin/llama-server" ]; then
        LLAMA_SERVER="$SCRIPT_DIR/llama.cpp/build/bin/llama-server"
    elif [ -f "/opt/llama.cpp/build/bin/llama-server" ]; then
        LLAMA_SERVER="/opt/llama.cpp/build/bin/llama-server"
    fi
fi
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
MODELS_DIR="${MODELS_DIR:-$SCRIPT_DIR/models}"
GEMMA_MODEL="${MODELS_DIR}/gemma-2-2b-it-Q4_K_M.gguf"
MISTRAL_MODEL="${MODELS_DIR}/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"

echo ""
echo "🤖 VIRON Hybrid AI — Starting..."
echo "═══════════════════════════════════════════════"

# Detect GPU for --n-gpu-layers flag
NGL=0
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    NGL=99
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "  GPU: $GPU (CUDA, ngl=$NGL)"
elif [ -f /proc/device-tree/model ] && grep -qi jetson /proc/device-tree/model 2>/dev/null; then
    NGL=99
    echo "  GPU: Jetson (CUDA, ngl=$NGL)"
else
    echo "  GPU: None — running CPU-only (slower but works)"
fi

# Check llama-server binary
if [ ! -f "$LLAMA_SERVER" ]; then
    echo "❌ llama-server not found at: $LLAMA_SERVER"
    echo "   Run: bash scripts/build_llamacpp.sh"
    echo "   Or set LLAMA_SERVER env var to your binary path."
    exit 1
fi

# Check models
if [ ! -f "$GEMMA_MODEL" ]; then
    echo "❌ Gemma model not found: $GEMMA_MODEL"
    echo "   Run: bash scripts/download_models.sh"
    exit 1
fi
if [ ! -f "$MISTRAL_MODEL" ]; then
    echo "❌ Mistral model not found: $MISTRAL_MODEL"
    echo "   Run: bash scripts/download_models.sh"
    exit 1
fi

PIDS=""

cleanup() {
    echo ""
    echo "🛑 Stopping VIRON Hybrid..."
    for p in $PIDS; do
        kill "$p" 2>/dev/null
    done
    wait 2>/dev/null
    echo "✓ Stopped."
    exit 0
}
trap cleanup INT TERM

# ─── 1. Start Router (Gemma 2B) ───────────────────
echo "🧠 Starting Router (Gemma 2B) on port 8081..."
$LLAMA_SERVER \
    --model "$GEMMA_MODEL" \
    --port 8081 \
    --host 0.0.0.0 \
    --n-gpu-layers $NGL \
    --ctx-size 2048 \
    --threads 4 \
    --parallel 2 \
    &>/tmp/viron_router.log &
PIDS="$PIDS $!"
echo "  PID: $! (log: /tmp/viron_router.log)"

# ─── 2. Start Tutor (Mistral 7B) ──────────────────
echo "🎓 Starting Tutor (Mistral 7B) on port 8082..."
$LLAMA_SERVER \
    --model "$MISTRAL_MODEL" \
    --port 8082 \
    --host 0.0.0.0 \
    --n-gpu-layers $NGL \
    --ctx-size 4096 \
    --threads 4 \
    --parallel 2 \
    &>/tmp/viron_tutor.log &
PIDS="$PIDS $!"
echo "  PID: $! (log: /tmp/viron_tutor.log)"

# Wait for models to load
echo ""
echo "⏳ Waiting for models to load (30-60s on first run)..."
for i in $(seq 1 60); do
    ROUTER_OK=false
    TUTOR_OK=false
    curl -sf http://localhost:8081/health >/dev/null 2>&1 && ROUTER_OK=true
    curl -sf http://localhost:8082/health >/dev/null 2>&1 && TUTOR_OK=true
    if $ROUTER_OK && $TUTOR_OK; then
        echo "  ✓ Both models loaded!"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "  ⚠ Timeout waiting for models. Check logs:"
        echo "    tail -20 /tmp/viron_router.log"
        echo "    tail -20 /tmp/viron_tutor.log"
    fi
    sleep 2
done

# ─── 3. Start Gateway ────────────────────────────
echo ""
echo "🌐 Starting Hybrid Gateway on port ${GATEWAY_PORT:-8080}..."
export ROUTER_URL=http://localhost:8081
export TUTOR_URL=http://localhost:8082
export DB_PATH="${DB_PATH:-$SCRIPT_DIR/data/viron.db}"
mkdir -p "$(dirname "$DB_PATH")"

cd "$SCRIPT_DIR/gateway"
python3 main.py &>/tmp/viron_gateway.log &
PIDS="$PIDS $!"
cd "$SCRIPT_DIR"
echo "  PID: $! (log: /tmp/viron_gateway.log)"
sleep 2

# ─── 4. Start Original Flask Backend ─────────────
echo ""
echo "🖥️  Starting Flask backend on port 5000..."
bash "$SCRIPT_DIR/run.sh" &
PIDS="$PIDS $!"

echo ""
echo "═══════════════════════════════════════════════"
echo "🤖 VIRON Hybrid AI is running!"
echo "═══════════════════════════════════════════════"
echo "  🌐 Gateway:  http://localhost:${GATEWAY_PORT:-8080}/v1/chat"
echo "  📚 API Docs: http://localhost:${GATEWAY_PORT:-8080}/docs"
echo "  🖥️  Face UI:  http://localhost:5000"
echo "  🧠 Router:   http://localhost:8081 (Gemma 2B)"
echo "  🎓 Tutor:    http://localhost:8082 (Mistral 7B)"
echo "  💊 Health:   http://localhost:${GATEWAY_PORT:-8080}/health"
echo "═══════════════════════════════════════════════"
echo ""
echo "  Test: curl -X POST http://localhost:${GATEWAY_PORT:-8080}/v1/chat \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"student_id\":\"test\",\"age\":10,\"message\":\"What is 2+2?\"}'"
echo ""
echo "  Press Ctrl+C to stop everything"
echo ""

wait
