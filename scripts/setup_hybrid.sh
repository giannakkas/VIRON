#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# VIRON — Full Hybrid AI Setup (Ubuntu PC or Jetson)
# ═══════════════════════════════════════════════════════════════
# This script:
#   1. Detects your hardware (GPU, CPU, RAM)
#   2. Installs dependencies (cmake, CUDA toolkit if GPU)
#   3. Builds llama.cpp (with CUDA if GPU, CPU-only otherwise)
#   4. Downloads GGUF models (~7GB)
#   5. Installs Python dependencies
#   6. Tests everything
#
# Usage:
#   cd ~/VIRON
#   bash scripts/setup_hybrid.sh
# ═══════════════════════════════════════════════════════════════

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
VIRON_DIR=$(dirname "$SCRIPT_DIR")
cd "$VIRON_DIR"

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  🤖 VIRON Hybrid AI — Full Setup${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════
# STEP 1: Detect Hardware
# ═══════════════════════════════════════════════════════════════
echo -e "${YELLOW}[1/6] Detecting hardware...${NC}"

CPU_CORES=$(nproc)
CPU_MODEL=$(lscpu | grep "Model name" | sed 's/Model name:\s*//')
RAM_TOTAL=$(free -h | awk '/Mem:/{print $2}')
DISK_FREE=$(df -h / | awk 'NR==2{print $4}')

echo "  CPU:  $CPU_MODEL ($CPU_CORES cores)"
echo "  RAM:  $RAM_TOTAL"
echo "  Disk: $DISK_FREE free"

# Detect GPU
HAS_NVIDIA=false
GPU_NAME=""
GPU_VRAM=""
IS_JETSON=false
USE_CUDA=false
CUDA_ARCH=""

if [ -f /proc/device-tree/model ] && grep -qi "jetson" /proc/device-tree/model 2>/dev/null; then
    IS_JETSON=true
    GPU_NAME="Jetson (integrated)"
    echo -e "  GPU:  ${GREEN}NVIDIA Jetson detected${NC}"
fi

if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "")
    if [ -n "$GPU_INFO" ]; then
        HAS_NVIDIA=true
        GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
        GPU_VRAM=$(echo "$GPU_INFO" | cut -d',' -f2 | xargs)
        echo -e "  GPU:  ${GREEN}$GPU_NAME ($GPU_VRAM)${NC}"
    fi
elif lspci 2>/dev/null | grep -qi nvidia; then
    HAS_NVIDIA=true
    GPU_NAME=$(lspci | grep -i nvidia | head -1 | sed 's/.*: //')
    echo -e "  GPU:  ${YELLOW}$GPU_NAME (driver not loaded?)${NC}"
fi

if ! $HAS_NVIDIA && ! $IS_JETSON; then
    echo -e "  GPU:  ${YELLOW}No NVIDIA GPU detected — will use CPU inference${NC}"
    echo -e "        ${YELLOW}(CPU mode works fine, just ~3-5x slower than GPU)${NC}"
fi

# Determine CUDA strategy
if $IS_JETSON; then
    USE_CUDA=true
    # Jetson Orin Nano = SM 87
    CUDA_ARCH="87"
    echo "  CUDA: Jetson (SM 87)"
elif $HAS_NVIDIA; then
    if command -v nvcc &>/dev/null; then
        USE_CUDA=true
        # Auto-detect CUDA arch
        CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' || echo "")
        if [ -z "$CUDA_ARCH" ]; then
            # Common architectures: RTX 30xx=86, RTX 40xx=89, GTX 10xx=61, RTX 20xx=75
            CUDA_ARCH="native"
        fi
        NVCC_VER=$(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | tr -d ',' || echo "unknown")
        echo "  CUDA: $NVCC_VER (arch: $CUDA_ARCH)"
    else
        echo -e "  CUDA: ${YELLOW}GPU found but CUDA toolkit not installed${NC}"
        echo -e "        ${YELLOW}Installing nvidia-cuda-toolkit...${NC}"
        sudo apt-get install -y nvidia-cuda-toolkit 2>/dev/null && USE_CUDA=true && CUDA_ARCH="native" || true
        if ! $USE_CUDA; then
            echo -e "        ${YELLOW}CUDA install failed — falling back to CPU${NC}"
        fi
    fi
fi

echo ""

# ═══════════════════════════════════════════════════════════════
# STEP 2: Install System Dependencies
# ═══════════════════════════════════════════════════════════════
echo -e "${YELLOW}[2/6] Installing system dependencies...${NC}"

sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential cmake git wget curl \
    libcurl4-openssl-dev \
    python3-pip python3-venv \
    2>/dev/null

echo -e "  ${GREEN}✓ System dependencies installed${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════
# STEP 3: Build llama.cpp
# ═══════════════════════════════════════════════════════════════
echo -e "${YELLOW}[3/6] Building llama.cpp...${NC}"

LLAMA_DIR="$VIRON_DIR/llama.cpp"

if [ -f "$LLAMA_DIR/build/bin/llama-server" ]; then
    echo -e "  ${GREEN}✓ llama.cpp already built${NC}"
    echo "    Binary: $LLAMA_DIR/build/bin/llama-server"
else
    if [ -d "$LLAMA_DIR" ]; then
        echo "  Updating existing llama.cpp..."
        cd "$LLAMA_DIR" && git pull --quiet
    else
        echo "  Cloning llama.cpp..."
        git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR"
    fi

    cd "$LLAMA_DIR"
    mkdir -p build && cd build

    if $USE_CUDA; then
        echo "  Building with CUDA (GPU acceleration)..."
        if [ "$CUDA_ARCH" = "native" ]; then
            cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
        else
            cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
        fi
    else
        echo "  Building CPU-only (no GPU)..."
        cmake .. -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
    fi

    echo "  Compiling (this takes 3-10 min)..."
    cmake --build . --config Release -j$(nproc) 2>&1 | tail -5

    if [ -f "$LLAMA_DIR/build/bin/llama-server" ]; then
        echo -e "  ${GREEN}✓ llama.cpp built successfully!${NC}"
        if $USE_CUDA; then
            echo -e "    Mode: ${GREEN}CUDA GPU${NC}"
        else
            echo -e "    Mode: ${YELLOW}CPU-only${NC}"
        fi
    else
        echo -e "  ${RED}❌ Build failed! Check output above.${NC}"
        exit 1
    fi
fi

cd "$VIRON_DIR"
echo ""

# ═══════════════════════════════════════════════════════════════
# STEP 4: Download Models
# ═══════════════════════════════════════════════════════════════
echo -e "${YELLOW}[4/6] Downloading GGUF models...${NC}"

MODELS_DIR="$VIRON_DIR/models"
mkdir -p "$MODELS_DIR"

# Gemma 2 2B IT
GEMMA_FILE="gemma-2-2b-it-Q4_K_M.gguf"
GEMMA_URL="https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"

if [ -f "$MODELS_DIR/$GEMMA_FILE" ]; then
    SIZE=$(du -h "$MODELS_DIR/$GEMMA_FILE" | cut -f1)
    echo -e "  ${GREEN}✓ Gemma 2B already downloaded ($SIZE)${NC}"
else
    echo "  📥 Downloading Gemma 2 2B IT (Q4_K_M) — ~1.5GB..."
    wget --progress=bar:force -O "$MODELS_DIR/$GEMMA_FILE" "$GEMMA_URL"
    echo -e "  ${GREEN}✓ Gemma 2B downloaded${NC}"
fi

# Mistral 7B Instruct
MISTRAL_FILE="Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
MISTRAL_URL="https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"

if [ -f "$MODELS_DIR/$MISTRAL_FILE" ]; then
    SIZE=$(du -h "$MODELS_DIR/$MISTRAL_FILE" | cut -f1)
    echo -e "  ${GREEN}✓ Mistral 7B already downloaded ($SIZE)${NC}"
else
    echo "  📥 Downloading Mistral 7B Instruct v0.3 (Q4_K_M) — ~4.4GB..."
    wget --progress=bar:force -O "$MODELS_DIR/$MISTRAL_FILE" "$MISTRAL_URL"
    echo -e "  ${GREEN}✓ Mistral 7B downloaded${NC}"
fi

echo ""

# ═══════════════════════════════════════════════════════════════
# STEP 5: Install Python Dependencies
# ═══════════════════════════════════════════════════════════════
echo -e "${YELLOW}[5/6] Installing Python dependencies...${NC}"

pip3 install --break-system-packages -q \
    fastapi uvicorn httpx pydantic 2>/dev/null

echo -e "  ${GREEN}✓ Gateway dependencies installed${NC}"

# Check existing VIRON deps too
pip3 install --break-system-packages -q \
    flask flask-cors requests python-dotenv 2>/dev/null

echo -e "  ${GREEN}✓ All Python dependencies ready${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════
# STEP 6: Create .env if missing
# ═══════════════════════════════════════════════════════════════
echo -e "${YELLOW}[6/6] Configuration...${NC}"

if [ ! -f "$VIRON_DIR/.env" ]; then
    cp "$VIRON_DIR/.env.example" "$VIRON_DIR/.env"
    echo -e "  ${YELLOW}⚠ Created .env from template — you need to add API keys!${NC}"
    echo "    Edit: nano $VIRON_DIR/.env"
else
    echo -e "  ${GREEN}✓ .env already exists${NC}"
fi

# Set the llama-server path in env
LLAMA_BIN="$LLAMA_DIR/build/bin/llama-server"
if ! grep -q "LLAMA_SERVER=" "$VIRON_DIR/.env" 2>/dev/null; then
    echo "" >> "$VIRON_DIR/.env"
    echo "# Auto-detected by setup script" >> "$VIRON_DIR/.env"
    echo "LLAMA_SERVER=$LLAMA_BIN" >> "$VIRON_DIR/.env"
fi

# Set models dir
if ! grep -q "MODELS_DIR=" "$VIRON_DIR/.env" 2>/dev/null; then
    echo "MODELS_DIR=$MODELS_DIR" >> "$VIRON_DIR/.env"
fi

echo ""

# ═══════════════════════════════════════════════════════════════
# DONE — Print Summary
# ═══════════════════════════════════════════════════════════════
echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ VIRON Hybrid AI Setup Complete!${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo "  Hardware:"
echo "    CPU:  $CPU_MODEL ($CPU_CORES cores)"
echo "    RAM:  $RAM_TOTAL"
if $USE_CUDA; then
    echo -e "    GPU:  ${GREEN}$GPU_NAME — CUDA enabled${NC}"
else
    echo -e "    GPU:  ${YELLOW}CPU-only mode${NC}"
fi
echo ""
echo "  Models:"
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null | awk '{print "    "$NF" ("$5")"}'
echo ""
echo "  llama.cpp: $LLAMA_BIN"
echo ""
echo -e "${CYAN}  ─── Next Steps ───────────────────────────────────${NC}"
echo ""
echo "  1. Add your API keys (if not done yet):"
echo "     nano $VIRON_DIR/.env"
echo ""
echo "  2. Quick test — start models manually:"
echo ""
echo "     # Terminal 1: Router (Gemma 2B)"
echo "     $LLAMA_BIN -m $MODELS_DIR/$GEMMA_FILE --port 8081 -ngl 99 --ctx-size 2048"
echo ""
echo "     # Terminal 2: Tutor (Mistral 7B)"
echo "     $LLAMA_BIN -m $MODELS_DIR/$MISTRAL_FILE --port 8082 -ngl 99 --ctx-size 4096"
echo ""
echo "     # Terminal 3: Gateway"
echo "     cd $VIRON_DIR/gateway && python3 main.py"
echo ""
echo "  3. Or start everything at once:"
echo "     bash $VIRON_DIR/run-hybrid.sh"
echo ""
echo "  4. Test it:"
echo "     curl -X POST http://localhost:8080/v1/chat \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"student_id\":\"chris\",\"age\":12,\"message\":\"What is 2+2?\"}'"
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
