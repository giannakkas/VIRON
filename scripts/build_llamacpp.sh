#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# VIRON — Build llama.cpp with CUDA for Jetson Orin Nano
# ═══════════════════════════════════════════════════════════════
# Builds llama.cpp from source with CUDA support for JetPack 6.x
# The resulting llama-server binary is used by docker-compose.
# ═══════════════════════════════════════════════════════════════

set -e

LLAMA_DIR="${LLAMA_DIR:-/opt/llama.cpp}"

echo ""
echo "🔨 Building llama.cpp with CUDA support"
echo "═══════════════════════════════════════════════"

# Check CUDA
if ! command -v nvcc &>/dev/null; then
    echo "⚠ CUDA toolkit not found. On Jetson with JetPack 6.x:"
    echo "  sudo apt install nvidia-cuda-toolkit"
    echo ""
    echo "Or set CUDA_PATH if installed in non-standard location."
    exit 1
fi

NVCC_VER=$(nvcc --version | grep release | awk '{print $5}' | tr -d ',')
echo "  CUDA: $NVCC_VER"
echo "  Target: $LLAMA_DIR"
echo ""

# Install build deps
echo "📦 Installing build dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential cmake git libcurl4-openssl-dev

# Clone or update
if [ -d "$LLAMA_DIR" ]; then
    echo "📂 Updating existing llama.cpp..."
    cd "$LLAMA_DIR"
    git pull --quiet
else
    echo "📥 Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR"
    cd "$LLAMA_DIR"
fi

# Build with CUDA
echo "🔨 Building (this takes ~5 min on Jetson)..."
mkdir -p build && cd build
cmake .. \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="87" \
    -DLLAMA_CURL=ON \
    -DCMAKE_BUILD_TYPE=Release

cmake --build . --config Release -j$(nproc)

# Verify
if [ -f "$LLAMA_DIR/build/bin/llama-server" ]; then
    echo ""
    echo "✅ llama.cpp built successfully!"
    echo "   Binary: $LLAMA_DIR/build/bin/llama-server"
    echo ""
    echo "   Test router:  $LLAMA_DIR/build/bin/llama-server -m /models/gemma-2-2b-it-Q4_K_M.gguf --port 8081 -ngl 99"
    echo "   Test tutor:   $LLAMA_DIR/build/bin/llama-server -m /models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf --port 8082 -ngl 99"
else
    echo "❌ Build failed! Check output above."
    exit 1
fi
