#!/bin/bash
# Build whisper.cpp with CUDA on Jetson Orin Nano
# This gives GPU-accelerated Whisper (~200-500ms vs 3s on CPU)

set -e
cd /home/test

echo "═══════════════════════════════════════"
echo "  Building whisper.cpp with CUDA"
echo "═══════════════════════════════════════"

# 1. Clone whisper.cpp
if [ ! -d "whisper.cpp" ]; then
    echo "Cloning whisper.cpp..."
    git clone https://github.com/ggerganov/whisper.cpp.git
else
    echo "whisper.cpp already exists, updating..."
    cd whisper.cpp && git pull && cd ..
fi

cd whisper.cpp

# 2. Build with CUDA
echo "Building with CUDA..."
mkdir -p build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

echo ""
echo "✅ Build complete!"
echo ""

# 3. Download small model (best speed/accuracy for Jetson)
cd ..
if [ ! -f "models/ggml-small.bin" ]; then
    echo "Downloading Whisper small model..."
    bash ./models/download-ggml-model.sh small
fi

# 4. Test
echo ""
echo "Testing GPU inference..."
echo "Generating test audio..."
ffmpeg -y -f lavfi -i "sine=frequency=440:duration=2" -ar 16000 -ac 1 /tmp/test_whisper.wav 2>/dev/null

./build/bin/whisper-cli -m models/ggml-small.bin -f /tmp/test_whisper.wav --no-prints -t 4 --gpu 2>/dev/null && echo "✅ GPU whisper works!" || echo "⚠ GPU failed, trying CPU..." && ./build/bin/whisper-cli -m models/ggml-small.bin -f /tmp/test_whisper.wav --no-prints -t 4

echo ""
echo "═══════════════════════════════════════"
echo "  whisper.cpp ready!"
echo "  Binary: $(pwd)/build/bin/whisper-cli"
echo "  Model:  $(pwd)/models/ggml-small.bin"
echo "═══════════════════════════════════════"
