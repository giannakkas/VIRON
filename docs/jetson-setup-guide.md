# VIRON → Jetson Orin Nano: Complete Transfer Guide

## OS: JetPack 6.2 (Ubuntu 22.04 + CUDA 12.6)

Why JetPack 6.2:
- Ubuntu 22.04 LTS (same as your current PC — maximum compatibility)
- CUDA 12.6, cuDNN 9.3, TensorRT 10.3 (pre-installed)
- "Super Mode" — up to 2x AI inference speed boost for LLMs
- 67 TOPS → ~100+ TOPS with Super Mode enabled

---

## PHASE 1: Flash the Jetson

### What you need
- Jetson Orin Nano Developer Kit
- microSD card (64GB+, UHS-I recommended) OR NVMe SSD (recommended for speed)
- USB-C cable (to connect Jetson to a host PC for flashing)
- Monitor + keyboard for initial setup

### Option A: SD Card (easiest)
```bash
# On your Windows/Mac/Linux PC:
# 1. Download JetPack 6.2 SD card image from:
#    https://developer.nvidia.com/embedded/jetpack-sdk-62
#
# 2. Download Balena Etcher: https://etcher.balena.io
#
# 3. Flash the SD card image using Etcher
#
# 4. Insert SD card into Jetson, connect power + monitor + keyboard
#
# 5. Boot and follow the on-screen setup (username, password, WiFi)
```

### Option B: NVMe SSD (recommended — much faster)
```bash
# Requires a Linux host PC (Ubuntu 20.04 or 22.04)
# 1. Install NVIDIA SDK Manager on host:
#    https://developer.nvidia.com/sdk-manager
#
# 2. Connect Jetson via USB-C to host
# 3. Put Jetson in recovery mode:
#    - Hold RECOVERY button
#    - Press RESET button
#    - Release both
# 4. In SDK Manager: select JetPack 6.2 → flash to NVMe
```

### IMPORTANT: Firmware update if coming from JetPack 5
If your Jetson came with JetPack 5 factory firmware, you MUST update firmware first. Follow NVIDIA's Initial Setup Guide before inserting the JetPack 6.2 SD card.

### After first boot
```bash
# Complete the Ubuntu setup wizard (username, password, etc.)
# Then:
sudo apt update && sudo apt upgrade -y

# Install full JetPack components
sudo apt install nvidia-jetpack -y

# Enable Super Mode (MAXN = maximum performance)
sudo nvpmodel -m 0
sudo jetson_clocks

# Verify CUDA
nvcc --version
# Should show: CUDA 12.6

# Verify GPU
nvidia-smi
# Or: sudo tegrastats
```

---

## PHASE 2: Install VIRON Dependencies

### System packages
```bash
sudo apt install -y \
    python3-pip python3-venv python3-dev \
    git curl wget \
    ffmpeg \
    build-essential cmake \
    libasound2-dev portaudio19-dev \
    nodejs npm \
    sqlite3 \
    libopenblas-dev
```

### Python packages
```bash
pip3 install --break-system-packages \
    flask flask-cors \
    faster-whisper \
    httpx \
    fastapi uvicorn pydantic \
    requests \
    edge-tts \
    numpy \
    opencv-python-headless
```

### Node.js (update to 18+ if needed)
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs
```

---

## PHASE 3: Build llama.cpp with CUDA

This is the KEY step — GPU-accelerated inference.

```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with CUDA support for Jetson
mkdir build && cd build
cmake .. \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc)

# Verify GPU build
./bin/llama-server --help | head -5
# Should work without errors

# Install globally
sudo cp build/bin/llama-server /usr/local/bin/
sudo cp build/bin/llama-cli /usr/local/bin/
```

**Note:** `-DCMAKE_CUDA_ARCHITECTURES=87` is for Orin Nano (Ampere SM 8.7). This is critical — wrong value = no GPU acceleration.

---

## PHASE 4: Clone VIRON

```bash
cd ~
git clone https://github.com/giannakkas/VIRON.git
cd VIRON
```

### Download models
```bash
mkdir -p models
cd models

# Router: Gemma 2B (small, fast classification)
wget https://huggingface.co/lmstudio-community/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf

# Tutor: Mistral 7B (conversational, multilingual)
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf

cd ..
```

### Create .env file
```bash
cat > .env << 'EOF'
# Cloud API Keys
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
GEMINI_API_KEY=your-gemini-key-here

# Local LLM
ROUTER_URL=http://localhost:8081
TUTOR_URL=http://localhost:8082

# Database
DB_PATH=/home/chris/VIRON/data/viron.db

# Gateway
GATEWAY_PORT=8080

# FORCE_CLOUD=0 means use local for casual chat (Jetson GPU is fast enough!)
FORCE_CLOUD=0
EOF
```

**Edit the file with your real API keys:**
```bash
nano .env
```

---

## PHASE 5: Create the Startup Script

```bash
cat > run-jetson.sh << 'SCRIPT'
#!/bin/bash
set -e

echo "🤖 VIRON — Jetson Orin Nano Startup"
echo "===================================="

# Enable max performance
sudo nvpmodel -m 0 2>/dev/null
sudo jetson_clocks 2>/dev/null

# Load environment
source /home/$USER/VIRON/.env
export OPENAI_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY
export DB_PATH ROUTER_URL TUTOR_URL GATEWAY_PORT FORCE_CLOUD

# Create data directory
mkdir -p /home/$USER/VIRON/data

# Kill any existing processes
echo "Cleaning up..."
pkill -f "llama-server" 2>/dev/null || true
pkill -f "gateway/main.py" 2>/dev/null || true
pkill -f "backend/server.py" 2>/dev/null || true
sleep 2

MODEL_DIR="/home/$USER/VIRON/models"

# ── Start Router (Gemma 2B) — GPU layers ──
echo "Starting Router (Gemma 2B on GPU)..."
llama-server \
    -m "$MODEL_DIR/gemma-2-2b-it-Q4_K_M.gguf" \
    --port 8081 \
    -ngl 99 \
    -c 2048 \
    --threads 4 \
    > /tmp/viron_router.log 2>&1 &

sleep 5

# ── Start Tutor (Mistral 7B) — GPU layers ──
echo "Starting Tutor (Mistral 7B on GPU)..."
llama-server \
    -m "$MODEL_DIR/mistral-7b-instruct-v0.2.Q4_K_M.gguf" \
    --port 8082 \
    -ngl 99 \
    -c 4096 \
    --threads 4 \
    > /tmp/viron_tutor.log 2>&1 &

sleep 8

# ── Start Gateway ──
echo "Starting Hybrid Gateway..."
cd /home/$USER/VIRON/gateway
python3 main.py > /tmp/viron_gateway.log 2>&1 &
sleep 3

# ── Start Flask (Face UI + TTS + STT) ──
echo "Starting Flask backend..."
cd /home/$USER/VIRON
python3 backend/server.py > /tmp/viron_flask.log 2>&1 &
sleep 3

echo ""
echo "🤖 VIRON is running on Jetson!"
echo "================================"
echo "📺 Face UI:  http://$(hostname -I | awk '{print $1}'):5000"
echo "🌐 Gateway:  http://localhost:8080"
echo "🔍 Router:   http://localhost:8081"
echo "📚 Tutor:    http://localhost:8082"
echo ""
echo "Logs: /tmp/viron_*.log"
echo "Stop: pkill -f llama-server; pkill -f python3"
SCRIPT

chmod +x run-jetson.sh
```

---

## PHASE 6: Test Everything

### Start VIRON
```bash
cd ~/VIRON
bash run-jetson.sh
```

### Test each component
```bash
# Test Router (should respond in <1 second on GPU)
curl -s http://localhost:8081/v1/chat/completions \
    -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":5}' | head

# Test Tutor (should respond in 3-5 seconds on GPU!)
curl -s http://localhost:8082/v1/chat/completions \
    -d '{"messages":[{"role":"system","content":"Reply briefly"},{"role":"user","content":"hi"}],"max_tokens":50}' | python3 -m json.tool

# Test Gateway
curl -s -X POST http://localhost:8080/v1/chat \
    -H 'Content-Type: application/json' \
    -d '{"student_id":"test","age":12,"message":"τι κάνεις;","language":"el"}' | python3 -m json.tool

# Test Face UI
curl -s http://localhost:5000/api/status
```

### Check GPU usage
```bash
# In another terminal:
sudo tegrastats
# Look for GR3D (GPU usage) — should show activity during inference

# Or check llama.cpp logs:
grep "GPU" /tmp/viron_router.log
grep "GPU" /tmp/viron_tutor.log
# Should show: "offloading XX layers to GPU"
```

---

## PHASE 7: Performance Tuning

### Memory optimization (8GB is tight with 2 models)
```bash
# Check memory usage
free -h

# If tight on RAM, reduce context sizes in run-jetson.sh:
# Router: -c 1024 (instead of 2048)
# Tutor:  -c 2048 (instead of 4096)

# Or use smaller tutor model:
# Mistral 7B Q3_K_M instead of Q4_K_M (saves ~1GB)
```

### Expected performance (GPU vs your current CPU)

| Task | Current PC (CPU) | Jetson (GPU) |
|------|-------------------|--------------|
| Router (Gemma 2B) | ~2-3s | **<0.5s** |
| Tutor greeting | ~18-30s | **3-5s** |
| Tutor explanation | 60s+ (timeout) | **8-15s** |
| Cloud (ChatGPT) | ~5s | ~5s (same) |
| Whisper STT | ~2s | ~1s |
| Total greeting flow | ~25s | **~5s** |

### Disable FORCE_CLOUD
Once local models work fast on GPU:
```bash
# In .env:
FORCE_CLOUD=0

# This lets casual chat ("τι κάνεις") stay local (3-5s)
# Educational questions still route to cloud
```

---

## PHASE 8: Auto-Start on Boot

```bash
# Create systemd service
sudo tee /etc/systemd/system/viron.service << EOF
[Unit]
Description=VIRON AI Study Buddy
After=network.target

[Service]
Type=forking
User=$USER
WorkingDirectory=/home/$USER/VIRON
ExecStart=/home/$USER/VIRON/run-jetson.sh
ExecStop=/usr/bin/pkill -f llama-server; /usr/bin/pkill -f "gateway/main.py"; /usr/bin/pkill -f "backend/server.py"
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable viron
sudo systemctl start viron

# Check status
sudo systemctl status viron
```

---

## PHASE 9: Hardware Setup

### USB peripherals to connect
- **USB microphone** — for voice input (or I2S MEMS mic for production)
- **USB speaker** — for TTS output (or I2S amp + speaker)
- **USB webcam** — for face detection (or CSI camera for better perf)

### Network
- **WiFi** — built-in on dev kit, or use Ethernet for reliability
- **Static IP** recommended for accessing Face UI from other devices

```bash
# Set static IP (example)
sudo nmcli connection modify "Wired connection 1" \
    ipv4.method manual \
    ipv4.addresses 192.168.100.200/24 \
    ipv4.gateway 192.168.100.1 \
    ipv4.dns "8.8.8.8,8.8.4.4"
sudo nmcli connection up "Wired connection 1"
```

---

## Quick Reference: Transfer Checklist

- [ ] Flash JetPack 6.2 to SD card or NVMe
- [ ] First boot, create user, connect WiFi
- [ ] `sudo apt install nvidia-jetpack`
- [ ] Enable Super Mode: `sudo nvpmodel -m 0 && sudo jetson_clocks`
- [ ] Install system packages (python, ffmpeg, cmake, etc.)
- [ ] Install Python packages (flask, faster-whisper, etc.)
- [ ] Build llama.cpp with CUDA (`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87`)
- [ ] Clone VIRON repo
- [ ] Download Gemma 2B + Mistral 7B models
- [ ] Create .env with API keys
- [ ] Create run-jetson.sh
- [ ] Run and test all 4 services
- [ ] Verify GPU offload in logs
- [ ] Set FORCE_CLOUD=0
- [ ] Test voice: speak Greek → hear Greek response in ~5 seconds
- [ ] Set up auto-start systemd service
- [ ] Connect mic, speaker, camera
