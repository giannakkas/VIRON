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
    pkill -f "mpv.*ytdl" 2>/dev/null
    # Audio holders (mic-stealing fix). Use SEPARATE pkill calls because
    # pkill -f with shell-escaped \| is interpreted as literal | by the
    # regex engine and matches nothing. -9 because some children ignore TERM.
    pkill -9 arecord 2>/dev/null
    pkill -9 aplay 2>/dev/null
    pkill -9 mpv 2>/dev/null
    # Chromium kiosk holds the mic via getUserMedia even after viron_kiosk.py dies.
    pkill -9 -f chromium 2>/dev/null
    pkill -9 -f "google-chrome" 2>/dev/null
    sleep 2
    echo "✓ Stopped"
    exit 0
fi

# Load API keys
if [ -f .env ]; then
    set -a; source .env; set +a
    echo "✓ Loaded .env"
fi
export GEMINI_API_KEY

# Kill any existing processes
echo "🧹 Cleaning up old processes..."
pkill -f "gemini_live_pipeline.py" 2>/dev/null
pkill -f "voice_pipeline.py" 2>/dev/null
pkill -f "llama-server" 2>/dev/null
pkill -f "backend/server.py" 2>/dev/null
pkill -f "gateway/main.py" 2>/dev/null
pkill -f "viron_kiosk.py" 2>/dev/null
# Audio holders that survive the python-process pkill above. -9 because a
# stuck arecord won't always exit on TERM, and a leftover from a Ctrl+Z
# suspend will hold /dev/snd/pcmC0D0c forcing 'Mic read failed' on next start.
pkill -9 arecord 2>/dev/null
pkill -9 aplay 2>/dev/null
pkill -9 mpv 2>/dev/null
pkill -9 -f chromium 2>/dev/null
pkill -9 -f "google-chrome" 2>/dev/null
# Verify mic device is actually free; warn loudly if not.
sleep 1
if command -v fuser >/dev/null 2>&1; then
    holder=$(fuser /dev/snd/pcmC0D0c 2>/dev/null | tr -d ' ')
    if [ -n "$holder" ]; then
        echo "⚠️  /dev/snd/pcmC0D0c still held by PID(s): $holder"
        echo "   Force-killing..."
        kill -9 $holder 2>/dev/null
        sleep 1
    fi
fi
# Kill ALL userspace audio servers — they grab the mic exclusively and starve
# arecord. PulseAudio is masked here; PipeWire (default on Ubuntu 22.04) gets
# the same treatment because pulseaudio.service being masked doesn't stop it.
echo "   Killing PulseAudio + PipeWire (they mask the mic)..."
pulseaudio --kill 2>/dev/null
systemctl --user stop pulseaudio.socket pulseaudio.service 2>/dev/null
systemctl --user mask pulseaudio.socket pulseaudio.service 2>/dev/null
# PipeWire on Ubuntu 22.04+ — needs all four units masked, otherwise socket
# activation re-spawns it the moment any process opens /dev/snd/*
systemctl --user stop pipewire.socket pipewire.service pipewire-pulse.socket pipewire-pulse.service 2>/dev/null
systemctl --user mask pipewire.socket pipewire.service pipewire-pulse.socket pipewire-pulse.service 2>/dev/null
# WirePlumber is the session manager that respawns PipeWire — also mask it
systemctl --user stop wireplumber.service 2>/dev/null
systemctl --user mask wireplumber.service 2>/dev/null
# Force-kill any survivors
pkill -9 pulseaudio 2>/dev/null
pkill -9 pipewire 2>/dev/null
pkill -9 wireplumber 2>/dev/null
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

# ─── Configure microphone ───
# The ReSpeaker XVF3800 exposes its capture controls as 'Headset',0
# (Front Left/Right) and 'Headset',1 (Mono). The previous 'Capture' /
# 'Mic' / 'Auto Gain Control' controls don't exist on this device, so
# those amixer calls silently failed (|| true) and 'Headset',1 was
# left at its default 67% / -20dB. Setting both to 100% recovers ~20dB
# of usable signal. AGC is firmware-side on the XMOS DSP; we compensate
# for its over-attenuation of ch1 in software via VIRON_MIC_BOOST.
echo ""
echo "🎤 Configuring microphone..."
amixer -c 0 sset 'Headset',0 100% 2>/dev/null || true
amixer -c 0 sset 'Headset',1 100% 2>/dev/null || true
echo "   ✅ Mic configured (Headset 0+1 → 100% / 0dB, software gain=${VIRON_MIC_GAIN:-0.4}, boost=${VIRON_MIC_BOOST:-1.0}x)"

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
