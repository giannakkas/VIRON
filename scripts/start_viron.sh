#!/bin/bash
# VIRON Start/Restart All Services
# Place on desktop for easy access

echo "🤖 Restarting VIRON..."

# Kill old kiosk
pkill -f viron_kiosk 2>/dev/null
pkill -f WebKitWebProcess 2>/dev/null
sleep 2

# Kill stale audio
pulseaudio --kill 2>/dev/null
pkill -f arecord 2>/dev/null
pkill -f ffplay 2>/dev/null
sleep 1

# Restart backend + pipeline
sudo systemctl restart viron-backend
sleep 2
sudo systemctl restart viron-pipeline
sleep 3

# Start face kiosk
DISPLAY=:0 python3 ~/viron_kiosk.py &
sleep 5

# Auto-click to unlock WebKit audio context
DISPLAY=:0 xdotool mousemove 640 360 click 1 2>/dev/null
sleep 1
DISPLAY=:0 xdotool mousemove 0 0 2>/dev/null

echo "✅ VIRON is running!"
echo "   Say 'Hey Jarvis' to talk"
