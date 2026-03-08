#!/bin/bash
# VIRON Start/Restart All Services

echo "🤖 Restarting VIRON..."

pkill -f viron_kiosk 2>/dev/null
pkill -f WebKitWebProcess 2>/dev/null
pkill -f ffplay 2>/dev/null
pulseaudio --kill 2>/dev/null
pkill -f arecord 2>/dev/null
sleep 2

cd ~/VIRON
git pull 2>/dev/null

echo "🔄 Restarting services..."
sudo systemctl restart viron-backend
sleep 2
sudo systemctl restart viron-pipeline
sleep 3

echo "🖥️ Starting face display..."
DISPLAY=:0 python3 ~/viron_kiosk.py &
sleep 5

DISPLAY=:0 xdotool mousemove 640 360 click 1 2>/dev/null
sleep 1
DISPLAY=:0 xdotool mousemove 0 0 2>/dev/null

echo ""
echo "✅ VIRON is running!"
echo "   Say 'Hey Jarvis' to talk"
echo ""
echo "📋 Live logs (Ctrl+C to stop watching):"
echo "═══════════════════════════════════════════"

# Show ALL pipeline logs unfiltered
sudo journalctl -u viron-pipeline -f --no-pager
