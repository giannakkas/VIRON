#!/bin/bash
# VIRON Start/Restart All Services + Logs

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
echo "📋 Live logs (Ctrl+C to stop):"
echo "═══════════════════════════════"

# Show important logs only — no HTTP noise
sudo journalctl -u viron-pipeline -f --no-pager | grep -v '"GET /\|HTTP/1.1" 200\|rd/poll\|ne/state\|ne/response\|/wakewo'
