#!/bin/bash
# VIRON Start/Restart All Services
# Shows live logs after starting

echo "🤖 Restarting VIRON..."

# Kill old processes
pkill -f viron_kiosk 2>/dev/null
pkill -f WebKitWebProcess 2>/dev/null
pkill -f ffplay 2>/dev/null
pulseaudio --kill 2>/dev/null
pkill -f arecord 2>/dev/null
sleep 2

# Pull latest code
cd ~/VIRON
git pull 2>/dev/null

# Restart services
echo "🔄 Restarting services..."
sudo systemctl restart viron-backend
sleep 2
sudo systemctl restart viron-pipeline
sleep 3

# Start face kiosk
echo "🖥️ Starting face display..."
DISPLAY=:0 python3 ~/viron_kiosk.py &
sleep 5

# Auto-click to unlock WebKit audio
DISPLAY=:0 xdotool mousemove 640 360 click 1 2>/dev/null
sleep 1
DISPLAY=:0 xdotool mousemove 0 0 2>/dev/null

echo ""
echo "✅ VIRON is running!"
echo "   Say 'Hey Jarvis' to talk"
echo ""
echo "📋 Live logs (Ctrl+C to stop watching):"
echo "═══════════════════════════════════════════"

# Show live pipeline logs
sudo journalctl -u viron-pipeline -f --no-pager | grep -E "Wake|Deepgram|whisper|Groq|Claude|Played|trigger|Whiteboard|Weather|News|Quiz|Music|💓|⚠|❌|🎯|📝|📰|🌤|🎵|ERROR|WARNING|sentence|word detected"
