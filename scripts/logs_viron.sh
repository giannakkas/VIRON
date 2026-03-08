#!/bin/bash
# VIRON Live Logs — View without restarting
# Usage: bash scripts/logs_viron.sh

echo "📋 VIRON Live Logs (Ctrl+C to stop)"
echo "═══════════════════════════════════════"
sudo journalctl -u viron-pipeline -f --no-pager | grep -E "Wake|Deepgram|whisper|Groq|Claude|Played|trigger|Whiteboard|Weather|News|Quiz|Music|💓|⚠|❌|🎯|📝|📰|🌤|🎵|ERROR|WARNING|sentence|word detected|rejected|stuck"
