#!/bin/bash
# VIRON — Start the companion
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Check config
if [ ! -f "$SCRIPT_DIR/backend/config.json" ]; then
    echo "❌ No config found. Run: sudo bash setup-local.sh"
    exit 1
fi

# Check API key
API_KEY=$(python3 -c "import json;print(json.load(open('$SCRIPT_DIR/backend/config.json'))['anthropic_api_key'])" 2>/dev/null)
if [ "$API_KEY" = "YOUR_API_KEY_HERE" ] || [ -z "$API_KEY" ]; then
    echo "⚠️  No API key set! Edit backend/config.json first."
    echo "   Get a key at: https://console.anthropic.com"
    exit 1
fi

echo "🤖 Starting VIRON..."
echo "   Open: http://localhost:5000"
echo "   Press Ctrl+C to stop"
echo ""

cd "$SCRIPT_DIR"
python3 backend/server.py
