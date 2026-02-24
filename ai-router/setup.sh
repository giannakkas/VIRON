#!/bin/bash
# VIRON AI Router — Setup
set -e
DIR=$(cd "$(dirname "$0")" && pwd)
echo "🧠 VIRON AI Router Setup"
echo "========================"

# Install Python packages
echo "[1/3] Installing Python packages..."
pip3 install -r "$DIR/requirements.txt" --break-system-packages -q 2>/dev/null || \
pip3 install -r "$DIR/requirements.txt" -q
echo "  ✓ Packages installed"

# Setup .env
echo "[2/3] Configuring API keys..."
if [ ! -f "$DIR/.env" ]; then
    cp "$DIR/.env.example" "$DIR/.env"
    echo ""
    echo "  Enter your API keys (press Enter to skip any):"
    echo ""
    read -p "  Anthropic API key (sk-ant-...): " ANT_KEY
    read -p "  OpenAI API key (sk-...): " OAI_KEY
    read -p "  Google/Gemini API key (AI...): " GEM_KEY
    [ -n "$ANT_KEY" ] && sed -i "s|^ANTHROPIC_API_KEY=.*|ANTHROPIC_API_KEY=$ANT_KEY|" "$DIR/.env"
    [ -n "$OAI_KEY" ] && sed -i "s|^OPENAI_API_KEY=.*|OPENAI_API_KEY=$OAI_KEY|" "$DIR/.env"
    [ -n "$GEM_KEY" ] && sed -i "s|^GOOGLE_API_KEY=.*|GOOGLE_API_KEY=$GEM_KEY|" "$DIR/.env"
    chmod 600 "$DIR/.env"
    echo "  ✓ Config saved"
else
    echo "  ✓ .env already exists"
fi

# Check Ollama
echo "[3/3] Checking Ollama..."
if command -v ollama &>/dev/null; then
    echo "  ✓ Ollama installed"
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "  ✓ Ollama running"
        if ollama list 2>/dev/null | grep -q phi3; then
            echo "  ✓ phi3 model ready"
        else
            echo "  ⚠ Pulling phi3 model (2.2GB)..."
            ollama pull phi3
            echo "  ✓ phi3 ready"
        fi
    else
        echo "  ⚠ Ollama not running. Start with: ollama serve"
    fi
else
    echo "  ⚠ Ollama not installed. Install: curl -fsSL https://ollama.com/install.sh | sh"
fi

echo ""
echo "✅ AI Router ready!"
echo "   Start: cd $DIR && python3 main.py"
echo "   Docs:  http://localhost:8000/docs"
echo ""
