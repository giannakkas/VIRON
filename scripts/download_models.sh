#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# VIRON — Download GGUF Models for Jetson Orin Nano
# ═══════════════════════════════════════════════════════════════
# Downloads Q4_K_M quantized models for local inference via llama.cpp
#
# Models directory: /models (configurable via MODELS_DIR env var)
# Storage required: ~2GB (Gemma) + ~5GB (Mistral) ≈ 7GB total
# ═══════════════════════════════════════════════════════════════

set -e

MODELS_DIR="${MODELS_DIR:-/models}"
mkdir -p "$MODELS_DIR"

echo ""
echo "🤖 VIRON Model Downloader"
echo "═══════════════════════════════════════════════"
echo "  Target: $MODELS_DIR"
echo "═══════════════════════════════════════════════"
echo ""

# ─── Check for HuggingFace CLI or wget ─────────────
HAS_HF=false
HAS_WGET=false
if command -v huggingface-cli &>/dev/null; then HAS_HF=true; fi
if command -v wget &>/dev/null; then HAS_WGET=true; fi

if ! $HAS_WGET && ! $HAS_HF; then
    echo "❌ Neither wget nor huggingface-cli found."
    echo "   Install: apt install wget  OR  pip install huggingface_hub"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════
# MODEL 1: Gemma 2 2B IT (Router — intent classification)
# ═══════════════════════════════════════════════════════════════
GEMMA_FILE="gemma-2-2b-it-Q4_K_M.gguf"
GEMMA_URL="https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"
GEMMA_REPO="bartowski/gemma-2-2b-it-GGUF"

if [ -f "$MODELS_DIR/$GEMMA_FILE" ]; then
    SIZE=$(du -h "$MODELS_DIR/$GEMMA_FILE" | cut -f1)
    echo "✅ Gemma 2 2B IT already exists ($SIZE)"
else
    echo "📥 Downloading Gemma 2 2B IT (Q4_K_M) — ~1.5GB..."
    echo "   Source: $GEMMA_REPO"
    if $HAS_WGET; then
        wget -q --show-progress -O "$MODELS_DIR/$GEMMA_FILE" "$GEMMA_URL"
    elif $HAS_HF; then
        huggingface-cli download "$GEMMA_REPO" "$GEMMA_FILE" \
            --local-dir "$MODELS_DIR" --local-dir-use-symlinks False
    fi
    if [ -f "$MODELS_DIR/$GEMMA_FILE" ]; then
        SIZE=$(du -h "$MODELS_DIR/$GEMMA_FILE" | cut -f1)
        echo "✅ Gemma 2 2B IT downloaded ($SIZE)"
    else
        echo "❌ Gemma download failed!"
        echo ""
        echo "   Manual download:"
        echo "   wget -O $MODELS_DIR/$GEMMA_FILE \\"
        echo "     $GEMMA_URL"
    fi
fi

echo ""

# ═══════════════════════════════════════════════════════════════
# MODEL 2: Mistral 7B Instruct v0.3 (Tutor — offline answers)
# ═══════════════════════════════════════════════════════════════
MISTRAL_FILE="Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
MISTRAL_URL="https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
MISTRAL_REPO="bartowski/Mistral-7B-Instruct-v0.3-GGUF"

if [ -f "$MODELS_DIR/$MISTRAL_FILE" ]; then
    SIZE=$(du -h "$MODELS_DIR/$MISTRAL_FILE" | cut -f1)
    echo "✅ Mistral 7B Instruct already exists ($SIZE)"
else
    echo "📥 Downloading Mistral 7B Instruct v0.3 (Q4_K_M) — ~4.4GB..."
    echo "   Source: $MISTRAL_REPO"
    if $HAS_WGET; then
        wget -q --show-progress -O "$MODELS_DIR/$MISTRAL_FILE" "$MISTRAL_URL"
    elif $HAS_HF; then
        huggingface-cli download "$MISTRAL_REPO" "$MISTRAL_FILE" \
            --local-dir "$MODELS_DIR" --local-dir-use-symlinks False
    fi
    if [ -f "$MODELS_DIR/$MISTRAL_FILE" ]; then
        SIZE=$(du -h "$MODELS_DIR/$MISTRAL_FILE" | cut -f1)
        echo "✅ Mistral 7B Instruct downloaded ($SIZE)"
    else
        echo "❌ Mistral download failed!"
        echo ""
        echo "   Manual download:"
        echo "   wget -O $MODELS_DIR/$MISTRAL_FILE \\"
        echo "     $MISTRAL_URL"
    fi
fi

echo ""

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════"
echo "  Model Files:"
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null || echo "  (no .gguf files found)"
echo "═══════════════════════════════════════════════"
echo ""
echo "  Next steps:"
echo "  1. Build llama.cpp with CUDA (see scripts/build_llamacpp.sh)"
echo "  2. docker-compose up  (or run manually)"
echo ""
