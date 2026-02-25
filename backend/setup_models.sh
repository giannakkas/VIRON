#!/bin/bash
# Download face detection and recognition models for VIRON
# Run this once: bash backend/setup_models.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"
mkdir -p "$MODELS_DIR"

DETECT_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
RECOG_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

DETECT_FILE="$MODELS_DIR/face_detection_yunet_2023mar.onnx"
RECOG_FILE="$MODELS_DIR/face_recognition_sface_2021dec.onnx"

echo "📥 Downloading Face Detector (YuNet)..."
curl -L --retry 3 -o "$DETECT_FILE" "$DETECT_URL" 2>/dev/null || \
  wget -q -O "$DETECT_FILE" "$DETECT_URL" 2>/dev/null || \
  python3 -c "import urllib.request; urllib.request.urlretrieve('$DETECT_URL', '$DETECT_FILE')"

echo "📥 Downloading Face Recognizer (SFace)..."
curl -L --retry 3 -o "$RECOG_FILE" "$RECOG_URL" 2>/dev/null || \
  wget -q -O "$RECOG_FILE" "$RECOG_URL" 2>/dev/null || \
  python3 -c "import urllib.request; urllib.request.urlretrieve('$RECOG_URL', '$RECOG_FILE')"

echo ""
echo "Checking downloads..."
for f in "$DETECT_FILE" "$RECOG_FILE"; do
  sz=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
  nm=$(basename "$f")
  if [ "$sz" -gt 100000 ] 2>/dev/null; then
    echo "  ✅ $nm: $(du -h "$f" | cut -f1)"
  else
    echo "  ❌ $nm: FAILED (${sz} bytes)"
    echo "     Manual download: https://github.com/opencv/opencv_zoo"
  fi
done
