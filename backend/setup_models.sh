#!/bin/bash
# Download face detection and recognition models for VIRON
# Run this once: bash backend/setup_models.sh

MODELS_DIR="$(dirname "$0")/models"
mkdir -p "$MODELS_DIR"

echo "📥 Downloading face detection model (YuNet)..."
curl -L -o "$MODELS_DIR/face_detection_yunet_2023mar.onnx" \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

echo "📥 Downloading face recognition model (SFace)..."
curl -L -o "$MODELS_DIR/face_recognition_sface_2021dec.onnx" \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# Verify downloads
DETECT_SIZE=$(stat -f%z "$MODELS_DIR/face_detection_yunet_2023mar.onnx" 2>/dev/null || stat -c%s "$MODELS_DIR/face_detection_yunet_2023mar.onnx" 2>/dev/null)
RECOG_SIZE=$(stat -f%z "$MODELS_DIR/face_recognition_sface_2021dec.onnx" 2>/dev/null || stat -c%s "$MODELS_DIR/face_recognition_sface_2021dec.onnx" 2>/dev/null)

if [ "$DETECT_SIZE" -gt 100000 ] 2>/dev/null && [ "$RECOG_SIZE" -gt 100000 ] 2>/dev/null; then
  echo "✅ Models downloaded successfully!"
  echo "  Detection model: $(du -h "$MODELS_DIR/face_detection_yunet_2023mar.onnx" | cut -f1)"
  echo "  Recognition model: $(du -h "$MODELS_DIR/face_recognition_sface_2021dec.onnx" | cut -f1)"
else
  echo "⚠ Download may have failed. Check the files in $MODELS_DIR"
  echo "  You can manually download from:"
  echo "  https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet"
  echo "  https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface"
fi
