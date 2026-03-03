"""
VIRON Wake Word Service — openWakeWord + Browser Audio Streaming
================================================================
Browser streams mic audio via WebSocket/HTTP to this service.
openWakeWord processes audio and returns wake word detections.

Usage:
  python3 wakeword/service.py

Environment:
  VIRON_WAKEWORD_THRESHOLD=0.7    Detection threshold (0-1)
  VIRON_WAKEWORD_MODEL=            Path to custom .onnx model (optional)
"""

import os
import sys
import time
import json
import threading
import logging
import numpy as np
from collections import deque

# ── Configuration ──
THRESHOLD = float(os.environ.get("VIRON_WAKEWORD_THRESHOLD", "0.5"))
CUSTOM_MODEL = os.environ.get("VIRON_WAKEWORD_MODEL", "")
PORT = int(os.environ.get("VIRON_WAKEWORD_PORT", "8085"))

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280  # 80ms frames — openWakeWord requirement

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("viron-wakeword")

# ── Wake Word Detection ──
class WakeWordDetector:
    def __init__(self):
        self.model = None
        self.model_names = []
        self.last_detection = 0
        self.detection_count = 0
        self.is_paused = False
        self._pending_detection = None
        self._lock = threading.Lock()
        
    def init_model(self):
        """Initialize openWakeWord model."""
        try:
            import openwakeword
            from openwakeword.model import Model
            
            # Download default models if needed
            openwakeword.utils.download_models()
            
            # Check for custom "hey viron" model
            custom_model_path = CUSTOM_MODEL or os.path.join(
                os.path.dirname(__file__), "models", "hey_viron.onnx"
            )
            
            if os.path.exists(custom_model_path):
                logger.info(f"Loading custom wake word model: {custom_model_path}")
                self.model = Model(
                    wakeword_models=[custom_model_path],
                    vad_threshold=0.5,
                )
                self.model_names = list(self.model.models.keys())
            else:
                logger.info("Using 'hey jarvis' model (train custom 'hey viron' for best results)")
                self.model = Model(
                    wakeword_models=["hey_jarvis_v0.1"],
                    vad_threshold=0.5,
                )
                self.model_names = list(self.model.models.keys())
            
            logger.info(f"openWakeWord loaded: models={self.model_names}, threshold={THRESHOLD}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to init openWakeWord: {e}")
            return False
    
    def process_audio(self, audio_int16: np.ndarray) -> dict:
        """Process int16 audio and return detection results."""
        if self.model is None or self.is_paused:
            return {"detected": False}
        
        try:
            # Energy check — skip quiet audio (TV through speakers)
            rms = np.sqrt(np.mean(audio_int16.astype(np.float32) ** 2))
            if rms < 300:  # int16 scale: direct speech ~2000+, TV through mic ~100-500
                return {"detected": False}
            
            # openWakeWord expects int16 numpy array
            predictions = self.model.predict(audio_int16)
            
            for name in self.model_names:
                score = predictions[name]
                
                if score >= THRESHOLD:
                    now = time.time()
                    # Debounce: min 2s between detections
                    if now - self.last_detection > 2.0:
                        self.last_detection = now
                        self.detection_count += 1
                        detection = {
                            "detected": True,
                            "model": name,
                            "score": float(score),
                            "time": now,
                            "count": self.detection_count,
                        }
                        with self._lock:
                            self._pending_detection = detection
                        logger.info(f"WAKE WORD DETECTED: {name} (score={score:.3f})")
                        return detection
            
            return {"detected": False}
            
        except Exception as e:
            return {"detected": False, "error": str(e)}
    
    def pause(self):
        self.is_paused = True
        try:
            if self.model:
                self.model.reset()
        except Exception:
            pass  # Reset can fail if audio is being processed — safe to ignore
    
    def resume(self):
        self.is_paused = False
        try:
            if self.model:
                self.model.reset()
        except Exception:
            pass
    
    def consume_detection(self) -> dict:
        with self._lock:
            d = self._pending_detection
            self._pending_detection = None
            return d


# ── HTTP API ──
def create_app(detector: WakeWordDetector):
    """Create Flask app for browser communication."""
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route("/wakeword/poll", methods=["GET"])
    def poll():
        """Browser polls this to check for wake word detection."""
        detection = detector.consume_detection()
        if detection:
            return jsonify({"wake": True, **detection})
        return jsonify({"wake": False})
    
    @app.route("/wakeword/audio", methods=["POST"])
    def receive_audio():
        """Receive audio chunk from browser and process it.
        
        Browser sends: raw PCM int16 mono 16kHz audio via POST body
        or base64-encoded audio in JSON.
        """
        try:
            content_type = request.content_type or ""
            
            if "application/octet-stream" in content_type:
                # Raw PCM bytes
                raw = request.get_data()
                audio = np.frombuffer(raw, dtype=np.int16)
            elif "application/json" in content_type:
                import base64
                data = request.get_json()
                raw = base64.b64decode(data["audio"])
                audio = np.frombuffer(raw, dtype=np.int16)
            else:
                # Try raw bytes
                raw = request.get_data()
                if len(raw) < 100:
                    return jsonify({"error": "No audio data"}), 400
                audio = np.frombuffer(raw, dtype=np.int16)
            
            if len(audio) < CHUNK_SAMPLES:
                return jsonify({"detected": False, "error": "too short"})
            
            # Process in chunks of CHUNK_SAMPLES (1280 = 80ms)
            result = {"detected": False}
            for i in range(0, len(audio) - CHUNK_SAMPLES + 1, CHUNK_SAMPLES):
                chunk = audio[i:i + CHUNK_SAMPLES]
                r = detector.process_audio(chunk)
                if r.get("detected"):
                    result = r
                    break
            
            return jsonify(result)
            
        except Exception as e:
            logger.warning(f"Audio processing error: {e}")
            return jsonify({"detected": False, "error": str(e)}), 400
    
    @app.route("/wakeword/pause", methods=["POST"])
    def pause():
        detector.pause()
        return jsonify({"status": "paused"})
    
    @app.route("/wakeword/resume", methods=["POST"])
    def resume():
        detector.resume()
        return jsonify({"status": "listening"})
    
    @app.route("/wakeword/status", methods=["GET"])
    def status():
        return jsonify({
            "ready": detector.model is not None,
            "paused": detector.is_paused,
            "models": detector.model_names,
            "threshold": THRESHOLD,
            "detections": detector.detection_count,
            "has_custom_model": bool(CUSTOM_MODEL) or os.path.exists(
                os.path.join(os.path.dirname(__file__), "models", "hey_viron.onnx")
            ),
        })
    
    @app.route("/wakeword/health", methods=["GET"])
    def health():
        return jsonify({"ok": True})
    
    return app


# ── Main ──
def main():
    print("""
═══════════════════════════════════════
   VIRON Wake Word Service (openWakeWord)
   Browser streams audio → server detects
═══════════════════════════════════════""")
    
    detector = WakeWordDetector()
    if not detector.init_model():
        logger.error("Failed to initialize wake word model")
        sys.exit(1)
    
    app = create_app(detector)
    
    print(f"""
   📡 http://0.0.0.0:{PORT}
   🎤 Audio endpoint: POST /wakeword/audio (PCM int16 16kHz)
   🔄 Poll: GET /wakeword/poll
   🔧 Status: GET /wakeword/status
   🎯 Models: {detector.model_names}
   📊 Threshold: {THRESHOLD}
═══════════════════════════════════════
""")
    
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
