"""
VIRON Wake Word Service — openWakeWord + PyAudio
=================================================
Captures mic audio directly on Jetson, runs openWakeWord detection.
Browser polls /poll endpoint to check for wake word activations.

Usage:
  python3 wakeword/service.py

Environment:
  VIRON_WAKEWORD_THRESHOLD=0.7    Detection threshold (0-1)
  VIRON_WAKEWORD_MODEL=            Path to custom .onnx model (optional)
  VIRON_MIC_DEVICE=                PyAudio device index (optional)
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
THRESHOLD = float(os.environ.get("VIRON_WAKEWORD_THRESHOLD", "0.65"))
CUSTOM_MODEL = os.environ.get("VIRON_WAKEWORD_MODEL", "")
MIC_DEVICE = os.environ.get("VIRON_MIC_DEVICE", "")
PORT = int(os.environ.get("VIRON_WAKEWORD_PORT", "8085"))

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # 80ms frames — openWakeWord requirement

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("viron-wakeword")

# ── Wake Word Detection ──
class WakeWordDetector:
    def __init__(self):
        self.model = None
        self.model_names = []
        self.last_detection = 0
        self.detection_count = 0
        self.detections = deque(maxlen=50)  # Last 50 detections
        self.is_listening = False
        self.is_paused = False
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
                logger.info(f"🎯 Loading custom wake word model: {custom_model_path}")
                self.model = Model(
                    wakeword_models=[custom_model_path],
                    vad_threshold=0.5,  # Built-in VAD filter
                )
                self.model_names = list(self.model.models.keys())
            else:
                # Use "hey jarvis" as temporary stand-in
                # It's phonetically the closest to "hey viron" among built-in models
                logger.info("🎯 Using 'hey jarvis' model (train custom 'hey viron' for best results)")
                self.model = Model(
                    wakeword_models=["hey_jarvis_v0.1"],
                    vad_threshold=0.5,
                )
                self.model_names = list(self.model.models.keys())
            
            logger.info(f"✅ openWakeWord loaded: models={self.model_names}, threshold={THRESHOLD}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to init openWakeWord: {e}")
            return False
    
    def process_audio(self, audio_chunk: np.ndarray) -> dict:
        """Process an audio chunk and return detection results."""
        if self.model is None or self.is_paused:
            return {}
        
        try:
            predictions = self.model.predict(audio_chunk)
            results = {}
            
            for name in self.model_names:
                score = predictions[name]
                results[name] = float(score)
                
                if score >= THRESHOLD:
                    now = time.time()
                    # Debounce: min 2s between detections
                    if now - self.last_detection > 2.0:
                        self.last_detection = now
                        self.detection_count += 1
                        detection = {
                            "model": name,
                            "score": float(score),
                            "time": now,
                            "count": self.detection_count,
                        }
                        self.detections.append(detection)
                        logger.info(f"🎯 WAKE WORD DETECTED: {name} (score={score:.3f})")
                        return {"detected": True, **detection, "scores": results}
            
            return {"detected": False, "scores": results}
            
        except Exception as e:
            logger.warning(f"Prediction error: {e}")
            return {"detected": False, "error": str(e)}
    
    def pause(self):
        """Pause detection (during VIRON speech)."""
        self.is_paused = True
        if self.model:
            self.model.reset()  # Reset state to avoid false triggers
    
    def resume(self):
        """Resume detection."""
        self.is_paused = False
        if self.model:
            self.model.reset()
    
    def get_latest_detection(self, since: float = 0) -> dict:
        """Get the latest detection after a given timestamp."""
        with self._lock:
            if self.detections:
                latest = self.detections[-1]
                if latest["time"] > since:
                    return latest
        return None


# ── Mic Capture Thread ──
class MicCapture:
    def __init__(self, detector: WakeWordDetector):
        self.detector = detector
        self.stream = None
        self.pa = None
        self.running = False
        self.thread = None
        self.device_index = int(MIC_DEVICE) if MIC_DEVICE else None
        # Pending detection for browser polling
        self._pending_detection = None
        self._pending_lock = threading.Lock()
    
    def find_mic(self):
        """Find the best microphone device."""
        import pyaudio
        self.pa = pyaudio.PyAudio()
        
        if self.device_index is not None:
            info = self.pa.get_device_info_by_index(self.device_index)
            logger.info(f"🎤 Using specified mic: [{self.device_index}] {info['name']}")
            return self.device_index
        
        # List all input devices and pick the best one
        best = None
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                name = info["name"]
                logger.info(f"  🎤 [{i}] {name} (channels={info['maxInputChannels']}, rate={info['defaultSampleRate']})")
                # Prefer USB mic, then default
                if "usb" in name.lower() or "webcam" in name.lower():
                    best = i
                elif best is None:
                    best = i
        
        if best is not None:
            info = self.pa.get_device_info_by_index(best)
            logger.info(f"🎤 Selected mic: [{best}] {info['name']}")
        else:
            logger.warning("⚠ No input device found, using default")
        
        return best
    
    def start(self):
        """Start mic capture in a background thread."""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop mic capture."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        if self.pa:
            self.pa.terminate()
    
    def _capture_loop(self):
        """Main capture loop — runs in background thread."""
        import pyaudio
        
        device = self.find_mic()
        
        try:
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=device,
                frames_per_buffer=CHUNK_SIZE,
            )
            logger.info("🎙️ Mic capture started — listening for wake word...")
            self.detector.is_listening = True
            
            while self.running:
                try:
                    data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    audio = np.frombuffer(data, dtype=np.int16)
                    
                    # Process with openWakeWord
                    result = self.detector.process_audio(audio)
                    
                    if result.get("detected"):
                        with self._pending_lock:
                            self._pending_detection = result
                        
                except IOError as e:
                    logger.warning(f"Audio read error: {e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"❌ Mic capture error: {e}")
            self.detector.is_listening = False
        finally:
            self.detector.is_listening = False
    
    def consume_detection(self) -> dict:
        """Get and clear pending detection (for browser polling)."""
        with self._pending_lock:
            d = self._pending_detection
            self._pending_detection = None
            return d


# ── HTTP API ──
def create_app(detector: WakeWordDetector, mic: MicCapture):
    """Create Flask app for browser communication."""
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route("/wakeword/poll", methods=["GET"])
    def poll():
        """Browser polls this to check for wake word detection."""
        detection = mic.consume_detection()
        if detection:
            return jsonify({"wake": True, **detection})
        return jsonify({"wake": False})
    
    @app.route("/wakeword/pause", methods=["POST"])
    def pause():
        """Pause detection (when VIRON is speaking)."""
        detector.pause()
        return jsonify({"status": "paused"})
    
    @app.route("/wakeword/resume", methods=["POST"])
    def resume():
        """Resume detection (after VIRON finishes speaking)."""
        detector.resume()
        return jsonify({"status": "listening"})
    
    @app.route("/wakeword/status", methods=["GET"])
    def status():
        return jsonify({
            "ready": detector.model is not None,
            "listening": detector.is_listening,
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
🎯 ═══════════════════════════════════════
   VIRON Wake Word Service (openWakeWord)
   ═══════════════════════════════════════""")
    
    detector = WakeWordDetector()
    if not detector.init_model():
        logger.error("Failed to initialize wake word model")
        sys.exit(1)
    
    mic = MicCapture(detector)
    mic.start()
    
    app = create_app(detector, mic)
    
    print(f"""
   📡 http://0.0.0.0:{PORT}
   🎤 Polling: http://0.0.0.0:{PORT}/wakeword/poll
   🔧 Status: http://0.0.0.0:{PORT}/wakeword/status
   🎯 Models: {detector.model_names}
   📊 Threshold: {THRESHOLD}
   ═══════════════════════════════════════
""")
    
    try:
        app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()


if __name__ == "__main__":
    main()
