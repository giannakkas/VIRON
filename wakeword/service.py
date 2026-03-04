"""
VIRON Wake Word Service — openWakeWord + ReSpeaker (Server-Side Mic)
====================================================================
Captures mic audio directly on Jetson via ReSpeaker array,
runs openWakeWord detection. Browser polls /wakeword/poll endpoint.

Usage:
  python3 wakeword/service.py

Environment:
  VIRON_WAKEWORD_THRESHOLD=0.5    Detection threshold (0-1)
  VIRON_WAKEWORD_MODEL=            Path to custom .onnx model (optional)
  VIRON_MIC_DEVICE=plughw:2,0      ALSA device (ReSpeaker = card 2)
  VIRON_WAKEWORD_PORT=8085         HTTP port
"""

import os
import sys
import time
import threading
import logging
import subprocess
import numpy as np
from collections import deque

# — Configuration —
THRESHOLD = float(os.environ.get("VIRON_WAKEWORD_THRESHOLD", "0.5"))
CUSTOM_MODEL = os.environ.get("VIRON_WAKEWORD_MODEL", "")
ALSA_DEVICE = os.environ.get("VIRON_MIC_DEVICE", "plughw:2,0")
PORT = int(os.environ.get("VIRON_WAKEWORD_PORT", "8085"))

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # 80ms frames — openWakeWord requirement

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("viron-wakeword")


class WakeWordDetector:
    def __init__(self):
        self.model = None
        self.model_names = []
        self.last_detection = 0
        self.detection_count = 0
        self.detections = deque(maxlen=50)
        self.is_listening = False
        self.is_paused = False
        self._lock = threading.Lock()

    def init_model(self):
        try:
            import openwakeword
            from openwakeword.model import Model

            openwakeword.utils.download_models()

            custom_model_path = CUSTOM_MODEL or os.path.join(
                os.path.dirname(__file__), "models", "hey_viron.onnx"
            )

            if os.path.exists(custom_model_path):
                logger.info(f"Loading custom model: {custom_model_path}")
                self.model = Model(wakeword_models=[custom_model_path], vad_threshold=0.5)
            else:
                logger.info("Using 'hey jarvis' model (train custom 'hey viron' for best results)")
                self.model = Model(wakeword_models=["hey_jarvis_v0.1"], vad_threshold=0.5)

            self.model_names = list(self.model.models.keys())
            logger.info(f"openWakeWord loaded: models={self.model_names}, threshold={THRESHOLD}")
            return True
        except Exception as e:
            logger.error(f"Failed to init openWakeWord: {e}")
            return False

    def process_audio(self, audio_int16: np.ndarray) -> dict:
        if self.model is None or self.is_paused:
            return {"detected": False}

        try:
            predictions = self.model.predict(audio_int16)

            for name in self.model_names:
                score = predictions[name]

                if score > 0.1:
                    logger.info(f"Score: {name}={score:.3f} (threshold={THRESHOLD})")

                if score >= THRESHOLD:
                    now = time.time()
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
                            self.detections.append(detection)
                        logger.info(f"WAKE WORD DETECTED: {name} (score={score:.3f})")
                        return detection

            return {"detected": False}
        except Exception as e:
            logger.warning(f"Prediction error: {e}")
            return {"detected": False}

    def pause(self):
        self.is_paused = True
        try:
            if self.model:
                self.model.reset()
        except Exception:
            pass

    def resume(self):
        self.is_paused = False
        try:
            if self.model:
                self.model.reset()
        except Exception:
            pass


class MicCapture:
    """Capture audio from ALSA device using arecord (no PyAudio needed)."""

    def __init__(self, detector: WakeWordDetector):
        self.detector = detector
        self.running = False
        self.thread = None
        self.proc = None
        self._pending_detection = None
        self._pending_lock = threading.Lock()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                pass
        if self.thread:
            self.thread.join(timeout=3)

    def _capture_loop(self):
        cmd = [
            "arecord", "-D", ALSA_DEVICE,
            "-f", "S16_LE", "-r", str(SAMPLE_RATE),
            "-c", "1", "-t", "raw",
        ]
        logger.info(f"Starting mic: {' '.join(cmd)}")

        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.detector.is_listening = True
            logger.info(f"Mic active on {ALSA_DEVICE} — listening for wake word...")

            bytes_per_chunk = CHUNK_SIZE * 2  # int16 = 2 bytes per sample

            while self.running:
                data = self.proc.stdout.read(bytes_per_chunk)
                if not data:
                    if not self.running:
                        break
                    # arecord died — try to restart
                    logger.warning("arecord stopped, restarting in 2s...")
                    time.sleep(2)
                    self.proc = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    )
                    continue

                if len(data) < bytes_per_chunk:
                    continue

                audio = np.frombuffer(data, dtype=np.int16)
                result = self.detector.process_audio(audio)

                if result.get("detected"):
                    with self._pending_lock:
                        self._pending_detection = result

        except Exception as e:
            logger.error(f"Mic capture error: {e}")
        finally:
            self.detector.is_listening = False

    def consume_detection(self) -> dict:
        with self._pending_lock:
            d = self._pending_detection
            self._pending_detection = None
            return d


def create_app(detector: WakeWordDetector, mic: MicCapture):
    from flask import Flask, jsonify
    from flask_cors import CORS

    app = Flask(__name__)
    CORS(app)

    @app.route("/wakeword/poll", methods=["GET"])
    def poll():
        detection = mic.consume_detection()
        if detection:
            return jsonify({"wake": True, **detection})
        return jsonify({"wake": False})

    @app.route("/wakeword/audio", methods=["POST"])
    def audio():
        # Legacy endpoint — no longer needed with server-side mic
        return jsonify({"detected": False})

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
            "listening": detector.is_listening,
            "paused": detector.is_paused,
            "models": detector.model_names,
            "threshold": THRESHOLD,
            "detections": detector.detection_count,
            "mic_device": ALSA_DEVICE,
            "server_side_mic": True,
            "has_custom_model": bool(CUSTOM_MODEL) or os.path.exists(
                os.path.join(os.path.dirname(__file__), "models", "hey_viron.onnx")
            ),
        })

    return app


def main():
    print(f"""
    ═══════════════════════════════════════
    VIRON Wake Word Service (Server-Side Mic)
    ReSpeaker → openWakeWord → Browser polls
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
    🎤 Mic: {ALSA_DEVICE} (ReSpeaker)
    🔍 Poll: GET /wakeword/poll
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
