"""
VIRON Wake Word Service — Whisper-Based "Hey VIRON" Detection
=============================================================
Captures mic from ReSpeaker, uses energy VAD to detect speech,
sends short clips to Whisper to check for "hey viron" wake phrase.

No openWakeWord needed — works directly with the actual wake phrase.

Usage:
  python3 wakeword/service.py

Environment:
  VIRON_MIC_DEVICE=plughw:2,0      ALSA device (ReSpeaker)
  VIRON_WAKEWORD_PORT=8085         HTTP port
  OPENAI_API_KEY=...               For Whisper API
"""

import os
import sys
import re
import time
import wave
import threading
import logging
import subprocess
import tempfile
import numpy as np
from collections import deque

# — Configuration —
ALSA_DEVICE = os.environ.get("VIRON_MIC_DEVICE", "plughw:2,0")
PORT = int(os.environ.get("VIRON_WAKEWORD_PORT", "8085"))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

SAMPLE_RATE = 16000
CHUNK_MS = 80  # 80ms chunks
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000  # 1280
BYTES_PER_CHUNK = CHUNK_SAMPLES * 2  # int16

# How many chunks of pre-roll to keep (captures "hey" before energy spike)
PRE_ROLL_CHUNKS = 12  # ~1 second
# Max speech duration before forced cutoff
MAX_SPEECH_CHUNKS = 38  # ~3 seconds
# Silence after speech to stop
SILENCE_CHUNKS = 5  # ~0.4 seconds

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("viron-wakeword")

# Wake word patterns
WAKE_PATTERNS = [
    # Greek
    r"βίρον", r"βιρον", r"γεια\s*(σου\s*)?βίρον", r"γεια\s*(σου\s*)?βιρον",
    r"hey?\s*βίρον", r"hey?\s*βιρον",
    # English  
    r"hey?\s*v[iy]ron", r"hey?\s*v[iy]r+on",
    r"hi\s*v[iy]ron", r"hay\s*v[iy]ron",
    # Common misheard variants
    r"hey?\s*[bv]iron", r"hey?\s*byron", r"hey?\s*myron",
    r"hey?\s*vi+ron", r"hey?\s*veron",
    # Just "viron" alone
    r"\bv[iy]ron\b", r"\bβίρον\b", r"\bβιρον\b",
]
WAKE_RE = re.compile("|".join(WAKE_PATTERNS), re.IGNORECASE)


class WakeWordDetector:
    def __init__(self):
        self.is_listening = False
        self.is_paused = False
        self.detection_count = 0
        self.last_detection = 0
        self.detections = deque(maxlen=50)
        self._lock = threading.Lock()

    def check_transcript(self, text: str) -> bool:
        """Check if transcription contains wake word."""
        if not text:
            return False
        text = text.strip().lower()
        if WAKE_RE.search(text):
            return True
        return False

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False


class MicCapture:
    """Continuous mic capture with energy VAD and Whisper wake word check."""

    def __init__(self, detector: WakeWordDetector):
        self.detector = detector
        self.running = False
        self.thread = None
        self.proc = None
        self._pending_detection = None
        self._pending_lock = threading.Lock()
        self._paused = False
        self._restart_event = threading.Event()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self._restart_event.set()
        self._kill_arecord()
        if self.thread:
            self.thread.join(timeout=3)

    def pause_mic(self):
        self._paused = True
        self._kill_arecord()
        logger.info("Mic released for conversation")

    def resume_mic(self):
        self._paused = False
        self._restart_event.set()
        logger.info("Mic resuming for wake word")

    def _kill_arecord(self):
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except:
                    pass
            self.proc = None

    def _capture_loop(self):
        cmd = ["arecord", "-D", ALSA_DEVICE, "-f", "S16_LE",
               "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw"]

        while self.running:
            if self._paused:
                self._restart_event.wait(timeout=1)
                self._restart_event.clear()
                continue

            try:
                self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.detector.is_listening = True
                logger.info(f"Listening on {ALSA_DEVICE} for 'Hey VIRON'...")

                # Calibrate noise floor
                noise_values = []
                for _ in range(6):
                    data = self.proc.stdout.read(BYTES_PER_CHUNK)
                    if not data:
                        break
                    samples = np.frombuffer(data, dtype=np.int16)
                    noise_values.append(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))

                noise_floor = np.mean(noise_values) if noise_values else 50
                speech_thresh = max(noise_floor * 3, 50)
                logger.info(f"Noise floor={noise_floor:.0f}, speech_thresh={speech_thresh:.0f}")

                # Rolling pre-buffer
                pre_buffer = deque(maxlen=PRE_ROLL_CHUNKS)
                speech_active = False
                speech_chunks = []
                silence_count = 0

                while self.running and not self._paused:
                    data = self.proc.stdout.read(BYTES_PER_CHUNK)
                    if not data:
                        break

                    if self.detector.is_paused:
                        continue

                    samples = np.frombuffer(data, dtype=np.int16)
                    rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))

                    if not speech_active:
                        pre_buffer.append(data)
                        if rms > speech_thresh:
                            speech_active = True
                            silence_count = 0
                            speech_chunks = list(pre_buffer) + [data]
                            logger.info(f"Speech start (RMS={rms:.0f})")
                    else:
                        speech_chunks.append(data)
                        if rms < speech_thresh * 0.5:
                            silence_count += 1
                        else:
                            silence_count = 0

                        # End of speech or max duration
                        if silence_count >= SILENCE_CHUNKS or len(speech_chunks) >= MAX_SPEECH_CHUNKS:
                            duration_ms = len(speech_chunks) * CHUNK_MS
                            logger.info(f"Speech end ({duration_ms}ms), checking Whisper...")

                            # Send to Whisper in background
                            audio_data = b''.join(speech_chunks)
                            threading.Thread(
                                target=self._check_whisper,
                                args=(audio_data,),
                                daemon=True
                            ).start()

                            # Reset
                            speech_active = False
                            speech_chunks = []
                            silence_count = 0
                            pre_buffer.clear()

                self._kill_arecord()
                self.detector.is_listening = False

            except Exception as e:
                logger.error(f"Capture error: {e}")
                self.detector.is_listening = False
                time.sleep(2)

    def _check_whisper(self, audio_data: bytes):
        """Send audio to Whisper API and check for wake word."""
        try:
            # Save to temp WAV
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                wav_path = tmp.name
                with wave.open(tmp, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(audio_data)

            text = ""

            # Try OpenAI Whisper API (fast)
            if OPENAI_API_KEY:
                try:
                    import urllib.request
                    import json

                    # Multipart form upload
                    boundary = '----WakeWordBoundary'
                    with open(wav_path, 'rb') as f:
                        file_data = f.read()

                    body = (
                        f'--{boundary}\r\n'
                        f'Content-Disposition: form-data; name="file"; filename="wake.wav"\r\n'
                        f'Content-Type: audio/wav\r\n\r\n'
                    ).encode() + file_data + (
                        f'\r\n--{boundary}\r\n'
                        f'Content-Disposition: form-data; name="model"\r\n\r\n'
                        f'whisper-1\r\n'
                        f'--{boundary}\r\n'
                        f'Content-Disposition: form-data; name="language"\r\n\r\n'
                        f'el\r\n'
                        f'--{boundary}--\r\n'
                    ).encode()

                    req = urllib.request.Request(
                        'https://api.openai.com/v1/audio/transcriptions',
                        data=body,
                        headers={
                            'Authorization': f'Bearer {OPENAI_API_KEY}',
                            'Content-Type': f'multipart/form-data; boundary={boundary}',
                        }
                    )
                    resp = urllib.request.urlopen(req, timeout=5)
                    result = json.loads(resp.read())
                    text = result.get('text', '').strip()
                    logger.info(f"Whisper: \"{text}\"")
                except Exception as e:
                    logger.warning(f"Whisper API error: {e}")

            # Fallback: local Whisper via Flask server
            if not text:
                try:
                    import urllib.request
                    import json

                    with open(wav_path, 'rb') as f:
                        file_data = f.read()

                    boundary = '----WakeWordLocal'
                    body = (
                        f'--{boundary}\r\n'
                        f'Content-Disposition: form-data; name="audio"; filename="wake.wav"\r\n'
                        f'Content-Type: audio/wav\r\n\r\n'
                    ).encode() + file_data + (
                        f'\r\n--{boundary}\r\n'
                        f'Content-Disposition: form-data; name="lang"\r\n\r\n'
                        f'el\r\n'
                        f'--{boundary}--\r\n'
                    ).encode()

                    req = urllib.request.Request(
                        'http://127.0.0.1:5000/api/stt',
                        data=body,
                        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
                    )
                    resp = urllib.request.urlopen(req, timeout=10)
                    result = json.loads(resp.read())
                    text = result.get('text', '').strip()
                    logger.info(f"Local Whisper: \"{text}\"")
                except Exception as e:
                    logger.warning(f"Local Whisper error: {e}")

            # Clean up
            try:
                os.unlink(wav_path)
            except:
                pass

            # Check for wake word
            if text and self.detector.check_transcript(text):
                now = time.time()
                if now - self.detector.last_detection > 2.0:
                    self.detector.last_detection = now
                    self.detector.detection_count += 1
                    detection = {
                        "detected": True,
                        "model": "whisper",
                        "transcript": text,
                        "score": 1.0,
                        "time": now,
                        "count": self.detector.detection_count,
                    }
                    with self.detector._lock:
                        self.detector.detections.append(detection)
                    with self._pending_lock:
                        self._pending_detection = detection
                    logger.info(f"🎯 WAKE WORD: \"{text}\"")
            elif text:
                logger.info(f"Not wake word: \"{text}\"")

        except Exception as e:
            logger.error(f"Whisper check error: {e}")

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
        return jsonify({"detected": False})

    @app.route("/wakeword/pause", methods=["POST"])
    def pause():
        detector.pause()
        mic.pause_mic()
        return jsonify({"status": "paused"})

    @app.route("/wakeword/resume", methods=["POST"])
    def resume():
        detector.resume()
        mic.resume_mic()
        return jsonify({"status": "listening"})

    @app.route("/wakeword/status", methods=["GET"])
    def status():
        return jsonify({
            "ready": True,
            "listening": detector.is_listening,
            "paused": detector.is_paused,
            "models": ["whisper-hey-viron"],
            "threshold": 0,
            "detections": detector.detection_count,
            "mic_device": ALSA_DEVICE,
            "server_side_mic": True,
            "method": "whisper",
        })

    return app


def main():
    print(f"""
    ═══════════════════════════════════════
    VIRON Wake Word: "Hey VIRON" (Whisper)
    ReSpeaker → Energy VAD → Whisper → Match
    ═══════════════════════════════════════""")

    if not OPENAI_API_KEY:
        logger.warning("No OPENAI_API_KEY — will use local Whisper (slower)")

    detector = WakeWordDetector()
    mic = MicCapture(detector)
    mic.start()

    app = create_app(detector, mic)

    print(f"""
    📡 http://0.0.0.0:{PORT}
    🎤 Mic: {ALSA_DEVICE} (ReSpeaker)
    🔍 Wake phrase: "Hey VIRON" / "Γεια σου VIRON"
    🎯 Method: Whisper speech recognition
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
