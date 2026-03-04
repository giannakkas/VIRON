"""
VIRON Wake Word Service — Whisper-Based (Server-Side Mic)
=========================================================
Captures mic from ReSpeaker, uses energy VAD to detect speech,
sends short segments to Whisper for "hey viron" recognition.

Architecture:
  arecord (ReSpeaker) → energy VAD → Whisper STT → pattern match → poll

Usage:
  python3 wakeword/service.py

Environment:
  VIRON_MIC_DEVICE=plughw:2,0      ALSA device (ReSpeaker = card 2)
  VIRON_WAKEWORD_PORT=8085         HTTP port
  VIRON_STT_URL=http://127.0.0.1:5000  Flask server for STT
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
STT_URL = os.environ.get("VIRON_STT_URL", "http://127.0.0.1:5000")

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # 80ms frames
BYTES_PER_CHUNK = CHUNK_SIZE * 2  # int16

# VAD settings
CALIBRATION_CHUNKS = 10    # 0.8s calibration
SPEECH_MULT = 2.5          # Speech = noise_floor * this
SILENCE_MULT = 1.3         # Silence = noise_floor * this
MIN_SPEECH_CHUNKS = 4      # Min 320ms to count as speech
MAX_WAKE_CHUNKS = 40       # Max 3.2s for a wake word (ignore longer)
SILENCE_END_CHUNKS = 8     # 640ms silence = end of phrase

# Wake word patterns
WAKE_PATTERNS = [
    # English
    r'\bh?e+y?\s*v[iy]r[oa]n\b',     # hey viron, viron
    r'\bv[iy]+r[oa]+n\b',             # viron, veron, byron
    r'\bb[iy]r[oa]n\b',               # biron, byron
    r'\bh?e+y?\s*j[aá]rv[iu]s\b',    # hey jarvis (legacy)
    # Greek transliterations
    r'\bβ[αά]ι?ρ[οό]ν\b',            # βάιρον
    r'\bγ[ει]α\s*β[αά]ι?ρ[οό]ν\b',  # γεια βάιρον
    r'\bχ[εέ]ι\s*β[αά]ι?ρ[οό]ν\b',  # χέι βάιρον
    r'\bβ[ιί]ρ[οό]ν\b',              # βίρον
]
WAKE_REGEX = re.compile('|'.join(WAKE_PATTERNS), re.IGNORECASE)

# Simple substring fallback
WAKE_SUBSTRINGS = [
    "viron", "veron", "byron", "biron", "vairon",
    "βάιρον", "βιρον", "βαιρον", "βέρον",
    "jarvis",  # legacy
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("viron-wakeword")


def matches_wake_word(text: str) -> bool:
    """Check if transcript contains a wake word."""
    if not text:
        return False
    t = text.lower().strip()
    # Regex check
    if WAKE_REGEX.search(t):
        return True
    # Substring check
    for sub in WAKE_SUBSTRINGS:
        if sub in t:
            return True
    return False


class WakeWordDetector:
    def __init__(self):
        self.is_listening = False
        self.is_paused = False
        self.detection_count = 0
        self.last_detection = 0
        self._pending_detection = None
        self._lock = threading.Lock()

    def set_detection(self, text: str, score: float = 1.0):
        now = time.time()
        if now - self.last_detection < 2.0:
            return  # debounce
        self.last_detection = now
        self.detection_count += 1
        with self._lock:
            self._pending_detection = {
                "detected": True,
                "model": "whisper",
                "text": text,
                "score": score,
                "time": now,
                "count": self.detection_count,
            }
        logger.info(f"WAKE WORD DETECTED: '{text}'")

    def consume_detection(self) -> dict:
        with self._lock:
            d = self._pending_detection
            self._pending_detection = None
            return d

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False


class MicCapture:
    """Capture audio, detect speech segments, send to Whisper."""

    def __init__(self, detector: WakeWordDetector):
        self.detector = detector
        self.running = False
        self.thread = None
        self.proc = None
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
            except:
                try:
                    self.proc.kill()
                except:
                    pass
            self.proc = None

    def _send_to_whisper(self, audio_frames: list) -> str:
        """Save audio to WAV, send to Flask /api/stt, return transcript."""
        try:
            raw = b''.join(audio_frames)
            duration_ms = len(audio_frames) * 80

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                wav_path = tmp.name
                with wave.open(tmp, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(raw)

            # POST to Flask STT
            import urllib.request
            import json

            boundary = '----WakeWordBoundary'
            with open(wav_path, 'rb') as f:
                file_data = f.read()
            os.unlink(wav_path)

            # Build multipart form data
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
                f'{STT_URL}/api/stt',
                data=body,
                headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
            )
            resp = urllib.request.urlopen(req, timeout=8)
            result = json.loads(resp.read())
            text = result.get('text', '').strip()
            logger.info(f"Whisper ({duration_ms}ms): '{text}'")
            return text

        except Exception as e:
            logger.warning(f"Whisper call failed: {e}")
            return ""

    def _capture_loop(self):
        cmd = [
            "arecord", "-D", ALSA_DEVICE,
            "-f", "S16_LE", "-r", str(SAMPLE_RATE),
            "-c", "1", "-t", "raw",
        ]

        while self.running:
            if self._paused:
                self._restart_event.wait(timeout=1)
                self._restart_event.clear()
                continue

            try:
                self.proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                self.detector.is_listening = True
                logger.info(f"Mic active on {ALSA_DEVICE}")

                # Calibrate noise floor
                noise_values = []
                for _ in range(CALIBRATION_CHUNKS):
                    data = self.proc.stdout.read(BYTES_PER_CHUNK)
                    if not data or len(data) < BYTES_PER_CHUNK:
                        break
                    samples = np.frombuffer(data, dtype=np.int16)
                    rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
                    noise_values.append(rms)

                noise_floor = np.mean(noise_values) if noise_values else 50
                speech_thresh = max(noise_floor * SPEECH_MULT, 40)
                silence_thresh = max(noise_floor * SILENCE_MULT, 25)
                logger.info(f"Noise={noise_floor:.0f}, speech>{speech_thresh:.0f}, silence<{silence_thresh:.0f}")
                logger.info("Listening for 'Hey VIRON'...")

                # Main detection loop
                speech_frames = []
                in_speech = False
                silence_count = 0
                whisper_busy = False

                while self.running and not self._paused:
                    data = self.proc.stdout.read(BYTES_PER_CHUNK)
                    if not data or len(data) < BYTES_PER_CHUNK:
                        break

                    if self.detector.is_paused:
                        continue

                    samples = np.frombuffer(data, dtype=np.int16)
                    rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))

                    if not in_speech:
                        if rms > speech_thresh:
                            in_speech = True
                            silence_count = 0
                            speech_frames = [data]  # Start collecting
                    else:
                        speech_frames.append(data)

                        if rms < silence_thresh:
                            silence_count += 1
                        else:
                            silence_count = 0

                        n = len(speech_frames)

                        # Too long → not a wake word, reset
                        if n > MAX_WAKE_CHUNKS:
                            in_speech = False
                            speech_frames = []
                            silence_count = 0
                            continue

                        # End of phrase detected
                        if silence_count >= SILENCE_END_CHUNKS and n >= MIN_SPEECH_CHUNKS:
                            # Trim trailing silence
                            trim = min(silence_count, len(speech_frames) - MIN_SPEECH_CHUNKS)
                            if trim > 0:
                                speech_frames = speech_frames[:-trim]

                            duration_ms = len(speech_frames) * 80
                            logger.info(f"Speech segment: {duration_ms}ms ({len(speech_frames)} chunks)")

                            # Send to Whisper in background
                            frames_copy = list(speech_frames)
                            def check_wake(frames=frames_copy):
                                text = self._send_to_whisper(frames)
                                if matches_wake_word(text):
                                    self.detector.set_detection(text)

                            t = threading.Thread(target=check_wake, daemon=True)
                            t.start()

                            in_speech = False
                            speech_frames = []
                            silence_count = 0

                self._kill_arecord()
                self.detector.is_listening = False

            except Exception as e:
                logger.error(f"Mic error: {e}")
                self.detector.is_listening = False
                time.sleep(2)


def create_app(detector: WakeWordDetector, mic: MicCapture):
    from flask import Flask, jsonify
    from flask_cors import CORS

    app = Flask(__name__)
    CORS(app)

    @app.route("/wakeword/poll", methods=["GET"])
    def poll():
        detection = detector.consume_detection()
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
            "models": ["whisper-wake"],
            "threshold": 0,
            "detections": detector.detection_count,
            "mic_device": ALSA_DEVICE,
            "server_side_mic": True,
            "mode": "whisper",
        })

    return app


def main():
    print(f"""
    ═══════════════════════════════════════
    VIRON Wake Word (Whisper-Based)
    Say "Hey VIRON" → ReSpeaker → Whisper
    ═══════════════════════════════════════""")

    detector = WakeWordDetector()
    mic = MicCapture(detector)
    mic.start()

    app = create_app(detector, mic)

    print(f"""
    📡 http://0.0.0.0:{PORT}
    🎤 Mic: {ALSA_DEVICE} (ReSpeaker)
    🔍 Poll: GET /wakeword/poll
    🎯 Wake: "Hey VIRON" (+ variants)
    🧠 STT: {STT_URL}/api/stt
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
