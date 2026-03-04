"""
VIRON Wake Word — Hybrid: openWakeWord (fast) + Whisper (hey viron)
===================================================================
Two parallel detectors:
  1. openWakeWord "hey jarvis" model → instant (~50ms) detection
  2. Speech segments → Whisper STT → "hey viron" pattern match (~1-2s)

Either trigger wakes VIRON.

Usage:
  python3 wakeword/service.py

Environment:
  VIRON_MIC_DEVICE=plughw:2,0
  VIRON_WAKEWORD_PORT=8085
  VIRON_STT_URL=http://127.0.0.1:5000
"""

import os, sys, re, time, wave, json
import threading, logging, subprocess, tempfile
import numpy as np
from collections import deque

ALSA_DEVICE = os.environ.get("VIRON_MIC_DEVICE", "plughw:2,0")
PORT = int(os.environ.get("VIRON_WAKEWORD_PORT", "8085"))
STT_URL = os.environ.get("VIRON_STT_URL", "http://127.0.0.1:5000")
OWW_THRESHOLD = float(os.environ.get("VIRON_WAKEWORD_THRESHOLD", "0.5"))

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # 80ms
BYTES_PER_CHUNK = CHUNK_SIZE * 2

# Whisper speech segment limits
MIN_SPEECH_CHUNKS = 3      # 240ms min
MAX_WAKE_CHUNKS = 25       # 2.0s max (wake word is short)
SILENCE_END_CHUNKS = 4     # 320ms silence = end of phrase

# "hey viron" patterns
WAKE_PATTERNS = [
    r'h?e+y?\s*v[iy]r[oa]n', r'v[iy]+r[oa]+n', r'b[iy]r[oa]n',
    r'h?e+y?\s*j[aá]rv[iu]s',
    r'β[αά]ι?ρ[οό]ν', r'γ[ει]α\s*β[αά]ι?ρ[οό]ν',
    r'βίρον', r'βέρον',
]
WAKE_REGEX = re.compile('|'.join(WAKE_PATTERNS), re.IGNORECASE)
WAKE_SUBS = ["viron","veron","byron","biron","vairon","βάιρον","βιρον","βαιρον","jarvis"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("viron-wakeword")


def matches_wake(text):
    if not text: return False
    t = text.lower().strip()
    if WAKE_REGEX.search(t): return True
    return any(s in t for s in WAKE_SUBS)


class Detector:
    def __init__(self):
        self.is_listening = False
        self.is_paused = False
        self.detection_count = 0
        self.last_detection = 0
        self._pending = None
        self._lock = threading.Lock()
        # openWakeWord
        self.oww_model = None
        self.oww_names = []

    def init_oww(self):
        """Try to load openWakeWord model."""
        try:
            import openwakeword
            from openwakeword.model import Model
            openwakeword.utils.download_models()

            custom = os.path.join(os.path.dirname(__file__), "models", "hey_viron.onnx")
            if os.path.exists(custom):
                logger.info(f"Loading custom model: {custom}")
                self.oww_model = Model(wakeword_models=[custom], vad_threshold=0.5)
            else:
                logger.info("Loading 'hey jarvis' model (say 'Hey Jarvis' OR 'Hey VIRON')")
                self.oww_model = Model(wakeword_models=["hey_jarvis_v0.1"], vad_threshold=0.5)

            self.oww_names = list(self.oww_model.models.keys())
            logger.info(f"openWakeWord ready: {self.oww_names}, threshold={OWW_THRESHOLD}")
            return True
        except Exception as e:
            logger.warning(f"openWakeWord not available: {e}")
            return False

    def process_oww(self, audio_int16):
        """Run openWakeWord on audio chunk. Returns True if wake detected."""
        if not self.oww_model or self.is_paused:
            return False
        try:
            preds = self.oww_model.predict(audio_int16)
            for name in self.oww_names:
                score = preds[name]
                if score > 0.1:
                    logger.info(f"OWW: {name}={score:.3f}")
                if score >= OWW_THRESHOLD:
                    self.set_detection(f"hey jarvis (OWW:{score:.2f})", score)
                    return True
        except Exception as e:
            logger.warning(f"OWW error: {e}")
        return False

    def set_detection(self, text, score=1.0):
        now = time.time()
        if now - self.last_detection < 2.0: return
        self.last_detection = now
        self.detection_count += 1
        with self._lock:
            self._pending = {
                "detected": True, "model": "hybrid",
                "text": text, "score": float(score),
                "time": now, "count": self.detection_count,
            }
        logger.info(f"WAKE WORD: '{text}'")

    def consume(self):
        with self._lock:
            d = self._pending
            self._pending = None
            return d

    def pause(self):
        self.is_paused = True
        try:
            if self.oww_model: self.oww_model.reset()
        except: pass

    def resume(self):
        self.is_paused = False
        try:
            if self.oww_model: self.oww_model.reset()
        except: pass


class MicCapture:
    def __init__(self, detector):
        self.det = detector
        self.running = False
        self.thread = None
        self.proc = None
        self._paused = False
        self._restart = threading.Event()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self._restart.set()
        self._kill()
        if self.thread: self.thread.join(timeout=3)

    def pause_mic(self):
        self._paused = True
        self._kill()
        logger.info("Mic released")

    def resume_mic(self):
        self._paused = False
        self._restart.set()
        logger.info("Mic resuming")

    def _kill(self):
        if self.proc:
            try: self.proc.terminate(); self.proc.wait(timeout=2)
            except:
                try: self.proc.kill()
                except: pass
            self.proc = None

    def _whisper_check(self, frames):
        """Send speech segment to Whisper, check for wake word."""
        try:
            raw = b''.join(frames)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                p = tmp.name
                with wave.open(tmp, 'wb') as wf:
                    wf.setnchannels(1); wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE); wf.writeframes(raw)

            import urllib.request
            boundary = '----WB'
            with open(p, 'rb') as f: fdata = f.read()
            os.unlink(p)

            body = (f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="w.wav"\r\n'
                    f'Content-Type: audio/wav\r\n\r\n').encode() + fdata + (
                    f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="lang"\r\n\r\nel'
                    f'\r\n--{boundary}--\r\n').encode()

            req = urllib.request.Request(f'{STT_URL}/api/stt', data=body,
                headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
            resp = urllib.request.urlopen(req, timeout=8)
            text = json.loads(resp.read()).get('text', '').strip()

            ms = len(frames) * 80
            logger.info(f"Whisper ({ms}ms): '{text}'")
            if matches_wake(text):
                self.det.set_detection(f"hey viron (Whisper: '{text}')")
        except Exception as e:
            logger.warning(f"Whisper error: {e}")

    def _loop(self):
        cmd = ["arecord", "-D", ALSA_DEVICE, "-f", "S16_LE",
               "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw"]

        while self.running:
            if self._paused:
                self._restart.wait(timeout=1)
                self._restart.clear()
                continue

            try:
                self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.det.is_listening = True
                logger.info(f"Mic active: {ALSA_DEVICE}")

                # Calibrate noise
                noise_vals = []
                for _ in range(8):
                    d = self.proc.stdout.read(BYTES_PER_CHUNK)
                    if not d or len(d) < BYTES_PER_CHUNK: break
                    s = np.frombuffer(d, dtype=np.int16)
                    noise_vals.append(np.sqrt(np.mean(s.astype(np.float32)**2)))

                nf = np.mean(noise_vals) if noise_vals else 50
                sp_thresh = max(nf * 2.0, 35)
                si_thresh = max(nf * 1.2, 20)
                logger.info(f"Noise={nf:.0f} speech>{sp_thresh:.0f} silence<{si_thresh:.0f}")
                logger.info("Listening... say 'Hey VIRON' or 'Hey Jarvis'")

                speech_frames = []
                in_speech = False
                sil_count = 0

                while self.running and not self._paused:
                    d = self.proc.stdout.read(BYTES_PER_CHUNK)
                    if not d or len(d) < BYTES_PER_CHUNK: break
                    if self.det.is_paused: continue

                    audio = np.frombuffer(d, dtype=np.int16)
                    rms = np.sqrt(np.mean(audio.astype(np.float32)**2))

                    # === openWakeWord (instant, every chunk) ===
                    if self.det.oww_model:
                        self.det.process_oww(audio)

                    # === Whisper path (speech segments) ===
                    if not in_speech:
                        if rms > sp_thresh:
                            in_speech = True
                            sil_count = 0
                            speech_frames = [d]
                    else:
                        speech_frames.append(d)
                        if rms < si_thresh:
                            sil_count += 1
                        else:
                            sil_count = 0

                        n = len(speech_frames)
                        if n > MAX_WAKE_CHUNKS:
                            in_speech = False; speech_frames = []; sil_count = 0
                            continue

                        if sil_count >= SILENCE_END_CHUNKS and n >= MIN_SPEECH_CHUNKS:
                            # Trim trailing silence
                            trim = min(sil_count, n - MIN_SPEECH_CHUNKS)
                            if trim > 0: speech_frames = speech_frames[:-trim]

                            # Send to Whisper in background
                            fc = list(speech_frames)
                            threading.Thread(target=self._whisper_check, args=(fc,), daemon=True).start()

                            in_speech = False; speech_frames = []; sil_count = 0

                self._kill()
                self.det.is_listening = False

            except Exception as e:
                logger.error(f"Mic error: {e}")
                self.det.is_listening = False
                time.sleep(2)


def create_app(det, mic):
    from flask import Flask, jsonify
    from flask_cors import CORS
    app = Flask(__name__)
    CORS(app)

    @app.route("/wakeword/poll", methods=["GET"])
    def poll():
        d = det.consume()
        if d: return jsonify({"wake": True, **d})
        return jsonify({"wake": False})

    @app.route("/wakeword/audio", methods=["POST"])
    def audio(): return jsonify({"detected": False})

    @app.route("/wakeword/pause", methods=["POST"])
    def pause():
        det.pause(); mic.pause_mic()
        return jsonify({"status": "paused"})

    @app.route("/wakeword/resume", methods=["POST"])
    def resume():
        det.resume(); mic.resume_mic()
        return jsonify({"status": "listening"})

    @app.route("/wakeword/status", methods=["GET"])
    def status():
        return jsonify({
            "ready": True, "listening": det.is_listening,
            "paused": det.is_paused, "models": det.oww_names + ["whisper"],
            "detections": det.detection_count,
            "mic_device": ALSA_DEVICE, "server_side_mic": True,
            "mode": "hybrid" if det.oww_model else "whisper-only",
        })

    return app


def main():
    det = Detector()
    has_oww = det.init_oww()

    mic = MicCapture(det)
    mic.start()

    app = create_app(det, mic)

    mode = "HYBRID (OWW instant + Whisper)" if has_oww else "WHISPER-ONLY"
    print(f"""
    ═══════════════════════════════════════
    VIRON Wake Word — {mode}
    Say "Hey VIRON" or "Hey Jarvis"
    ═══════════════════════════════════════
    📡 http://0.0.0.0:{PORT}
    🎤 {ALSA_DEVICE} (ReSpeaker)
    ═══════════════════════════════════════
""")
    try:
        app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
    except KeyboardInterrupt: pass
    finally: mic.stop()


if __name__ == "__main__":
    main()
