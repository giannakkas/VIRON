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
# LOWERED from 0.35 → 0.28 to catch quieter wake words at distance
OWW_THRESHOLD = float(os.environ.get("VIRON_WAKEWORD_THRESHOLD", "0.28"))
# Which stereo channel to use: 0=left (beamformed+AEC), 1=right (ASR beam)
MIC_CHANNEL = int(os.environ.get("VIRON_MIC_CHANNEL", "1"))
# Adaptive threshold: adjusts OWW threshold based on ambient noise
ADAPTIVE_ENABLED = os.environ.get("VIRON_WAKEWORD_ADAPTIVE", "1") == "1"
ADAPTIVE_MIN = 0.18
ADAPTIVE_MAX = 0.45

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # 80ms mono samples
# XVF3800: stereo output — ch0=beamformed+AEC, ch1=ASR-optimized beam (best for STT)
BYTES_PER_CHUNK = CHUNK_SIZE * 4  # stereo: 2ch × 2 bytes/sample

# Whisper speech segment limits
MIN_SPEECH_CHUNKS = 8      # 640ms min (was 240ms — "Hey VIRON" needs ~800ms)
MAX_WAKE_CHUNKS = 25       # 2.0s max (wake word is short)
SILENCE_END_CHUNKS = 6     # 480ms silence = end of phrase (was 320ms — allows brief pauses)

# "hey viron" patterns — MUST catch common Whisper misrecognitions
# Real Whisper outputs observed: "Hey, Vero", "Bye", "Hey Jarvis"
WAKE_PATTERNS = [
    r'h?e+y?,?\s*v[iy]r[oa]n',  # hey viron, hey vyron
    r'v[iy]+r[oa]+n',            # viron, vyron
    r'b[iy]r[oa]n',              # biron, byron
    r'h?e+y?,?\s*j[aá]rv[iu]s', # hey jarvis
    r'β[αά]ι?ρ[οό]ν',           # βαίρον (Greek)
    r'γ[ει]α\s*β[αά]ι?ρ[οό]ν', # γεια βαίρον (Greek)
    r'βίρον', r'βέρον',
    # Common Whisper misrecognitions (observed from real testing)
    r'h?e+y?,?\s*ver[oa]',       # "hey vero" ← MOST COMMON MISS
    r'h?e+y?,?\s*b[iy]r[oa]n?',  # "hey biron", "hey biro"
    r'h?e+y?,?\s*v[ae]r[oa]n?',  # "hey varon", "hey varo"
    r'h?e+y?,?\s*iron',           # "hey iron"
    r'h?e+y?,?\s*v[iy]r[oa]$',   # "hey viro" (truncated)
    r'h?e+y?,?\s*v[iy]run',      # "hey virun"
    r'h?e+y?,?\s*ver[oa]n',      # "hey veron"
    r'h?e+y?,?\s*byr[oa]n',      # "hey byron"
    r'h?e+y?,?\s*bar[oa]n',      # "hey baron"
    r'h?e+y?,?\s*brian',          # "hey brian"
]
WAKE_REGEX = re.compile('|'.join(WAKE_PATTERNS), re.IGNORECASE)
WAKE_SUBS = ["viron","veron","byron","biron","vairon","βάιρον","βιρον","βαιρον",
             "jarvis","iron","varon","vyron",
             # Critical additions from real Whisper testing:
             "vero","viro","biro","vera","baron","brian","verone","virun"]

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
        # Custom mel classifier (trained on user's voice)
        self.custom_model = None
        self.custom_threshold = 0.70  # high threshold to avoid false triggers

    def init_custom_model(self):
        """Load custom hey_viron mel classifier (JSON or ONNX)."""
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        json_path = os.path.join(model_dir, "hey_viron_simple.json")
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    data = json.load(f)
                self.custom_model = {
                    "type": "json",
                    "weights": np.array(data["weights"], dtype=np.float32),
                    "bias": float(data["bias"]),
                    "mean": np.array(data["mean"], dtype=np.float32),
                    "std": np.array(data["std"], dtype=np.float32),
                    "n_mels": data.get("n_mels", 40),
                    "n_fft": data.get("n_fft", 512),
                    "hop": data.get("hop", 160),
                }
                logger.info(f"Custom 'hey viron' model loaded (acc={data.get('accuracy','?')}, {data.get('n_positive',0)} positive clips)")
                return True
            except Exception as e:
                logger.warning(f"Failed to load custom model: {e}")
        return False

    def _compute_mel_features(self, audio_float):
        """Compute mel-spectrogram features from audio (float32, -1 to 1)."""
        if self.custom_model is None:
            return None
        m = self.custom_model
        n_mels, n_fft, hop = m["n_mels"], m["n_fft"], m["hop"]

        # Simple STFT
        frames = []
        for start in range(0, len(audio_float) - n_fft, hop):
            frame = audio_float[start:start + n_fft]
            windowed = frame * np.hanning(n_fft)
            spectrum = np.abs(np.fft.rfft(windowed)) ** 2
            frames.append(spectrum)
        if not frames:
            return None
        power = np.array(frames).T  # shape: (n_fft//2+1, n_frames)

        # Mel filterbank
        mel_freqs = np.linspace(0, 2595 * np.log10(1 + SAMPLE_RATE / 2 / 700), n_mels + 2)
        mel_freqs = 700 * (10 ** (mel_freqs / 2595) - 1)
        bin_freqs = np.floor((n_fft + 1) * mel_freqs / SAMPLE_RATE).astype(int)

        fb = np.zeros((n_mels, power.shape[0]))
        for i in range(n_mels):
            s, c, e = bin_freqs[i], bin_freqs[i + 1], bin_freqs[i + 2]
            if s < power.shape[0] and e < power.shape[0]:
                for j in range(s, c):
                    if c > s: fb[i, j] = (j - s) / (c - s)
                for j in range(c, e):
                    if e > c: fb[i, j] = (e - j) / (e - c)

        mel = np.dot(fb, power)
        mel = np.log(mel + 1e-10)
        return mel.mean(axis=1)  # average over time → fixed-size vector

    def process_custom(self, audio_buffer_int16):
        """Score audio buffer with custom mel classifier. Returns score 0-1."""
        if self.custom_model is None or self.is_paused:
            return 0.0
        try:
            audio = audio_buffer_int16.astype(np.float32) / 32768.0
            features = self._compute_mel_features(audio)
            if features is None:
                return 0.0
            m = self.custom_model
            normed = (features - m["mean"]) / m["std"]
            logit = float(normed @ m["weights"] + m["bias"])
            score = 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))
            return score
        except Exception as e:
            logger.warning(f"Custom model error: {e}")
            return 0.0

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

    def process_oww(self, audio_int16, threshold=None):
        """Run openWakeWord on audio chunk. Returns True if wake detected."""
        if not self.oww_model or self.is_paused:
            return False
        thresh = threshold if threshold is not None else OWW_THRESHOLD
        try:
            preds = self.oww_model.predict(audio_int16)
            for name in self.oww_names:
                score = preds[name]
                if score > 0.08:
                    logger.info(f"OWW: {name}={score:.3f}")
                if score >= thresh:
                    self.set_detection(f"hey jarvis (OWW:{score:.2f})", score)
                    return True
        except Exception as e:
            logger.warning(f"OWW error: {e}")
        return False

    def set_detection(self, text, score=1.0):
        now = time.time()
        # REDUCED debounce from 2.0s → 1.5s for faster re-detection
        if now - self.last_detection < 1.5: return
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
        # Track how many chunks to discard after resume (echo flush)
        self._flush_chunks = 0
        self._current_oww_thresh = OWW_THRESHOLD

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
        # Flush first 3 chunks (~240ms) after resume to discard echo remnants
        self._flush_chunks = 3
        self._restart.set()
        logger.info("Mic resuming (flushing 3 chunks for echo)")

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

            # Prompt hint helps Whisper expect "Hey VIRON" instead of hallucinating
            wake_prompt = "Hey VIRON. Hey Viron. Hey Byron. Hey Veron."

            body = (f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="w.wav"\r\n'
                    f'Content-Type: audio/wav\r\n\r\n').encode() + fdata + (
                    f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="lang"\r\n\r\nen'
                    f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="prompt"\r\n\r\n{wake_prompt}'
                    f'\r\n--{boundary}--\r\n').encode()

            req = urllib.request.Request(f'{STT_URL}/api/stt', data=body,
                headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
            resp = urllib.request.urlopen(req, timeout=8)
            text = json.loads(resp.read()).get('text', '').strip()

            ms = len(frames) * 80
            logger.info(f"Whisper ({ms}ms): '{text}'")
            
            # Filter common Whisper hallucinations on short clips
            WHISPER_HALLUCINATIONS = {
                "thank you", "thank you.", "thanks", "thank you so much",
                "thank you. that's great.", "cool", "cool.", "bye", "bye.",
                "oh", "oh.", "you", "you.", "yeah", "okay", "ok",
                "subtitles by", "like and subscribe", "music",
            }
            if text.lower().strip().rstrip('.!,') in {h.rstrip('.!,') for h in WHISPER_HALLUCINATIONS}:
                logger.debug(f"Filtered hallucination: '{text}'")
                return
            
            if matches_wake(text):
                self.det.set_detection(f"hey viron (Whisper: '{text}')")
        except Exception as e:
            logger.warning(f"Whisper error: {e}")

    def _loop(self):
        # Record stereo — XVF3800 outputs processed audio on both channels
        # ch0 (left)  = beamformed + AEC + noise suppressed
        # ch1 (right) = ASR-optimized beam (best for speech recognition)
        cmd = ["arecord", "-D", ALSA_DEVICE, "-f", "S16_LE",
               "-r", str(SAMPLE_RATE), "-c", "2", "-t", "raw"]

        while self.running:
            if self._paused:
                self._restart.wait(timeout=1)
                self._restart.clear()
                continue

            try:
                self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.det.is_listening = True
                logger.info(f"Mic active: {ALSA_DEVICE}")

                # Calibrate noise — REDUCED from 8 to 5 chunks (400ms → 250ms)
                noise_vals = []
                for _ in range(5):
                    d = self.proc.stdout.read(BYTES_PER_CHUNK)
                    if not d or len(d) < BYTES_PER_CHUNK: break
                    # Extract selected channel from stereo interleaved data
                    stereo = np.frombuffer(d, dtype=np.int16)
                    s = stereo[MIC_CHANNEL::2]  # configurable via VIRON_MIC_CHANNEL
                    noise_vals.append(np.sqrt(np.mean(s.astype(np.float32)**2)))

                nf = np.mean(noise_vals) if noise_vals else 50
                # Cap noise floor — if ambient is very high (TV, music), 
                # use fixed thresholds instead of adaptive
                if nf > 300:
                    logger.warning(f"High ambient noise ({nf:.0f})! Using capped thresholds. Turn off TV for best results.")
                    sp_thresh = min(nf * 1.5, 400)  # Cap speech threshold
                    si_thresh = min(nf * 1.1, 250)
                else:
                    # ADJUSTED: 1.8→1.5 for better sensitivity to short wake words
                    sp_thresh = max(nf * 1.5, 35)
                    si_thresh = max(nf * 1.2, 20)
                logger.info(f"Noise={nf:.0f} speech>{sp_thresh:.0f} silence<{si_thresh:.0f}")
                
                # Adaptive OWW threshold: lower in quiet rooms, raise in noisy ones
                oww_thresh = OWW_THRESHOLD
                if ADAPTIVE_ENABLED and self.det.oww_model:
                    if nf < 50:
                        oww_thresh = max(OWW_THRESHOLD - 0.06, ADAPTIVE_MIN)
                    elif nf > 300:
                        oww_thresh = min(OWW_THRESHOLD + 0.08, ADAPTIVE_MAX)
                    if oww_thresh != OWW_THRESHOLD:
                        logger.info(f"Adaptive OWW threshold: {OWW_THRESHOLD:.2f} → {oww_thresh:.2f} (noise={nf:.0f})")
                self._current_oww_thresh = oww_thresh
                
                logger.info("Listening... say 'Hey VIRON' or 'Hey Jarvis'")

                speech_frames = []
                in_speech = False
                sil_count = 0
                # Sliding buffer for custom mel classifier (~1.5s = 19 chunks at 80ms)
                custom_buf = deque(maxlen=19)
                custom_score_interval = 6  # score every ~480ms
                custom_chunk_count = 0
                # Pre-roll buffer: keeps last few chunks before speech starts
                preroll_buf = deque(maxlen=3)  # 240ms pre-roll

                while self.running and not self._paused:
                    d_stereo = self.proc.stdout.read(BYTES_PER_CHUNK)
                    if not d_stereo or len(d_stereo) < BYTES_PER_CHUNK: break
                    if self.det.is_paused: continue

                    # Flush echo chunks after resume
                    if self._flush_chunks > 0:
                        self._flush_chunks -= 1
                        continue

                    # Extract selected channel (configurable via VIRON_MIC_CHANNEL)
                    stereo = np.frombuffer(d_stereo, dtype=np.int16)
                    audio = stereo[MIC_CHANNEL::2]   # mono: selected channel
                    d = audio.tobytes()    # mono bytes for speech_frames / Whisper
                    rms = np.sqrt(np.mean(audio.astype(np.float32)**2))

                    # === openWakeWord (instant, every chunk) ===
                    if self.det.oww_model:
                        self.det.process_oww(audio, oww_thresh)

                    # === Custom mel classifier (sliding window, every ~480ms) ===
                    if self.det.custom_model:
                        custom_buf.append(audio)
                        custom_chunk_count += 1
                        if custom_chunk_count >= custom_score_interval and len(custom_buf) >= 12:
                            custom_chunk_count = 0
                            # Concatenate buffer into ~1.5s clip
                            combined = np.concatenate(list(custom_buf))
                            score = self.det.process_custom(combined)
                            if score > 0.15:
                                logger.info(f"Custom: score={score:.3f} (threshold={self.det.custom_threshold})")
                            if score >= self.det.custom_threshold:
                                self.det.set_detection(f"hey viron (custom:{score:.2f})", score)

                    # === Whisper path (speech segments) ===
                    if not in_speech:
                        preroll_buf.append(d)
                        if rms > sp_thresh:
                            in_speech = True
                            sil_count = 0
                            # Prepend pre-roll to capture start of "Hey"
                            speech_frames = list(preroll_buf)
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

                            dur_ms = len(speech_frames) * 80
                            logger.info(f"Speech segment: {dur_ms}ms ({len(speech_frames)} chunks)")

                            # Check peak RMS — skip if not significantly louder than noise
                            peak_rms = max(
                                np.sqrt(np.mean(np.frombuffer(f, dtype=np.int16).astype(np.float32)**2))
                                for f in speech_frames
                            )
                            if peak_rms < sp_thresh * 1.0:
                                logger.debug(f"Skipped weak segment (peak={peak_rms:.0f})")
                                in_speech = False; speech_frames = []; sil_count = 0
                                continue

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
        models = det.oww_names + ["whisper"]
        if det.custom_model:
            models = ["hey_viron_custom"] + models
        return jsonify({
            "ready": True, "listening": det.is_listening,
            "paused": det.is_paused, "models": models,
            "detections": det.detection_count,
            "mic_device": ALSA_DEVICE, "mic_channel": MIC_CHANNEL,
            "server_side_mic": True,
            "mode": "custom+whisper" if det.custom_model else ("hybrid" if det.oww_model else "whisper-only"),
            "adaptive": ADAPTIVE_ENABLED,
            "threshold": OWW_THRESHOLD,
            "custom_threshold": det.custom_threshold if det.custom_model else None,
        })

    return app


def main():
    det = Detector()
    has_oww = det.init_oww()
    has_custom = det.init_custom_model()

    mic = MicCapture(det)
    mic.start()

    app = create_app(det, mic)

    if has_custom:
        mode = "CUSTOM + Whisper"
    elif has_oww:
        mode = "HYBRID (OWW + Whisper)"
    else:
        mode = "WHISPER-ONLY"
    adaptive_tag = " [adaptive]" if ADAPTIVE_ENABLED else ""
    print(f"""
    ═══════════════════════════════════════
    VIRON Wake Word — {mode}
    Say "Hey VIRON" or "Hey Jarvis"
    Threshold: {OWW_THRESHOLD}{adaptive_tag}
    ═══════════════════════════════════════
    📡 http://0.0.0.0:{PORT}
    🎤 {ALSA_DEVICE} ch{MIC_CHANNEL} (ReSpeaker)
    ═══════════════════════════════════════
""")
    try:
        app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
    except KeyboardInterrupt: pass
    finally: mic.stop()


if __name__ == "__main__":
    main()
