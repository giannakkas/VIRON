#!/usr/bin/env python3
"""
VIRON Voice Pipeline v2 — Clean Architecture
=============================================
Single-file voice assistant pipeline:

  MIC (XVF3800) → Porcupine Wake → Silero VAD → Faster-Whisper (GPU) → LLM → Streaming TTS

Requirements:
  pip install pvporcupine faster-whisper torch torchaudio flask requests

Environment:
  PICOVOICE_ACCESS_KEY  — required for Porcupine (get free at picovoice.ai)
  OPENAI_API_KEY        — for cloud STT fallback and LLM
  VIRON_MIC_DEVICE      — ALSA device (default: plughw:2,0)
  VIRON_MIC_CHANNEL     — 0=beamformed, 1=ASR beam (default: 1)

Usage:
  source ~/VIRON/.env
  python3 voice_pipeline.py
"""

import os
import sys
import time
import wave
import json
import struct
import logging
import tempfile
import threading
import subprocess
import numpy as np
from pathlib import Path
from collections import deque
from flask import Flask, jsonify, request

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

ALSA_DEVICE = os.environ.get("VIRON_MIC_DEVICE", "plughw:2,0")
MIC_CHANNEL = int(os.environ.get("VIRON_MIC_CHANNEL", "1"))
SAMPLE_RATE = 16000
FRAME_LENGTH = 1280  # 80ms at 16kHz — matches openWakeWord expectations

WHISPER_MODEL = os.environ.get("VIRON_WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("VIRON_WHISPER_DEVICE", "auto")  # auto, cuda, cpu

GATEWAY_URL = os.environ.get("VIRON_GATEWAY_URL", "http://127.0.0.1:8080")
BACKEND_URL = os.environ.get("VIRON_BACKEND_URL", "http://127.0.0.1:5000")
TTS_URL = os.environ.get("VIRON_TTS_URL", "http://127.0.0.1:5000/api/tts")

ECHO_COOLDOWN = float(os.environ.get("VIRON_ECHO_COOLDOWN", "2.0"))
MAX_SPEECH_SEC = float(os.environ.get("VIRON_MAX_SPEECH", "15.0"))
SILENCE_TIMEOUT = float(os.environ.get("VIRON_SILENCE_TIMEOUT", "1.0"))
NO_SPEECH_TIMEOUT = float(os.environ.get("VIRON_NO_SPEECH_TIMEOUT", "4.0"))

PORT = int(os.environ.get("VIRON_PIPELINE_PORT", "8085"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("viron-pipeline")

# ═══════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════

class PipelineState:
    def __init__(self):
        self.is_speaking = False        # TTS playing
        self.is_listening = False       # recording user command
        self.is_processing = False      # LLM thinking
        self.is_paused = False          # externally paused
        self.wake_detected = False
        self.last_tts_end = 0           # time TTS finished (for echo cooldown)
        self.last_wake = 0
        self.detection_count = 0
        self._pending_wake = None
        self._lock = threading.Lock()
    
    def set_wake(self, source="porcupine", score=1.0):
        now = time.time()
        # Suppress during TTS + cooldown
        if self.is_speaking: return False
        if now - self.last_tts_end < ECHO_COOLDOWN: return False
        # Suppress rapid re-triggers
        if now - self.last_wake < 5.0: return False
        
        self.last_wake = now
        self.detection_count += 1
        self.wake_detected = True
        with self._lock:
            self._pending_wake = {
                "wake": True, "model": source,
                "score": score, "time": now,
                "count": self.detection_count,
            }
        log.info(f"🔔 WAKE WORD detected ({source}, score={score:.2f})")
        return True
    
    def consume_wake(self):
        with self._lock:
            d = self._pending_wake
            self._pending_wake = None
            return d
    
    def tts_start(self):
        self.is_speaking = True
    
    def tts_end(self):
        self.is_speaking = False
        self.last_tts_end = time.time()

state = PipelineState()

# ═══════════════════════════════════════════════════════════
# 1. WAKE WORD (openWakeWord — works on ARM64/Jetson)
# ═══════════════════════════════════════════════════════════

oww_model = None
oww_names = []
OWW_THRESHOLD = float(os.environ.get("VIRON_WAKEWORD_THRESHOLD", "0.3"))

def init_wake():
    global oww_model, oww_names
    try:
        import openwakeword
        from openwakeword.model import Model
        openwakeword.utils.download_models()
        
        # Use hey_jarvis built-in (closest to "Hey VIRON")
        oww_model = Model(wakeword_models=["hey_jarvis_v0.1"], vad_threshold=0.5)
        oww_names = list(oww_model.models.keys())
        log.info(f"✅ openWakeWord ready: {oww_names}, threshold={OWW_THRESHOLD}")
        return True
    except Exception as e:
        log.error(f"❌ openWakeWord failed: {e}")
        return False


def check_wake(audio_int16):
    """Check audio frame for wake word. Returns True if detected."""
    if oww_model is None:
        return False
    try:
        preds = oww_model.predict(audio_int16)
        for name in oww_names:
            score = preds[name]
            if score > 0.08:
                log.debug(f"OWW: {name}={score:.3f}")
            if score >= OWW_THRESHOLD:
                log.info(f"🔔 OWW: {name}={score:.3f} (threshold={OWW_THRESHOLD})")
                return True
        return False
    except Exception:
        return False

# ═══════════════════════════════════════════════════════════
# 2. SILERO VAD
# ═══════════════════════════════════════════════════════════

silero_vad = None
silero_get_speech_ts = None

def init_silero_vad():
    global silero_vad
    try:
        import torch
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=True,
        )
        silero_vad = model
        log.info("✅ Silero VAD ready (ONNX)")
        return True
    except Exception as e:
        log.warning(f"⚠ Silero VAD not available: {e}")
        log.warning("  Falling back to RMS-based VAD")
        return False


def vad_is_speech(audio_int16, threshold=0.5):
    """Check if audio chunk contains speech using Silero VAD."""
    if silero_vad is None:
        # Fallback: simple RMS threshold
        rms = np.sqrt(np.mean(audio_int16.astype(np.float32) ** 2))
        return rms > 150  # rough threshold
    
    try:
        import torch
        audio_float = torch.FloatTensor(audio_int16.astype(np.float32) / 32768.0)
        # Silero expects 512 samples at 16kHz
        if len(audio_float) < 512:
            audio_float = torch.nn.functional.pad(audio_float, (0, 512 - len(audio_float)))
        prob = silero_vad(audio_float[:512], SAMPLE_RATE).item()
        return prob > threshold
    except Exception:
        rms = np.sqrt(np.mean(audio_int16.astype(np.float32) ** 2))
        return rms > 150

# ═══════════════════════════════════════════════════════════
# 3. FASTER-WHISPER (GPU/CPU)
# ═══════════════════════════════════════════════════════════

whisper_model = None

def init_whisper():
    global whisper_model
    try:
        from faster_whisper import WhisperModel
        
        # Auto-detect CUDA
        device = WHISPER_DEVICE
        compute_type = "float16"
        
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                    log.info(f"  CUDA available: {torch.cuda.get_device_name(0)}")
                else:
                    device = "cpu"
                    compute_type = "int8"
            except ImportError:
                device = "cpu"
                compute_type = "int8"
        
        if device == "cpu":
            compute_type = "int8"
        
        log.info(f"Loading Whisper '{WHISPER_MODEL}' on {device} ({compute_type})...")
        whisper_model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)
        log.info(f"✅ Whisper ready: {WHISPER_MODEL} on {device}")
        return True
    except Exception as e:
        log.warning(f"⚠ Local Whisper not available: {e}")
        log.warning("  Will use OpenAI cloud STT")
        return False


def transcribe(audio_int16, lang=None):
    """Transcribe audio using local Whisper or OpenAI cloud fallback."""
    # Try local Whisper first
    if whisper_model is not None:
        try:
            audio_float = audio_int16.astype(np.float32) / 32768.0
            segments, info = whisper_model.transcribe(
                audio_float,
                language=lang,
                beam_size=3,
                best_of=3,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300),
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            detected_lang = info.language
            log.info(f"📝 Local Whisper: \"{text[:80]}\" (lang={detected_lang})")
            return text, detected_lang
        except Exception as e:
            log.warning(f"Local Whisper error: {e}, trying cloud...")
    
    # Cloud fallback: OpenAI Whisper API
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        log.error("❌ No STT available (no local Whisper, no OpenAI key)")
        return "", ""
    
    try:
        import requests
        # Save as WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
            with wave.open(tmp, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_int16.tobytes())
        
        with open(wav_path, 'rb') as f:
            resp = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {openai_key}"},
                files={"file": ("speech.wav", f, "audio/wav")},
                data={
                    "model": "whisper-1",
                    **({"language": lang} if lang else {}),
                    "temperature": 0.0,
                },
                timeout=15,
            )
        os.unlink(wav_path)
        
        if resp.status_code == 200:
            text = resp.json().get("text", "").strip()
            log.info(f"☁️ Cloud Whisper: \"{text[:80]}\"")
            return text, lang or "auto"
        else:
            log.error(f"Cloud Whisper error {resp.status_code}: {resp.text[:100]}")
            return "", ""
    except Exception as e:
        log.error(f"Cloud STT failed: {e}")
        return "", ""

# ═══════════════════════════════════════════════════════════
# 4. LLM (via Gateway)
# ═══════════════════════════════════════════════════════════

def chat(user_message, system="You are VIRON, a helpful AI companion. Reply concisely.", lang="en"):
    """Send message to LLM via gateway. Returns reply text."""
    try:
        import requests
        
        if lang in ("el", "el-GR"):
            system += " Απάντα στα Ελληνικά."
        
        resp = requests.post(
            f"{GATEWAY_URL}/v1/chat",
            json={
                "messages": [{"role": "user", "content": user_message}],
                "system": system,
            },
            timeout=30,
        )
        
        if resp.status_code == 200:
            data = resp.json()
            reply = data.get("reply", data.get("text", ""))
            provider = data.get("provider", "?")
            log.info(f"💬 LLM ({provider}): \"{reply[:80]}\"")
            return reply
        else:
            log.error(f"LLM error {resp.status_code}: {resp.text[:100]}")
            return "Συγγνώμη, δεν μπορώ να απαντήσω τώρα." if lang == "el" else "Sorry, I can't respond right now."
    except Exception as e:
        log.error(f"LLM failed: {e}")
        return "Sorry, I can't respond right now."

# ═══════════════════════════════════════════════════════════
# 5. TTS
# ═══════════════════════════════════════════════════════════

def speak(text, lang="el"):
    """Send text to TTS and play audio. Blocks until playback complete."""
    if not text:
        return
    
    state.tts_start()
    try:
        import requests
        
        # Split into sentences for streaming feel
        import re
        sentences = re.split(r'(?<=[.!;?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        for i, sentence in enumerate(sentences):
            if not state.is_speaking:  # interrupted
                break
            
            try:
                resp = requests.post(
                    TTS_URL,
                    json={"text": sentence, "lang": lang, "speed": "normal"},
                    timeout=15,
                )
                
                if resp.status_code == 200 and len(resp.content) > 1000:
                    # Save to temp file and play
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                        tmp.write(resp.content)
                        tmp_path = tmp.name
                    
                    # Play with ffplay (non-blocking detection of end)
                    proc = subprocess.run(
                        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path],
                        timeout=30,
                    )
                    os.unlink(tmp_path)
                else:
                    log.warning(f"TTS failed for sentence {i}: status={resp.status_code}")
            except Exception as e:
                log.warning(f"TTS/playback error: {e}")
    finally:
        state.tts_end()
        log.info("🔇 TTS playback complete")

# ═══════════════════════════════════════════════════════════
# MIC CAPTURE + MAIN LOOP
# ═══════════════════════════════════════════════════════════

class MicStream:
    """Streams audio from ALSA device, extracts single channel."""
    
    def __init__(self):
        self.proc = None
        self.running = False
    
    def start(self):
        cmd = ["arecord", "-D", ALSA_DEVICE, "-f", "S16_LE",
               "-r", str(SAMPLE_RATE), "-c", "2", "-t", "raw"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.running = True
        log.info(f"🎤 Mic started: {ALSA_DEVICE} ch{MIC_CHANNEL}")
    
    def read_frame(self, frame_length=FRAME_LENGTH):
        """Read one frame of mono audio. Returns int16 numpy array."""
        if not self.proc:
            return None
        bytes_needed = frame_length * 4  # stereo int16 = 4 bytes per sample
        raw = self.proc.stdout.read(bytes_needed)
        if len(raw) < bytes_needed:
            return None
        stereo = np.frombuffer(raw, dtype=np.int16)
        mono = stereo[MIC_CHANNEL::2]
        return mono
    
    def read_seconds(self, seconds):
        """Read N seconds of mono audio."""
        total_samples = int(SAMPLE_RATE * seconds)
        frames = []
        while len(frames) * FRAME_LENGTH < total_samples:
            frame = self.read_frame()
            if frame is None:
                break
            frames.append(frame)
        return np.concatenate(frames) if frames else np.array([], dtype=np.int16)
    
    def stop(self):
        self.running = False
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except:
                self.proc.kill()
            self.proc = None


def record_command(mic, timeout=MAX_SPEECH_SEC):
    """Record user's command using Silero VAD for endpoint detection.
    
    Returns: numpy int16 array of speech audio
    """
    log.info("👂 Listening for command...")
    state.is_listening = True
    
    speech_frames = []
    silence_frames = 0
    speech_started = False
    no_speech_frames = 0
    
    silence_needed = int(SILENCE_TIMEOUT * SAMPLE_RATE / FRAME_LENGTH)
    no_speech_needed = int(NO_SPEECH_TIMEOUT * SAMPLE_RATE / FRAME_LENGTH)
    max_frames = int(timeout * SAMPLE_RATE / FRAME_LENGTH)
    
    # Pre-roll buffer (capture audio before speech starts)
    preroll = deque(maxlen=6)  # ~192ms
    
    try:
        for _ in range(max_frames):
            frame = mic.read_frame()
            if frame is None:
                break
            
            is_speech = vad_is_speech(frame)
            
            if not speech_started:
                preroll.append(frame)
                if is_speech:
                    speech_started = True
                    speech_frames = list(preroll)  # include pre-roll
                    silence_frames = 0
                    log.info("  🗣️ Speech started")
                else:
                    no_speech_frames += 1
                    if no_speech_frames >= no_speech_needed:
                        log.info("  🔇 No speech detected")
                        return np.array([], dtype=np.int16)
            else:
                speech_frames.append(frame)
                if is_speech:
                    silence_frames = 0
                else:
                    silence_frames += 1
                    if silence_frames >= silence_needed:
                        log.info(f"  ✅ Speech ended ({len(speech_frames) * FRAME_LENGTH / SAMPLE_RATE:.1f}s)")
                        break
        
        if speech_frames:
            return np.concatenate(speech_frames)
        return np.array([], dtype=np.int16)
    finally:
        state.is_listening = False


def detect_language(text):
    """Simple Greek vs English detection."""
    greek_chars = sum(1 for c in text if '\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF')
    return "el" if greek_chars > len(text) * 0.3 else "en"


def conversation_turn(mic, text, lang):
    """Handle one conversation turn: user text → LLM → TTS."""
    state.is_processing = True
    try:
        reply = chat(text, lang=lang)
        if reply:
            state.is_processing = False
            speak(reply, lang=lang)
    finally:
        state.is_processing = False


def main_loop(mic):
    """Main voice assistant loop — openWakeWord + Silero VAD + Whisper."""
    if oww_model is None:
        log.error("❌ Wake word engine not available!")
        sys.exit(1)
    
    log.info("=" * 50)
    log.info("🤖 VIRON Voice Pipeline Active")
    log.info(f"   Wake: openWakeWord (threshold={OWW_THRESHOLD})")
    log.info(f"   VAD: {'Silero' if silero_vad else 'RMS'}")
    log.info(f"   STT: {'Local Whisper ('+WHISPER_DEVICE+')' if whisper_model else 'Cloud'}")
    log.info(f"   Mic: {ALSA_DEVICE} ch{MIC_CHANNEL}")
    log.info("=" * 50)
    log.info("🎤 Say 'Hey VIRON'...")
    
    while True:
        try:
            # Skip if speaking or in cooldown
            if state.is_speaking or state.is_processing or state.is_paused:
                time.sleep(0.1)
                continue
            
            if time.time() - state.last_tts_end < ECHO_COOLDOWN:
                mic.read_frame()  # drain echo audio
                continue
            
            frame = mic.read_frame()
            if frame is None:
                log.error("Mic read failed, restarting...")
                mic.stop()
                time.sleep(1)
                mic.start()
                continue
            
            # === openWakeWord detection ===
            if check_wake(frame):
                if state.set_wake("openwakeword", 1.0):
                    log.info("🎯 Wake word detected!")
                    
                    # Reset OWW model state
                    try:
                        oww_model.reset()
                    except:
                        pass
                    
                    # Record command with Silero VAD
                    audio = record_command(mic)
                    if len(audio) > SAMPLE_RATE * 0.3:
                        text, lang = transcribe(audio)
                        if text and len(text) > 1:
                            lang = detect_language(text)
                            log.info(f"📝 Command: \"{text}\" (lang={lang})")
                            conversation_turn(mic, text, lang)
                        else:
                            log.info("  (empty transcription)")
                    else:
                        log.info("  (no speech in command)")
                    
                    # Reset OWW after conversation
                    try:
                        oww_model.reset()
                    except:
                        pass
                    log.info("🎤 Listening for wake word...")
        
        except KeyboardInterrupt:
            log.info("Shutting down...")
            break
        except Exception as e:
            log.error(f"Pipeline error: {e}")
            time.sleep(1)


import re
WAKE_PATTERNS = re.compile(
    r'hey\s*v[iy]r[oa]n|hey\s*jar[vu]is|hey\s*byr[oa]n|hey\s*ver[oa]n?'
    r'|hey\s*iron|hey\s*brian|hey\s*bar[oa]n|hey\s*v[iy]ro$',
    re.IGNORECASE
)

def _is_wake_word(text):
    t = text.lower().strip()
    if WAKE_PATTERNS.search(t):
        return True
    for w in ["viron", "jarvis", "byron", "veron", "biron", "viro", "baron", "brian"]:
        if w in t:
            return True
    return False

# ═══════════════════════════════════════════════════════════
# HTTP API (for browser frontend)
# ═══════════════════════════════════════════════════════════

app = Flask(__name__)

@app.route("/wakeword/status", methods=["GET"])
def ww_status():
    return jsonify({
        "ready": oww_model is not None,
        "listening": not state.is_speaking and not state.is_paused,
        "paused": state.is_paused or state.is_speaking,
        "models": oww_names or ["none"],
        "detections": state.detection_count,
        "mode": "openwakeword",
        "mic_device": ALSA_DEVICE,
        "mic_channel": MIC_CHANNEL,
        "server_side_mic": True,
        "pipeline_version": 2,
    })

@app.route("/wakeword/poll", methods=["GET"])
def ww_poll():
    d = state.consume_wake()
    if d:
        return jsonify(d)
    return jsonify({"wake": False})

@app.route("/wakeword/pause", methods=["POST"])
def ww_pause():
    state.is_paused = True
    return jsonify({"status": "paused"})

@app.route("/wakeword/resume", methods=["POST"])
def ww_resume():
    state.is_paused = False
    state.last_tts_end = time.time()  # reset echo cooldown
    if silero_vad:
        try:
            silero_vad.reset_states()
        except:
            pass
    return jsonify({"status": "resumed"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "pipeline_version": 2,
        "wake_engine": "openwakeword" if oww_model else "none",
        "vad_engine": "silero" if silero_vad else "rms",
        "stt_engine": f"faster-whisper-{WHISPER_DEVICE}" if whisper_model else "cloud",
        "speaking": state.is_speaking,
        "listening": state.is_listening,
        "processing": state.is_processing,
    })


def run_http_server():
    """Run Flask API in background thread."""
    app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print()
    print("═" * 50)
    print("  🤖 VIRON Voice Pipeline v2")
    print("═" * 50)
    
    # Init components
    print("\n📦 Initializing components...")
    has_wake = init_wake()
    has_vad = init_silero_vad()
    has_whisper = init_whisper()
    
    print()
    print(f"  Wake:    {'✅ openWakeWord' if has_wake else '❌ NOT AVAILABLE'}")
    print(f"  VAD:     {'✅ Silero' if has_vad else '⚠ RMS fallback'}")
    print(f"  STT:     {'✅ Local Whisper (' + WHISPER_DEVICE + ')' if has_whisper else '⚠ Cloud only'}")
    print(f"  Gateway: {GATEWAY_URL}")
    print(f"  TTS:     {TTS_URL}")
    print(f"  Mic:     {ALSA_DEVICE} ch{MIC_CHANNEL}")
    print()
    
    # Start HTTP API
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()
    log.info(f"📡 HTTP API on port {PORT}")
    
    # Start mic
    mic = MicStream()
    mic.start()
    time.sleep(0.5)
    
    # Calibrate noise
    log.info("🔇 Calibrating noise level (1s)...")
    noise_audio = mic.read_seconds(1.0)
    if len(noise_audio) > 0:
        noise_rms = np.sqrt(np.mean(noise_audio.astype(np.float32) ** 2))
        log.info(f"  Noise floor: RMS={noise_rms:.0f}")
    
    try:
        main_loop(mic)
    finally:
        mic.stop()
        log.info("Pipeline stopped.")


if __name__ == "__main__":
    main()
