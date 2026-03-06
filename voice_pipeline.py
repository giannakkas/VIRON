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
# LOAD .env FILE DIRECTLY (no python-dotenv needed)
# ═══════════════════════════════════════════════════════════

def _load_env():
    for p in [Path.home() / "VIRON" / ".env", Path(__file__).parent / ".env"]:
        if p.exists():
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, val = line.partition("=")
                        key = key.strip()
                        val = val.strip().strip("'\"")
                        # Always override - env may have stale/empty values
                        os.environ[key] = val
            print(f"  ✅ Loaded env from {p}")
            return
    print("  ⚠ No .env file found!")
_load_env()

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

ALSA_DEVICE = os.environ.get("VIRON_MIC_DEVICE", "plughw:2,0")
MIC_CHANNEL = int(os.environ.get("VIRON_MIC_CHANNEL", "1"))
SAMPLE_RATE = 16000
FRAME_LENGTH = 512  # Porcupine requires 512 samples at 16kHz (32ms)

WHISPER_MODEL = os.environ.get("VIRON_WHISPER_MODEL", "small")

GATEWAY_URL = os.environ.get("VIRON_GATEWAY_URL", "http://127.0.0.1:8080")
BACKEND_URL = os.environ.get("VIRON_BACKEND_URL", "http://127.0.0.1:5000")
TTS_URL = os.environ.get("VIRON_TTS_URL", "http://127.0.0.1:5000/api/tts")

ECHO_COOLDOWN = float(os.environ.get("VIRON_ECHO_COOLDOWN", "2.0"))
MAX_SPEECH_SEC = float(os.environ.get("VIRON_MAX_SPEECH", "15.0"))
SILENCE_TIMEOUT = float(os.environ.get("VIRON_SILENCE_TIMEOUT", "1.0"))
NO_SPEECH_TIMEOUT = float(os.environ.get("VIRON_NO_SPEECH_TIMEOUT", "6.0"))

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
# 1. PORCUPINE WAKE WORD
# ═══════════════════════════════════════════════════════════

porcupine = None
PICOVOICE_KEY = os.environ.get("PICOVOICE_ACCESS_KEY", "")
PORCUPINE_KEYWORD = os.environ.get("VIRON_WAKE_KEYWORD", "jarvis")
PORCUPINE_CUSTOM_PATH = os.environ.get("VIRON_WAKE_MODEL", "")
PORCUPINE_SENSITIVITY = float(os.environ.get("VIRON_WAKE_SENSITIVITY", "0.6"))

def init_wake():
    global porcupine
    if not PICOVOICE_KEY:
        log.error("❌ PICOVOICE_ACCESS_KEY not set!")
        log.error("   Check ~/VIRON/.env has: PICOVOICE_ACCESS_KEY=your_key")
        log.error("   Current env value: '%s'" % os.environ.get("PICOVOICE_ACCESS_KEY", "EMPTY"))
        # List .env contents (redacted)
        env_path = Path.home() / "VIRON" / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if "KEY" in line and "=" in line:
                        k, _, v = line.strip().partition("=")
                        log.error(f"   .env has: {k}={v[:8]}...")
        return False
    
    try:
        import pvporcupine
        
        # Try custom .ppn first, fallback to built-in keyword
        if PORCUPINE_CUSTOM_PATH and os.path.exists(PORCUPINE_CUSTOM_PATH):
            log.info(f"Loading custom wake word: {PORCUPINE_CUSTOM_PATH}")
            porcupine = pvporcupine.create(
                access_key=PICOVOICE_KEY,
                keyword_paths=[PORCUPINE_CUSTOM_PATH],
                sensitivities=[PORCUPINE_SENSITIVITY],
            )
            log.info(f"✅ Porcupine ready: custom model, sensitivity={PORCUPINE_SENSITIVITY}")
        else:
            log.info(f"Using built-in keyword: '{PORCUPINE_KEYWORD}'")
            porcupine = pvporcupine.create(
                access_key=PICOVOICE_KEY,
                keywords=[PORCUPINE_KEYWORD],
                sensitivities=[PORCUPINE_SENSITIVITY],
            )
            log.info(f"✅ Porcupine ready: '{PORCUPINE_KEYWORD}', sensitivity={PORCUPINE_SENSITIVITY}")
        
        return True
    except Exception as e:
        log.error(f"❌ Porcupine init failed: {e}")
        return False


def check_wake(audio_int16):
    """Check audio frame for wake word. Returns True if detected."""
    if porcupine is None:
        return False
    try:
        result = porcupine.process(audio_int16)
        return result >= 0
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
# 3. STT: Deepgram streaming (~300ms) > whisper.cpp GPU > local CPU
# ═══════════════════════════════════════════════════════════

whisper_model = None
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
WHISPER_CPP_BIN = os.environ.get("VIRON_WHISPER_CPP", os.path.expanduser("~/whisper.cpp/build/bin/whisper-cli"))
WHISPER_CPP_MODEL = os.environ.get("VIRON_WHISPER_MODEL_PATH", os.path.expanduser("~/whisper.cpp/models/ggml-small.bin"))
WHISPER_CPP_AVAILABLE = os.path.exists(WHISPER_CPP_BIN) and os.path.exists(WHISPER_CPP_MODEL)

def init_whisper():
    global whisper_model, WHISPER_CPP_AVAILABLE
    
    if DEEPGRAM_API_KEY:
        log.info(f"✅ Deepgram STT ready (streaming, ~300ms)")
    
    if WHISPER_CPP_AVAILABLE:
        log.info(f"✅ whisper.cpp GPU ready (offline fallback)")
    
    # Also try loading faster-whisper as last resort
    try:
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        log.info(f"✅ faster-whisper CPU ready (offline fallback)")
    except:
        pass
    
    return bool(DEEPGRAM_API_KEY) or WHISPER_CPP_AVAILABLE or whisper_model is not None


def _transcribe_deepgram(wav_path, lang="el"):
    """Transcribe using Deepgram API. ~300ms, best accuracy for Greek."""
    try:
        import requests
        t0 = time.time()
        
        with open(wav_path, 'rb') as f:
            audio_data = f.read()
        
        resp = requests.post(
            "https://api.deepgram.com/v1/listen",
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "audio/wav",
            },
            params={
                "model": "nova-2",
                "language": lang or "el",
                "smart_format": "true",
                "punctuate": "true",
            },
            data=audio_data,
            timeout=10,
        )
        
        ms = int((time.time() - t0) * 1000)
        
        if resp.status_code == 200:
            result = resp.json()
            text = result.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "").strip()
            confidence = result.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("confidence", 0)
            log.info(f"⚡ Deepgram ({ms}ms, conf={confidence:.2f}): \"{text[:80]}\"")
            return text, lang
        else:
            log.warning(f"Deepgram error {resp.status_code}: {resp.text[:100]}")
            return None, None
    except Exception as e:
        log.warning(f"Deepgram failed: {e}")
        return None, None


def _transcribe_whisper_cpp(wav_path, lang=None):
    """Transcribe using whisper.cpp with GPU. ~300-500ms offline."""
    cmd = [
        WHISPER_CPP_BIN, "-m", WHISPER_CPP_MODEL,
        "-f", wav_path, "--no-prints", "-t", "4", "--no-timestamps", "-otxt",
    ]
    if lang:
        lang_map = {"el": "el", "en": "en", "el-GR": "el", "en-US": "en"}
        cmd.extend(["-l", lang_map.get(lang, lang)])
    
    try:
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, timeout=15, text=True)
        ms = int((time.time() - t0) * 1000)
        
        if result.returncode == 0:
            text = result.stdout.strip()
            if not text:
                txt_path = wav_path + ".txt"
                if os.path.exists(txt_path):
                    text = open(txt_path).read().strip()
                    os.unlink(txt_path)
            log.info(f"🖥️ whisper.cpp ({ms}ms): \"{text[:80]}\"")
            return text, lang or "auto"
        return None, None
    except Exception as e:
        log.warning(f"whisper.cpp failed: {e}")
        return None, None


def _check_internet():
    """Quick internet check (~100ms)."""
    try:
        import requests
        requests.head("https://api.deepgram.com", timeout=1)
        return True
    except:
        return False


def transcribe(audio_int16, lang="el"):
    """Transcribe: Deepgram (~300ms) > whisper.cpp GPU (~500ms) > local CPU (~3s)."""
    
    # Save audio to WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
        with wave.open(tmp, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
    
    try:
        # 1. Deepgram (~300ms, best accuracy for Greek)
        if DEEPGRAM_API_KEY:
            text, detected = _transcribe_deepgram(wav_path, lang)
            if text:
                return text, detected
        
        # 2. whisper.cpp GPU (~500ms, offline)
        if WHISPER_CPP_AVAILABLE:
            text, detected = _transcribe_whisper_cpp(wav_path, lang)
            if text:
                return text, detected
        
        # 3. Cloud Whisper fallback (~1s)
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if openai_key:
            try:
                import requests
                t0 = time.time()
                with open(wav_path, 'rb') as f:
                    resp = requests.post(
                        "https://api.openai.com/v1/audio/transcriptions",
                        headers={"Authorization": f"Bearer {openai_key}"},
                        files={"file": ("speech.wav", f, "audio/wav")},
                        data={"model": "whisper-1", "temperature": 0.0,
                              **({"language": lang} if lang else {})},
                        timeout=15,
                    )
                ms = int((time.time() - t0) * 1000)
                if resp.status_code == 200:
                    text = resp.json().get("text", "").strip()
                    log.info(f"☁️ OpenAI Whisper ({ms}ms): \"{text[:80]}\"")
                    return text, lang or "auto"
            except:
                pass
        
        # 4. Local faster-whisper CPU (~3s, always works)
        if whisper_model is not None:
            try:
                audio_float = audio_int16.astype(np.float32) / 32768.0
                t0 = time.time()
                segments, info = whisper_model.transcribe(
                    audio_float, language=lang, beam_size=3, best_of=3,
                    vad_filter=True, vad_parameters=dict(min_silence_duration_ms=300),
                )
                text = " ".join(s.text.strip() for s in segments).strip()
                ms = int((time.time() - t0) * 1000)
                log.info(f"🐌 Local CPU ({ms}ms): \"{text[:80]}\"")
                return text, info.language
            except Exception as e:
                log.error(f"Local Whisper error: {e}")
        
        log.error("❌ No STT available")
        return "", ""
    finally:
        try: os.unlink(wav_path)
        except: pass

# ═══════════════════════════════════════════════════════════
# 4. LLM (via Gateway)
# ═══════════════════════════════════════════════════════════

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# Keywords that trigger Claude (tutoring, complex reasoning)
CLAUDE_TRIGGERS = [
    # Greek tutoring keywords
    "εξήγησέ", "εξήγησε", "μάθε", "δίδαξε", "πώς", "γιατί", "τι είναι",
    "βοήθησε με", "homework", "μάθημα", "σχολείο", "άσκηση", "εργασία",
    "υπολόγισε", "λύσε", "ανάλυσε", "σύγκρινε", "περίγραψε",
    # English tutoring keywords  
    "explain", "teach", "help me understand", "how does", "why does",
    "what is", "homework", "lesson", "exercise", "calculate", "solve",
    "analyze", "compare", "describe", "write an essay", "summarize",
    # Complex tasks
    "code", "program", "κώδικα", "πρόγραμμα", "debug",
    "story", "ιστορία", "poem", "ποίημα",
]


def _needs_claude(text):
    """Determine if the query needs Claude (complex) or Groq (quick)."""
    t = text.lower()
    # Short greetings/simple questions → Groq
    if len(t) < 20:
        return False
    # Check for tutoring/complex keywords
    for trigger in CLAUDE_TRIGGERS:
        if trigger in t:
            return True
    # Long questions are more likely complex
    if len(t) > 100:
        return True
    return False


def chat(user_message, system="You are VIRON, a helpful AI companion.", lang="en"):
    """Smart routing: Groq for quick answers, Claude for tutoring/complex."""
    try:
        import requests
        
        system += " Απάντα πάντα στα Ελληνικά. Είσαι ο ΒΙΡΟΝ, ένας φιλικός βοηθός. Απάντα σύντομα."
        lang = "el"
        
        use_claude = _needs_claude(user_message)
        
        if use_claude:
            log.info(f"🧠 Routing to Claude (complex/tutoring)")
        else:
            log.info(f"⚡ Routing to Groq (quick answer)")
        
        # 1. Claude for complex/tutoring
        if use_claude and ANTHROPIC_API_KEY:
            try:
                t0 = time.time()
                resp = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": ANTHROPIC_API_KEY,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": ANTHROPIC_MODEL,
                        "max_tokens": 500,
                        "system": system,
                        "messages": [{"role": "user", "content": user_message}],
                    },
                    timeout=30,
                )
                ms = int((time.time() - t0) * 1000)
                if resp.status_code == 200:
                    reply = resp.json()["content"][0]["text"].strip()
                    log.info(f"🧠 Claude ({ms}ms): \"{reply[:80]}\"")
                    return reply
                else:
                    log.warning(f"Claude error {resp.status_code}: {resp.text[:100]}")
            except Exception as e:
                log.warning(f"Claude failed: {e}")
        
        # 2. Groq for quick answers (~200ms)
        if GROQ_API_KEY:
            try:
                t0 = time.time()
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": GROQ_MODEL,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user_message},
                        ],
                        "max_tokens": 200,
                        "temperature": 0.7,
                    },
                    timeout=10,
                )
                ms = int((time.time() - t0) * 1000)
                if resp.status_code == 200:
                    reply = resp.json()["choices"][0]["message"]["content"].strip()
                    log.info(f"⚡ Groq ({ms}ms): \"{reply[:80]}\"")
                    return reply
                else:
                    log.warning(f"Groq error {resp.status_code}")
            except Exception as e:
                log.warning(f"Groq failed: {e}")
        
        # 3. Gateway fallback
        try:
            resp = requests.post(
                f"{GATEWAY_URL}/v1/chat",
                json={
                    "message": user_message,
                    "messages": [{"role": "user", "content": user_message}],
                    "system": system,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("reply", data.get("text", ""))
        except:
            pass
        
        # 4. Local Gemma 2B (offline)
        ROUTER_URL = os.environ.get("ROUTER_URL", "http://127.0.0.1:8081")
        try:
            resp = requests.post(
                f"{ROUTER_URL}/v1/completions",
                json={"prompt": f"System: {system}\nUser: {user_message}\nAssistant:",
                      "max_tokens": 200, "temperature": 0.7},
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json().get("choices", [{}])[0].get("text", "").strip()
        except:
            pass
        
        return "Συγγνώμη, δεν μπορώ να απαντήσω."
    except Exception as e:
        log.error(f"LLM failed: {e}")
        return "Συγγνώμη."

# ═══════════════════════════════════════════════════════════
# 5. TTS — Queue response for browser playback
# ═══════════════════════════════════════════════════════════

# Response queue: pipeline stores LLM replies here, browser picks them up
_response_queue = []
_response_lock = threading.Lock()

def speak(text, lang="el"):
    """Queue response for browser TTS playback."""
    if not text:
        return
    state.tts_start()
    
    with _response_lock:
        _response_queue.append({"text": text, "lang": lang, "time": time.time()})
    log.info(f"📤 Response: \"{text[:60]}\"")
    
    # Wait for browser to play
    wait_time = min(len(text) * 0.06, 25)
    time.sleep(wait_time)
    state.tts_end()

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
    """Record user's command using Silero VAD for endpoint detection."""
    log.info("👂 Listening for command...")
    state.is_listening = True
    
    speech_frames = []
    silence_frames = 0
    speech_started = False
    no_speech_frames = 0
    
    silence_needed = int(SILENCE_TIMEOUT * SAMPLE_RATE / FRAME_LENGTH)
    no_speech_needed = int(NO_SPEECH_TIMEOUT * SAMPLE_RATE / FRAME_LENGTH)
    max_frames = int(timeout * SAMPLE_RATE / FRAME_LENGTH)
    
    preroll = deque(maxlen=6)
    frame_count = 0
    
    try:
        for _ in range(max_frames):
            frame = mic.read_frame()
            if frame is None:
                log.error("  Mic read returned None!")
                break
            
            frame_count += 1
            rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
            
            # Log every 30 frames (~1s) to show mic is alive
            if frame_count % 30 == 0:
                log.info(f"  ... listening (frame {frame_count}, RMS={rms:.0f}, speech_started={speech_started})")
            
            is_speech = vad_is_speech(frame)
            
            if not speech_started:
                preroll.append(frame)
                if is_speech:
                    speech_started = True
                    speech_frames = list(preroll)
                    silence_frames = 0
                    log.info(f"  🗣️ Speech started (RMS={rms:.0f})")
                else:
                    no_speech_frames += 1
                    if no_speech_frames >= no_speech_needed:
                        log.info(f"  🔇 No speech detected after {no_speech_frames} frames ({NO_SPEECH_TIMEOUT}s)")
                        return np.array([], dtype=np.int16)
            else:
                speech_frames.append(frame)
                if is_speech:
                    silence_frames = 0
                else:
                    silence_frames += 1
                    if silence_frames >= silence_needed:
                        dur = len(speech_frames) * FRAME_LENGTH / SAMPLE_RATE
                        log.info(f"  ✅ Speech ended ({dur:.1f}s)")
                        break
        
        if speech_frames:
            audio = np.concatenate(speech_frames)
            log.info(f"  Captured {len(audio)/SAMPLE_RATE:.1f}s of audio")
            return audio
        return np.array([], dtype=np.int16)
    finally:
        state.is_listening = False


def detect_language(text):
    """Simple Greek vs English detection."""
    greek_chars = sum(1 for c in text if '\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF')
    return "el" if greek_chars > len(text) * 0.3 else "en"


def conversation_turn(mic, text, lang):
    """Handle one conversation turn with smart routing."""
    state.is_processing = True
    try:
        # Check internet first
        if not _check_internet():
            log.warning("⚠ No internet connection!")
            speak("Σε παρακαλώ σύνδεσέ με με το ίντερνετ για να σε βοηθήσω.", lang="el")
            return
        
        # Non-streaming chat (Groq is already ~200ms, no need for streaming complexity)
        reply = chat(text, lang=lang)
        if reply:
            state.is_processing = False
            speak(reply, lang="el")
    finally:
        state.is_processing = False


def _groq_streaming_chat(user_message, lang):
    """Groq streaming: start TTS as soon as first sentence arrives."""
    import requests
    
    system = "You are VIRON, a helpful AI companion. Απάντα πάντα στα Ελληνικά. Είσαι ο ΒΙΡΟΝ, ένας φιλικός βοηθός. Απάντα σύντομα."
    
    try:
        t0 = time.time()
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": 300,
                "temperature": 0.7,
                "stream": True,
            },
            timeout=15,
            stream=True,
        )
        
        if resp.status_code != 200:
            log.warning(f"Groq stream error {resp.status_code}")
            return None
        
        state.tts_start()
        
        # Accumulate tokens, send to browser as soon as a sentence is complete
        buffer = ""
        sentence_count = 0
        import re
        
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode('utf-8', errors='replace')
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            
            try:
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    buffer += token
                    
                    # Check if we have a complete sentence
                    if re.search(r'[.!;?·]\s*$', buffer) or len(buffer) > 150:
                        sentence = buffer.strip()
                        if sentence:
                            sentence_count += 1
                            ms = int((time.time() - t0) * 1000)
                            log.info(f"⚡ Groq stream sentence {sentence_count} ({ms}ms): \"{sentence[:60]}\"")
                            with _response_lock:
                                _response_queue.append({
                                    "text": sentence, "lang": "el",
                                    "time": time.time(), "part": sentence_count
                                })
                        buffer = ""
            except json.JSONDecodeError:
                continue
        
        # Flush remaining buffer
        if buffer.strip():
            sentence_count += 1
            with _response_lock:
                _response_queue.append({
                    "text": buffer.strip(), "lang": "el",
                    "time": time.time(), "part": sentence_count
                })
        
        ms = int((time.time() - t0) * 1000)
        log.info(f"⚡ Groq streaming complete: {sentence_count} sentences in {ms}ms")
        
        # Wait for browser TTS playback
        wait_time = min(sentence_count * 3, 20)
        time.sleep(wait_time)
        state.tts_end()
        return True
        
    except Exception as e:
        log.warning(f"Groq streaming failed: {e}")
        state.tts_end()
        return None


def main_loop(mic):
    """Main voice assistant loop — Porcupine + Silero VAD + Whisper."""
    if porcupine is None:
        log.error("❌ Porcupine not available! Cannot start.")
        sys.exit(1)
    
    wake_name = PORCUPINE_CUSTOM_PATH.split('/')[-1] if PORCUPINE_CUSTOM_PATH else f"'{PORCUPINE_KEYWORD}'"
    
    log.info("=" * 50)
    log.info("🤖 VIRON Voice Pipeline Active")
    log.info(f"   Wake: Porcupine ({wake_name})")
    log.info(f"   VAD: {'Silero' if silero_vad else 'RMS'}")
    log.info(f"   STT: {'Deepgram streaming' if DEEPGRAM_API_KEY else ('whisper.cpp GPU' if WHISPER_CPP_AVAILABLE else ('faster-whisper CPU' if whisper_model else 'Cloud'))}")
    log.info(f"   Mic: {ALSA_DEVICE} ch{MIC_CHANNEL}")
    log.info("=" * 50)
    log.info(f"🎤 Say 'Hey {PORCUPINE_KEYWORD.title()}'...")
    
    while True:
        try:
            if state.is_speaking or state.is_processing or state.is_paused:
                time.sleep(0.1)
                continue
            
            if time.time() - state.last_tts_end < ECHO_COOLDOWN:
                mic.read_frame()
                continue
            
            frame = mic.read_frame()
            if frame is None:
                log.error("Mic read failed, restarting...")
                mic.stop()
                time.sleep(1)
                mic.start()
                continue
            
            if check_wake(frame):
                if state.set_wake("porcupine", 1.0):
                    log.info("🎯 Wake word detected!")
                    
                    # Say "Ορίστε" through browser speakers
                    with _response_lock:
                        _response_queue.append({"text": "Ορίστε;", "lang": "el", "time": time.time()})
                    log.info("📤 Sent 'Ορίστε' to browser")
                    
                    audio = record_command(mic)
                    if len(audio) > SAMPLE_RATE * 0.3:
                        text, lang = transcribe(audio, lang="el")  # Default Greek
                        if text and len(text) > 1:
                            lang = detect_language(text)
                            log.info(f"📝 Command: \"{text}\" (lang={lang})")
                            conversation_turn(mic, text, lang)
                        else:
                            log.info("  (empty transcription)")
                    else:
                        log.info("  (no speech in command)")
                    
                    log.info(f"🎤 Listening for wake word...")
        
        except KeyboardInterrupt:
            log.info("Shutting down...")
            break
        except Exception as e:
            log.error(f"Pipeline error: {e}")
            time.sleep(1)
        
        except KeyboardInterrupt:
            log.info("Shutting down...")
            break
        except Exception as e:
            log.error(f"Pipeline error: {e}")
            time.sleep(1)


# ═══════════════════════════════════════════════════════════
# HTTP API (for browser frontend)
# ═══════════════════════════════════════════════════════════

app = Flask(__name__)

@app.route("/wakeword/status", methods=["GET"])
def ww_status():
    return jsonify({
        "ready": porcupine is not None,
        "listening": not state.is_speaking and not state.is_paused,
        "paused": state.is_paused or state.is_speaking,
        "models": ["porcupine"],
        "detections": state.detection_count,
        "mode": "porcupine",
        "mic_device": ALSA_DEVICE,
        "mic_channel": MIC_CHANNEL,
        "server_side_mic": True,
        "pipeline_version": 2,
    })

@app.route("/wakeword/poll", methods=["GET"])
def ww_poll():
    # Pipeline handles full conversation (wake → record → STT → LLM → TTS)
    # Don't tell browser about wake - it would try to record and steal the mic
    return jsonify({"wake": False})

@app.route("/pipeline/response", methods=["GET"])
def pipeline_response():
    """Browser polls this to get LLM responses for TTS playback."""
    with _response_lock:
        if _response_queue:
            resp = _response_queue.pop(0)
            return jsonify({"has_response": True, "text": resp["text"], "lang": resp["lang"]})
    return jsonify({"has_response": False})

@app.route("/pipeline/state", methods=["GET"])
def pipeline_state():
    """Browser polls this for visual indicators (listening, processing, speaking)."""
    return jsonify({
        "listening": state.is_listening,
        "processing": state.is_processing,
        "speaking": state.is_speaking,
    })

@app.route("/wakeword/pause", methods=["POST"])
def ww_pause():
    return jsonify({"status": "paused"})

@app.route("/wakeword/resume", methods=["POST"])
def ww_resume():
    return jsonify({"status": "resumed"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "pipeline_version": 2,
        "wake_engine": "porcupine" if porcupine else "none",
        "vad_engine": "silero" if silero_vad else "rms",
        "stt_engine": "deepgram" if DEEPGRAM_API_KEY else ("whisper.cpp-gpu" if WHISPER_CPP_AVAILABLE else ("faster-whisper-cpu" if whisper_model else "cloud")),
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
    print(f"  Wake:    {'✅ Porcupine' if has_wake else '❌ PORCUPINE FAILED - check PICOVOICE_ACCESS_KEY'}")
    print(f"  VAD:     {'✅ Silero' if has_vad else '⚠ RMS fallback'}")
    stt_status = "✅ Deepgram streaming" if DEEPGRAM_API_KEY else ("✅ whisper.cpp GPU" if WHISPER_CPP_AVAILABLE else ("✅ faster-whisper CPU" if whisper_model else "☁️ Cloud only"))
    print(f"  STT:     {stt_status}")
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
        if porcupine:
            porcupine.delete()
        log.info("Pipeline stopped.")


if __name__ == "__main__":
    main()
