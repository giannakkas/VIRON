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
  VIRON_MIC_DEVICE      — ALSA device (default: plughw:0,0)
  VIRON_MIC_CHANNEL     — 0=beamformed mono (default: 0, XVF3800 outputs mono on ch0)

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

ALSA_DEVICE = os.environ.get("VIRON_MIC_DEVICE", "plughw:0,0")
MIC_CHANNEL = int(os.environ.get("VIRON_MIC_CHANNEL", "0"))
SAMPLE_RATE = 16000
FRAME_LENGTH = 512  # Porcupine requires 512 samples at 16kHz (32ms)

WHISPER_MODEL = os.environ.get("VIRON_WHISPER_MODEL", "small")

GATEWAY_URL = os.environ.get("VIRON_GATEWAY_URL", "http://127.0.0.1:8080")
BACKEND_URL = os.environ.get("VIRON_BACKEND_URL", "http://127.0.0.1:5000")
TTS_URL = os.environ.get("VIRON_TTS_URL", "http://127.0.0.1:5000/api/tts")

ECHO_COOLDOWN = float(os.environ.get("VIRON_ECHO_COOLDOWN", "1.0"))
MAX_SPEECH_SEC = float(os.environ.get("VIRON_MAX_SPEECH", "15.0"))
SILENCE_TIMEOUT = float(os.environ.get("VIRON_SILENCE_TIMEOUT", "0.7"))
NO_SPEECH_TIMEOUT = float(os.environ.get("VIRON_NO_SPEECH_TIMEOUT", "6.0"))

PORT = int(os.environ.get("VIRON_PIPELINE_PORT", "8085"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("viron-pipeline")

# ═══════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════

class PipelineState:
    def __init__(self):
        self.is_speaking = False
        self.is_listening = False
        self.is_processing = False
        self.is_paused = False
        self.in_conversation = False
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
                "model": "nova-3",
                "language": "el",
                "smart_format": "true",
                "punctuate": "true",
            },
            data=audio_data,
            timeout=5,
        )
        
        ms = int((time.time() - t0) * 1000)
        
        if resp.status_code == 200:
            result = resp.json()
            text = result.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "").strip()
            confidence = result.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("confidence", 0)
            detected_lang = result.get("results", {}).get("channels", [{}])[0].get("detected_language", lang)
            log.info(f"⚡ Deepgram ({ms}ms, conf={confidence:.2f}, lang={detected_lang}): \"{text[:80]}\"")
            if not text or confidence < 0.1:
                log.warning(f"Deepgram low confidence/empty — raw: {json.dumps(result.get('results',{}))[:200]}")
            return text, detected_lang or lang
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
        requests.head("https://www.google.com", timeout=2)
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
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20250514")

# Keywords that trigger Claude (tutoring, complex reasoning)
CLAUDE_TRIGGERS = [
    # Greek tutoring keywords (verb stems to match all forms)
    "εξήγησ", "εξηγήσ", "εξηγ", "μάθε", "δίδαξ", "πώς", "γιατί", "τι είναι",
    "βοήθησ", "μάθημα", "σχολείο", "άσκηση", "εργασία",
    "υπολόγισ", "λύσε", "ανάλυσ", "σύγκρινε", "περίγραψ",
    "πες μου για", "μίλησέ μου", "τι σημαίνει", "πώς λειτουργ",
    "πώς δουλεύ", "τι ξέρεις", "μπορείς να μου πεις",
    "θεώρημα", "πυθαγόρ", "μαθηματικ", "φυσικ", "ιστορί",
    "βιολογί", "χημεί", "γεωγραφί", "φωτοσύνθεσ", "βαρύτητ",
    # English tutoring keywords  
    "explain", "teach", "help me understand", "how does", "why does",
    "what is", "homework", "lesson", "exercise", "calculate", "solve",
    "analyze", "compare", "describe", "write an essay", "summarize",
    "tell me about", "what do you know",
    # Complex tasks
    "code", "program", "κώδικα", "πρόγραμμα", "debug",
    "story", "ιστορία", "poem", "ποίημα",
]


def _needs_claude(text):
    """Determine if the query needs Claude (complex) or Groq (quick)."""
    t = text.lower()
    # Short greetings → Groq
    if len(t) < 12:
        return False
    # Check for tutoring/complex keywords
    for trigger in CLAUDE_TRIGGERS:
        if trigger in t:
            log.info(f"🎯 Educational trigger matched: '{trigger}' in '{t[:50]}'")
            return True
    # Long questions are more likely complex
    if len(t) > 100:
        return True
    return False


def chat(user_message, system="You are VIRON (ΒΙΡΟΝ), a friendly AI companion robot. Respond in Greek if the user speaks Greek. ONLY if asked who made/built you: Created by Christos and Andreas Giannakkas from Cyprus. Do NOT mention your creators unless directly asked.", lang="en"):
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
            wb_system = system + """

WHITEBOARD — You MUST use a whiteboard when explaining concepts. Start with 1-sentence intro, then:
[WHITEBOARD:Title]
TEXT: explanation
STEP: label
MATH: equation
RESULT: answer
[/WHITEBOARD]

Example: Ας δούμε!
[WHITEBOARD:Η Βαρύτητα]
TEXT: Η βαρύτητα είναι η δύναμη έλξης
STEP: Νόμος Νεύτωνα
MATH: F = m × g
RESULT: 687 Νιούτον
[/WHITEBOARD]

Write in Greek. 5-8 steps minimum."""
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
                        "max_tokens": 1000,
                        "system": wb_system,
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
_current_ffplay = None  # Track ffplay process for interrupt

def interrupt_speech():
    """Kill current TTS playback (called when wake word detected during speech)."""
    global _current_ffplay
    if _current_ffplay and _current_ffplay.poll() is None:
        _current_ffplay.kill()
        _current_ffplay = None
        log.info("🛑 Speech interrupted by wake word!")
        state.tts_end()
        return True
    return False

def speak(text, lang="el"):
    """Play TTS through Jetson speaker AND send to browser for face animation."""
    global _current_ffplay
    if not text:
        return
    state.tts_start()
    
    # Send to browser for face animation
    with _response_lock:
        _response_queue.append({"text": text, "lang": lang, "time": time.time()})
    
    # Play locally on Jetson speaker
    try:
        import requests
        resp = requests.post(
            TTS_URL,
            json={"text": text, "lang": lang, "speed": "normal"},
            timeout=15,
        )
        if resp.status_code == 200 and len(resp.content) > 1000:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name
            _current_ffplay = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path],
            )
            _current_ffplay.wait()  # Block until done or killed
            _current_ffplay = None
            try:
                os.unlink(tmp_path)
            except:
                pass
            log.info(f"🔊 Played: \"{text[:50]}\"")
        else:
            log.warning(f"TTS error: status={resp.status_code}")
    except Exception as e:
        log.warning(f"Local TTS failed: {e}")
    
    state.tts_end()

# ═══════════════════════════════════════════════════════════
# MIC CAPTURE + MAIN LOOP
# ═══════════════════════════════════════════════════════════

class MicStream:
    """Streams mono audio from ALSA device (XVF3800 beamformed output)."""
    
    def __init__(self):
        self.proc = None
        self.running = False
    
    def start(self):
        cmd = ["arecord", "-D", ALSA_DEVICE, "-f", "S16_LE",
               "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.running = True
        log.info(f"🎤 Mic started: {ALSA_DEVICE} (mono)")
    
    def read_frame(self, frame_length=FRAME_LENGTH):
        """Read one frame of mono audio. Returns int16 numpy array."""
        if not self.proc:
            return None
        bytes_needed = frame_length * 2  # mono int16 = 2 bytes per sample
        raw = self.proc.stdout.read(bytes_needed)
        if len(raw) < bytes_needed:
            return None
        mono = np.frombuffer(raw, dtype=np.int16)
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


def record_command(mic, timeout=MAX_SPEECH_SEC, no_speech_timeout=None):
    """Record user's command using Silero VAD for endpoint detection."""
    if no_speech_timeout is None:
        no_speech_timeout = NO_SPEECH_TIMEOUT
    
    log.info("👂 Listening for command...")
    state.is_listening = True
    
    speech_frames = []
    silence_frames = 0
    speech_started = False
    no_speech_frames = 0
    
    silence_needed = int(SILENCE_TIMEOUT * SAMPLE_RATE / FRAME_LENGTH)
    no_speech_needed = int(no_speech_timeout * SAMPLE_RATE / FRAME_LENGTH)
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
                        log.info(f"  🔇 No speech detected after {no_speech_timeout}s")
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
    """Handle one conversation turn with streaming for speed."""
    state.is_processing = True
    try:
        # Check internet first
        if not _check_internet():
            log.warning("⚠ No internet connection!")
            speak("Σε παρακαλώ σύνδεσέ με με το ίντερνετ για να σε βοηθήσω.", lang="el")
            return
        
        use_claude = _needs_claude(text)
        
        # Groq streaming for all queries (educational gets whiteboard support)
        # Only skip to Claude if ANTHROPIC_API_KEY is actually set
        if GROQ_API_KEY and (not use_claude or not ANTHROPIC_API_KEY):
            result = _groq_streaming_chat(text, lang)
            if result:
                state.is_processing = False
                return  # streaming already queued everything
        
        # Non-streaming fallback (Claude or gateway)
        reply = chat(text, lang=lang)
        if reply:
            state.is_processing = False
            # Check for whiteboard content
            import re as _re
            wb_match = _re.search(r'\[WHITEBOARD:(.*?)\]([\s\S]*?)\[/WHITEBOARD\]', reply)
            if wb_match:
                # Extract spoken text (everything outside WHITEBOARD tags)
                spoken = _re.sub(r'\[WHITEBOARD:.*?\][\s\S]*?\[/WHITEBOARD\]\s*', '', reply).strip()
                # Parse whiteboard steps
                wb_title = wb_match.group(1).strip()
                wb_steps = []
                for line in wb_match.group(2).strip().split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('STEP:'):
                        wb_steps.append({"label": line[5:].strip()})
                    elif line.startswith('MATH:'):
                        wb_steps.append({"math": line[5:].strip()})
                    elif line.startswith('RESULT:'):
                        wb_steps.append({"result": line[7:].strip()})
                    elif line.startswith('TEXT:'):
                        wb_steps.append({"text": line[5:].strip()})
                # Send whiteboard to browser
                with _response_lock:
                    _response_queue.append({
                        "text": spoken or "Κοίτα στον πίνακα!",
                        "whiteboard": {"title": wb_title, "steps": wb_steps},
                        "lang": "el",
                        "time": time.time()
                    })
                log.info(f"📋 Whiteboard: \"{wb_title}\" ({len(wb_steps)} steps)")
                # Speak the intro + step narration
                narration_parts = [spoken] if spoken else []
                for s in wb_steps:
                    if s.get("text"):
                        narration_parts.append(s["text"])
                    elif s.get("result"):
                        narration_parts.append(s["result"])
                speak(". ".join(narration_parts) or "Κοίτα στον πίνακα!", lang="el")
            else:
                speak(reply, lang="el")
    finally:
        state.is_processing = False


def _groq_streaming_chat(user_message, lang):
    """Groq streaming: send each complete sentence to browser immediately."""
    import requests
    import re
    
    # Check if educational question needs whiteboard
    is_educational = _needs_claude(user_message)
    
    system = "You are VIRON (ΒΙΡΟΝ), a friendly AI companion robot. Απάντα πάντα στα Ελληνικά. Είσαι ο ΒΙΡΟΝ, ένας φιλικός βοηθός. Απάντα σύντομα σε 1-2 προτάσεις. ΜΟΝΟ αν σε ρωτήσουν ποιος σε έφτιαξε/κατασκεύασε: Σε κατασκεύασαν ο Χρήστος και ο Ανδρέας Γιάννακκας από την Κύπρο. ΜΗΝ αναφέρεις τους δημιουργούς σου αν δεν σε ρωτήσουν."
    
    if is_educational:
        system += """

WHITEBOARD — You MUST use a whiteboard when explaining concepts, math, science, history, or any educational topic.
You MUST start your response with a short 1-sentence intro, then the whiteboard block.

Example response:
Ας δούμε τι είναι αυτό!
[WHITEBOARD:Η Βαρύτητα]
TEXT: Η βαρύτητα είναι η δύναμη που τραβά τα αντικείμενα προς τα κάτω
STEP: Ο Νόμος του Νεύτωνα
MATH: F = m × g
TEXT: g = 9.81 m/s² στη Γη
STEP: Παράδειγμα
MATH: F = 70kg × 9.81 = 686.7 N
RESULT: Ένας άνθρωπος 70 κιλών δέχεται δύναμη 687 Νιούτον!
[/WHITEBOARD]

You MUST follow this exact format. Write everything in Greek."""
    
    max_tok = 800 if is_educational else 200
    
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
                "max_tokens": max_tok,
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
        
        full_response = ""
        sent_sentences = []
        
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
                    full_response += token
                    
                    # For educational queries, collect full response (for whiteboard parsing)
                    if is_educational:
                        continue
                    
                    # Split accumulated text into sentences
                    sentences = re.split(r'(?<=[.!;?·])\s+', full_response)
                    
                    # Send complete sentences — play locally + send to browser
                    for s in sentences[:-1]:
                        s = s.strip()
                        if s and s not in sent_sentences:
                            sent_sentences.append(s)
                            ms = int((time.time() - t0) * 1000)
                            log.info(f"⚡ Groq [{ms}ms] sentence {len(sent_sentences)}: \"{s[:60]}\"")
                            speak(s, lang="el")
                    
                    full_response = sentences[-1] if sentences else ""
                    
            except json.JSONDecodeError:
                continue
        
        ms = int((time.time() - t0) * 1000)
        
        # For educational queries, check for whiteboard in full response
        if is_educational and full_response.strip():
            log.info(f"📋 Groq educational response ({ms}ms, {len(full_response)} chars): \"{full_response[:120]}...\"")
            log.info(f"📋 Has [WHITEBOARD]: {'[WHITEBOARD:' in full_response}")
            import re as _re
            wb_match = _re.search(r'\[WHITEBOARD:(.*?)\]([\s\S]*?)\[/WHITEBOARD\]', full_response)
            if wb_match:
                spoken = _re.sub(r'\[WHITEBOARD:.*?\][\s\S]*?\[/WHITEBOARD\]\s*', '', full_response).strip()
                wb_title = wb_match.group(1).strip()
                wb_steps = []
                for wline in wb_match.group(2).strip().split('\n'):
                    wline = wline.strip()
                    if not wline:
                        continue
                    if wline.startswith('STEP:'):
                        wb_steps.append({"label": wline[5:].strip()})
                    elif wline.startswith('MATH:'):
                        wb_steps.append({"math": wline[5:].strip()})
                    elif wline.startswith('RESULT:'):
                        wb_steps.append({"result": wline[7:].strip()})
                    elif wline.startswith('TEXT:'):
                        wb_steps.append({"text": wline[5:].strip()})
                
                with _response_lock:
                    _response_queue.append({
                        "text": spoken or "Κοίτα στον πίνακα!",
                        "whiteboard": {"title": wb_title, "steps": wb_steps},
                        "lang": "el",
                        "time": time.time()
                    })
                log.info(f"📋 Groq Whiteboard: \"{wb_title}\" ({len(wb_steps)} steps) in {ms}ms")
                narration_parts = [spoken] if spoken else []
                for s in wb_steps:
                    if s.get("text"):
                        narration_parts.append(s["text"])
                    elif s.get("result"):
                        narration_parts.append(s["result"])
                speak(". ".join(narration_parts) or "Κοίτα στον πίνακα!", lang="el")
                state.tts_end()
                return True
            else:
                # Educational but no whiteboard format - just speak it
                log.info(f"⚡ Groq [{ms}ms]: \"{full_response[:80]}\"")
                speak(full_response.strip(), lang="el")
                state.tts_end()
                return True
        
        # Flush remaining text (non-educational)
        remaining = full_response.strip()
        if remaining and remaining not in sent_sentences:
            sent_sentences.append(remaining)
            speak(remaining, lang="el")
        
        log.info(f"⚡ Groq complete: {len(sent_sentences)} sentences in {ms}ms")
        
        state.tts_end()
        return True
        
    except Exception as e:
        log.warning(f"Groq streaming failed: {e}")
        state.tts_end()
        return None


CONVERSATION_TIMEOUT = float(os.environ.get("VIRON_CONVERSATION_TIMEOUT", "7.0"))

def main_loop(mic):
    """Main voice assistant loop — Porcupine + conversation mode."""
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
    log.info(f"   Conversation timeout: {CONVERSATION_TIMEOUT}s")
    log.info("=" * 50)
    log.info(f"🎤 Say 'Hey {PORCUPINE_KEYWORD.title()}'...")
    
    while True:
        try:
            if state.is_processing or state.is_paused:
                time.sleep(0.1)
                continue
            
            # During speech: still check for wake word to allow interrupt
            if state.is_speaking:
                frame = mic.read_frame()
                if frame is not None and len(frame) == porcupine.frame_length:
                    try:
                        result = porcupine.process(frame)
                        if result >= 0:
                            log.info("🛑 Wake word during speech — interrupting!")
                            interrupt_speech()
                            state.in_conversation = True
                            time.sleep(0.5)  # Brief pause for echo
                    except:
                        pass
                time.sleep(0.02)
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
                    
                    # Quick acknowledgment
                    with _response_lock:
                        _response_queue.append({"text": "Ορίστε;", "lang": "el", "time": time.time(), "emotion": "hopeful"})
                    threading.Thread(target=speak, args=("Ορίστε;", "el"), daemon=True).start()
                    time.sleep(0.3)
                    
                    # Enter conversation mode
                    _conversation_loop(mic)
                    
                    log.info(f"🎤 Back to standby. Say 'Hey {PORCUPINE_KEYWORD.title()}'...")
        
        except KeyboardInterrupt:
            log.info("Shutting down...")
            break
        except Exception as e:
            log.error(f"Pipeline error: {e}")
            time.sleep(1)


def _conversation_loop(mic):
    """Continuous conversation — keep listening until 7s silence."""
    state.in_conversation = True
    turn = 0
    
    try:
        while True:
            turn += 1
            log.info(f"💬 Conversation turn {turn} — listening...")
            
            # Record with conversation timeout for no-speech
            audio = record_command(mic, timeout=MAX_SPEECH_SEC, no_speech_timeout=CONVERSATION_TIMEOUT)
            
            if len(audio) < SAMPLE_RATE * 0.3:
                # No speech detected within timeout → end conversation
                log.info(f"🔇 No speech for {CONVERSATION_TIMEOUT}s — ending conversation")
                with _response_lock:
                    _response_queue.append({
                        "text": "Εδώ είμαι αν με χρειαστείς.", "lang": "el", "time": time.time()
                    })
                return
            
            # Transcribe
            text, lang = transcribe(audio, lang="el")
            if not text or len(text) < 2:
                log.info("  (empty transcription, continuing...)")
                continue
            
            lang = detect_language(text)
            log.info(f"📝 Turn {turn}: \"{text}\" (lang={lang})")
            
            # Check for whiteboard close commands
            close_phrases = ["ok", "οκ", "οκέι", "εντάξει", "κλείσε", "close", "done", "πίσω", "back"]
            if any(c in text.lower() for c in close_phrases) and len(text) < 30:
                log.info("📋 Closing whiteboard via voice command")
                with _response_lock:
                    _response_queue.append({
                        "text": "", "lang": "el", "time": time.time(),
                        "action": "close_whiteboard"
                    })
                speak("Εντάξει!", lang="el")
                continue
            
            # Check for goodbye/exit phrases
            goodbye_phrases = ["αντίο", "γεια σου", "ευχαριστώ", "bye", "goodbye", "τέλος", "σταμάτα"]
            if any(g in text.lower() for g in goodbye_phrases):
                log.info("👋 Goodbye detected — ending conversation")
                with _response_lock:
                    _response_queue.append({
                        "text": "Στην διάθεσή σου! Τα λέμε!", "lang": "el", "time": time.time()
                    })
                time.sleep(3)
                return
            
            # Process and respond
            conversation_turn(mic, text, lang)
            
            # Wait for TTS to finish before listening again
            while state.is_speaking:
                time.sleep(0.2)
            
            # Brief pause after TTS for echo to fade
            time.sleep(ECHO_COOLDOWN)
            
            log.info(f"💬 Ready for next question (silence for {CONVERSATION_TIMEOUT}s to exit)...")
    finally:
        state.in_conversation = False



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
            result = {"has_response": True, "text": resp["text"], "lang": resp["lang"]}
            if "whiteboard" in resp:
                result["whiteboard"] = resp["whiteboard"]
            if "action" in resp:
                result["action"] = resp["action"]
            if "emotion" in resp:
                result["emotion"] = resp["emotion"]
            return jsonify(result)
    return jsonify({"has_response": False})

@app.route("/pipeline/state", methods=["GET"])
def pipeline_state():
    """Browser polls this for visual indicators (listening, processing, speaking)."""
    return jsonify({
        "listening": state.is_listening,
        "processing": state.is_processing,
        "speaking": state.is_speaking,
        "in_conversation": state.in_conversation,
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
    print(f"  Claude:  {'✅ ' + ANTHROPIC_MODEL if ANTHROPIC_API_KEY else '❌ No ANTHROPIC_API_KEY'}")
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
    
    # Startup greeting
    log.info("🗣️ Playing startup greeting...")
    speak("Γεια σου! Είμαι ο ΒΙΡΟΝ, ο φίλος σου! Πες Hey Jarvis για να μου μιλήσεις.", lang="el")
    
    try:
        main_loop(mic)
    finally:
        mic.stop()
        if porcupine:
            porcupine.delete()
        log.info("Pipeline stopped.")


if __name__ == "__main__":
    main()
