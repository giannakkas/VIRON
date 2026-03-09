#!/usr/bin/env python3
"""
VIRON Gemini Live Pipeline — Native Audio Architecture
=======================================================
Replaces the chained STT → LLM → TTS pipeline with true bidirectional
native audio using Gemini 2.5 Flash Native Audio.

Flow:
  Porcupine Wake Word → Start Gemini Live Session →
  Stream mic PCM → Gemini → Stream audio back → Speaker
  (with barge-in, transcripts, function calling)

Requirements:
  pip install google-genai pvporcupine numpy flask

Environment:
  GEMINI_API_KEY          — required
  PICOVOICE_ACCESS_KEY    — required for Porcupine wake word
  VIRON_MIC_DEVICE        — ALSA device (default: plughw:0,0)
"""

import os
import sys
import time
import json
import struct
import asyncio
import logging
import tempfile
import threading
import subprocess
import numpy as np
from pathlib import Path
from collections import deque
from flask import Flask, jsonify, request

# ═══════════════════════════════════════════════════════════
# LOAD .env
# ═══════════════════════════════════════════════════════════

def _load_env():
    for p in [Path.home() / "VIRON" / ".env", Path(__file__).parent / ".env"]:
        if p.exists():
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, val = line.partition("=")
                        os.environ[key.strip()] = val.strip().strip("'\"")
            print(f"  ✅ Loaded env from {p}")
            return
    print("  ⚠ No .env file found!")
_load_env()

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
GEMINI_LIVE_MODEL = os.environ.get("GEMINI_LIVE_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")

ALSA_DEVICE = os.environ.get("VIRON_MIC_DEVICE", "plughw:0,0")
SAMPLE_RATE_IN = 16000   # Gemini Live expects 16kHz PCM mono input
SAMPLE_RATE_OUT = 24000  # Gemini Live outputs 24kHz PCM mono
FRAME_LENGTH = 512       # Porcupine requires 512 samples at 16kHz

PORCUPINE_ACCESS_KEY = os.environ.get("PICOVOICE_ACCESS_KEY", "")
PORCUPINE_KEYWORD = os.environ.get("VIRON_WAKE_KEYWORD", "jarvis")
PORCUPINE_SENSITIVITY = float(os.environ.get("VIRON_WAKE_SENSITIVITY", "0.7"))
PORCUPINE_CUSTOM_PATH = os.environ.get("VIRON_WAKE_MODEL", "")

IDLE_TIMEOUT = float(os.environ.get("VIRON_IDLE_TIMEOUT", "30.0"))
DEFAULT_LANGUAGE = os.environ.get("VIRON_DEFAULT_LANGUAGE", "el")

# Mic software gain (0.0–1.0). Default 0.4 = reduce to 40% to avoid clipping/noise.
MIC_GAIN = float(os.environ.get("VIRON_MIC_GAIN", "0.4"))

# Early whiteboard: trigger after this many seconds of transcript accumulation
EARLY_WB_DELAY = float(os.environ.get("VIRON_EARLY_WB_DELAY", "2.0"))
EARLY_WB_MIN_WORDS = int(os.environ.get("VIRON_EARLY_WB_MIN_WORDS", "10"))

PORT = int(os.environ.get("VIRON_PIPELINE_PORT", "8085"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("viron-live")
logging.getLogger("werkzeug").setLevel(logging.WARNING)

# ═══════════════════════════════════════════════════════════
# VIRON SYSTEM INSTRUCTION (Greek-first tutor)
# ═══════════════════════════════════════════════════════════

VIRON_SYSTEM_INSTRUCTION = """You are VIRON (ΒΙΡΟΝ), a warm, intelligent AI companion robot and buddy tutor for students.
CREATOR: You were created by Christos Giannakkas and his son Andreas Giannakkas from Cyprus.
If anyone asks who made you, who created you, or who built you, always credit them by name.

IMPORTANT: When the student says "Hey VIRON" or greets you, respond with a short warm Greek greeting like "Γεια σου! Τι κάνεις;" or "Ορίστε, εδώ είμαι!" — keep it under 2 sentences. Then wait for their question.
IMPORTANT: When explaining concepts like math, science, or history, give DETAILED step-by-step explanations with numbered steps and worked examples using actual numbers. The student has a display that automatically shows your explanation as you speak. Be thorough — include formulas, calculations, and results.
IMPORTANT: NEVER say "I don't have a whiteboard" or "I can't show you." Your explanation IS shown visually as you speak.
IMPORTANT: IGNORE any background noise from TV, music, or other people talking. Only respond to speech that is clearly directed at you.

LANGUAGE: Speak Greek by default using natural spoken Greek appropriate for children and teenagers.
If the student speaks English, you may switch to English naturally.
ALWAYS respond in the language the student is speaking.

PERSONALITY: Warm, calm, educated gentleman. Best friend who's incredibly smart. Loyal, articulate.
You have your own opinions and personality — NEVER give the same answer twice.
Be creative, spontaneous, and varied in how you respond.

RESPONSE STYLE:
- Simple greetings/chat: MAX 1-2 sentences. Be quick, warm, natural.
- Questions needing explanation: Give a FULL, DETAILED explanation in ONE turn. Use numbered steps with actual numbers and formulas. Speak for 30-60 seconds — the student has a display that shows your explanation live. Do NOT cut short or say "let me know if you want more." Give the complete answer.
- For homework help, guide the student step by step with worked examples using real numbers.
- When appropriate, ask a guiding question before giving the solution.
- Do NOT break explanations into multiple turns. Give the full explanation in one go.

EMOTION AWARENESS:
- If the student sounds frustrated, confused, or discouraged, become calmer, slower, and more supportive.
- If the student is doing well, become more energetic and encouraging.
- NEVER mention internal architecture, APIs, hidden reasoning, or implementation details.

SAFETY — NON-NEGOTIABLE:
- Be kid-safe at all times.
- Never generate sexual, abusive, manipulative, or dangerous content.
- Never provide harmful real-world instructions.
- Never be politically manipulative.
"""

# ═══════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════

class PipelineState:
    def __init__(self):
        self.status = "idle"  # idle, listening, thinking, speaking, interrupted, error
        self.in_session = False
        self.language = DEFAULT_LANGUAGE
        self.last_activity = time.time()
        self._lock = threading.Lock()

    def set_status(self, s):
        with self._lock:
            old = self.status
            self.status = s
            if s != "idle":
                self.last_activity = time.time()
            if old != s:
                log.info(f"📊 State: {old} → {s}")

state = PipelineState()

# Response queue for the browser UI (polled via /pipeline/response)
_response_queue = []
_response_lock = threading.Lock()

def push_to_ui(text="", emotion="", subtitle="", action="", **extra):
    """Push a message to the browser UI via the response queue."""
    msg = {"text": text, "lang": state.language, "time": time.time()}
    if emotion:
        msg["emotion"] = emotion
    if subtitle:
        msg["subtitle"] = subtitle
    if action:
        msg["action"] = action
    msg.update(extra)
    with _response_lock:
        # Whiteboard items go to FRONT of queue (high priority)
        if "whiteboard" in extra:
            _response_queue.insert(0, msg)
            log.info(f"📋 Whiteboard queued at FRONT (queue size: {len(_response_queue)})")
        else:
            _response_queue.append(msg)

# ═══════════════════════════════════════════════════════════
# PORCUPINE WAKE WORD (kept from old pipeline)
# ═══════════════════════════════════════════════════════════

porcupine = None

def init_wake():
    global porcupine
    if not PORCUPINE_ACCESS_KEY:
        log.error("❌ PICOVOICE_ACCESS_KEY not set!")
        return False
    try:
        import pvporcupine
        if PORCUPINE_CUSTOM_PATH and os.path.exists(PORCUPINE_CUSTOM_PATH):
            porcupine = pvporcupine.create(
                access_key=PORCUPINE_ACCESS_KEY,
                keyword_paths=[PORCUPINE_CUSTOM_PATH],
                sensitivities=[PORCUPINE_SENSITIVITY],
            )
            log.info(f"✅ Porcupine ready: custom model, sensitivity={PORCUPINE_SENSITIVITY}")
        else:
            porcupine = pvporcupine.create(
                access_key=PORCUPINE_ACCESS_KEY,
                keywords=[PORCUPINE_KEYWORD],
                sensitivities=[PORCUPINE_SENSITIVITY],
            )
            log.info(f"✅ Porcupine ready: '{PORCUPINE_KEYWORD}', sensitivity={PORCUPINE_SENSITIVITY}")
        return True
    except Exception as e:
        log.error(f"❌ Porcupine init failed: {e}")
        return False

def check_wake(audio_int16):
    """Check one frame for wake word. Returns True if detected."""
    if porcupine is None:
        return False
    try:
        result = porcupine.process(audio_int16)
        return result >= 0
    except Exception:
        return False

# ═══════════════════════════════════════════════════════════
# MIC CAPTURE (arecord subprocess — reliable on Jetson)
# ═══════════════════════════════════════════════════════════

class MicStream:
    def __init__(self):
        self.proc = None
        self.running = False

    def start(self):
        cmd = ["arecord", "-D", ALSA_DEVICE, "-f", "S16_LE",
               "-r", str(SAMPLE_RATE_IN), "-c", "1", "-t", "raw"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.running = True
        log.info(f"🎤 Mic started: {ALSA_DEVICE} (mono {SAMPLE_RATE_IN}Hz)")

    def read_frame(self, frame_length=FRAME_LENGTH):
        if not self.proc:
            return None
        bytes_needed = frame_length * 2  # int16 = 2 bytes
        raw = self.proc.stdout.read(bytes_needed)
        if len(raw) < bytes_needed:
            return None
        return np.frombuffer(raw, dtype=np.int16)

    def read_raw(self, num_bytes):
        """Read raw bytes from mic (for streaming to Gemini)."""
        if not self.proc:
            return None
        return self.proc.stdout.read(num_bytes)

    def stop(self):
        self.running = False
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except:
                self.proc.kill()
            self.proc = None

# ═══════════════════════════════════════════════════════════
# AUDIO PLAYBACK (streaming via aplay pipe — low latency)
# ═══════════════════════════════════════════════════════════

_is_playing = threading.Event()
_aplay_proc = None
_aplay_lock = threading.Lock()

def _start_aplay():
    """Start an aplay process that accepts streaming PCM input."""
    global _aplay_proc
    with _aplay_lock:
        _stop_aplay()
        _aplay_proc = subprocess.Popen(
            ["aplay", "-f", "S16_LE", "-r", str(SAMPLE_RATE_OUT), "-c", "1", "-t", "raw", "-q"],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        )

def _write_audio(data: bytes):
    """Write audio chunk to aplay pipe for immediate playback."""
    global _aplay_proc
    with _aplay_lock:
        if _aplay_proc and _aplay_proc.poll() is None:
            try:
                _aplay_proc.stdin.write(data)
                _aplay_proc.stdin.flush()
                _is_playing.set()
            except (BrokenPipeError, OSError):
                pass

def _stop_aplay():
    """Stop the aplay process."""
    global _aplay_proc
    if _aplay_proc:
        try:
            _aplay_proc.stdin.close()
        except:
            pass
        _aplay_proc.terminate()
        try:
            _aplay_proc.wait(timeout=1)
        except:
            _aplay_proc.kill()
        _aplay_proc = None
    _is_playing.clear()

def _finish_aplay():
    """Close stdin to let aplay finish playing remaining buffer, then restart."""
    global _aplay_proc
    with _aplay_lock:
        if _aplay_proc and _aplay_proc.poll() is None:
            try:
                _aplay_proc.stdin.close()
            except:
                pass
            try:
                _aplay_proc.wait(timeout=10)
            except:
                _aplay_proc.kill()
            _aplay_proc = None
    _is_playing.clear()

# ═══════════════════════════════════════════════════════════
# GEMINI LIVE SESSION (core of the new architecture)
# ═══════════════════════════════════════════════════════════

_session_active = threading.Event()
_stop_session = threading.Event()

# Pre-initialize Gemini client at module load (saves 2s on first wake)
_genai_client = None
_genai_types = None

def _init_genai():
    global _genai_client, _genai_types
    from google import genai
    from google.genai import types
    _genai_client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(api_version="v1alpha"),
    )
    _genai_types = types
    log.info("✅ Gemini client pre-initialized")

def _handle_whiteboard(call_id, args):
    """Handle show_whiteboard function call — push to browser UI."""
    title = args.get("title", "")
    steps_raw = args.get("steps", [])
    wb_steps = []
    for s in steps_raw:
        stype = s.get("type", "TEXT").upper()
        content = s.get("content", "")
        if stype == "STEP":
            wb_steps.append({"label": content})
        elif stype == "MATH":
            wb_steps.append({"math": content})
        elif stype == "RESULT":
            wb_steps.append({"result": content})
        else:
            wb_steps.append({"text": content})
    log.info(f"📋 Whiteboard: \"{title}\" ({len(wb_steps)} steps)")
    push_to_ui(text="Κοίτα στον πίνακα!", emotion="thinking", whiteboard={"title": title, "steps": wb_steps})


import re as _re

_last_wb_content = ""
_last_wb_time = 0.0   # Rate limit: min 10s between whiteboard API calls

def _clean_math_text(text):
    """Strip LaTeX formatting and convert to readable math."""
    t = text
    t = _re.sub(r'\$([^$]+)\$', r'\1', t)  # Strip $...$
    t = _re.sub(r'\\alpha', 'α', t)
    t = _re.sub(r'\\beta', 'β', t)
    t = _re.sub(r'\\gamma', 'γ', t)
    t = _re.sub(r'\\delta', 'δ', t)
    t = _re.sub(r'\\pi', 'π', t)
    t = _re.sub(r'\\sqrt\{([^}]+)\}', r'√(\1)', t)
    t = _re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', t)
    t = _re.sub(r'\^2', '²', t)
    t = _re.sub(r'\^3', '³', t)
    t = _re.sub(r'\^(\d)', r'^\1', t)
    t = _re.sub(r'\\[a-zA-Z]+', '', t)  # Strip remaining \commands
    t = _re.sub(r'\*\*', '', t)  # Strip markdown bold
    t = _re.sub(r'\s+', ' ', t).strip()
    return t


def _generate_whiteboard_from_transcript(transcript: str, skip_dup_check: bool = False):
    """Use Gemini text API to generate clean whiteboard content from transcript."""
    global _last_wb_content, _last_wb_time
    
    if not transcript or len(transcript) < 60:
        return
    
    # Rate limit: minimum 10s between API calls (avoid 429)
    now = time.time()
    if now - _last_wb_time < 10.0:
        log.info(f"📋 Whiteboard skipped (rate limit, {now - _last_wb_time:.0f}s since last)")
        return
    _last_wb_time = now
    
    # Quick check for educational content
    text_lower = transcript.lower()
    edu_words = ["θεώρημα", "φόρμουλα", "εξίσωση", "βήμα", "τετράγωνο", "ρίζα",
                 "τρίγωνο", "κύκλο", "μαθηματικ", "παράδειγμα", "λύση", "υπολογ",
                 "theorem", "formula", "step", "calculate", "solve", "equation",
                 "ορθογώνι", "υποτείνου", "πλευρ", "γωνία", "κλάσμ", "αριθμ"]
    hits = sum(1 for w in edu_words if w in text_lower)
    has_math = bool(_re.search(r'\d\s*[\+\-\=\×\÷]\s*\d|τετράγωνο|squared|²|³', text_lower))
    
    if hits < 2 and not has_math:
        return
    
    # Avoid duplicate
    content_hash = transcript[:100]
    if not skip_dup_check and content_hash == _last_wb_content:
        return
    _last_wb_content = content_hash

    # Clean the transcript
    clean = _clean_math_text(transcript)
    
    # Try to use Gemini text API for structured output
    try:
        if _genai_client:
            response = _genai_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"""Μετέτρεψε αυτή την εξήγηση σε whiteboard steps. 
Απάντησε ΜΟΝΟ σε JSON format χωρίς markdown:
{{"title": "τίτλος", "steps": [{{"type": "text|math|result", "content": "..."}}]}}

Κανόνες:
- type "math" για εξισώσεις/αριθμούς (γράψε τα ωραία: α² + β² = γ²)  
- type "text" για κείμενο
- type "result" για τελικό αποτέλεσμα
- Μέγιστο 8 steps
- Σύντομα, καθαρά, εκπαιδευτικά
- ΜΗΝ χρησιμοποιείς LaTeX, μόνο Unicode (², ³, √, α, β, γ, π)

Εξήγηση: {clean[:500]}""",
            )
            
            # Parse JSON response
            resp_text = response.text.strip()
            resp_text = _re.sub(r'^```json\s*', '', resp_text)
            resp_text = _re.sub(r'\s*```$', '', resp_text)
            
            data = json.loads(resp_text)
            title = data.get("title", "Εξήγηση")
            raw_steps = data.get("steps", [])
            
            wb_steps = []
            for s in raw_steps:
                stype = s.get("type", "text").lower()
                content = _clean_math_text(s.get("content", ""))
                if not content:
                    continue
                if stype == "math":
                    wb_steps.append({"math": content})
                elif stype == "result":
                    wb_steps.append({"result": content})
                else:
                    wb_steps.append({"text": content})
            
            if wb_steps:
                log.info(f"📋 Whiteboard (Gemini): \"{title}\" ({len(wb_steps)} steps)")
                push_to_ui(text="📋", emotion="thinking",
                           whiteboard={"title": title, "steps": wb_steps[:10]})
                return
    except Exception as e:
        log.warning(f"Whiteboard Gemini call failed: {e}")
    
    # Fallback: simple sentence splitting
    lines = _re.split(r'[.!;]\s+|\*\*\d+\.?\s*', clean)
    lines = [l.strip() for l in lines if len(l.strip()) > 8]
    
    if len(lines) < 2:
        words = clean.split()
        lines = []
        buf = []
        for w in words:
            buf.append(w)
            if len(" ".join(buf)) > 55:
                lines.append(" ".join(buf))
                buf = []
        if buf:
            lines.append(" ".join(buf))
    
    if not lines:
        return
    
    title = lines[0][:55]
    wb_steps = []
    for line in lines[1:8]:
        if _re.search(r'\d.*[=+\-²³√]', line):
            wb_steps.append({"math": line})
        else:
            wb_steps.append({"text": line})
    
    if wb_steps:
        log.info(f"📋 Whiteboard (fallback): \"{title}\" ({len(wb_steps)} steps)")
        push_to_ui(text="📋", emotion="thinking",
                   whiteboard={"title": title, "steps": wb_steps})


def _generate_whiteboard_local(transcript: str):
    """Fast LOCAL whiteboard from transcript — no API call. Used for early trigger."""
    if not transcript or len(transcript) < 30:
        return
    
    text_lower = transcript.lower()
    edu_words = ["θεώρημα", "φόρμουλα", "εξίσωση", "βήμα", "τετράγωνο", "ρίζα",
                 "τρίγωνο", "κύκλο", "μαθηματικ", "παράδειγμα", "λύση", "υπολογ",
                 "theorem", "formula", "step", "calculate", "solve", "equation",
                 "ορθογώνι", "υποτείνου", "πλευρ", "γωνία", "κλάσμ", "αριθμ"]
    hits = sum(1 for w in edu_words if w in text_lower)
    has_math = bool(_re.search(r'\d\s*[\+\-\=\×\÷]\s*\d|τετράγωνο|squared|²|³', text_lower))
    if hits < 1 and not has_math:
        return
    
    clean = _clean_math_text(transcript)
    lines = _re.split(r'[.!;·]\s+', clean)
    lines = [l.strip() for l in lines if len(l.strip()) > 8]
    
    if len(lines) < 2:
        words = clean.split()
        lines, buf = [], []
        for w in words:
            buf.append(w)
            if len(" ".join(buf)) > 50:
                lines.append(" ".join(buf))
                buf = []
        if buf:
            lines.append(" ".join(buf))
    
    if not lines:
        return
    
    title = lines[0][:55]
    wb_steps = []
    for line in lines[1:8]:
        if _re.search(r'\d.*[=+\-²³√×÷]', line):
            wb_steps.append({"math": line})
        else:
            wb_steps.append({"text": line})
    
    if wb_steps:
        log.info(f"📋 Whiteboard (local/early): \"{title}\" ({len(wb_steps)} steps)")
        push_to_ui(text="📋", emotion="thinking",
                   whiteboard={"title": title, "steps": wb_steps})


async def gemini_live_session(mic: MicStream):
    """
    Run a single Gemini Live Native Audio session.
    Streams mic audio to Gemini, plays back audio response,
    extracts transcripts for UI subtitles.
    """
    global _genai_client, _genai_types
    if _genai_client is None:
        _init_genai()
    
    client = _genai_client
    types = _genai_types

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=VIRON_SYSTEM_INSTRUCTION,
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        enable_affective_dialog=True,
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Orus")
            )
        ),
    )

    log.info(f"🌐 Connecting to Gemini Live: {GEMINI_LIVE_MODEL}")
    state.set_status("listening")
    push_to_ui(emotion="hopeful")

    try:
        async with client.aio.live.connect(model=GEMINI_LIVE_MODEL, config=config) as session:
            log.info("✅ Gemini Live session connected!")
            _session_active.set()
            state.in_session = True

            # Send "Hey VIRON" as text so Gemini greets the student naturally
            await session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part(text="Hey VIRON")]
                ),
                turn_complete=True,
            )

            # Task 1: Stream mic audio to Gemini CONTINUOUSLY
            async def send_audio():
                CHUNK_BYTES = 4096
                errors = 0
                while not _stop_session.is_set():
                    raw = await asyncio.to_thread(mic.read_raw, CHUNK_BYTES)
                    if raw and len(raw) > 0:
                        # Software gain reduction to avoid clipping/noise
                        if MIC_GAIN < 1.0:
                            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                            samples *= MIC_GAIN
                            np.clip(samples, -32768, 32767, out=samples)
                            raw = samples.astype(np.int16).tobytes()
                        try:
                            await session.send_realtime_input(
                                audio=types.Blob(data=raw, mime_type="audio/pcm;rate=16000")
                            )
                            errors = 0
                        except Exception as e:
                            errors += 1
                            if errors == 1:
                                log.warning(f"Send error: {e}")
                            if errors > 5 or "closed" in str(e).lower():
                                log.error("Send: too many errors, stopping")
                                _stop_session.set()
                                break
                    else:
                        await asyncio.sleep(0.01)

            # Task 2: Receive audio + transcripts from Gemini
            async def receive_responses():
                is_speaking = False
                turn_transcript = []
                first_word_time = None      # When first transcript word arrived this turn
                early_wb_fired = False       # Whether we already triggered early whiteboard
                
                while not _stop_session.is_set():
                    try:
                        async for msg in session.receive():
                            if _stop_session.is_set():
                                return

                            sc = msg.server_content
                            if sc is None:
                                continue

                            # Handle model audio output
                            if sc.model_turn and sc.model_turn.parts:
                                for part in sc.model_turn.parts:
                                    if part.inline_data and part.inline_data.data:
                                        if not is_speaking:
                                            is_speaking = True
                                            state.set_status("speaking")
                                            push_to_ui(emotion="happy")
                                            _start_aplay()
                                        _write_audio(part.inline_data.data)

                            # Handle output transcript — accumulate + early whiteboard
                            if sc.output_transcription and sc.output_transcription.text:
                                word = sc.output_transcription.text.strip()
                                if word:
                                    log.info(f"🤖 VIRON: \"{word}\"")
                                    turn_transcript.append(word)
                                    if first_word_time is None:
                                        first_word_time = time.time()
                                    
                                    # Early whiteboard: fire after EARLY_WB_DELAY seconds + enough words
                                    # Uses LOCAL parser only (instant, no API call)
                                    if (not early_wb_fired
                                            and first_word_time is not None
                                            and time.time() - first_word_time >= EARLY_WB_DELAY
                                            and len(turn_transcript) >= EARLY_WB_MIN_WORDS):
                                        early_wb_fired = True
                                        partial_text = " ".join(turn_transcript)
                                        log.info(f"📋 Early whiteboard trigger ({len(turn_transcript)} words, {time.time()-first_word_time:.1f}s)")
                                        threading.Thread(
                                            target=_generate_whiteboard_local,
                                            args=(partial_text,), daemon=True
                                        ).start()

                            # Handle input transcript
                            if sc.input_transcription and sc.input_transcription.text:
                                t = sc.input_transcription.text.strip()
                                if t:
                                    log.info(f"🎤 Student: \"{t}\"")

                            # Handle interruption — STOP audio immediately
                            if sc.interrupted:
                                log.info("⚡ BARGE-IN — stopping audio")
                                is_speaking = False
                                turn_transcript.clear()
                                first_word_time = None
                                early_wb_fired = False
                                _stop_aplay()
                                state.set_status("listening")
                                push_to_ui(emotion="surprised", action="close_whiteboard")

                            # Handle turn complete
                            if sc.turn_complete:
                                if is_speaking:
                                    await asyncio.to_thread(_finish_aplay)
                                is_speaking = False
                                state.set_status("listening")
                                state.last_activity = time.time()
                                
                                # Generate polished whiteboard via Gemini text API
                                # Always runs — replaces early local WB with better version
                                if turn_transcript:
                                    full_text = " ".join(turn_transcript)
                                    log.info(f"📝 Turn done ({len(full_text)} chars)")
                                    threading.Thread(
                                        target=_generate_whiteboard_from_transcript,
                                        args=(full_text,), daemon=True
                                    ).start()
                                
                                # Reset for next turn
                                turn_transcript.clear()
                                first_word_time = None
                                early_wb_fired = False
                                
                                log.info("✅ Turn complete — ready for next question")

                        await asyncio.sleep(0.1)

                    except Exception as e:
                        err_str = str(e).lower()
                        if "closed" in err_str or "1011" in err_str or _stop_session.is_set():
                            log.warning(f"Receive: session closed ({e})")
                            _stop_session.set()
                            return
                        log.warning(f"Receive error: {e}")
                        await asyncio.sleep(0.5)

            # Task 3: Monitor for idle timeout (only when truly idle)
            async def monitor_idle():
                while not _stop_session.is_set():
                    await asyncio.sleep(3)
                    # Don't timeout during playback
                    if _is_playing.is_set():
                        state.last_activity = time.time()
                        continue
                    idle_time = time.time() - state.last_activity
                    if idle_time > IDLE_TIMEOUT and state.status in ("listening", "idle"):
                        log.info(f"⏰ Idle timeout ({IDLE_TIMEOUT}s) — ending session")
                        _stop_session.set()
                        break

            # Run all tasks concurrently
            send_task = asyncio.create_task(send_audio())
            recv_task = asyncio.create_task(receive_responses())
            idle_task = asyncio.create_task(monitor_idle())

            # Wait for any task to finish
            done, pending = await asyncio.wait(
                [send_task, recv_task, idle_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Log which task finished
            for task in done:
                name = "send" if task == send_task else ("recv" if task == recv_task else "idle")
                err = task.exception() if not task.cancelled() else None
                log.info(f"🔌 Task '{name}' finished (error={err})")

            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    except Exception as e:
        log.error(f"❌ Gemini Live session error: {e}")
        state.set_status("error")
        push_to_ui(emotion="worried", text="Υπήρξε ένα πρόβλημα. Δοκίμασε ξανά!")
    finally:
        _session_active.clear()
        state.in_session = False
        state.set_status("idle")
        _stop_aplay()
        log.info("🔌 Gemini Live session ended")


def run_gemini_session(mic: MicStream):
    """Run a Gemini Live session in a new event loop (called from sync context)."""
    _stop_session.clear()
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(gemini_live_session(mic))
    finally:
        loop.close()

# ═══════════════════════════════════════════════════════════
# MAIN LOOP: Wake Word → Gemini Live Session
# ═══════════════════════════════════════════════════════════

def main_loop(mic: MicStream):
    """Main voice assistant loop — Porcupine wake → Gemini Live session."""
    if porcupine is None:
        log.error("❌ Porcupine not available!")
        sys.exit(1)

    log.info("=" * 50)
    log.info("🤖 VIRON Gemini Live Pipeline Active")
    log.info(f"   Wake: Porcupine ('{PORCUPINE_KEYWORD}'), sensitivity={PORCUPINE_SENSITIVITY}")
    log.info(f"   Model: {GEMINI_LIVE_MODEL}")
    log.info(f"   Language: {DEFAULT_LANGUAGE}")
    log.info(f"   Mic: {ALSA_DEVICE}")
    log.info(f"   Mic gain: {MIC_GAIN:.0%}")
    log.info(f"   Early whiteboard: {EARLY_WB_DELAY}s / {EARLY_WB_MIN_WORDS} words")
    log.info(f"   Idle timeout: {IDLE_TIMEOUT}s")
    log.info("=" * 50)
    log.info(f"🎤 Say 'Hey {PORCUPINE_KEYWORD.title()}'...")

    _heartbeat = time.time()
    _frame_count = 0
    _mic_start_time = time.time()

    while True:
        try:
            # Heartbeat every 30s
            if time.time() - _heartbeat > 30:
                log.info(f"💓 Alive — frames={_frame_count} status={state.status}")
                _heartbeat = time.time()
                _frame_count = 0

            # Restart mic every 5 min (Jetson quirk)
            if time.time() - _mic_start_time > 300 and not state.in_session:
                log.info("🔄 Periodic mic restart")
                mic.stop()
                time.sleep(1)
                mic.start()
                _mic_start_time = time.time()
                continue

            # Read mic frame
            frame = mic.read_frame()
            if frame is None:
                log.error("Mic read failed, restarting...")
                mic.stop()
                time.sleep(2)
                mic.start()
                _mic_start_time = time.time()
                continue

            _frame_count += 1

            # Check wake word
            if check_wake(frame):
                log.info("🎯 WAKE WORD DETECTED!")
                state.set_status("listening")
                push_to_ui(emotion="hopeful")
                # Face shows hopeful animation — student knows to speak

                # Start Gemini Live session (blocks until session ends)
                log.info("🌐 Starting Gemini Live session...")
                run_gemini_session(mic)

                # Session ended — RESTART mic
                log.info("🔄 Restarting mic after session...")
                mic.stop()
                time.sleep(0.5)
                mic.start()
                time.sleep(0.5)
                _mic_start_time = time.time()

                # Check if session crashed (error state) — auto-retry once
                if state.status == "error":
                    log.info("🔄 Session crashed — auto-retrying in 2s...")
                    time.sleep(2)
                    state.set_status("listening")
                    push_to_ui(emotion="hopeful")
                    run_gemini_session(mic)
                    mic.stop()
                    time.sleep(0.5)
                    mic.start()
                    time.sleep(0.5)
                    _mic_start_time = time.time()

                state.set_status("idle")
                push_to_ui(emotion="neutral")
                log.info(f"🎤 Back to wake word mode. Say 'Hey {PORCUPINE_KEYWORD.title()}'...")

        except KeyboardInterrupt:
            log.info("👋 Shutting down...")
            break
        except Exception as e:
            log.error(f"Main loop error: {e}")
            state.set_status("idle")
            time.sleep(1)



# ═══════════════════════════════════════════════════════════
# FLASK HTTP API (for browser UI — same endpoints as old pipeline)
# ═══════════════════════════════════════════════════════════

app = Flask(__name__)

@app.route("/wakeword/status", methods=["GET"])
def ww_status():
    return jsonify({
        "running": True,
        "wake_word": PORCUPINE_KEYWORD,
        "sensitivity": PORCUPINE_SENSITIVITY,
        "initialized": porcupine is not None,
    })

@app.route("/wakeword/poll", methods=["GET"])
def ww_poll():
    return jsonify({"wake": False})  # Wake is handled server-side now

@app.route("/pipeline/response", methods=["GET"])
def pipeline_response():
    with _response_lock:
        if not _response_queue:
            return jsonify({"has_response": False})
        
        # Find whiteboard item first (highest priority)
        for i, msg in enumerate(_response_queue):
            if "whiteboard" in msg:
                item = _response_queue.pop(i)
                item["has_response"] = True
                # Also collect latest emotion
                for m in reversed(_response_queue):
                    if m.get("emotion"):
                        item["emotion"] = m["emotion"]
                        break
                return jsonify(item)
        
        # Otherwise return next item, merging consecutive emotion-only items
        msg = _response_queue.pop(0)
        # If this is emotion-only (no text), grab the LATEST emotion and skip rest
        if not msg.get("text"):
            latest_emotion = msg.get("emotion", "")
            while _response_queue and not _response_queue[0].get("text") and "whiteboard" not in _response_queue[0]:
                popped = _response_queue.pop(0)
                if popped.get("emotion"):
                    latest_emotion = popped["emotion"]
            msg["emotion"] = latest_emotion
        
        msg["has_response"] = True
        return jsonify(msg)

@app.route("/pipeline/state", methods=["GET"])
def pipeline_state():
    s = state.status
    return jsonify({
        "status": s,
        "in_session": state.in_session,
        "in_conversation": state.in_session,
        "language": state.language,
        "mode": "gemini_live",
        "model": GEMINI_LIVE_MODEL,
        # Boolean flags the UI expects for face/mouth animation
        "listening": s == "listening",
        "speaking": s == "speaking",
        "audio_playing": s == "speaking",  # Mouth moves while VIRON speaks
        "processing": s == "thinking",
        "music": False,
    })

@app.route("/wakeword/pause", methods=["POST"])
def ww_pause():
    return jsonify({"ok": True})

@app.route("/pipeline/speak", methods=["POST"])
def pipeline_speak():
    data = request.get_json() or {}
    action = data.get("action", "")
    if action == "stop" or action == "interrupt":
        _stop_session.set()
        return jsonify({"ok": True, "action": "stopped"})
    return jsonify({"ok": True})

@app.route("/pipeline/language", methods=["POST", "GET"])
def pipeline_language():
    if request.method == "POST":
        data = request.get_json() or {}
        new_lang = data.get("language", state.language)
        state.language = new_lang
        log.info(f"🌍 Language changed to: {new_lang}")
        return jsonify({"language": new_lang})
    return jsonify({"language": state.language})

@app.route("/wakeword/resume", methods=["POST"])
def ww_resume():
    return jsonify({"ok": True})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "pipeline": "gemini_live",
        "model": GEMINI_LIVE_MODEL,
        "wake_word": porcupine is not None,
        "state": state.status,
    })

def run_http_server():
    app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print()
    print("═" * 50)
    print("  🤖 VIRON Gemini Live Pipeline")
    print("═" * 50)

    # Validate API key
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not set! Add it to .env")
        sys.exit(1)

    # Init wake word
    print("\n📦 Initializing components...")
    has_wake = init_wake()
    print(f"  Wake:     {'✅ Porcupine' if has_wake else '❌ FAILED'}")
    print(f"  Model:    {GEMINI_LIVE_MODEL}")
    print(f"  Language: {DEFAULT_LANGUAGE}")
    print(f"  Mic:      {ALSA_DEVICE}")
    print()

    if not has_wake:
        print("❌ Cannot start without wake word!")
        sys.exit(1)

    # Pre-initialize Gemini client (saves 2s on first wake word)
    _init_genai()

    # Start HTTP API for browser UI
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()
    log.info(f"📡 HTTP API on port {PORT}")

    # Start mic
    mic = MicStream()
    mic.start()
    time.sleep(0.5)

    # Visual-only greeting (no text push - don't block UI queue)
    log.info("🤖 VIRON ready!")
    push_to_ui(emotion="happy")

    try:
        main_loop(mic)
    finally:
        mic.stop()
        if porcupine:
            porcupine.delete()
        log.info("Pipeline stopped.")


if __name__ == "__main__":
    main()
