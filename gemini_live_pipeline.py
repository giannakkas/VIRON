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

LANGUAGE: Speak Greek by default using natural spoken Greek appropriate for children and teenagers.
If the student speaks English, you may switch to English naturally.
ALWAYS respond in the language the student is speaking.

PERSONALITY: Warm, calm, educated gentleman. Best friend who's incredibly smart. Loyal, articulate.
You have your own opinions and personality — NEVER give the same answer twice.
Be creative, spontaneous, and varied in how you respond.

RESPONSE STYLE:
- Simple greetings/chat: MAX 1-2 sentences. Be quick, warm, natural.
- Questions needing explanation: Be detailed but concise. Guide step by step.
- For homework help, guide the student step by step instead of just giving the final answer.
- When appropriate, ask a guiding question before giving the solution.

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
# AUDIO PLAYBACK (write PCM to temp file, play with ffplay)
# ═══════════════════════════════════════════════════════════

def _play_pcm_audio(pcm_data: bytes):
    """Write raw 24kHz PCM audio to a temp file and play with ffplay."""
    if not pcm_data or len(pcm_data) < 100:
        return
    try:
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
            tmp.write(pcm_data)
            tmp_path = tmp.name
        # ffplay can play raw PCM with explicit format
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
             "-f", "s16le", "-ar", str(SAMPLE_RATE_OUT), "-ac", "1", tmp_path],
            timeout=30,
        )
        os.unlink(tmp_path)
    except subprocess.TimeoutExpired:
        log.warning("🔊 Playback timeout")
    except Exception as e:
        log.warning(f"🔊 Playback error: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass

# ═══════════════════════════════════════════════════════════
# GEMINI LIVE SESSION (core of the new architecture)
# ═══════════════════════════════════════════════════════════

_session_active = threading.Event()
_stop_session = threading.Event()

async def gemini_live_session(mic: MicStream):
    """
    Run a single Gemini Live Native Audio session.
    Streams mic audio to Gemini, plays back audio response,
    extracts transcripts for UI subtitles.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(api_version="v1alpha"),
    )

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

            # Task 1: Stream mic audio to Gemini
            async def send_audio():
                CHUNK_BYTES = 4096  # 2048 samples at 16-bit = ~128ms
                while not _stop_session.is_set():
                    raw = await asyncio.to_thread(mic.read_raw, CHUNK_BYTES)
                    if raw and len(raw) > 0:
                        try:
                            await session.send_realtime_input(
                                audio=types.Blob(data=raw, mime_type="audio/pcm;rate=16000")
                            )
                        except Exception as e:
                            if "closed" in str(e).lower():
                                break
                            log.warning(f"Send error: {e}")
                    else:
                        await asyncio.sleep(0.01)

            # Task 2: Receive audio + transcripts from Gemini
            async def receive_responses():
                audio_buffer = bytearray()
                is_speaking = False
                try:
                    async for msg in session.receive():
                        if _stop_session.is_set():
                            break

                        sc = msg.server_content
                        if sc is None:
                            continue

                        # Handle model audio output — buffer chunks
                        if sc.model_turn and sc.model_turn.parts:
                            for part in sc.model_turn.parts:
                                if part.inline_data and part.inline_data.data:
                                    if not is_speaking:
                                        is_speaking = True
                                        state.set_status("speaking")
                                        push_to_ui(emotion="happy")
                                    audio_buffer.extend(part.inline_data.data)

                        # Handle input transcript (what the student said)
                        if sc.input_transcription and sc.input_transcription.text:
                            transcript = sc.input_transcription.text.strip()
                            if transcript:
                                log.info(f"🎤 Student: \"{transcript}\"")
                                push_to_ui(subtitle=transcript)

                        # Handle output transcript (what VIRON said)
                        if sc.output_transcription and sc.output_transcription.text:
                            transcript = sc.output_transcription.text.strip()
                            if transcript:
                                log.info(f"🤖 VIRON: \"{transcript}\"")
                                push_to_ui(text=transcript)

                        # Handle interruption
                        if sc.interrupted:
                            log.info("⚡ Interrupted by student (barge-in)")
                            is_speaking = False
                            audio_buffer.clear()
                            state.set_status("listening")
                            push_to_ui(emotion="surprised")

                        # Handle turn complete — play accumulated audio
                        if sc.turn_complete:
                            if audio_buffer and len(audio_buffer) > 100:
                                log.info(f"🔊 Playing {len(audio_buffer)} bytes of audio...")
                                await asyncio.to_thread(_play_pcm_audio, bytes(audio_buffer))
                                log.info("🔊 Playback done")
                            audio_buffer.clear()
                            is_speaking = False
                            state.set_status("listening")
                            state.last_activity = time.time()
                            log.info("✅ Turn complete — listening for next question...")

                except Exception as e:
                    if "closed" not in str(e).lower():
                        log.error(f"Receive error: {e}")
                finally:
                    # Play any remaining audio
                    if audio_buffer and len(audio_buffer) > 100:
                        await asyncio.to_thread(_play_pcm_audio, bytes(audio_buffer))

            # Task 3: Monitor for idle timeout
            async def monitor_idle():
                while not _stop_session.is_set():
                    await asyncio.sleep(2)
                    idle_time = time.time() - state.last_activity
                    if idle_time > IDLE_TIMEOUT and state.status == "listening":
                        log.info(f"⏰ Idle timeout ({IDLE_TIMEOUT}s) — ending session")
                        _stop_session.set()
                        break

            # Run all tasks concurrently
            send_task = asyncio.create_task(send_audio())
            recv_task = asyncio.create_task(receive_responses())
            idle_task = asyncio.create_task(monitor_idle())

            # Wait for any task to finish (usually idle timeout or stop signal)
            done, pending = await asyncio.wait(
                [send_task, recv_task, idle_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

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
                push_to_ui(emotion="hopeful", text="Ορίστε;")

                # Play acknowledgment
                _play_ack()

                # Start Gemini Live session (blocks until session ends)
                log.info("🌐 Starting Gemini Live session...")
                run_gemini_session(mic)

                # Session ended — flush mic buffer
                log.info("🔄 Flushing mic buffer...")
                for _ in range(20):
                    mic.read_frame()

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


def _play_ack():
    """Play a short acknowledgment sound/TTS when wake word is detected."""
    try:
        # Use edge-tts for a quick "Ορίστε;" acknowledgment
        import tempfile
        ack_text = "Ορίστε;" if state.language == "el" else "Yes?"
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.run(
            ["edge-tts", "--voice", "el-GR-AthinaNeural" if state.language == "el" else "en-US-AriaNeural",
             "--text", ack_text, "--write-media", tmp_path],
            capture_output=True, timeout=5,
        )
        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 500:
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path],
                timeout=5,
            )
        os.unlink(tmp_path)
    except Exception as e:
        log.warning(f"Ack playback failed: {e}")

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
        if _response_queue:
            responses = list(_response_queue)
            _response_queue.clear()
            return jsonify(responses)
    return jsonify([])

@app.route("/pipeline/state", methods=["GET"])
def pipeline_state():
    return jsonify({
        "status": state.status,
        "in_session": state.in_session,
        "language": state.language,
        "mode": "gemini_live",
        "model": GEMINI_LIVE_MODEL,
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

    # Start HTTP API for browser UI
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()
    log.info(f"📡 HTTP API on port {PORT}")

    # Start mic
    mic = MicStream()
    mic.start()
    time.sleep(0.5)

    # Startup greeting
    log.info("🗣️ Playing startup greeting...")
    time.sleep(3)  # Wait for face to load
    push_to_ui(text="Γεια σου! Είμαι ο ΒΙΡΟΝ!", emotion="excited")
    _play_ack_text("Γεια σου! Είμαι ο ΒΙΡΟΝ, ο φίλος σου! Πες Hey Jarvis για να μου μιλήσεις.")

    try:
        main_loop(mic)
    finally:
        mic.stop()
        if porcupine:
            porcupine.delete()
        log.info("Pipeline stopped.")


def _play_ack_text(text):
    """Play TTS for startup greeting."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.run(
            ["edge-tts", "--voice", "el-GR-AthinaNeural", "--text", text, "--write-media", tmp_path],
            capture_output=True, timeout=10,
        )
        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 500:
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path],
                timeout=15,
            )
        os.unlink(tmp_path)
    except Exception as e:
        log.warning(f"Greeting TTS failed: {e}")


if __name__ == "__main__":
    main()
