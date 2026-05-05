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
  pip install google-genai openwakeword numpy flask

Environment:
  GEMINI_API_KEY          — required
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
FRAME_LENGTH = 1280      # OpenWakeWord requires 1280 samples (80ms) at 16kHz

OWW_MODEL_NAME = os.environ.get("VIRON_WAKE_MODEL", "hey_jarvis_v0.1")
OWW_THRESHOLD = float(os.environ.get("VIRON_WAKE_SENSITIVITY", "0.5"))
# Kept for UI/log compatibility:
PORCUPINE_KEYWORD = os.environ.get("VIRON_WAKE_KEYWORD", "jarvis")
PORCUPINE_SENSITIVITY = OWW_THRESHOLD

IDLE_TIMEOUT = float(os.environ.get("VIRON_IDLE_TIMEOUT", "30.0"))
DEFAULT_LANGUAGE = os.environ.get("VIRON_DEFAULT_LANGUAGE", "el")

# Mic software gain (0.0–1.0). Default 0.4 = reduce to 40% to avoid clipping/noise.
MIC_GAIN = float(os.environ.get("VIRON_MIC_GAIN", "0.4"))

# Mic software BOOST applied to the WAKE-WORD DETECTION path only
# (inside read_frame, AFTER stereo->mono extraction). Default 1.0 = no
# change. Use values > 1.0 to compensate for the XMOS XVF3800 DSP's
# internal AGC keeping the beamformed channel (ch1) below useful levels
# for OpenWakeWord — that AGC is firmware-side and not controllable via
# ALSA mixer. Values are multiplied with int16-range clipping protection.
#
# IMPORTANT: this boost does NOT affect audio sent to Gemini. Gemini
# receives the natural AEC-cleaned ch1 via read_raw so its barge-in
# detector and ASR aren't fooled by amplified speaker echo of VIRON's
# own output.
MIC_BOOST = float(os.environ.get("VIRON_MIC_BOOST", "1.0"))

# Early whiteboard: trigger after this many seconds of transcript accumulation
EARLY_WB_DELAY = float(os.environ.get("VIRON_EARLY_WB_DELAY", "4.0"))
EARLY_WB_MIN_WORDS = int(os.environ.get("VIRON_EARLY_WB_MIN_WORDS", "15"))

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
IMPORTANT: When explaining concepts like math, science, or history, give DETAILED step-by-step explanations with numbered steps and worked examples using actual numbers. The student has a display that can show a whiteboard with your steps when you ask for one (see WHITEBOARD section below). Be thorough — include formulas, calculations, and results.
IMPORTANT: IGNORE any background noise from TV, music, or other people talking. Only respond to speech that is clearly directed at you.

LANGUAGE: Speak Greek by default using natural spoken Greek appropriate for children and teenagers.
If the student speaks English, you may switch to English naturally.
ALWAYS respond in the language the student is speaking.

PERSONALITY: Warm, calm, educated gentleman. Best friend who's incredibly smart. Loyal, articulate.
You have your own opinions and personality — NEVER give the same answer twice.
Be creative, spontaneous, and varied in how you respond.

RESPONSE STYLE:
- Simple greetings/chat: MAX 1-2 sentences. Be quick, warm, natural.
- Questions needing explanation: Give a FULL explanation in ONE turn. Use numbered steps with actual numbers and formulas. Keep it under 40 seconds of speech total — be thorough but efficient.
- For homework help, guide the student step by step with worked examples using real numbers.
- When appropriate, ask a guiding question before giving the solution.
- Do NOT break explanations into multiple turns. Give the full explanation in one go.
- IMPORTANT: Keep responses concise to avoid disconnection. Maximum 40 seconds of speech.

EMOTION AWARENESS:
- If the student sounds frustrated, confused, or discouraged, become calmer, slower, and more supportive.
- If the student is doing well, become more energetic and encouraging.
- NEVER mention internal architecture, APIs, hidden reasoning, or implementation details.

SAFETY — NON-NEGOTIABLE:
- Be kid-safe at all times.
- Never generate sexual, abusive, manipulative, or dangerous content.
- Never provide harmful real-world instructions.
- Never be politically manipulative.

MUSIC — You CAN play music! When the student asks for music — by song name, artist, genre, or mood — do BOTH of these:
1. Say a brief, warm acknowledgment in Greek/English (1 short sentence). Examples: "Φυσικά, ορίστε!" or "Sure, here you go!"
2. End your response with a [MUSIC:search query] tag. The tag is silent — the student does NOT hear or see it. It tells VIRON what to look up on YouTube.

EXAMPLES:
- Student: "Παίξε Bohemian Rhapsody" → "Καλή επιλογή! [MUSIC:Bohemian Rhapsody Queen]"
- Student: "play something happy" → "Sure, something cheerful coming up! [MUSIC:happy upbeat pop music]"
- Student: "βάλε ελληνική μουσική" → "Έρχεται! [MUSIC:Greek folk music traditional]"
- Student: "play Despacito" → "Ωραία επιλογή! [MUSIC:Despacito Luis Fonsi]"

STOPPING MUSIC: When the student says "Hey VIRON" while music is playing, the music stops automatically before you hear the request. So if the student then asks you to stop the music (e.g. "σταμάτα τη μουσική", "stop the music", "turn it off"), simply acknowledge briefly: "Έγινε!" or "Done!" — do NOT emit a [MUSIC:...] tag.
NEVER include the [MUSIC:...] tag in non-music conversations. Only when actually STARTING new music playback.

WHITEBOARD — You CAN show a visual whiteboard for teaching content. Use it ONLY when actually teaching or explaining structured material.

For teaching responses, BEGIN your response with a [BOARD:title] tag — write it FIRST, before any other words. The tag is a silent control marker that opens a study sheet panel immediately while you explain. The tag itself is NOT spoken aloud and is invisible to the student. After the tag, deliver your full explanation as normal.

USE the [BOARD:...] tag when:
- Teaching math, physics, chemistry, or science with formulas/steps
- Explaining a concept that has stages, parts, or a process (e.g. water cycle, photosynthesis, how an engine works)
- Working through a problem step-by-step
- Listing key points the student should memorize for school

DO NOT use [BOARD:...] for:
- Casual conversation, greetings, jokes
- Yes/no answers, single-fact answers ("What's the capital of France?" → just "Paris", no board)
- Weather, news, sports scores, current events
- Music requests (use [MUSIC:...] only)
- Emotional support or personal chat
- Stories or creative writing

EXAMPLES (notice: [BOARD:title] FIRST, then explanation):
- Student: "How do I solve 2x + 5 = 11?" → "[BOARD:Λύση εξίσωσης] Λοιπόν, για να λύσουμε αυτή την εξίσωση, ξεκινάμε αφαιρώντας 5 από τις δύο πλευρές..."
- Student: "Πες μου για τη φωτοσύνθεση" → "[BOARD:Φωτοσύνθεση] Η φωτοσύνθεση είναι η διαδικασία με την οποία τα φυτά..."
- Student: "Τι καιρός έχει σήμερα;" → answer the weather, NO BOARD
- Student: "Πες μου ένα αστείο" → tell joke, NO BOARD
- Student: "Γεια σου" → "Γεια σου!" — NO BOARD

CURRENT EVENTS & FRESH INFORMATION — You have a Google Search tool available. USE IT for anything that needs up-to-date information:
- News (today's headlines, recent events)
- Weather (current conditions, forecast)
- Sports scores and results
- Current prices, exchange rates
- Recently released movies, books, products
- Anything where the answer changes over time

Process: when the student asks one of these, use the search tool, then give a clear, brief answer in their language. Don't say "I don't have access to the internet" — you DO have search; use it. Don't read URLs out loud — just summarize the answer naturally.

CAMERA / VISION — Sometimes you receive a webcam snapshot of the student at the start of a turn. The image is for YOUR awareness only — use it to shape HOW you respond, never to describe what you see.

IDENTITY — Sometimes you also receive a "LIVE CONTEXT" block at the bottom of these instructions that tells you which family members are visible right now (e.g. "Visible in camera right now: Andreas (student)."). When this is present:
- Greet recognized people warmly BY NAME ("Γεια σου Ανδρέα!")
- The person tagged as "student" is your primary tutee — focus your teaching on them by name
- Parents/siblings present in frame are observers — be friendly to them but don't redirect lessons toward them unless they speak up
- If the LIVE CONTEXT says no registered family member is visible, be slightly more guarded — avoid using personal details, keep responses helpful but generic, never bring up past lessons or memories that belong to the student
- NEVER announce that you "recognized" someone or explain how you knew their name — just use it naturally, the way a person who knows them would

USE the visual context to:
- Match your tone to mood: warmer if the student looks tired or sad, more energetic if they look engaged
- Notice attention: if the student is looking away or distracted, you can gently check in ("Είσαι μαζί μου;" / "Are you still there?")
- Pick up on confusion or frustration during a lesson — slow down, simplify, encourage

NEVER do any of these — they would be creepy and break trust:
- Describe the student's appearance ("I see you have...", "you're wearing...", "your hair...")
- Comment on what they're wearing, their face, their body
- Mention other people, pets, or objects in the room unprompted
- Say anything that makes the student feel watched or surveilled
- Describe surroundings ("I see you're in the kitchen", etc.)

The visual is invisible context. Let it shape WHAT TONE you take, not WHAT you say. If there's no image (camera off, dark room, etc.), just respond normally based on audio.

CONVERSATION CONTINUITY: If the conversation has prior history visible above (i.e. you're resuming where you left off — not just starting fresh), do NOT greet the student again or say "Γεια σου". Just continue naturally as if the conversation never paused. Only greet on a truly fresh start (when there's no prior conversation history).
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

oww_model = None
porcupine = None  # Kept as alias for backward compat in main() shutdown

def init_wake():
    """Initialize OpenWakeWord. Returns True on success."""
    global oww_model, porcupine
    try:
        from openwakeword.model import Model
        oww_model = Model(
            wakeword_models=[OWW_MODEL_NAME],
            inference_framework="onnx",
        )
        porcupine = oww_model  # alias so cleanup code still works
        log.info(f"✅ OpenWakeWord ready: '{OWW_MODEL_NAME}', threshold={OWW_THRESHOLD}")
        return True
    except Exception as e:
        log.error(f"❌ OpenWakeWord init failed: {e}")
        return False

def check_wake(audio_int16):
    """Check one frame for wake word. Returns True if detected."""
    if oww_model is None:
        return False
    try:
        scores = oww_model.predict(audio_int16)
        max_score = max(scores.values()) if scores else 0.0

        # DEBUG: log audio level and OWW score periodically + on near-hits
        check_wake._counter = getattr(check_wake, "_counter", 0) + 1
        try:
            audio_max = int(abs(audio_int16).max())
        except Exception:
            audio_max = 0
        if check_wake._counter % 60 == 0 or max_score >= 0.2:
            log.info(f"🎤 audio_max={audio_max} oww_score={max_score:.3f} (threshold={OWW_THRESHOLD})")

        for name, score in scores.items():
            if score >= OWW_THRESHOLD:
                # Reset model state to prevent immediate re-trigger
                oww_model.reset()
                return True
        return False
    except Exception as e:
        log.error(f"check_wake error: {e}")
        return False

# ═══════════════════════════════════════════════════════════
# MIC CAPTURE (arecord subprocess — reliable on Jetson)
# ═══════════════════════════════════════════════════════════

class MicStream:
    def __init__(self):
        self.proc = None
        self.running = False
        # The ReSpeaker XVF3800 is a native-stereo device: channel 0 is the raw
        # mic feed, channel 1 is the on-board beamformed/AEC-processed output.
        # ALSA's plug plugin is supposed to downmix stereo->mono when we ask
        # for `-c 1`, but for THIS device the conversion silently produces a
        # constant value (audio_max=1088 stuck) — likely because it picks the
        # wrong channel or hits a format mismatch with the XMOS USB descriptor.
        # Workaround: record stereo and extract one channel ourselves.
        self.mic_stereo = os.environ.get("VIRON_MIC_STEREO", "1") == "1"
        # Which channel to keep when stereo: 0 = raw mic, 1 = beamformed.
        self.mic_channel = int(os.environ.get("VIRON_MIC_CHANNEL", "1"))
        self._stereo_buf = b""  # leftover bytes from the previous read

    def start(self):
        if self.mic_stereo:
            cmd = ["arecord", "-D", ALSA_DEVICE, "-f", "S16_LE",
                   "-r", str(SAMPLE_RATE_IN), "-c", "2", "-t", "raw"]
            log.info(f"🎤 Mic started: {ALSA_DEVICE} (stereo {SAMPLE_RATE_IN}Hz, "
                     f"extracting ch{self.mic_channel})")
        else:
            cmd = ["arecord", "-D", ALSA_DEVICE, "-f", "S16_LE",
                   "-r", str(SAMPLE_RATE_IN), "-c", "1", "-t", "raw"]
            log.info(f"🎤 Mic started: {ALSA_DEVICE} (mono {SAMPLE_RATE_IN}Hz)")
        # stderr=DEVNULL: arecord writes verbose status messages to stderr that
        # we never read; if it's a PIPE the buffer eventually fills and arecord
        # blocks, causing the mic to feed identical buffers repeatedly.
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.running = True
        self._stereo_buf = b""

    def _extract_mono(self, stereo_bytes: bytes) -> bytes:
        """Take interleaved S16_LE stereo (LRLRLR...) and return mono bytes
        for self.mic_channel. Each sample is 2 bytes; one stereo frame is 4
        bytes. We slice every other 16-bit word starting at the right offset.

        NO software boost is applied here. Both wake-detect (read_frame)
        and Gemini-send (read_raw) call this for stereo->mono conversion,
        but only read_frame applies MIC_BOOST afterwards. Gemini receives
        natural-level AEC-cleaned audio so its barge-in detector and ASR
        aren't fooled by amplified speaker echo."""
        # Trim to a whole number of stereo frames (4 bytes each)
        usable = len(stereo_bytes) - (len(stereo_bytes) % 4)
        if usable <= 0:
            return b""
        arr = np.frombuffer(stereo_bytes[:usable], dtype=np.int16)
        # arr is [L0, R0, L1, R1, L2, R2, ...]. Take every other starting from
        # the chosen channel offset.
        mono = arr[self.mic_channel::2]
        return mono.tobytes()

    def read_frame(self, frame_length=FRAME_LENGTH):
        """Read one frame of mono int16 samples for WAKE WORD DETECTION.

        Applies MIC_BOOST so the local OpenWakeWord model can pick up
        quiet 'Hey Jarvis' utterances even though the XMOS DSP's internal
        AGC keeps the beamformed channel naturally quiet. Audio sent to
        Gemini goes through read_raw and is intentionally NOT boosted —
        otherwise the boosted echo of VIRON's own speaker output would
        false-trigger Gemini's barge-in detector and corrupt ASR."""
        if not self.proc:
            return None
        if self.mic_stereo:
            # We need `frame_length` mono samples, so read 2x the bytes.
            bytes_needed = frame_length * 2 * 2  # mono samples * 2 bytes/sample * 2 channels
            raw = self.proc.stdout.read(bytes_needed)
            if len(raw) < bytes_needed:
                return None
            mono_bytes = self._extract_mono(raw)
            samples = np.frombuffer(mono_bytes, dtype=np.int16)
        else:
            bytes_needed = frame_length * 2
            raw = self.proc.stdout.read(bytes_needed)
            if len(raw) < bytes_needed:
                return None
            samples = np.frombuffer(raw, dtype=np.int16)
        if MIC_BOOST != 1.0:
            boosted = samples.astype(np.float32) * MIC_BOOST
            np.clip(boosted, -32768, 32767, out=boosted)
            samples = boosted.astype(np.int16)
        return samples

    def read_raw(self, num_bytes):
        """Read num_bytes of MONO PCM (what Gemini Live expects). When
        mic_stereo=True we read 2x bytes from the device, downmix, and
        cache any leftover bytes for the next call so we never desync."""
        if not self.proc:
            return None
        if not self.mic_stereo:
            return self.proc.stdout.read(num_bytes)

        # Stereo path: keep reading + downmixing until we have num_bytes
        # of mono. Stash any half-frame leftover so we don't lose the
        # right channel of a partial stereo frame across calls.
        out = bytearray()
        # We need num_bytes mono → 2*num_bytes stereo, but we may have
        # leftover stereo bytes from last call.
        while len(out) < num_bytes:
            need_mono = num_bytes - len(out)
            need_stereo = need_mono * 2 - len(self._stereo_buf)
            if need_stereo > 0:
                chunk = self.proc.stdout.read(need_stereo)
                if not chunk:
                    return None
                self._stereo_buf += chunk
            # Convert as much as we can in whole stereo frames
            usable = len(self._stereo_buf) - (len(self._stereo_buf) % 4)
            mono = self._extract_mono(self._stereo_buf[:usable])
            self._stereo_buf = self._stereo_buf[usable:]
            out += mono
        # Trim if we overshot (we won't — mono is exactly half of stereo)
        return bytes(out[:num_bytes])

    def stop(self):
        """Terminate arecord and REAP it. Critical: must wait() after kill() or
        we leak a zombie that holds /dev/snd/pcmC0D0c forever."""
        self.running = False
        proc = self.proc
        self.proc = None
        if not proc:
            return
        # Close stdout pipe to unblock any pending read
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
        # Try graceful then forceful termination — always wait() at the end
        try:
            proc.terminate()
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=1)  # ← this reap is what was missing
            except Exception:
                pass
        except Exception:
            pass

# ═══════════════════════════════════════════════════════════
# CAMERA CAPTURE (USB webcam via OpenCV)
# ═══════════════════════════════════════════════════════════
# Background thread continuously grabs frames at low FPS so a snapshot
# is always immediately available with no warmup delay. JPEG is encoded
# once per capture, kept in a single-buffer in-memory store. No frames
# are ever written to disk.

CAMERA_ENABLED = os.environ.get("VIRON_CAMERA_ENABLED", "1") == "1"
CAMERA_DEVICE_ENV = os.environ.get("VIRON_CAMERA_DEVICE", "")  # e.g. "0" or "/dev/video0"
CAMERA_WIDTH = int(os.environ.get("VIRON_CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.environ.get("VIRON_CAMERA_HEIGHT", "480"))
CAMERA_FPS = int(os.environ.get("VIRON_CAMERA_FPS", "24"))   # capture rate (preview only)
CAMERA_JPEG_QUALITY = int(os.environ.get("VIRON_CAMERA_JPEG_QUALITY", "70"))

class CameraStream:
    def __init__(self):
        self.cap = None
        self.latest_jpeg = None     # bytes, latest encoded frame
        self.latest_ts = 0.0        # time.time() of latest frame
        self._lock = threading.Lock()
        self._thread = None
        self._running = False
        self._cv2 = None
        self._device_used = None
        self._enabled = CAMERA_ENABLED   # runtime toggle

    def _try_import_cv2(self):
        if self._cv2 is not None:
            return True
        try:
            import cv2  # type: ignore
            self._cv2 = cv2
            return True
        except ImportError:
            log.warning("📷 OpenCV (cv2) not installed — camera disabled. "
                        "Install: sudo apt install -y python3-opencv")
            return False

    def _autodetect_device(self):
        """Find the first /dev/videoN that opens AND returns a real frame."""
        cv2 = self._cv2
        # If user pinned a device via env, use only that
        if CAMERA_DEVICE_ENV:
            try:
                idx = int(CAMERA_DEVICE_ENV)
                return [idx]
            except ValueError:
                return [CAMERA_DEVICE_ENV]
        return [0, 1, 2, 3]  # try /dev/video0..3

    def start(self):
        if not self._enabled:
            log.info("📷 Camera disabled (VIRON_CAMERA_ENABLED=0)")
            return False
        if not self._try_import_cv2():
            return False
        cv2 = self._cv2
        for dev in self._autodetect_device():
            try:
                # Explicitly request V4L2 backend. JetPack-shipped OpenCV
                # builds include the OBSensor (Orbbec) backend, which gets
                # picked first by VideoCapture(idx) and then fails with
                # "Camera index out of range" because the Brio is a UVC
                # webcam, not a depth camera. cv2.CAP_V4L2 = 200.
                cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
                if not cap.isOpened():
                    # Fall back to default backend in case V4L2 isn't built
                    # into THIS opencv (rare but cheap to try)
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = cv2.VideoCapture(dev)
                if not cap.isOpened():
                    cap.release()
                    continue
                # CRITICAL for the Brio (and most modern UVC webcams):
                # Force MJPG codec. The OpenCV default on Linux is YUYV
                # (uncompressed) which crushes USB bandwidth — at 1280x720
                # YUYV the cam will only do ~10fps because the bus is full.
                # Brio outputs MJPG natively at much higher framerates;
                # OpenCV decodes it on the fly.
                try:
                    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                except Exception:
                    pass
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
                # CRITICAL: keep only the latest frame in the driver buffer.
                # Without this, the V4L2 backend buffers ~5 frames and read()
                # walks through them oldest-first — preview lags behind reality
                # by hundreds of ms and updates feel choppy because we're
                # always working through a backlog.
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                # First frame may be black/empty — grab a few to verify
                ok = False
                for _ in range(5):
                    r, f = cap.read()
                    if r and f is not None and f.size > 0:
                        ok = True
                        break
                if not ok:
                    cap.release()
                    continue
                self.cap = cap
                self._device_used = dev
                # Log what the driver actually negotiated (might differ from request)
                actual_fps = cap.get(cv2.CAP_PROP_FPS) or CAMERA_FPS
                aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or CAMERA_WIDTH)
                ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or CAMERA_HEIGHT)
                log.info(f"📷 Camera negotiated: {aw}x{ah} @ {actual_fps:.0f}fps (requested {CAMERA_FPS}fps)")
                break
            except Exception as e:
                log.warning(f"📷 Could not probe camera {dev}: {e}")
        if self.cap is None:
            log.warning("📷 No working camera found — VIRON will run without vision")
            return False

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        log.info(f"📷 Camera ready: device={self._device_used}, "
                 f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}, ~{CAMERA_FPS}fps")
        return True

    def _capture_loop(self):
        cv2 = self._cv2
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), CAMERA_JPEG_QUALITY]
        # NOTE: no time.sleep() throttle here. cap.read() blocks until the
        # camera delivers the next frame, which is paced by the driver at
        # the negotiated FPS. Adding our own sleep just stacks on top of
        # that and means we're always one cycle behind reality.
        while self._running:
            try:
                if not self._enabled:
                    time.sleep(0.2)
                    continue
                ret, frame = self.cap.read()
                if not ret or frame is None or frame.size == 0:
                    time.sleep(0.05)
                    continue
                success, buf = cv2.imencode('.jpg', frame, encode_params)
                if success:
                    with self._lock:
                        self.latest_jpeg = buf.tobytes()
                        self.latest_ts = time.time()
            except Exception as e:
                log.warning(f"📷 Capture error: {e}")
                time.sleep(0.5)

    def get_jpeg(self, max_age_sec: float = 5.0):
        """Return (bytes, age_seconds) for the latest frame, or (None, None)
        if no frame has been captured or it's stale."""
        with self._lock:
            if not self.latest_jpeg:
                return None, None
            age = time.time() - self.latest_ts
            if age > max_age_sec:
                return None, age
            return self.latest_jpeg, age

    def set_enabled(self, enabled: bool):
        """Runtime camera mute. Capture loop pauses, latest frame is cleared."""
        self._enabled = bool(enabled)
        if not self._enabled:
            with self._lock:
                self.latest_jpeg = None
                self.latest_ts = 0.0
            log.info("📷 Camera muted (frames cleared)")
        else:
            log.info("📷 Camera unmuted")

    def stop(self):
        self._running = False
        if self._thread:
            try:
                self._thread.join(timeout=2)
            except Exception:
                pass
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        log.info("📷 Camera stopped")


camera = CameraStream()


# ═══════════════════════════════════════════════════════════
# FACE RECOGNITION (multi-person family identification)
# ═══════════════════════════════════════════════════════════
# Optional. If `face_recognition` (dlib) isn't installed, the engine
# silently disables itself and the rest of the pipeline keeps working.
# Enrolled faces persist in ~/VIRON/faces.db (next to this script).

from face_db import FaceDB
from face_engine import FaceEngine

_FACE_DB_PATH = os.environ.get(
    "VIRON_FACES_DB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces.db"),
)
face_db = FaceDB(_FACE_DB_PATH)
face_engine = FaceEngine(face_db)

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
# MUSIC PLAYBACK (mpv + yt-dlp for YouTube streaming)
# ═══════════════════════════════════════════════════════════
# Requires:  sudo apt install mpv yt-dlp
#
# VIRON emits a [MUSIC:query] tag at end of response to request playback.
# We parse the tag from output transcript at turn_complete and stream
# the matching YouTube audio. New wake word stops playback.

import re as _re_music
import json as _json_music
import socket as _socket_music

_music_proc = None
_music_lock = threading.Lock()
_music_query = ""
_music_paused = False
MPV_IPC_SOCKET = "/tmp/viron-mpv.sock"

def _mpv_ipc(*command_args):
    """Send a JSON command to mpv via Unix socket. Returns True on success."""
    if not os.path.exists(MPV_IPC_SOCKET):
        return False
    payload = (_json_music.dumps({"command": list(command_args)}) + "\n").encode()
    try:
        s = _socket_music.socket(_socket_music.AF_UNIX, _socket_music.SOCK_STREAM)
        s.settimeout(0.5)
        s.connect(MPV_IPC_SOCKET)
        s.sendall(payload)
        s.close()
        return True
    except Exception as e:
        log.warning(f"mpv IPC failed: {e}")
        return False

def _push_music_state(playing: bool, title: str = "", paused: bool = False):
    """Notify the UI about music playback state."""
    try:
        push_to_ui(music={"playing": playing, "title": title, "paused": paused})
    except Exception as e:
        log.warning(f"Music UI push failed: {e}")

def _duck_music(should_duck: bool):
    """Lower mpv volume while VIRON is speaking, restore when done.
    No-op if no music is playing."""
    if _music_proc is None or _music_proc.poll() is not None:
        return
    target_vol = 25 if should_duck else 100
    _mpv_ipc("set_property", "volume", target_vol)

def _stop_music():
    """Terminate any currently playing music. Safe to call repeatedly."""
    global _music_proc, _music_query, _music_paused
    was_playing = False
    with _music_lock:
        if _music_proc and _music_proc.poll() is None:
            log.info("🎵 Stopping music")
            was_playing = True
            try:
                _music_proc.terminate()
                _music_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                _music_proc.kill()
            except Exception:
                pass
        _music_proc = None
        _music_query = ""
        _music_paused = False
        # Clean up stale socket
        try:
            if os.path.exists(MPV_IPC_SOCKET):
                os.unlink(MPV_IPC_SOCKET)
        except Exception:
            pass
    if was_playing:
        _push_music_state(False)

def _pause_music() -> bool:
    """Toggle pause/resume on the running mpv process."""
    global _music_paused
    if _music_proc is None or _music_proc.poll() is not None:
        return False
    new_state = not _music_paused
    if _mpv_ipc("set_property", "pause", new_state):
        _music_paused = new_state
        log.info(f"🎵 Music {'paused' if _music_paused else 'resumed'}")
        _push_music_state(True, _music_query, _music_paused)
        return True
    return False

def _play_music(query: str):
    """Search YouTube and stream audio. Background, non-blocking."""
    global _music_proc, _music_query, _music_paused
    _stop_music()
    cmd = [
        "mpv",
        "--no-video",
        "--no-terminal",
        "--audio-display=no",
        "--really-quiet",
        f"--input-ipc-server={MPV_IPC_SOCKET}",
        f"ytdl://ytsearch1:{query}",
    ]
    try:
        with _music_lock:
            _music_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            _music_query = query
            _music_paused = False
            this_proc = _music_proc
        log.info(f"🎵 Playing: '{query}' (PID {this_proc.pid})")
        _push_music_state(True, query, False)

        # Watch for natural end-of-song to clear the UI bar.
        def _watch_end():
            try:
                this_proc.wait()
            except Exception:
                return
            # Only clear if THIS playback is still the active one
            with _music_lock:
                if _music_proc is this_proc:
                    log.info("🎵 Music ended naturally")
                    _push_music_state(False)
        threading.Thread(target=_watch_end, daemon=True).start()
    except FileNotFoundError:
        log.error("❌ mpv not installed — run: sudo apt install mpv yt-dlp")
    except Exception as e:
        log.error(f"🎵 Music playback failed: {e}")

# Pattern: [MUSIC:query] anywhere in transcript. Tolerant to spacing/case.
_MUSIC_TAG_RE = _re_music.compile(r'\[\s*MUSIC\s*:\s*([^\]]+?)\s*\]', _re_music.IGNORECASE)
# Pattern: [BOARD:title] — VIRON emits this only when the response is teaching content.
# When this tag is present we generate a whiteboard; when absent, no board.
_BOARD_TAG_RE = _re_music.compile(r'\[\s*BOARD\s*:\s*([^\]]+?)\s*\]', _re_music.IGNORECASE)

def _extract_music_query(transcript: str):
    """Return the music search query if [MUSIC:...] tag is present, else None."""
    if not transcript:
        return None
    m = _MUSIC_TAG_RE.search(transcript)
    return m.group(1).strip() if m else None

def _extract_board_title(transcript: str):
    """Return the whiteboard title if [BOARD:...] tag is present, else None."""
    if not transcript:
        return None
    m = _BOARD_TAG_RE.search(transcript)
    return m.group(1).strip() if m else None

def _strip_control_tags(transcript: str) -> str:
    """Remove [MUSIC:...] and [BOARD:...] tags from text. Used before sending
    transcripts to the whiteboard generator and before saving to history."""
    if not transcript:
        return ""
    cleaned = _MUSIC_TAG_RE.sub("", transcript)
    cleaned = _BOARD_TAG_RE.sub("", cleaned)
    return cleaned.strip()

# ═══════════════════════════════════════════════════════════
# GEMINI LIVE SESSION (core of the new architecture)
# ═══════════════════════════════════════════════════════════

_session_active = threading.Event()
_stop_session = threading.Event()

# ═══════════════════════════════════════════════════════════
# CONVERSATION HISTORY (across sessions, 5-min window)
# ═══════════════════════════════════════════════════════════
# Keep recent turns so a follow-up "Hey Jarvis" within 5 minutes
# resumes context instead of restarting from a greeting.

_conv_history = []  # list of (role, text): role = "user" | "model"
_last_session_end_time = 0.0
_history_lock = threading.Lock()
HISTORY_WINDOW_SEC = 300.0   # 5 minutes
MAX_HISTORY_TURNS = 30        # cap to avoid bloat (~15 exchanges)

def _append_history(role: str, text: str):
    """Add a turn to conversation history. Thread-safe."""
    if not text or not text.strip():
        return
    with _history_lock:
        _conv_history.append((role, text.strip()))
        if len(_conv_history) > MAX_HISTORY_TURNS:
            del _conv_history[: len(_conv_history) - MAX_HISTORY_TURNS]

def _should_resume_history() -> bool:
    """True if we have recent history within the resume window."""
    with _history_lock:
        if not _conv_history:
            return False
        return (time.time() - _last_session_end_time) < HISTORY_WINDOW_SEC

def _mark_session_ended():
    global _last_session_end_time
    _last_session_end_time = time.time()

def _clear_history(reason: str = ""):
    with _history_lock:
        if _conv_history:
            log.info(f"🗑️  Cleared {len(_conv_history)} history turns ({reason})")
            _conv_history.clear()

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
_api_backoff_until = 0.0  # After 429, skip API calls for 120s


def _smart_parse_transcript(transcript: str):
    """Smart parser for Gemini Live transcripts.
    Detects numbered steps, math ($...$), bold headers (**...**), bullets.
    Works on RAW transcript (before bold/LaTeX stripping).
    Returns (title, steps) or (None, None)."""
    
    text = transcript.strip()
    if len(text) < 30:
        return None, None
    
    # Try splitting on numbered step markers: "1." "2." "3." 
    # Only match when preceded by sentence end (. ! ;) or start of text
    numbered = _re.split(r'(?:^|[.!;)°]\s+)(\d+)\.\s+', text)
    
    if len(numbered) >= 4:  # "intro", "1", "content1", "2", "content2"...
        raw_title = _clean_math_text(numbered[0].strip()) if numbered[0].strip() else "Εξήγηση"
        # Shorten: first sentence or 40 chars
        tm = _re.match(r'^(.{8,40}?)[.!;]', raw_title)
        title = (tm.group(1) if tm else raw_title[:40]).strip() or "Εξήγηση"
        steps = []
        # numbered = [intro, "1", content, "2", content, "3", content...]
        i = 1
        while i < len(numbered) - 1:
            step_num = numbered[i]
            step_content = _clean_math_text(numbered[i+1].strip())
            if step_content:
                # Detect math
                has_math = bool(_re.search(r'\d\s*[=+\-×÷]\s*\d|²|³|√|α²|β²|γ²', step_content))
                is_result = bool(_re.search(r'(?:αποτέλεσμα|result|answer|λύση|άρα|therefore)', step_content.lower()))
                
                if is_result:
                    steps.append({"result": step_content})
                elif has_math:
                    steps.append({"label": f"Βήμα {step_num}", "math": step_content})
                else:
                    steps.append({"label": f"Βήμα {step_num}", "text": step_content})
            i += 2
        if steps:
            return title, steps[:10]
    
    # Try splitting on bold markers: **Title**: content  
    bold_parts = _re.split(r'\*\*([^*]+)\*\*:?\s*', text)
    if len(bold_parts) >= 3:
        title = _clean_math_text(bold_parts[0].strip()) if bold_parts[0].strip() else _clean_math_text(bold_parts[1])
        title = title[:60] if title else "Εξήγηση"
        steps = []
        i = 1
        while i < len(bold_parts) - 1:
            label = _clean_math_text(bold_parts[i].strip())
            content = _clean_math_text(bold_parts[i+1].strip()) if i+1 < len(bold_parts) else ""
            if content:
                has_math = bool(_re.search(r'\d\s*[=+\-×÷]\s*\d|²|³|√|α²|β²|γ²', content))
                if has_math:
                    steps.append({"label": label, "math": content})
                else:
                    steps.append({"label": label, "text": content})
            elif label:
                steps.append({"text": label})
            i += 2
        if steps:
            return title, steps[:10]
    
    # Last resort: chunk cleaned text by ~7 words
    clean = _clean_math_text(text)
    words = clean.split()
    if len(words) < 6:
        return None, None
    
    title = " ".join(words[:4])
    remaining = words[4:]
    steps = []
    for i in range(0, len(remaining), 7):
        chunk = " ".join(remaining[i:i+7])
        if _re.search(r'\d.*[=+\-]|²|³|√|α²|β²|γ²', chunk):
            steps.append({"math": chunk})
        else:
            steps.append({"text": chunk})
    return title, steps[:10] if steps else (None, None)

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
    global _last_wb_content, _last_wb_time, _api_backoff_until
    
    if not transcript or len(transcript) < 40:
        return
    
    # Rate limit: minimum 10s between API calls
    now = time.time()
    if now - _last_wb_time < 10.0:
        log.info(f"📋 Whiteboard skipped (rate limit, {now - _last_wb_time:.0f}s since last)")
        return
    _last_wb_time = now
    
    # Quick check for educational content
    # NOTE: removed the old keyword filter (θεώρημα/εξίσωση/βήμα/etc) — it
    # only matched math content and silently rejected biology/history/language
    # lessons even when VIRON had explicitly tagged the response with [BOARD:].
    # If the function is being called, VIRON already decided this is teaching
    # material; trust that signal and generate the sheet.
    
    # Avoid duplicate
    content_hash = transcript[:100]
    if not skip_dup_check and content_hash == _last_wb_content:
        return
    _last_wb_content = content_hash

    # Clean the transcript
    clean = _clean_math_text(transcript)
    
    # Try Gemini text API ONLY if not in 429 backoff
    if now > _api_backoff_until:
        try:
            if _genai_client:
                # Detect language to instruct the model in the right tongue
                is_greek = bool(_re.search(r'[Α-Ωα-ωάέήίόύώϊϋΐΰ]', clean[:400]))
                lang_instr = "Greek (Ελληνικά)" if is_greek else "the same language as the input"

                prompt = f"""You are a teacher's assistant building a STUDY SHEET for a student.
The teacher just gave a spoken explanation; the student is listening and ALSO needs a visual reference on a whiteboard.

Your job: extract the KEY CONCEPTS from the spoken explanation and present them as a structured study sheet — NOT a transcript. The student already heard the teacher; the board complements the audio with visual structure.

Respond in: {lang_instr}.
Output STRICT JSON, no markdown fences, no commentary:

{{
  "title": "Short topic title (max 40 chars)",
  "definition": "ONE clear sentence defining the concept (or empty string if N/A)",
  "formula": "The MAIN formula/equation in clean Unicode (e.g. 'α² + β² = γ²', 'F = m·a'). Empty string if not applicable.",
  "example": {{
    "problem": "Concrete worked-example problem with real numbers (e.g. 'a=3, b=4, c=?')",
    "steps": [
      {{"label": "Step 1", "math": "3² + 4² = c²"}},
      {{"label": "Step 2", "math": "9 + 16 = c²"}}
    ],
    "result": "Final numeric answer (e.g. 'c = 5')"
  }},
  "key_points": [
    "Short bullet takeaway 1",
    "Short bullet takeaway 2",
    "Short bullet takeaway 3"
  ]
}}

Rules:
- DO NOT transcribe the explanation. EXTRACT and STRUCTURE.
- Use Unicode math only: ² ³ √ π α β γ θ ° → ≈ ≠ ≤ ≥ × ÷. NEVER LaTeX.
- The "example" block must use ACTUAL numbers a student can verify, not letters or placeholders.
- 3-5 key_points maximum, each under 12 words.
- If a field doesn't apply (e.g. no formula for a history lesson), use empty string "" or empty array [].
- For history/language/social-studies topics: skip "formula" and "example", focus on definition + key_points.

Spoken explanation:
{clean[:1200]}"""

                response = _genai_client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt,
                )
                
                resp_text = response.text.strip()
                resp_text = _re.sub(r'^```json\s*', '', resp_text)
                resp_text = _re.sub(r'\s*```$', '', resp_text)
                
                data = json.loads(resp_text)
                title = (data.get("title") or "Εξήγηση").strip()
                definition = (data.get("definition") or "").strip()
                formula = (data.get("formula") or "").strip()
                example = data.get("example") or {}
                key_points = [p.strip() for p in (data.get("key_points") or []) if p and p.strip()]

                # Sanitize example
                example_clean = None
                if isinstance(example, dict):
                    ex_problem = (example.get("problem") or "").strip()
                    ex_result = (example.get("result") or "").strip()
                    ex_raw_steps = example.get("steps") or []
                    ex_steps = []
                    for s in ex_raw_steps[:6]:
                        if not isinstance(s, dict):
                            continue
                        label = (s.get("label") or "").strip()
                        math = (s.get("math") or s.get("content") or "").strip()
                        if math:
                            ex_steps.append({"label": label, "math": _clean_math_text(math)})
                    if ex_problem or ex_steps or ex_result:
                        example_clean = {
                            "problem": ex_problem,
                            "steps": ex_steps,
                            "result": ex_result,
                        }

                # Build legacy-compatible "steps" array so older frontends still
                # show something even if they don't know about the new fields.
                legacy_steps = []
                if definition:
                    legacy_steps.append({"text": definition})
                if formula:
                    legacy_steps.append({"math": _clean_math_text(formula)})
                if example_clean:
                    if example_clean["problem"]:
                        legacy_steps.append({"label": "Παράδειγμα" if is_greek else "Example",
                                            "text": example_clean["problem"]})
                    for st in example_clean["steps"]:
                        legacy_steps.append({"math": st["math"]})
                    if example_clean["result"]:
                        legacy_steps.append({"result": example_clean["result"]})
                for kp in key_points:
                    legacy_steps.append({"text": "• " + kp})

                if not legacy_steps:
                    log.info("📋 Whiteboard skipped — model returned no usable content")
                    return

                log.info(f"📋 Whiteboard (study sheet): \"{title}\" "
                         f"def={'Y' if definition else 'N'} "
                         f"formula={'Y' if formula else 'N'} "
                         f"example={'Y' if example_clean else 'N'} "
                         f"points={len(key_points)}")
                push_to_ui(text="📋", emotion="thinking",
                           whiteboard={
                               "title": title,
                               "definition": definition,
                               "formula": formula,
                               "example": example_clean,
                               "key_points": key_points,
                               "steps": legacy_steps,  # backward compat
                           })
                return
        except Exception as e:
            err_msg = str(e)
            log.warning(f"Whiteboard Gemini call failed: {e}")
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                _api_backoff_until = time.time() + 120  # Skip API for 2 min
                log.info("📋 API 429 — backing off for 120s, using local parser")
    else:
        log.info(f"📋 API in backoff ({_api_backoff_until - now:.0f}s left), using local parser")
    
    # Smart local fallback — parse numbered steps, math, etc
    title, wb_steps = _smart_parse_transcript(clean)
    if title and wb_steps:
        log.info(f"📋 Whiteboard (local): \"{title}\" ({len(wb_steps)} steps)")
        push_to_ui(text="📋", emotion="thinking",
                   whiteboard={"title": title, "steps": wb_steps})


def _generate_whiteboard_local(transcript: str):
    """Fast LOCAL whiteboard from transcript — no API call. Used for early trigger."""
    if not transcript or len(transcript) < 40:
        return
    
    text_lower = transcript.lower()
    edu_words = ["θεώρημα", "φόρμουλα", "εξίσωση", "βήμα", "τετράγωνο", "ρίζα",
                 "τρίγωνο", "κύκλο", "μαθηματικ", "παράδειγμα", "λύση", "υπολογ",
                 "theorem", "formula", "step", "calculate", "solve", "equation",
                 "ορθογώνι", "υποτείνου", "πλευρ", "γωνία", "κλάσμ", "αριθμ",
                 "μοιρ", "ίσο", "αποτέλεσμα", "δηλαδή", "number", "equal"]
    hits = sum(1 for w in edu_words if w in text_lower)
    has_math = bool(_re.search(r'\d|²|³|√|α|β|γ|\+|\=|\$', text_lower))
    if hits < 1 and not has_math:
        return
    
    clean = _clean_math_text(transcript)
    title, wb_steps = _smart_parse_transcript(clean)
    
    if title and wb_steps:
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

    # Gemini 3.1 has breaking changes vs 2.5:
    #   - Does NOT support enable_affective_dialog
    #   - Requires history_config.initial_history_in_client_content=True
    #     to allow the initial "Hey VIRON" seed via send_client_content
    is_v31 = "3.1" in GEMINI_LIVE_MODEL or "3-1" in GEMINI_LIVE_MODEL

    # ─── Pre-session vision: identify who's in front of the camera RIGHT NOW.
    # We capture the snapshot once here, identify any registered family
    # members, and inject the identity into the system prompt so VIRON greets
    # by name and adjusts tone per person. The same snapshot is reused later
    # as the visual input we send to Gemini at session start.
    pre_snap_jpeg, pre_snap_age = camera.get_jpeg(max_age_sec=3.0)
    vision_context = ""
    recognized_faces = []
    if pre_snap_jpeg and face_engine.is_available():
        try:
            recognized_faces = face_engine.detect_and_identify(pre_snap_jpeg)
            if recognized_faces:
                names = ", ".join(
                    f"{r['name']}({r['confidence']})" if r['face_id'] else "Unknown"
                    for r in recognized_faces
                )
                log.info(f"📸 Faces in frame: {names}")
            vision_context = face_engine.vision_context_for_prompt(recognized_faces)
        except Exception as e:
            log.warning(f"📸 Face recognition failed (non-fatal): {e}")

    # Compose the actual system instruction for THIS session.
    # The base prompt is constant; we append a small live-context block when
    # we know who's in the frame. Gemini Live builds its persona once at
    # session creation, so injecting identity here is what binds "Andreas =
    # the student you're talking to" for the duration of the conversation.
    session_system_instruction = VIRON_SYSTEM_INSTRUCTION
    if vision_context:
        session_system_instruction += (
            "\n\nLIVE CONTEXT (current camera, this session only):\n"
            + vision_context
            + "\nGreet the recognized person(s) by name in your first reply."
        )

    config_kwargs = dict(
        response_modalities=["AUDIO"],
        system_instruction=session_system_instruction,
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Orus")
            )
        ),
    )

    if is_v31:
        # Allow initial-seed send_client_content for the wake greeting trigger.
        # If the SDK doesn't expose HistoryConfig yet, skip it — model still works,
        # just without our explicit greeting trigger.
        HistoryConfig = getattr(types, "HistoryConfig", None)
        history_config_set = False
        if HistoryConfig is not None:
            try:
                config_kwargs["history_config"] = HistoryConfig(
                    initial_history_in_client_content=True
                )
                history_config_set = True
            except Exception as e:
                log.warning(f"HistoryConfig setup failed: {e}")
        log.info(f"🧠 Using Gemini 3.1 config (no affective_dialog, history_config={'enabled' if history_config_set else 'unavailable'})")
    else:
        # 2.5 native audio supports affective dialog
        config_kwargs["enable_affective_dialog"] = True
        log.info("🧠 Using Gemini 2.5 config (affective_dialog enabled)")

    # Google Search grounding — gives VIRON access to fresh info (news, weather,
    # sports, current events). Try the typed API first, fall back to dict format.
    # If neither path is supported by the running SDK, we just skip it; the model
    # still works without search and answers from training data.
    search_enabled = False
    try:
        GoogleSearch = getattr(types, "GoogleSearch", None)
        Tool = getattr(types, "Tool", None)
        if GoogleSearch is not None and Tool is not None:
            config_kwargs["tools"] = [Tool(google_search=GoogleSearch())]
            search_enabled = True
    except Exception as e:
        log.warning(f"Typed Google Search setup failed: {e}")
    if not search_enabled:
        try:
            config_kwargs["tools"] = [{"google_search": {}}]
            search_enabled = True
        except Exception as e:
            log.warning(f"Dict Google Search setup failed: {e}")
    log.info(f"🔍 Google Search grounding: {'enabled' if search_enabled else 'unavailable'}")

    config = types.LiveConnectConfig(**config_kwargs)

    log.info(f"🌐 Connecting to Gemini Live: {GEMINI_LIVE_MODEL}")
    state.set_status("listening")
    push_to_ui(emotion="hopeful")

    try:
        async with client.aio.live.connect(model=GEMINI_LIVE_MODEL, config=config) as session:
            log.info("✅ Gemini Live session connected!")
            _session_active.set()
            state.in_session = True

            # Decide: resume previous conversation or fresh start with greeting?
            resume = _should_resume_history()
            if resume:
                with _history_lock:
                    history_snapshot = list(_conv_history)
                age = int(time.time() - _last_session_end_time)
                log.info(f"📚 Resuming conversation ({len(history_snapshot)} turns, {age}s since last)")
                try:
                    history_turns = [
                        types.Content(role=role, parts=[types.Part(text=text)])
                        for role, text in history_snapshot
                    ]
                    # Seed history WITHOUT marking the turn complete: the last
                    # entry is typically a 'model' turn (VIRON's previous reply),
                    # and signaling turn_complete=True would push the model to
                    # immediately generate a follow-up turn on top of its own
                    # last reply — which Gemini rejects with 1008 policy
                    # violation. With turn_complete=False this just seeds context;
                    # the next student utterance (streamed via send_realtime_input)
                    # naturally triggers the model's response.
                    await session.send_client_content(
                        turns=history_turns,
                        turn_complete=False,
                    )
                    log.info("📚 History seeded — model knows the context")
                    # No greeting — student will speak first to continue conversation
                except Exception as e:
                    log.warning(f"History resume failed: {e} — falling back to fresh greeting")
                    resume = False  # fall through to greeting branch below

            if not resume:
                # Fresh start: trigger greeting
                # 2.5: send_client_content triggers a model turn with the text.
                # 3.1: send_client_content only SEEDS history — to actually trigger
                #      a response, we must use send_realtime_input(text=...).
                _clear_history("fresh session")

                # Capture a single snapshot so VIRON can read the student's mood
                # at the start of the conversation. We already grabbed one for
                # face recognition at the top of this function — reuse it if
                # still fresh, otherwise grab a new one.
                snap_jpeg, snap_age = pre_snap_jpeg, pre_snap_age
                if not snap_jpeg or (snap_age and snap_age > 3.0):
                    snap_jpeg, snap_age = camera.get_jpeg(max_age_sec=3.0)
                if snap_jpeg:
                    log.info(f"📷 Snapshot ready ({len(snap_jpeg)} bytes, {snap_age:.1f}s old)")
                else:
                    log.info("📷 No snapshot available for this session start")

                try:
                    if is_v31:
                        # 3.1: realtime_input handles both media (image) and text.
                        # Send the image first so the model has visual context
                        # before the trigger word.
                        if snap_jpeg:
                            try:
                                await session.send_realtime_input(
                                    video=types.Blob(data=snap_jpeg, mime_type="image/jpeg")
                                )
                            except Exception as e:
                                log.warning(f"📷 Snapshot send failed (3.1): {e}")
                        await session.send_realtime_input(text="Hey VIRON")
                        log.info("👋 Sent greeting via send_realtime_input (3.1)")
                    else:
                        # 2.5: image goes in the same multi-part Content.
                        parts = []
                        if snap_jpeg:
                            try:
                                parts.append(types.Part(
                                    inline_data=types.Blob(data=snap_jpeg, mime_type="image/jpeg")
                                ))
                            except Exception as e:
                                log.warning(f"📷 Snapshot Part build failed (2.5): {e}")
                        parts.append(types.Part(text="Hey VIRON"))
                        await session.send_client_content(
                            turns=types.Content(role="user", parts=parts),
                            turn_complete=True,
                        )
                        log.info("👋 Sent greeting via send_client_content (2.5)")
                except Exception as e:
                    log.warning(f"Greeting trigger failed: {e} — model will respond when student speaks")

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
                            if errors > 5 or "clos" in str(e).lower() or "1011" in str(e).lower():
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
                early_wb_fired = False       # Whether we already pushed the skeleton whiteboard this turn
                
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
                                            _duck_music(True)  # lower music while VIRON talks
                                        _write_audio(part.inline_data.data)

                            # Handle output transcript — accumulate + EARLY skeleton-board reveal
                            if sc.output_transcription and sc.output_transcription.text:
                                word = sc.output_transcription.text.strip()
                                if word:
                                    log.info(f"🤖 VIRON: \"{word}\"")
                                    turn_transcript.append(word)
                                    if first_word_time is None:
                                        first_word_time = time.time()
                                    
                                    # As soon as VIRON emits the [BOARD:title] tag (it should appear
                                    # at the start of teaching responses), open a skeleton board
                                    # immediately so the student sees the panel WHILE VIRON keeps
                                    # explaining. The full study sheet is generated at turn_complete
                                    # and updates the open panel in place.
                                    if not early_wb_fired:
                                        partial = " ".join(turn_transcript)
                                        early_title = _extract_board_title(partial)
                                        if early_title:
                                            early_wb_fired = True
                                            log.info(f"📋 Skeleton whiteboard early reveal: '{early_title}'")
                                            # Detect language for the loading copy
                                            is_greek_partial = bool(_re.search(r'[Α-Ωα-ωάέήίόύώϊϋΐΰ]', partial))
                                            loading_msg = "Ετοιμάζω τις σημειώσεις..." if is_greek_partial else "Preparing study notes..."
                                            push_to_ui(emotion="thinking", whiteboard={
                                                "title": early_title,
                                                "definition": "⏳ " + loading_msg,
                                                "formula": "",
                                                "example": None,
                                                "key_points": [],
                                                "steps": [{"text": "⏳ " + loading_msg}],
                                            })

                            # Handle input transcript
                            if sc.input_transcription and sc.input_transcription.text:
                                t = sc.input_transcription.text.strip()
                                if t:
                                    log.info(f"🎤 Student: \"{t}\"")
                                    _append_history("user", t)

                            # Handle interruption — STOP audio immediately
                            if sc.interrupted:
                                log.info("⚡ BARGE-IN — stopping audio")
                                is_speaking = False
                                turn_transcript.clear()
                                first_word_time = None
                                early_wb_fired = False
                                _stop_aplay()
                                _duck_music(False)  # restore music volume
                                state.set_status("listening")
                                push_to_ui(emotion="surprised", action="close_whiteboard")

                            # Handle turn complete
                            if sc.turn_complete:
                                if is_speaking:
                                    await asyncio.to_thread(_finish_aplay)
                                    _duck_music(False)  # restore music volume
                                is_speaking = False
                                state.set_status("listening")
                                state.last_activity = time.time()
                                
                                if turn_transcript:
                                    full_text = " ".join(turn_transcript)
                                    
                                    # [MUSIC:query] — start playback
                                    music_query = _extract_music_query(full_text)
                                    if music_query:
                                        log.info(f"🎵 Music request detected: '{music_query}'")
                                        threading.Thread(
                                            target=_play_music,
                                            args=(music_query,), daemon=True
                                        ).start()
                                    
                                    # [BOARD:title] — only generate the whiteboard when VIRON
                                    # explicitly tagged the response as teaching content.
                                    # Without the tag: NO whiteboard. Casual chat, weather,
                                    # news, jokes, music requests stay clean.
                                    board_title = _extract_board_title(full_text)
                                    if board_title:
                                        log.info(f"📋 Whiteboard requested: '{board_title}'")
                                        clean_for_board = _strip_control_tags(full_text)
                                        log.info(f"📝 Turn done ({len(clean_for_board)} chars)")
                                        threading.Thread(
                                            target=_generate_whiteboard_from_transcript,
                                            args=(clean_for_board,), daemon=True
                                        ).start()
                                    else:
                                        log.info(f"📝 Turn done ({len(full_text)} chars) — no [BOARD] tag, skipping whiteboard")
                                    
                                    # Save to history with all control tags stripped
                                    clean_text = _strip_control_tags(full_text)
                                    if clean_text:
                                        _append_history("model", clean_text)
                                
                                # Reset for next turn
                                turn_transcript.clear()
                                first_word_time = None
                                early_wb_fired = False
                                
                                log.info("✅ Turn complete — ready for next question")

                        await asyncio.sleep(0.1)

                    except Exception as e:
                        err_str = str(e).lower()
                        if "clos" in err_str or "1011" in err_str or "1006" in err_str or _stop_session.is_set():
                            log.warning(f"Receive: session closed ({e})")
                            # Generate whiteboard from whatever we accumulated before crash
                            if turn_transcript and len(turn_transcript) >= 5:
                                crash_text = " ".join(turn_transcript)
                                log.info(f"📋 Generating whiteboard from crash transcript ({len(crash_text)} chars)")
                                threading.Thread(
                                    target=_generate_whiteboard_from_transcript,
                                    args=(crash_text,), daemon=True
                                ).start()
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
        _mark_session_ended()
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
    log.info(f"   Mic boost: {MIC_BOOST:.1f}x")
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
                # Stop any music that's currently playing — student wants attention
                _stop_music()
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
        
        # Otherwise return next item, merging consecutive emotion-only items.
        # CRITICAL: when merging, preserve any rich payload (music, quiz, weather,
        # news, action) from items we're collapsing. The previous version only
        # carried over `emotion`, so a music event sandwiched between two emotion
        # pushes was silently discarded — UI never saw it.
        msg = _response_queue.pop(0)
        if not msg.get("text"):
            latest_emotion = msg.get("emotion", "")
            preserve_keys = ("music", "quiz", "weather", "news", "action", "subtitle")
            while _response_queue and not _response_queue[0].get("text") and "whiteboard" not in _response_queue[0]:
                popped = _response_queue.pop(0)
                if popped.get("emotion"):
                    latest_emotion = popped["emotion"]
                # Carry forward any rich payloads this message had
                for k in preserve_keys:
                    if k in popped and k not in msg:
                        msg[k] = popped[k]
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
    if action == "stop_music":
        _stop_music()
        return jsonify({"ok": True, "action": "music_stopped"})
    if action == "pause_music":
        ok = _pause_music()
        return jsonify({"ok": ok, "action": "music_paused" if ok else "no_music"})
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

@app.route("/camera/snapshot.jpg", methods=["GET"])
def camera_snapshot():
    """Return the latest webcam frame for the UI preview corner.
    Returns 503 (Service Unavailable) if no frame is available — the
    frontend treats that as 'hide preview'."""
    from flask import Response
    jpeg, age = camera.get_jpeg(max_age_sec=2.0)
    if not jpeg:
        return Response(b"", status=503, headers={"Cache-Control": "no-store"})
    return Response(jpeg, mimetype="image/jpeg", headers={
        "Cache-Control": "no-store",
        "X-Frame-Age": f"{age:.2f}",
    })

@app.route("/camera/stream", methods=["GET"])
def camera_stream():
    """MJPEG (multipart/x-mixed-replace) stream for smooth UI preview.
    The browser <img> stays connected and the kernel pushes frames as
    they're encoded — no polling overhead, no per-frame flicker."""
    from flask import Response
    boundary = b"--vironframe"

    def gen():
        last_ts = 0.0
        deadline = time.time() + 600  # 10 min per stream connection
        # Tight loop: as soon as the capture thread updates latest_ts,
        # forward the frame. Sleep is tiny (5ms) so we stay responsive
        # without spinning the CPU when the camera stalls.
        while time.time() < deadline:
            with camera._lock:
                ts = camera.latest_ts
                jpeg = camera.latest_jpeg if ts != last_ts else None
            if jpeg:
                last_ts = ts
                yield (boundary + b"\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                       + jpeg + b"\r\n")
            else:
                time.sleep(0.005)  # 5ms yield — wakes within 1 frame at 60fps

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=vironframe",
                    headers={"Cache-Control": "no-store, no-cache, must-revalidate",
                             "Pragma": "no-cache",
                             "Connection": "close"})

@app.route("/camera/state", methods=["GET"])
def camera_state():
    return jsonify({
        "available": camera.cap is not None,
        "enabled": camera._enabled,
        "device": camera._device_used,
    })

@app.route("/camera/toggle", methods=["POST"])
def camera_toggle():
    """Mute/unmute the camera. POST {'enabled': true|false}."""
    data = request.get_json(silent=True) or {}
    if "enabled" in data:
        camera.set_enabled(bool(data["enabled"]))
    else:
        camera.set_enabled(not camera._enabled)  # toggle
    return jsonify({"ok": True, "enabled": camera._enabled})

# ─── Face recognition (family identification) ───────────────
@app.route("/faces/list", methods=["GET"])
def faces_list():
    """Return all enrolled family members + recognition stats."""
    return jsonify({
        "available": face_engine.is_available(),
        "faces": face_db.list_faces(),
    })

@app.route("/faces/enroll", methods=["POST"])
def faces_enroll():
    """Enroll a new face from the CURRENT camera frame.
    POST { 'name': 'Andreas', 'role': 'student'|'parent'|'sibling'|'other' }"""
    if not face_engine.is_available():
        return jsonify({"success": False,
                        "error": "face_recognition not installed on this device"}), 400
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    role = (data.get("role") or "other").strip()
    if not name:
        return jsonify({"success": False, "error": "Name is required"}), 400
    if role not in ("student", "parent", "sibling", "other"):
        role = "other"
    snap_jpeg, _ = camera.get_jpeg(max_age_sec=3.0)
    if not snap_jpeg:
        return jsonify({"success": False,
                        "error": "No camera frame available — is the camera muted?"}), 400
    result = face_engine.enroll(name, role, snap_jpeg)
    return jsonify(result)

@app.route("/faces/<int:face_id>", methods=["DELETE"])
def faces_delete(face_id: int):
    ok = face_engine.delete(face_id)
    return jsonify({"ok": ok})

@app.route("/faces/<int:face_id>/thumbnail", methods=["GET"])
def faces_thumbnail(face_id: int):
    from flask import Response
    thumb = face_db.get_thumbnail(face_id)
    if not thumb:
        return Response(b"", status=404)
    return Response(thumb, mimetype="image/jpeg",
                    headers={"Cache-Control": "private, max-age=86400"})

@app.route("/faces/preview", methods=["GET"])
def faces_preview():
    """Diagnostic: return what the engine sees in the CURRENT frame.
    Useful for debugging recognition issues from the settings panel."""
    if not face_engine.is_available():
        return jsonify({"available": False, "results": []})
    snap_jpeg, _ = camera.get_jpeg(max_age_sec=3.0)
    if not snap_jpeg:
        return jsonify({"available": True, "results": [], "error": "No frame"})
    results = face_engine.detect_and_identify(snap_jpeg)
    return jsonify({"available": True, "results": results})

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

    # Start camera (best-effort — pipeline still works without it)
    try:
        camera.start()
    except Exception as e:
        log.warning(f"📷 Camera start failed: {e}")

    # Warm the face recognition engine in the background. First call to
    # face_recognition loads the dlib model (~100ms-2s on Orin Nano), so
    # we want that done before the first wake — otherwise the first
    # recognition adds noticeable latency to the greeting.
    threading.Thread(target=face_engine.warm, daemon=True).start()

    # Visual-only greeting (no text push - don't block UI queue)
    log.info("🤖 VIRON ready!")
    push_to_ui(emotion="happy")

    try:
        main_loop(mic)
    finally:
        _stop_music()
        try:
            camera.stop()
        except Exception:
            pass
        mic.stop()
        if oww_model is not None:
            try:
                oww_model.reset()
            except Exception:
                pass
        log.info("Pipeline stopped.")


if __name__ == "__main__":
    main()
