#!/usr/bin/env python3
"""
VIRON AI Companion Backend
Works on: Ubuntu Desktop (dev) + Jetson Orin Nano (production)
Features: AI proxy, student emotion detection, hardware APIs
"""

from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
import subprocess, json, os, re, time, threading, sys

# ============ WHISPER SPEECH-TO-TEXT ============
HAS_WHISPER = False
whisper_model = None
_whisper_lock = threading.Lock()

def _load_whisper():
    global HAS_WHISPER, whisper_model
    try:
        from faster_whisper import WhisperModel
        print("🎙️ Loading Whisper STT model (small — better Greek accuracy)...")
        whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        HAS_WHISPER = True
        print("✅ Whisper STT ready (small model, CPU)")
    except ImportError:
        print("⚠ faster-whisper not installed. Install: pip3 install faster-whisper --break-system-packages")
    except Exception as e:
        print(f"⚠ Whisper load error: {e}")

# Load in background so server starts fast
threading.Thread(target=_load_whisper, daemon=True).start()

try:
    from flask_socketio import SocketIO, emit
    HAS_SOCKETIO = True
except ImportError:
    HAS_SOCKETIO = False

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠ OpenCV not installed — emotion detection disabled")

try:
    import requests as http_requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Face recognition
HAS_FACE_REC = False
face_recognizer = None
_face_rec_error = ""
try:
    # Add backend dir to path (handles running from different directories)
    _backend_dir = os.path.dirname(os.path.abspath(__file__))
    if _backend_dir not in sys.path:
        sys.path.insert(0, _backend_dir)
    from viron_faces import face_recognizer
    HAS_FACE_REC = True
    print("✅ Face recognition module loaded")
except Exception as e:
    _face_rec_error = f"{type(e).__name__}: {e}"
    print(f"⚠ Face recognition module not available: {_face_rec_error}")
    import traceback
    traceback.print_exc()
    print("⚠ requests not installed — AI proxy disabled")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")

DEFAULT_CONFIG = {
    "anthropic_api_key": "YOUR_API_KEY_HERE",
    "model": "claude-opus-4-20250514",
    "camera_index": 4,
    "volume": 75,
    "brightness": 80,
    "emotion_detection": True,
    "proactive_care": True,
    "port": 5000
}

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            for k, v in DEFAULT_CONFIG.items():
                if k not in cfg:
                    cfg[k] = v
            return cfg
        except Exception as e:
            print(f"⚠ Config error: {e}")
    return DEFAULT_CONFIG.copy()

config = load_config()
app = Flask(__name__, static_folder=ROOT_DIR)
CORS(app)

# ============ TTS CACHE (eliminates cloud round-trip for repeated phrases) ============
import hashlib
_tts_cache = {}  # key: hash(text+lang+speed) -> audio bytes
_tts_cache_lock = threading.Lock()

def _tts_cache_key(text, lang, speed):
    return hashlib.md5(f"{text}|{lang}|{speed}".encode()).hexdigest()

def _generate_tts_audio(text, lang='el', speed='normal'):
    """Generate TTS audio bytes (blocking). Returns (audio_bytes, mimetype) or None."""
    rate_map = {'slow': '+10%', 'normal': '+15%', 'fast': '+25%'}
    tts_rate = rate_map.get(speed, '+15%')
    try:
        import edge_tts, asyncio, io
        voice = "el-GR-NestorasNeural" if lang == "el" else "en-GB-RyanNeural"
        async def gen():
            communicate = edge_tts.Communicate(text, voice, rate=tts_rate, pitch="-10Hz")
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            buf.seek(0)
            return buf
        buf = asyncio.run(gen())
        return buf.read()
    except Exception as e:
        print(f"⚠ TTS cache gen error: {e}")
        import traceback
        traceback.print_exc()
        return None

def _prewarm_tts_cache():
    """Pre-generate common phrases so they're instant."""
    phrases = [
        # Ack phrases (mhm expression)
        ("Ορίστε;", "el"), ("Ορίστε!", "el"),
        ("Yes?", "en"), ("Yes!", "en"),
        # Common greetings
        ("Γεια σου! Πες Hey VIRON για να μου μιλήσεις!", "el"),
        ("Γεια σου!", "el"),
        ("Γεια σου! Πώς είσαι;", "el"),
        ("Γεια! Τι κάνεις;", "el"),
        ("Χαίρομαι πολύ!", "el"),
        ("Γεια! Τι νέα;", "el"),
        ("Τι κάνεις;", "el"),
        ("Ε, γεια! Χαίρομαι που σε βλέπω!", "el"),
        ("Hi there! Say Hey VIRON to talk to me!", "en"),
        ("You're welcome!", "en"),
        ("No problem!", "en"),
    ]
    print("🔥 Pre-warming TTS cache for common phrases...")
    for text, lang in phrases:
        key = _tts_cache_key(text, lang, 'normal')
        audio = _generate_tts_audio(text, lang, 'normal')
        if audio:
            with _tts_cache_lock:
                _tts_cache[key] = audio
            print(f"  ✓ Cached: \"{text}\" ({len(audio)} bytes)")
        else:
            print(f"  ⚠ Failed to cache: \"{text}\"")
    print(f"✅ TTS cache ready ({len(_tts_cache)} phrases)")

# Pre-warm in background thread so server starts immediately
threading.Thread(target=_prewarm_tts_cache, daemon=True).start()

# Global error handler — prevent unhandled exceptions from crashing the server
@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    print(f"  ❌ UNHANDLED ERROR: {e}")
    traceback.print_exc()
    return jsonify({"error": str(e), "content": [{"type": "text", "text": "[confused] Something went wrong!"}]}), 500

# ============ STUDENT EMOTION DETECTOR ============
class StudentEmotionDetector:
    def __init__(self):
        global HAS_CV2
        self.running = False
        self.thread = None
        self.cap = None
        self.current_state = {
            "emotion": "neutral", "confidence": 0, "engagement_score": 100,
            "face_detected": False, "attention": "center", "blink_rate": 0,
            "mouth_open": False, "dominant_emotion": "neutral",
            "emotion_streak": 0, "yawn_count": 0,
            "recognized_person": None, "recognition_confidence": 0,
            "recognized_persons": [], "face_count": 0
        }
        self.frame_count = 0
        self.last_frame = None
        self.emotion_history = []
        self.streak_emotion = "neutral"
        self.streak_count = 0
        self.last_face_time = time.time()
        self.engagement = 100
        self.yawn_count = 0
        self.blink_timestamps = []
        self.last_eye_state = True
        self.looking_away_since = 0
        if HAS_CV2:
            # Find haarcascades path (varies by OpenCV version)
            cascade_dir = ""
            if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                cascade_dir = cv2.data.haarcascades
            else:
                # Common paths on Ubuntu
                for p in ['/usr/share/opencv4/haarcascades/', '/usr/share/opencv/haarcascades/',
                          '/usr/local/share/opencv4/haarcascades/', '/usr/share/OpenCV/haarcascades/']:
                    if os.path.exists(p):
                        cascade_dir = p
                        break
                # Last resort: find it
                if not cascade_dir:
                    try:
                        result = subprocess.run(['find', '/usr', '-name', 'haarcascade_frontalface_default.xml', '-type', 'f'],
                                                capture_output=True, text=True, timeout=5)
                        if result.stdout.strip():
                            cascade_dir = os.path.dirname(result.stdout.strip().split('\n')[0]) + '/'
                    except:
                        pass
            if not cascade_dir:
                print("⚠ Cannot find haarcascade files — emotion detection disabled")
                HAS_CV2 = False
                return
            self.face_cascade = cv2.CascadeClassifier(cascade_dir + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cascade_dir + 'haarcascade_eye.xml')
            self.smile_cascade = cv2.CascadeClassifier(cascade_dir + 'haarcascade_smile.xml')
            self.mouth_cascade = cv2.CascadeClassifier(cascade_dir + 'haarcascade_mcs_mouth.xml')

    def start(self, camera_index=0):
        if self.running or not HAS_CV2:
            return False
        
        # Try V4L2 first (more reliable than GStreamer for USB cameras)
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            print(f"  V4L2 failed, trying default backend...")
            self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"⚠ Cannot open camera {camera_index}")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        
        # Warm up — read a few frames to let camera stabilize
        for _ in range(5):
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame.copy()
        
        self.running = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        print(f"📷 Emotion detection started (camera {camera_index})")
        
        # Initialize face recognition
        if HAS_FACE_REC:
            face_recognizer.initialize()
        
        return True

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def _detection_loop(self):
        frames_ok = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            self.last_frame = frame.copy()  # Store for face registration
            frames_ok += 1
            if frames_ok == 1:
                print(f"  📷 First frame captured OK ({frame.shape})")
            self.frame_count += 1
            if self.frame_count % 3 != 0:
                continue
            try:
                self._analyze_frame(frame)
            except:
                pass
            time.sleep(0.033)

    def _analyze_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
        if len(faces) == 0:
            if time.time() - self.last_face_time > 5:
                self._update_emotion("absent")
                self.engagement = max(0, self.engagement - 2)
            self.current_state["face_detected"] = False
            return
        self.last_face_time = time.time()
        self.current_state["face_detected"] = True
        
        # Face recognition (periodically check who's in front)
        if HAS_FACE_REC and face_recognizer.initialized:
            person, confidence = face_recognizer.recognize(frame)
            self.current_state["recognized_person"] = person
            self.current_state["recognition_confidence"] = round(confidence, 3)
            # Multi-face: recognize ALL faces
            try:
                all_persons = face_recognizer.recognize_all(frame)
                self.current_state["recognized_persons"] = all_persons
                self.current_state["face_count"] = len(faces)
            except:
                self.current_state["recognized_persons"] = []
                self.current_state["face_count"] = len(faces)
        
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y+h, x:x+w]
        fw = frame.shape[1]
        fc = x + w / 2
        attention = "right" if fc < fw * 0.35 else ("left" if fc > fw * 0.65 else "center")
        self.current_state["attention"] = attention
        eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 5, minSize=(20, 20))
        ne = len(eyes)
        eo = ne >= 2
        if not eo and self.last_eye_state:
            self.blink_timestamps.append(time.time())
        self.last_eye_state = eo
        now = time.time()
        self.blink_timestamps = [t for t in self.blink_timestamps if now - t < 60]
        br = len(self.blink_timestamps) / 60.0
        smiles = self.smile_cascade.detectMultiScale(face_roi, 1.8, 20, minSize=(40, 40))
        sr = len(smiles) / max(1, w / 80)
        lower = face_roi[int(h * 0.5):, :]
        mouths = self.mouth_cascade.detectMultiScale(lower, 1.5, 11, minSize=(30, 20))
        mo = False
        if len(mouths) > 0:
            mx, my, mw, mh = max(mouths, key=lambda m: m[2] * m[3])
            mo = mh > mw * 0.5
        self.current_state["blink_rate"] = round(br, 2)
        self.current_state["mouth_open"] = mo
        fr = h / max(w, 1)
        ht = fr > 1.4 or fr < 0.9
        we = ne >= 2 and any(eh > h * 0.15 for (_, _, _, eh) in eyes)
        sq = ne >= 2 and all(eh < h * 0.08 for (_, _, _, eh) in eyes)
        if attention != "center":
            if self.looking_away_since == 0:
                self.looking_away_since = time.time()
        else:
            self.looking_away_since = 0
        la = (time.time() - self.looking_away_since) if self.looking_away_since > 0 else 0

        if sr > 0.6:
            emotion = "happy"; self.engagement = min(100, self.engagement + 1)
        elif br > 0.5 and sr <= 0.3:
            if mo:
                emotion = "bored"; self.yawn_count += 1; self.engagement = max(0, self.engagement - 2)
            else:
                emotion = "sleepy"; self.engagement = max(0, self.engagement - 1)
        elif mo and we:
            emotion = "confused"
        elif sq and sr <= 0.3:
            emotion = "frustrated"
        elif ht and sr <= 0.3:
            emotion = "thinking"
        elif la > 3:
            emotion = "distracted"; self.engagement = max(0, self.engagement - 1)
        elif attention == "center" and ne >= 2:
            emotion = "attentive"; self.engagement = min(100, self.engagement + 0.5)
        else:
            emotion = "neutral"
        self._update_emotion(emotion)

    def _update_emotion(self, emotion):
        if emotion == self.streak_emotion:
            self.streak_count += 1
        else:
            self.streak_emotion = emotion
            self.streak_count = 1
        if self.streak_count >= 5:
            self.current_state["emotion"] = emotion
            self.current_state["confidence"] = min(1.0, self.streak_count / 10)
        self.current_state["engagement_score"] = max(0, min(100, int(self.engagement)))
        self.current_state["dominant_emotion"] = emotion
        self.current_state["emotion_streak"] = self.streak_count
        self.current_state["yawn_count"] = self.yawn_count
        self.emotion_history.append({"emotion": emotion, "time": time.time()})
        if len(self.emotion_history) > 60:
            self.emotion_history = self.emotion_history[-60:]

    def get_state(self):
        return self.current_state.copy()

detector = StudentEmotionDetector()

# ============ AI SMART ROUTER ============
# ============ AI BRAIN — Multi-LLM Orchestrator ============
# Routes by SUBJECT to best AI: Math→ChatGPT, Greek→Gemini, Literature→Claude
# 4 strategies: Turbo, Race, Check, Smart (consensus from all 3)
# Student can switch mode by voice: "turbo mode", "smart mode" etc.

import logging
logging.basicConfig(level=logging.INFO)

try:
    from viron_ai_router import VironAIRouterSync, RouterConfig
    _router_cfg = RouterConfig.from_env()
    # Also load API key from config.json
    _api_key = config.get("anthropic_api_key", "")
    if not _api_key or _api_key == "YOUR_API_KEY_HERE":
        _api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    _router_cfg.anthropic_api_key = _api_key
    _router_cfg.claude_model = config.get("model", "claude-opus-4-20250514")
    if not _router_cfg.google_api_key:
        _router_cfg.google_api_key = os.environ.get("GOOGLE_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    if not _router_cfg.openai_api_key:
        _router_cfg.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    ai_router = VironAIRouterSync(_router_cfg)
    HAS_ROUTER = True
    _prov = []
    _prov.append(f"Ollama ({_router_cfg.ollama_model})")
    if _router_cfg.anthropic_api_key: _prov.append(f"Claude ({_router_cfg.claude_model})")
    if _router_cfg.google_api_key: _prov.append(f"Gemini ({_router_cfg.gemini_model})")
    if _router_cfg.openai_api_key: _prov.append(f"ChatGPT ({_router_cfg.chatgpt_model})")
    print(f"✅ AI Brain loaded — {' + '.join(_prov)}")
    print(f"  🎛️ Strategy: {_router_cfg.strategy} | Subject routing: Math→ChatGPT, Greek→Gemini, Literature→Claude")
except Exception as _router_err:
    HAS_ROUTER = False
    ai_router = None
    print(f"⚠ AI Brain not available: {_router_err}")
    import traceback; traceback.print_exc()

@app.route('/api/chat', methods=['POST'])
def chat_proxy():
    """
    Smart AI chat endpoint — VIRON Brain.
    Routes by SUBJECT to the best AI:
      Math/Science/Code → ChatGPT | Greek/Translation → Gemini | Literature/History → Claude
    Strategies: Turbo (1 AI), Race (2 AIs), Check (verify), Smart (all 3 + merge)
    Voice commands: "turbo mode", "race mode", "check mode", "smart mode"
    Fallback: cloud → Ollama (offline)
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    user_msg = data.get("messages", [{}])[-1].get("content", "")
    forced_lang = data.get("language", "")  # 'el' or 'en' — forced by frontend
    print(f"\n💬 CHAT: '{user_msg[:80]}' (lang={forced_lang})")

    # ── Try Hybrid Gateway (local + cloud routing via port 8080) ──
    _gw_url = os.environ.get("HYBRID_GATEWAY_URL", "http://localhost:8080")
    try:
        # Get student info from face recognition
        _student_id = "anonymous"
        _student_age = int(os.environ.get("DEFAULT_STUDENT_AGE", "12"))
        if HAS_PROFILES and detector.current_state.get("recognized_person"):
            _student_id = detector.current_state["recognized_person"]
            try:
                _prof = get_student_profile(_student_id)
                if _prof and _prof.get("age"):
                    _student_age = _prof["age"]
            except Exception:
                pass

        _gw_resp = http_requests.post(
            f"{_gw_url}/v1/chat",
            json={
                "student_id": _student_id,
                "age": _student_age,
                "message": user_msg,
                "language": forced_lang or "en",
            },
            timeout=90,
        )
        if _gw_resp.status_code == 200:
            _gw = _gw_resp.json()
            _reply = _gw.get("reply", "")
            if _reply and len(_reply.strip()) > 2:
                _router = _gw.get("router", {})
                _provider = _gw.get("cloud_provider", "none")
                if _provider == "none":
                    _provider = "local-mistral"
                _subject = _router.get("subject", "general")
                _mode = _gw.get("mode", "local")
                _latency = _gw.get("latency_ms", 0)
                print(f"  ✅ Gateway [{_mode}] | {_provider} | 📚 {_subject} | ⏱ {_latency:.0f}ms")
                print(f"  → '{_reply[:120]}'")

                # Log interaction for points
                if HAS_PROFILES and _student_id != "anonymous":
                    try:
                        points_result = log_interaction(
                            student_name=_student_id,
                            type="question",
                            subject=_subject,
                            language=forced_lang or "en",
                            question=user_msg[:500],
                            answer=_reply[:500],
                            emotion=detector.current_state.get("emotion", ""),
                        )
                        print(f"  🎮 +{points_result['points_earned']}pts → {points_result['total_points']} total")
                        if points_result.get("new_achievements"):
                            for ach in points_result["new_achievements"]:
                                print(f"  🏆 NEW: {ach['icon']} {ach['name_en']}")
                    except Exception as e:
                        print(f"  ⚠ Points error: {e}")

                return jsonify({
                    "content": [{"type": "text", "text": _reply}],
                    "provider": _provider,
                    "subject": _subject,
                    "strategy": f"hybrid-{_mode}",
                    "language": forced_lang or _router.get("language", "en"),
                    "confidence": 0.95 if _mode == "cloud" else 0.80,
                })
            else:
                print(f"  ⚠ Gateway returned empty reply, falling back to old router")
        else:
            print(f"  ⚠ Gateway returned {_gw_resp.status_code}, falling back to old router")
    except Exception as _gw_err:
        print(f"  ⚠ Gateway unavailable ({_gw_err}), falling back to old router")

    # ── Old router fallback (if gateway is down) ──
    system_prompt = data.get("system", "")
    history = data.get("messages", [])[:-1]
    conversation_id = data.get("conversation_id", "")

    # ── Inject student memory into system prompt ──
    student_name = ""
    if HAS_PROFILES and detector.current_state.get("recognized_person"):
        student_name = detector.current_state["recognized_person"]
        try:
            ctx = get_student_context(student_name)
            if ctx:
                system_prompt += f"\n\n{ctx}"
                print(f"  👤 Student context injected for: {student_name}")
        except Exception as e:
            print(f"  ⚠ Student context error: {e}")

    # ── Smart Router (Subject-based multi-LLM routing) ──
    if HAS_ROUTER and ai_router:
        try:
            reply, provider = ai_router.chat(
                message=user_msg,
                history=history,
                system_prompt=system_prompt,
                force_language=forced_lang,
            )
            if reply and len(reply.strip()) > 2:
                print(f"  ✅ {provider} | 📚 {ai_router.last_subject} | 🎛️ {ai_router.last_strategy} | 🌐 {ai_router.last_language} | conf:{ai_router.last_confidence:.2f}")
                print(f"  → '{reply[:120]}'")
                
                # Log interaction for points (non-blocking)
                subject_str = ai_router.last_subject or "general"
                lang_str = ai_router.last_language or "el"
                if HAS_PROFILES and student_name:
                    try:
                        points_result = log_interaction(
                            student_name=student_name,
                            type="question",
                            subject=subject_str,
                            language=lang_str,
                            question=user_msg[:500],
                            answer=reply[:500],
                            emotion=detector.current_state.get("emotion", ""),
                        )
                        print(f"  🎮 +{points_result['points_earned']}pts → {points_result['total_points']} total")
                        if points_result.get("new_achievements"):
                            for ach in points_result["new_achievements"]:
                                print(f"  🏆 NEW: {ach['icon']} {ach['name_en']}")
                    except Exception as e:
                        print(f"  ⚠ Points error: {e}")
                
                return jsonify({
                    "content": [{"type": "text", "text": reply}],
                    "provider": provider,
                    "subject": ai_router.last_subject,
                    "strategy": ai_router.last_strategy,
                    "language": ai_router.last_language,
                    "confidence": ai_router.last_confidence,
                })
            else:
                print(f"  ⚠ Router returned empty, falling back to direct API")
        except Exception as e:
            print(f"  ⚠ Router error: {e}, falling back to direct API")

    # ── Fallback: direct Anthropic API (if router fails completely) ──
    if not HAS_REQUESTS:
        return jsonify({
            "content": [{"type": "text", "text": "[confused] Ο εγκέφαλός μου δεν είναι συνδεδεμένος!" if forced_lang == "el" else "[confused] My brain isn't connected!"}],
            "provider": "none",
        })

    api_key = config.get("anthropic_api_key", "")
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("  ❌ No API key!")
        return jsonify({
            "content": [{"type": "text", "text": "[confused] Δεν έχω κλειδί API!" if forced_lang == "el" else "[confused] No API key configured!"}],
            "provider": "none",
        })
    try:
        print(f"  → Direct Anthropic API fallback...")
        model = config.get("model", "claude-opus-4-20250514")
        resp = http_requests.post("https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"},
            json={"model": model, "max_tokens": data.get("max_tokens", 1500),
                  "system": system_prompt, "messages": data.get("messages", [])},
            timeout=45)
        if resp.status_code == 529 and "opus" in model:
            print(f"  ⚠ Opus 529, trying Sonnet...")
            resp = http_requests.post("https://api.anthropic.com/v1/messages",
                headers={"Content-Type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"},
                json={"model": "claude-sonnet-4-20250514", "max_tokens": data.get("max_tokens", 1500),
                      "system": system_prompt, "messages": data.get("messages", [])},
                timeout=30)
        return Response(resp.content, status=resp.status_code, content_type="application/json")
    except Exception as e:
        print(f"  ❌ API error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/status', methods=['GET'])
def chat_router_status():
    """Debug: show AI router stats, provider status, and routing info."""
    if HAS_ROUTER and ai_router:
        return jsonify(ai_router.get_status())
    return jsonify({"error": "Router not loaded"}), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    safe = {k: v for k, v in config.items() if 'key' not in k.lower()}
    return jsonify(safe)

@app.route('/test/tts', methods=['GET'])
def test_tts():
    """Quick test: GET /test/tts to hear a male Greek voice"""
    try:
        import edge_tts, asyncio, io
        async def gen():
            communicate = edge_tts.Communicate("Γεια σου! Είμαι ο VIRON.", "el-GR-NestorasNeural", rate="+15%", pitch="-10Hz")
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            buf.seek(0)
            return buf
        buf = asyncio.run(gen())
        return Response(buf.read(), mimetype='audio/mpeg')
    except Exception as e:
        return f"edge-tts error: {e}", 500

@app.route('/test/chat', methods=['GET'])
def test_chat():
    """Quick test: GET /test/chat to test AI response"""
    try:
        api_key = config.get("anthropic_api_key", "")
        if not api_key:
            import os
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return "No API key configured", 500
        resp = http_requests.post("https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"},
            json={"model": config.get("model", "claude-opus-4-20250514"), "max_tokens": 100,
                  "messages": [{"role": "user", "content": "Say hello in 5 words"}]},
            timeout=15)
        return f"Status: {resp.status_code}\n{resp.text[:500]}", resp.status_code
    except Exception as e:
        return f"Error: {e}", 500

# ============ STUDENT EMOTION ============
@app.route('/api/student/emotion', methods=['GET'])
def student_emotion():
    try:
        return jsonify(detector.get_state())
    except Exception as e:
        return jsonify({"emotion": "neutral", "confidence": 0, "face_count": 0, "error": str(e)})

@app.route('/api/student/start-detection', methods=['POST'])
def start_detection():
    cam = request.json.get('camera_index', config.get('camera_index', 0)) if request.json else config.get('camera_index', 0)
    return jsonify({"status": "started" if detector.start(cam) else "failed"})

@app.route('/api/student/stop-detection', methods=['POST'])
def stop_detection():
    detector.stop()
    return jsonify({"status": "stopped"})

# ============ FACE RECOGNITION ============
@app.route('/api/faces/debug', methods=['GET'])
def face_debug():
    """Debug face recognition import issues"""
    info = {
        "HAS_FACE_REC": HAS_FACE_REC,
        "import_error": _face_rec_error,
        "python_path": sys.path[:5],
        "backend_dir": os.path.dirname(os.path.abspath(__file__)),
        "viron_faces_exists": os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "viron_faces.py")),
        "cv2_version": cv2.__version__ if HAS_CV2 else "not installed",
        "has_FaceDetectorYN": hasattr(cv2, 'FaceDetectorYN') if HAS_CV2 else False,
        "has_FaceRecognizerSF": hasattr(cv2, 'FaceRecognizerSF') if HAS_CV2 else False,
    }
    # Try live import
    try:
        bd = os.path.dirname(os.path.abspath(__file__))
        if bd not in sys.path:
            sys.path.insert(0, bd)
        import importlib
        mod = importlib.import_module("viron_faces")
        info["live_import"] = "SUCCESS"
        info["face_recognizer_type"] = str(type(mod.face_recognizer))
    except Exception as e:
        info["live_import"] = f"FAILED: {type(e).__name__}: {e}"
    return jsonify(info)

@app.route('/api/faces/status', methods=['GET'])
def face_status():
    """Get face recognition status"""
    if not HAS_FACE_REC:
        return jsonify({"initialized": False, "error": _face_rec_error or "Module not loaded"})
    return jsonify(face_recognizer.get_status())

@app.route('/api/faces/init', methods=['POST'])
def face_init():
    """Initialize/download face recognition models"""
    global HAS_FACE_REC, face_recognizer, _face_rec_error
    
    # Try to import if not loaded yet
    if not HAS_FACE_REC:
        try:
            bd = os.path.dirname(os.path.abspath(__file__))
            if bd not in sys.path:
                sys.path.insert(0, bd)
            import importlib
            mod = importlib.import_module("viron_faces")
            face_recognizer = mod.face_recognizer
            HAS_FACE_REC = True
            _face_rec_error = ""
            print("✅ Face recognition module loaded (retry)")
        except Exception as e:
            _face_rec_error = f"{type(e).__name__}: {e}"
            return jsonify({"success": False, "message": f"Module error: {_face_rec_error}"}), 503
    
    ok = face_recognizer.initialize()
    if ok:
        return jsonify({"success": True, "message": "Face recognition initialized!"})
    return jsonify({"success": False, "message": "Failed to initialize. Models may not have downloaded."}), 500

@app.route('/api/faces/register', methods=['POST'])
def face_register():
    """Register a face from camera or uploaded image"""
    if not HAS_FACE_REC:
        return jsonify({"success": False, "message": "Face recognition not available"}), 503
    
    if not face_recognizer.initialized:
        # Try to auto-initialize
        if not face_recognizer.initialize():
            return jsonify({"success": False, "message": "Models not ready. Click 'Setup' first."}), 503
    
    data = request.json
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"success": False, "message": "Name is required"}), 400
    
    # Option 1: Image provided as base64
    image_b64 = data.get("image")
    if image_b64:
        success, message = face_recognizer.register_face_from_base64(image_b64, name)
        return jsonify({"success": success, "message": message})
    
    # Option 2: Capture from camera (use latest frame from emotion detector)
    if detector.last_frame is not None:
        success, message = face_recognizer.register_face(detector.last_frame, name)
        return jsonify({"success": success, "message": message})
    
    return jsonify({"success": False, "message": "No image provided and no camera frame available"}), 400

@app.route('/api/faces/register-multi', methods=['POST'])
def face_register_multi():
    """Register multiple samples of a face (better accuracy)"""
    if not HAS_FACE_REC:
        return jsonify({"success": False, "message": "Face recognition not available"}), 503
    
    if not face_recognizer.initialized:
        if not face_recognizer.initialize():
            return jsonify({"success": False, "message": "Models not ready. Click 'Setup' to download them."}), 503
    
    data = request.json
    name = data.get("name", "").strip()
    count = data.get("count", 5)  # Number of samples to take
    
    if not name:
        return jsonify({"success": False, "message": "Name is required"}), 400
    
    if detector.last_frame is None:
        return jsonify({"success": False, "message": "Camera not capturing frames yet"}), 503
    
    registered = 0
    errors = []
    max_attempts = count * 3  # Try up to 3x more frames to get enough samples
    attempts = 0
    for i in range(max_attempts):
        if registered >= count:
            break
        attempts += 1
        frame = detector.last_frame
        if frame is None:
            errors.append(f"Attempt {attempts}: no frame")
            time.sleep(0.5)
            continue
        try:
            success, msg = face_recognizer.register_face(frame.copy(), name)
            if success:
                registered += 1
                print(f"  ✅ Face sample {registered}/{count} for {name}")
            else:
                if attempts <= 3:  # Only log first few errors
                    errors.append(f"Attempt {attempts}: {msg}")
        except Exception as e:
            errors.append(f"Attempt {attempts}: {type(e).__name__}: {e}")
        time.sleep(0.5)  # Wait for new frame
    
    return jsonify({
        "success": registered > 0,
        "message": f"Registered {registered}/{count} samples for {name}",
        "registered": registered,
        "errors": errors
    })

@app.route('/api/faces/list', methods=['GET'])
def face_list():
    """List all registered faces"""
    if not HAS_FACE_REC:
        return jsonify({"faces": {}})
    return jsonify({"faces": face_recognizer.list_faces()})

@app.route('/api/faces/delete', methods=['POST'])
def face_delete():
    """Delete a registered face"""
    if not HAS_FACE_REC:
        return jsonify({"success": False, "message": "Face recognition not available"})
    data = request.json
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"success": False, "message": "Name required"}), 400
    success, message = face_recognizer.delete_face(name)
    return jsonify({"success": success, "message": message})

@app.route('/api/faces/snapshot', methods=['GET'])
def face_snapshot():
    """Get a camera snapshot as JPEG (for registration preview)"""
    if detector.last_frame is None:
        return jsonify({"error": "No camera frame available"}), 503
    _, jpeg = cv2.imencode('.jpg', detector.last_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(jpeg.tobytes(), mimetype='image/jpeg')


# ============ STUDENT PROFILES & GAMIFICATION ============
try:
    from student_profiles import (
        get_or_create_student, get_student_profile, get_all_students,
        update_student, start_session, end_session, log_interaction,
        save_quiz_result, get_quiz_history, log_homework,
        get_student_context, get_greeting_context, get_leaderboard,
        get_level, ACHIEVEMENTS, POINTS
    )
    HAS_PROFILES = True
    print("✅ Student profiles loaded")
except ImportError as e:
    HAS_PROFILES = False
    print(f"⚠ Student profiles not available: {e}")


@app.route('/api/student/profile', methods=['GET'])
def student_profile_get():
    """Get student profile by name."""
    name = request.args.get('name', '')
    if not name:
        return jsonify({"error": "Name required"}), 400
    if not HAS_PROFILES:
        return jsonify({"error": "Profiles not available"}), 503
    profile = get_student_profile(name)
    if not profile:
        return jsonify({"error": "Student not found"}), 404
    return jsonify(profile)


@app.route('/api/student/profile', methods=['POST'])
def student_profile_update():
    """Update student profile fields."""
    data = request.get_json()
    name = data.get('name', '')
    if not name or not HAS_PROFILES:
        return jsonify({"error": "Name required"}), 400
    updates = {k: v for k, v in data.items() if k != 'name'}
    success = update_student(name, **updates)
    return jsonify({"success": success})


@app.route('/api/student/profiles', methods=['GET'])
def student_profiles_list():
    """List all student profiles."""
    if not HAS_PROFILES:
        return jsonify([])
    return jsonify(get_all_students())


@app.route('/api/student/session/start', methods=['POST'])
def session_start():
    """Start a study session for a student."""
    data = request.get_json() or {}
    name = data.get('name', '')
    if not name or not HAS_PROFILES:
        return jsonify({"error": "Name required"}), 400
    mood = data.get('mood', '')
    session_id = start_session(name, mood)
    context = get_greeting_context(name)
    return jsonify({"session_id": session_id, "greeting_context": context})


@app.route('/api/student/session/end', methods=['POST'])
def session_end():
    """End a study session."""
    data = request.get_json() or {}
    session_id = data.get('session_id')
    if not session_id or not HAS_PROFILES:
        return jsonify({"error": "Session ID required"}), 400
    end_session(session_id, data.get('mood', ''))
    return jsonify({"success": True})


@app.route('/api/student/interaction', methods=['POST'])
def log_student_interaction():
    """Log a question/quiz/homework interaction and get points."""
    data = request.get_json() or {}
    name = data.get('name', '')
    if not name or not HAS_PROFILES:
        return jsonify({"points_earned": 0})
    result = log_interaction(
        student_name=name,
        session_id=data.get('session_id'),
        type=data.get('type', 'question'),
        subject=data.get('subject', 'general'),
        language=data.get('language', 'el'),
        question=data.get('question', ''),
        answer=data.get('answer', ''),
        was_correct=data.get('was_correct', -1),
        emotion=data.get('emotion', ''),
    )
    return jsonify(result)


@app.route('/api/student/quiz/save', methods=['POST'])
def save_quiz():
    """Save a completed quiz result."""
    data = request.get_json() or {}
    name = data.get('name', '')
    if not name or not HAS_PROFILES:
        return jsonify({"error": "Name required"}), 400
    result = save_quiz_result(
        student_name=name,
        session_id=data.get('session_id'),
        subject=data.get('subject', ''),
        difficulty=data.get('difficulty', 'normal'),
        total_questions=data.get('total_questions', 0),
        correct_answers=data.get('correct_answers', 0),
        questions_json=data.get('questions', []),
        time_taken=data.get('time_taken_seconds', 0),
    )
    return jsonify(result)


@app.route('/api/student/quiz/history', methods=['GET'])
def quiz_history():
    """Get quiz history for a student."""
    name = request.args.get('name', '')
    if not name or not HAS_PROFILES:
        return jsonify([])
    return jsonify(get_quiz_history(name))


@app.route('/api/student/leaderboard', methods=['GET'])
def leaderboard():
    """Get the top students leaderboard."""
    if not HAS_PROFILES:
        return jsonify([])
    limit = int(request.args.get('limit', 10))
    return jsonify(get_leaderboard(limit))


@app.route('/api/student/context', methods=['GET'])
def student_context():
    """Get AI context string for a student (injected into system prompt)."""
    name = request.args.get('name', '')
    if not name or not HAS_PROFILES:
        return jsonify({"context": ""})
    return jsonify({"context": get_student_context(name)})


@app.route('/api/student/achievements', methods=['GET'])
def student_achievements():
    """Get all available achievements and student's earned ones."""
    name = request.args.get('name', '')
    all_achievements = []
    earned = []
    if HAS_PROFILES and name:
        profile = get_student_profile(name)
        if profile:
            earned = profile.get("achievements", [])
    for aid, info in ACHIEVEMENTS.items():
        all_achievements.append({
            "id": aid, **info,
            "earned": aid in earned,
        })
    return jsonify({"achievements": all_achievements, "earned_count": len(earned)})


# ============ VOICE VERIFICATION ============
try:
    from voice_verify import voice_verifier
    HAS_VOICE_VERIFY = voice_verifier.initialize()
    if HAS_VOICE_VERIFY:
        print("✅ Voice verification ready")
except ImportError as e:
    HAS_VOICE_VERIFY = False
    print(f"⚠ Voice verification not available: {e}")


@app.route('/api/voice/status', methods=['GET'])
def voice_status():
    """Get voice verification status."""
    if not HAS_VOICE_VERIFY:
        return jsonify({"initialized": False, "enabled": False, "profiles": {}, "profile_count": 0})
    return jsonify(voice_verifier.get_status())


@app.route('/api/voice/register', methods=['POST'])
def voice_register():
    """Register a voice sample for a student.
    Accepts: multipart form with 'audio' file and 'name' field,
    or JSON with 'name' and 'audio_base64' (base64 encoded audio).
    """
    if not HAS_VOICE_VERIFY:
        return jsonify({"success": False, "message": "Voice verification not available"}), 503

    name = None
    audio_data = None
    sample_rate = 16000

    if request.content_type and 'multipart' in request.content_type:
        name = request.form.get('name', '')
        audio_file = request.files.get('audio')
        if audio_file:
            audio_data = audio_file.read()
    else:
        data = request.get_json() or {}
        name = data.get('name', '')
        audio_b64 = data.get('audio_base64', '')
        sample_rate = data.get('sample_rate', 16000)
        if audio_b64:
            import base64
            audio_data = base64.b64decode(audio_b64)

    if not name:
        return jsonify({"success": False, "message": "Name required"}), 400
    if not audio_data:
        return jsonify({"success": False, "message": "Audio data required"}), 400

    result = voice_verifier.register_voice(name, audio_data, sample_rate)
    return jsonify(result)


@app.route('/api/voice/verify', methods=['POST'])
def voice_verify_endpoint():
    """Verify who is speaking from an audio sample.
    Accepts: multipart form with 'audio' file,
    or JSON with 'audio_base64' (base64 encoded audio).
    """
    if not HAS_VOICE_VERIFY:
        return jsonify({"verified": True, "name": None, "confidence": 0.0, "reason": "not_available"})

    audio_data = None
    sample_rate = 16000

    if request.content_type and 'multipart' in request.content_type:
        audio_file = request.files.get('audio')
        if audio_file:
            audio_data = audio_file.read()
    else:
        data = request.get_json() or {}
        audio_b64 = data.get('audio_base64', '')
        sample_rate = data.get('sample_rate', 16000)
        if audio_b64:
            import base64
            audio_data = base64.b64decode(audio_b64)

    if not audio_data:
        return jsonify({"verified": True, "name": None, "confidence": 0.0, "reason": "no_audio"})

    result = voice_verifier.verify_speaker(audio_data, sample_rate)
    return jsonify(result)


@app.route('/api/voice/delete', methods=['POST'])
def voice_delete():
    """Delete a voice profile."""
    data = request.get_json() or {}
    name = data.get('name', '')
    if not name:
        return jsonify({"success": False}), 400
    if not HAS_VOICE_VERIFY:
        return jsonify({"success": False}), 503
    success = voice_verifier.delete_voice(name)
    return jsonify({"success": success})


@app.route('/api/voice/toggle', methods=['POST'])
def voice_toggle():
    """Enable/disable voice verification."""
    data = request.get_json() or {}
    if not HAS_VOICE_VERIFY:
        return jsonify({"enabled": False}), 503
    voice_verifier.enabled = data.get('enabled', True)
    return jsonify({"enabled": voice_verifier.enabled})


# ============ HARDWARE ============
@app.route('/api/wifi/list', methods=['GET'])
def wifi_list():
    try:
        r = subprocess.run(['nmcli', '-t', '-f', 'SSID,SIGNAL,SECURITY', 'dev', 'wifi', 'list'], capture_output=True, text=True, timeout=10)
        nets = []; seen = set()
        for line in r.stdout.strip().split('\n'):
            if line:
                p = line.split(':')
                if len(p) >= 3 and p[0] and p[0] not in seen:
                    seen.add(p[0])
                    nets.append({"ssid": p[0], "signal": int(p[1]) if p[1].isdigit() else 0, "security": p[2]})
        return jsonify(sorted(nets, key=lambda x: x["signal"], reverse=True))
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/wifi/connect', methods=['POST'])
def wifi_connect():
    d = request.get_json(); cmd = ['nmcli', 'dev', 'wifi', 'connect', d.get('ssid', '')]
    if d.get('password'): cmd += ['password', d['password']]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return jsonify({"status": "connected" if r.returncode == 0 else "failed"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/wifi/status', methods=['GET'])
def wifi_status():
    try:
        r = subprocess.run(['nmcli', '-t', '-f', 'DEVICE,STATE,CONNECTION', 'dev', 'status'], capture_output=True, text=True, timeout=5)
        for line in r.stdout.strip().split('\n'):
            p = line.split(':')
            if len(p) >= 3 and p[1] == 'connected' and p[0].startswith('wl'):
                # Get IP address
                ip_r = subprocess.run(['hostname', '-I'], capture_output=True, text=True, timeout=3)
                ip = ip_r.stdout.strip().split()[0] if ip_r.stdout.strip() else ""
                return jsonify({"connected": True, "ssid": p[2], "ip": ip})
        return jsonify({"connected": False})
    except:
        return jsonify({"connected": False})


@app.route('/api/wifi/qr', methods=['GET'])
def wifi_qr():
    """Generate a QR code PNG for WiFi connection to VIRON.
    Mode 1 (default): QR code linking to VIRON's web interface (current IP)
    Mode 2 (?type=hotspot): QR code to connect to VIRON's hotspot
    Mode 3 (?type=wifi&ssid=X&password=Y): QR code with WiFi credentials"""
    qr_type = request.args.get('type', 'url')
    
    try:
        import qrcode
        import io
    except ImportError:
        # Auto-install qrcode
        try:
            subprocess.run(['pip', 'install', 'qrcode[pil]', '--break-system-packages', '-q'],
                           capture_output=True, timeout=30)
            import qrcode
            import io
        except:
            return jsonify({"error": "qrcode not available, install with: pip install qrcode[pil]"}), 500
    
    if qr_type == 'wifi':
        # Standard WiFi QR code format
        ssid = request.args.get('ssid', 'VIRON')
        password = request.args.get('password', '')
        security = 'WPA' if password else 'nopass'
        qr_data = f"WIFI:T:{security};S:{ssid};P:{password};;"
    elif qr_type == 'hotspot':
        qr_data = "WIFI:T:WPA;S:VIRON-Setup;P:viron2024;;"
    else:
        # URL to VIRON's web interface
        try:
            ip_r = subprocess.run(['hostname', '-I'], capture_output=True, text=True, timeout=3)
            ip = ip_r.stdout.strip().split()[0] if ip_r.stdout.strip() else "127.0.0.1"
        except:
            ip = "127.0.0.1"
        qr_data = f"http://{ip}:5000/viron-complete.html"
    
    qr = qrcode.QRCode(version=1, box_size=8, border=2)
    qr.add_data(qr_data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="white", back_color="black")
    
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return Response(buf.read(), mimetype='image/png')


@app.route('/api/wifi/hotspot/start', methods=['POST'])
def wifi_hotspot_start():
    """Create a WiFi hotspot so phones can connect to VIRON for setup."""
    try:
        # Create hotspot using nmcli
        r = subprocess.run([
            'nmcli', 'dev', 'wifi', 'hotspot',
            'ifname', 'wlan0',
            'ssid', 'VIRON-Setup',
            'password', 'viron2024'
        ], capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            return jsonify({"success": True, "ssid": "VIRON-Setup", "password": "viron2024",
                            "message": "Connect to WiFi 'VIRON-Setup' with password 'viron2024'"})
        return jsonify({"success": False, "error": r.stderr})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/wifi/hotspot/stop', methods=['POST'])
def wifi_hotspot_stop():
    """Stop the hotspot and reconnect to normal WiFi."""
    try:
        subprocess.run(['nmcli', 'con', 'down', 'Hotspot'], capture_output=True, timeout=10)
        # Try to reconnect to previous WiFi
        subprocess.run(['nmcli', 'con', 'up', '--ask'], capture_output=True, timeout=15)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# ============ SETUP WIZARD ============
@app.route('/api/setup/status', methods=['GET'])
def setup_status():
    """Check if initial setup has been completed."""
    if HAS_PROFILES:
        students = get_all_students()
        return jsonify({
            "setup_complete": len(students) > 0,
            "student_count": len(students),
            "has_faces": bool(HAS_FACE_REC and face_recognizer and face_recognizer.initialized and face_recognizer.list_faces()),
        })
    return jsonify({"setup_complete": False, "student_count": 0, "has_faces": False})


@app.route('/api/setup/register', methods=['POST'])
def setup_register_student():
    """Register a new student during setup wizard."""
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({"success": False, "error": "Name required"}), 400
    
    # Create student profile
    if HAS_PROFILES:
        student = get_or_create_student(name)
        update_student(name,
                       display_name=data.get('display_name', name),
                       age=int(data.get('age', 0)),
                       grade=data.get('grade', ''),
                       language=data.get('language', 'el'))
    
    # Register face if photo provided
    face_registered = False
    if data.get('register_face') and HAS_FACE_REC and face_recognizer:
        if data.get('photo_base64'):
            # From uploaded photo
            success, msg = face_recognizer.register_face_from_base64(data['photo_base64'], name)
            face_registered = success
        elif detector.last_frame is not None:
            # From camera
            success, msg = face_recognizer.register_face(detector.last_frame, name)
            face_registered = success
    
    return jsonify({
        "success": True,
        "name": name,
        "face_registered": face_registered,
        "message": f"Welcome {name}!"
    })


@app.route('/api/setup/reset', methods=['POST'])
def setup_reset():
    """Reset VIRON — clear all student data, faces, and voice profiles.
    Next startup will show the setup wizard again."""
    errors = []

    # 1. Clear student profiles database
    if HAS_PROFILES:
        try:
            import sqlite3 as _sql
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viron_students.db")
            if os.path.exists(db_path):
                conn = _sql.connect(db_path)
                for table in ['students', 'sessions', 'interactions', 'quiz_results', 'homework_scans', 'subject_progress']:
                    try:
                        conn.execute(f"DELETE FROM {table}")
                    except:
                        pass
                conn.commit()
                conn.close()
                print("🗑️ Reset: Student database cleared")
        except Exception as e:
            errors.append(f"students: {e}")

    # 2. Clear face recognition data
    if HAS_FACE_REC and face_recognizer:
        try:
            names = face_recognizer.list_faces()
            for name in list(names):
                face_recognizer.delete_face(name)
            print(f"🗑️ Reset: {len(names)} face(s) deleted")
        except Exception as e:
            errors.append(f"faces: {e}")

    # 3. Clear voice profiles
    if HAS_VOICE_VERIFY:
        try:
            names = list(voice_verifier.voice_profiles.keys())
            for name in names:
                voice_verifier.delete_voice(name)
            voice_verifier.enabled = False
            print(f"🗑️ Reset: {len(names)} voice profile(s) deleted")
        except Exception as e:
            errors.append(f"voice: {e}")

    if errors:
        print(f"⚠ Reset errors: {errors}")

    return jsonify({
        "success": len(errors) == 0,
        "message": "VIRON reset complete. Restart to run setup wizard.",
        "errors": errors
    })

@app.route('/api/battery', methods=['GET'])
def battery_status():
    for path in ['/sys/class/power_supply/BAT0/capacity', '/sys/class/power_supply/BAT1/capacity']:
        try:
            if os.path.exists(path):
                with open(path) as f: pct = int(f.read().strip())
                status_path = path.replace('capacity', 'status')
                charging = False
                if os.path.exists(status_path):
                    with open(status_path) as f: charging = 'charging' in f.read().strip().lower()
                return jsonify({"percent": pct, "charging": charging, "source": "sysfs"})
        except: pass
    return jsonify({"percent": 100, "charging": True, "source": "simulated"})

@app.route('/api/volume', methods=['POST'])
def set_volume():
    d = request.get_json(); level = d.get('level', 75)
    try:
        subprocess.run(['amixer', 'set', 'Master', f'{level}%'], capture_output=True, timeout=5)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/system/info', methods=['GET'])
def system_info():
    try:
        up = subprocess.run(['uptime', '-p'], capture_output=True, text=True, timeout=5).stdout.strip()
        hn = subprocess.run(['hostname'], capture_output=True, text=True, timeout=5).stdout.strip()
        is_jetson = os.path.exists('/proc/device-tree/model')
        return jsonify({"hostname": hn, "uptime": up, "platform": "jetson" if is_jetson else "desktop"})
    except:
        return jsonify({"hostname": "viron", "uptime": "unknown"})

@app.route('/api/system/shutdown', methods=['POST'])
def shutdown():
    subprocess.Popen(['sudo', 'shutdown', '-h', 'now']); return jsonify({"status": "ok"})

@app.route('/api/system/restart', methods=['POST'])
def restart():
    subprocess.Popen(['sudo', 'reboot']); return jsonify({"status": "ok"})

@app.route('/api/ping', methods=['GET'])
def ping():
    return jsonify({"status": "ok", "version": "1.0.0", "opencv": HAS_CV2, "ai_proxy": HAS_REQUESTS})

# ============ TEXT-TO-SPEECH ============
try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False
    print("⚠ gTTS not installed — server-side TTS disabled. Install: pip3 install gTTS")

@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    """Generate speech audio from text. Returns MP3. Uses cache for repeated phrases, edge-tts streaming for new text, gTTS fallback."""
    data = request.get_json()
    if not data or not data.get('text'):
        return jsonify({"error": "No text"}), 400
    text = data['text']
    lang = data.get('lang', 'el')
    speed = data.get('speed', 'normal')  # 'normal', 'slow', 'fast'
    
    # Check cache first (instant response for ack phrases and repeated text)
    cache_key = _tts_cache_key(text, lang, speed)
    with _tts_cache_lock:
        cached = _tts_cache.get(cache_key)
    if cached:
        print(f"⚡ TTS cache hit: '{text[:50]}' ({len(cached)} bytes)")
        return Response(cached, mimetype='audio/mpeg',
                       headers={'Content-Disposition': 'inline'})
    
    # Speed presets: normal for chat, slow for whiteboard teaching
    rate_map = {'slow': '+10%', 'normal': '+15%', 'fast': '+25%'}
    tts_rate = rate_map.get(speed, '+15%')
    
    # Try edge-tts CLI subprocess (streams audio via pipe — much faster than Python async)
    try:
        import shutil
        if shutil.which('edge-tts'):
            voice = "el-GR-NestorasNeural" if lang == "el" else "en-GB-RyanNeural"
            print(f"🎙️ edge-tts CLI: voice={voice}, rate={tts_rate}, text='{text[:50]}'")
            
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                tmp_path = tmp.name
            
            t_start = time.time()
            proc = subprocess.run(
                ['edge-tts', '--voice', voice, '--rate', tts_rate, '--pitch', '-10Hz',
                 '--text', text, '--write-media', tmp_path],
                capture_output=True, timeout=15
            )
            
            if proc.returncode == 0 and os.path.exists(tmp_path):
                with open(tmp_path, 'rb') as f:
                    audio_bytes = f.read()
                os.unlink(tmp_path)
                elapsed = time.time() - t_start
                print(f"✅ edge-tts CLI OK: {len(audio_bytes)} bytes in {elapsed:.1f}s")
                
                # Cache short phrases for future instant playback
                if len(text) < 200:
                    with _tts_cache_lock:
                        _tts_cache[cache_key] = audio_bytes
                    print(f"  💾 Cached for next time: '{text[:50]}'")
                
                return Response(audio_bytes, mimetype='audio/mpeg',
                               headers={'Content-Disposition': 'inline'})
            else:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                print(f"⚠ edge-tts CLI failed: {proc.stderr.decode()[:200]}")
    except Exception as e:
        print(f"⚠ edge-tts CLI error: {e}")
    
    # Fallback: edge-tts Python API
    try:
        import edge_tts, asyncio, io
        voice = "el-GR-NestorasNeural" if lang == "el" else "en-GB-RyanNeural"
        print(f"🎙️ edge-tts Python API fallback: voice={voice}, text='{text[:50]}'")
        
        async def gen():
            communicate = edge_tts.Communicate(text, voice, rate=tts_rate, pitch="-10Hz")
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            buf.seek(0)
            return buf
        
        buf = asyncio.run(gen())
        audio_bytes = buf.read()
        print(f"✅ edge-tts Python OK: {len(audio_bytes)} bytes")
        
        if len(text) < 200:
            with _tts_cache_lock:
                _tts_cache[cache_key] = audio_bytes
        
        return Response(audio_bytes, mimetype='audio/mpeg',
                       headers={'Content-Disposition': 'inline'})
    except ImportError:
        print("⚠ edge-tts not installed")
    except Exception as e:
        import traceback
        print(f"⚠ edge-tts error: {e}")
        traceback.print_exc()
    
    # Final fallback: gTTS (Google Translate TTS — slower but always works)
    if HAS_GTTS:
        try:
            import tempfile
            tts_lang = 'el' if lang == 'el' else 'en'
            print(f"🎙️ gTTS fallback: lang={tts_lang}, text='{text[:50]}'")
            tts = gTTS(text=text, lang=tts_lang)
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                tmp_path = tmp.name
                tts.save(tmp_path)
            with open(tmp_path, 'rb') as f:
                audio_bytes = f.read()
            os.unlink(tmp_path)
            print(f"✅ gTTS OK: {len(audio_bytes)} bytes")
            if len(text) < 200:
                with _tts_cache_lock:
                    _tts_cache[cache_key] = audio_bytes
            return Response(audio_bytes, mimetype='audio/mpeg',
                           headers={'Content-Disposition': 'inline'})
        except Exception as e:
            print(f"⚠ gTTS error: {e}")
    
    return jsonify({"error": "All TTS engines failed"}), 500

@app.route('/api/tts/prewarm', methods=['POST'])
def tts_prewarm():
    """Pre-cache TTS for a name so ack is instant. Called when face is recognized."""
    data = request.get_json() or {}
    name = data.get('name', '')
    if not name:
        return jsonify({"status": "no name"}), 200
    phrases = [
        (f"Ορίστε {name};", "el"),
        (f"Yes {name}?", "en"),
    ]
    def _warm():
        for text, lang in phrases:
            key = _tts_cache_key(text, lang, 'normal')
            with _tts_cache_lock:
                if key in _tts_cache:
                    continue
            audio = _generate_tts_audio(text, lang, 'normal')
            if audio:
                with _tts_cache_lock:
                    _tts_cache[key] = audio
                print(f"  ⚡ Pre-cached ack for {name}: \"{text}\"")
    threading.Thread(target=_warm, daemon=True).start()
    return jsonify({"status": "warming", "name": name}), 200

# ============ WHISPER SPEECH-TO-TEXT ENDPOINT ============
def _clean_whisper_text(text: str) -> str:
    """Clean up Whisper hallucinations: repeated chars, stutters."""
    if not text:
        return text
    # Remove runs of 3+ repeated characters: "κατασκευάζειεεεεε" → "κατασκευάζει"
    cleaned = re.sub(r'(.)\1{2,}', r'\1', text)
    # Remove repeated words: "hello hello hello" → "hello"
    cleaned = re.sub(r'\b(\w+)(\s+\1){2,}\b', r'\1', cleaned)
    return cleaned.strip()


@app.route('/api/stt', methods=['POST'])
def speech_to_text():
    """Transcribe audio — tries OpenAI Whisper API first, local fallback."""
    audio_file = request.files.get('audio')
    if not audio_file:
        return jsonify({"error": "No audio", "text": ""}), 400
    
    import tempfile
    tmp_path = None
    wav_path = None
    try:
        # Save uploaded audio to temp file
        orig_name = audio_file.filename or 'speech.webm'
        suffix = '.wav' if orig_name.endswith('.wav') else '.webm'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            audio_file.save(tmp)
        
        file_size = os.path.getsize(tmp_path)
        hint_lang = request.form.get('lang', 'el')
        whisper_lang = 'el' if hint_lang in ('el', 'el-GR') else 'en' if hint_lang in ('en', 'en-US', 'en-GB') else None
        print(f"🎙️ STT: received {file_size//1024}KB audio ({suffix}, lang={whisper_lang})")
        
        t_start = time.time()
        
        # ── Try OpenAI Whisper API first (much more accurate for Greek) ──
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if openai_key and file_size > 500:  # skip tiny files
            try:
                import requests as req_lib
                print("  ☁️ Trying OpenAI Whisper API (large-v3)...")
                with open(tmp_path, 'rb') as f:
                    resp = req_lib.post(
                        "https://api.openai.com/v1/audio/transcriptions",
                        headers={"Authorization": f"Bearer {openai_key}"},
                        files={"file": (f"speech{suffix}", f, f"audio/{'wav' if suffix=='.wav' else 'webm'}")},
                        data={
                            "model": "whisper-1",
                            "language": whisper_lang or "el",
                            "temperature": 0.0,
                            "prompt": "Αυτό είναι Ελληνικά." if whisper_lang == "el" else "",
                        },
                        timeout=15,
                    )
                if resp.status_code == 200:
                    text = resp.json().get("text", "").strip()
                    # Clean up repeated characters (Whisper hallucination from trailing silence)
                    text = _clean_whisper_text(text)
                    elapsed = time.time() - t_start
                    print(f"  ✅ OpenAI Whisper: \"{text[:80]}\" ({elapsed:.1f}s)")
                    # Filter hallucinations (common Whisper ghost outputs)
                    HALLUCINATION_BLACKLIST = [
                        "υπότιτλοι", "authorwave", "σας ευχαριστούμε",
                        "ευχαριστώ που παρακολουθήσατε", "εγγραφείτε στο κανάλι",
                        "like and subscribe", "thank you for watching",
                        "subtitles by", "translated by", "amara.org",
                        "σας ευχαριστώ για την παρακολούθηση",
                        "παρακαλώ εγγραφείτε", "κάντε like",
                        "music", "♪", "...", "…",
                    ]
                    if text and any(bl in text.lower() for bl in HALLUCINATION_BLACKLIST):
                        print(f"  ⚠ Filtered hallucination: '{text[:60]}'")
                        return jsonify({"text": "", "language": whisper_lang or "el", "duration": round(elapsed, 2), "engine": "openai", "filtered": "hallucination"})
                    if text and len(text) > 1:
                        return jsonify({"text": text, "language": whisper_lang or "el", "duration": round(elapsed, 2), "engine": "openai"})
                    else:
                        print("  ⚠ OpenAI returned empty, trying local...")
                else:
                    print(f"  ⚠ OpenAI API error {resp.status_code}: {resp.text[:100]}")
            except Exception as e:
                print(f"  ⚠ OpenAI Whisper failed: {e}, trying local...")
        
        # ── Fallback to local faster-whisper ──
        if not HAS_WHISPER or whisper_model is None:
            return jsonify({"error": "No STT available", "text": ""}), 503
        
        print("  🏠 Using local Whisper (small)...")
        
        # If already WAV (from Silero VAD), use directly — no ffmpeg needed
        if suffix == '.wav':
            audio_path = tmp_path
            print(f"  ✅ Direct WAV from Silero VAD (no conversion needed)")
        else:
            # Convert webm → wav with ffmpeg (better Whisper compatibility + amplify)
            wav_path = tmp_path.replace('.webm', '.wav')
            try:
                result = subprocess.run([
                    'ffmpeg', '-y', '-i', tmp_path,
                    '-af', 'volume=3.0,highpass=f=100,lowpass=f=8000',  # Amplify 3x + band-pass filter
                    '-ar', '16000', '-ac', '1',  # 16kHz mono (Whisper optimal)
                    wav_path
                ], capture_output=True, timeout=10)
                if result.returncode == 0 and os.path.exists(wav_path):
                    audio_path = wav_path
                    print(f"  ✅ Converted to WAV ({os.path.getsize(wav_path)//1024}KB), amplified 3x")
                else:
                    audio_path = tmp_path
                    print(f"  ⚠ ffmpeg failed, using raw webm")
            except Exception as e:
                audio_path = tmp_path
                print(f"  ⚠ ffmpeg error: {e}, using raw webm")
        
        t_local = time.time()
        
        with _whisper_lock:
            # First try with Silero VAD (server-side neural speech detection)
            segments, info = whisper_model.transcribe(
                audio_path,
                language=whisper_lang,  # Force language instead of auto-detect
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=400,    # Tighter silence detection
                    speech_pad_ms=200,               # Less padding
                    threshold=0.35,                  # Silero VAD threshold
                ),
                no_speech_threshold=0.5,            # Filter low-confidence segments
                condition_on_previous_text=False,   # Reduce hallucination chaining
            )
            text_parts = []
            for segment in segments:
                t = segment.text.strip()
                # Skip Whisper hallucinations: repeated words, music tags, nonsense
                if t and not re.match(r'^[\s\.\,\!\?\;\:…]+$', t) and '[' not in t:
                    # Skip if same phrase repeated 3+ times
                    words = t.split()
                    if len(words) >= 3:
                        # Check for repeating patterns
                        unique_words = set(w.lower() for w in words)
                        if len(unique_words) <= 2 and len(words) > 4:
                            print(f"  ⚠ Skipping hallucination (repetition): '{t[:60]}'")
                            continue
                    text_parts.append(t)
            
            # If VAD filtered everything out, retry WITHOUT VAD
            if not text_parts:
                print("  ⚠ VAD filtered all audio, retrying without VAD...")
                segments2, info = whisper_model.transcribe(
                    audio_path,
                    language=whisper_lang,
                    beam_size=5,
                    vad_filter=False,
                )
                for segment in segments2:
                    t = segment.text.strip()
                    # Skip Whisper hallucinations (repeated punctuation, music tags, etc)
                    if t and not re.match(r'^[\s\.\,\!\?\;\:…]+$', t) and '[' not in t:
                        text_parts.append(t)
        
        text = _clean_whisper_text(" ".join(text_parts).strip())
        elapsed = time.time() - t_local
        lang = info.language if info else "unknown"
        
        print(f"🎙️ Local Whisper: \"{text[:80]}\" (lang={lang}, {elapsed:.1f}s)")
        
        return jsonify({
            "text": text,
            "language": lang,
            "duration": round(elapsed, 2),
            "engine": "local"
        })
    except Exception as e:
        import traceback
        print(f"⚠ Whisper STT error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "text": ""}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if wav_path and wav_path != tmp_path and os.path.exists(wav_path):
            os.unlink(wav_path)

@app.route('/api/stt/status', methods=['GET'])
def stt_status():
    """Check if any STT engine is ready."""
    local_ready = HAS_WHISPER and whisper_model is not None
    cloud_ready = bool(os.environ.get("OPENAI_API_KEY", ""))
    return jsonify({"ready": local_ready or cloud_ready, "local": local_ready, "cloud": cloud_ready})


# ============ SERVER-SIDE MIC RECORDING (ReSpeaker) ============
RECORD_ALSA_DEVICE = os.environ.get("VIRON_MIC_DEVICE", "plughw:2,0")

@app.route('/api/record', methods=['POST'])
def record_from_mic():
    """Record audio from ReSpeaker mic with energy-based VAD.
    Returns WAV audio blob that browser can send to /api/stt.
    
    POST params (JSON):
      max_duration: max recording seconds (default 15)
      silence_duration: seconds of silence to stop (default 1.5)
      min_duration: minimum recording seconds (default 0.5)
    """
    import wave, struct, tempfile
    
    params = request.get_json(silent=True) or {}
    max_duration = min(float(params.get('max_duration', 10)), 30)
    silence_duration = float(params.get('silence_duration', 0.6))
    min_duration = float(params.get('min_duration', 0.2))
    
    sample_rate = 16000
    chunk_ms = 80  # 80ms chunks
    chunk_samples = sample_rate * chunk_ms // 1000  # 1280
    bytes_per_chunk = chunk_samples * 2  # int16
    
    # Initial thresholds — overridden by adaptive calibration below
    SPEECH_THRESHOLD = 100
    SILENCE_THRESHOLD = 60
    
    cmd = [
        "arecord", "-D", RECORD_ALSA_DEVICE,
        "-f", "S16_LE", "-r", str(sample_rate),
        "-c", "1", "-t", "raw",
    ]
    
    try:
        # Pause wakeword service to release ALSA device
        oww_port = int(os.environ.get('VIRON_WAKEWORD_PORT', '8085'))
        try:
            import urllib.request
            req = urllib.request.Request(f'http://127.0.0.1:{oww_port}/wakeword/pause', method='POST',
                                        data=b'{}', headers={'Content-Type': 'application/json'})
            urllib.request.urlopen(req, timeout=2)
            time.sleep(0.3)  # Give arecord time to release device
        except Exception as e:
            print(f"⚠ Could not pause wakeword: {e}")
        
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        audio_frames = []
        speech_started = False
        silence_chunks = 0
        silence_chunks_needed = int(silence_duration * 1000 / chunk_ms)
        max_chunks = int(max_duration * 1000 / chunk_ms)
        min_chunks = int(min_duration * 1000 / chunk_ms)
        total_chunks = 0
        
        print(f"🎤 Recording from {RECORD_ALSA_DEVICE} (max {max_duration}s, silence {silence_duration}s)")
        
        # Phase 1: Measure noise floor (first 0.5s = ~6 chunks)
        noise_rms_values = []
        CALIBRATION_CHUNKS = 6
        
        for _ in range(CALIBRATION_CHUNKS):
            data = proc.stdout.read(bytes_per_chunk)
            if not data or len(data) < bytes_per_chunk:
                break
            audio_frames.append(data)
            total_chunks += 1
            samples = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
            noise_rms_values.append(rms)
        
        if noise_rms_values:
            noise_floor = np.mean(noise_rms_values)
            # Speech = 3x above noise floor (minimum 50)
            SPEECH_THRESHOLD = max(noise_floor * 3, 50)
            # Silence = 1.5x above noise floor
            SILENCE_THRESHOLD = max(noise_floor * 1.5, 30)
            print(f"🎤 Noise floor={noise_floor:.0f}, speech_thresh={SPEECH_THRESHOLD:.0f}, silence_thresh={SILENCE_THRESHOLD:.0f}")
        
        # Phase 2: Record with adaptive VAD
        while total_chunks < max_chunks:
            data = proc.stdout.read(bytes_per_chunk)
            if not data or len(data) < bytes_per_chunk:
                break
            
            audio_frames.append(data)
            total_chunks += 1
            
            samples = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
            
            # Log every ~0.5s
            if total_chunks % 6 == 0:
                print(f"🎤 chunk={total_chunks} rms={rms:.0f} speech={speech_started} silence_count={silence_chunks}")
            
            if not speech_started:
                if rms > SPEECH_THRESHOLD:
                    speech_started = True
                    silence_chunks = 0
                    print(f"🎤 Speech START (RMS={rms:.0f}) at {total_chunks * chunk_ms}ms")
                elif total_chunks > int(3000 / chunk_ms):  # 3s wait max
                    print(f"🎤 No speech after 3s (last RMS={rms:.0f})")
                    break
            else:
                if rms < SILENCE_THRESHOLD:
                    silence_chunks += 1
                    if silence_chunks >= silence_chunks_needed:
                        print(f"🎤 Silence END, stopping ({total_chunks * chunk_ms}ms recorded)")
                        break
                else:
                    silence_chunks = 0
        
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except:
            proc.kill()
        
        if not audio_frames or not speech_started:
            # Resume wakeword before returning
            _resume_wakeword()
            return jsonify({"error": "no_speech", "duration_ms": 0}), 204
        
        # Trim trailing silence to prevent Whisper hallucinations
        # Remove chunks from the end where RMS < SILENCE_THRESHOLD
        while len(audio_frames) > 3:  # Keep at least ~240ms
            last_chunk = np.frombuffer(audio_frames[-1], dtype=np.int16)
            rms = np.sqrt(np.mean(last_chunk.astype(np.float32) ** 2))
            if rms < SILENCE_THRESHOLD:
                audio_frames.pop()
            else:
                break
        
        # Package as WAV
        total_chunks = len(audio_frames)
        duration_ms = total_chunks * chunk_ms
        raw_audio = b''.join(audio_frames)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            wav_path = tmp.name
            with wave.open(tmp, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(raw_audio)
        
        print(f"🎤 Recorded {duration_ms}ms ({len(raw_audio)//1024}KB)")
        
        # Note: wakeword resumes when browser calls /wakeword/resume after conversation
        from flask import send_file
        return send_file(wav_path, mimetype='audio/wav', 
                        download_name='recording.wav',
                        as_attachment=False)
    
    except Exception as e:
        print(f"❌ Record error: {e}")
        _resume_wakeword()
        return jsonify({"error": str(e)}), 500


def _resume_wakeword():
    """Resume wakeword service after recording."""
    try:
        import urllib.request
        oww_port = int(os.environ.get('VIRON_WAKEWORD_PORT', '8085'))
        req = urllib.request.Request(f'http://127.0.0.1:{oww_port}/wakeword/resume', method='POST',
                                    data=b'{}', headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass

@app.route('/api/record/status', methods=['GET'])
def record_status():
    """Check if server-side mic is available."""
    import shutil
    has_arecord = shutil.which('arecord') is not None
    # Check if ALSA device node exists (don't try to open it — wakeword service has it)
    device_exists = os.path.exists('/dev/snd') and has_arecord
    
    return jsonify({
        "available": device_exists,
        "device": RECORD_ALSA_DEVICE,
        "has_arecord": has_arecord,
    })


# ============ CAMERA (Brio 4K on /dev/video0) ============
CAMERA_DEVICE = os.environ.get("VIRON_CAMERA_DEVICE", "/dev/video0")

@app.route('/api/camera/snapshot', methods=['GET'])
def camera_snapshot():
    """Capture a single JPEG frame from Brio camera."""
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            snap_path = tmp.name
        
        # Use fswebcam or v4l2 to capture a frame
        result = subprocess.run([
            'fswebcam', '-d', CAMERA_DEVICE,
            '-r', '1280x720', '--no-banner',
            '--jpeg', '85', snap_path
        ], capture_output=True, timeout=5)
        
        if result.returncode != 0:
            # Fallback: try ffmpeg
            result = subprocess.run([
                'ffmpeg', '-f', 'v4l2', '-i', CAMERA_DEVICE,
                '-frames:v', '1', '-y', snap_path
            ], capture_output=True, timeout=5)
        
        if os.path.exists(snap_path) and os.path.getsize(snap_path) > 0:
            from flask import send_file
            return send_file(snap_path, mimetype='image/jpeg')
        else:
            return jsonify({"error": "Failed to capture frame"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/camera/status', methods=['GET'])
def camera_status():
    """Check if camera is available."""
    cam_exists = os.path.exists(CAMERA_DEVICE)
    return jsonify({
        "available": cam_exists,
        "device": CAMERA_DEVICE,
    })


# ============ DEBUG LOGGING ============
DEBUG_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "debug.log")

# Initialize debug log
with open(DEBUG_LOG, "w") as f:
    f.write(f"=== VIRON Debug Log - {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")

def debug_log(source, message, level="INFO"):
    timestamp = time.strftime("%H:%M:%S")
    line = f"[{timestamp}] [{level}] [{source}] {message}"
    print(f"  DEBUG: {line}")
    try:
        with open(DEBUG_LOG, "a") as f:
            f.write(line + "\n")
    except:
        pass

@app.route('/debug/log', methods=['POST', 'OPTIONS'])
def debug_receive_log():
    if request.method == 'OPTIONS':
        return '', 204
    data = request.get_json()
    if data:
        debug_log(data.get('source', 'browser'), data.get('message', ''), data.get('level', 'INFO'))
    return jsonify({"ok": True})

@app.route('/debug/save', methods=['POST', 'OPTIONS'])
def debug_save_log():
    if request.method == 'OPTIONS':
        return '', 204
    data = request.get_json()
    if data and 'log' in data:
        try:
            with open(DEBUG_LOG, "w") as f:
                f.write("=== VIRON Browser Debug Log ===\n")
                f.write(data['log'] + "\n")
            return jsonify({"ok": True, "size": len(data['log'])})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"ok": False}), 400

@app.route('/debug/viewlog', methods=['GET'])
def debug_view_log():
    try:
        with open(DEBUG_LOG, "r") as f:
            return f.read(), 200, {'Content-Type': 'text/plain'}
    except:
        return "No log file yet", 200

@app.route('/debug/status', methods=['GET'])
def debug_status():
    import urllib.request
    checks = {}
    # Flask
    checks["flask_5000"] = "✅ Running (you're seeing this)"
    # AI Router
    try:
        urllib.request.urlopen("http://localhost:8000/health", timeout=2)
        checks["ai_router_8000"] = "✅ Running"
    except:
        checks["ai_router_8000"] = "❌ Down"
    # Wake Word
    try:
        result = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True, timeout=3)
        checks["wake_word_9000"] = "✅ Running" if ":9000" in result.stdout else "❌ Down"
    except:
        checks["wake_word_9000"] = "❓ Unknown"
    # Ollama
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        checks["ollama"] = "✅ Running"
    except:
        checks["ollama"] = "❌ Down"
    return jsonify(checks)

@app.route('/debug')
def debug_page():
    return send_from_directory(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug.html')

# ============ SERVE FILES ============
@app.route('/')
def index():
    return send_from_directory(SCRIPT_DIR, 'boot.html')

@app.route('/viron-complete.html')
def main_face():
    return send_from_directory(ROOT_DIR, 'viron-complete.html')

@app.route('/viron-logo.png')
def logo():
    return send_from_directory(SCRIPT_DIR, 'viron-logo.png')

@app.route('/static/<path:filename>')
def serve_static(filename):
    static_dir = os.path.join(ROOT_DIR, 'static')
    return send_from_directory(static_dir, filename)

# ============ WAKE WORD SERVICE PROXY ============
# Proxy requests to openWakeWord service on port 8085
OWW_PORT = int(os.environ.get('VIRON_WAKEWORD_PORT', '8085'))

@app.route('/wakeword/<path:path>', methods=['GET', 'POST'])
def wakeword_proxy(path):
    import urllib.request
    import urllib.error
    try:
        url = f'http://127.0.0.1:{OWW_PORT}/wakeword/{path}'
        req = urllib.request.Request(url, method=request.method)
        if request.method == 'POST':
            data = request.get_data() or b'{}'
            ct = request.content_type or 'application/json'
            req.add_header('Content-Type', ct)
            resp = urllib.request.urlopen(req, data, timeout=2)
        else:
            resp = urllib.request.urlopen(req, timeout=2)
        return jsonify(json.loads(resp.read()))
    except urllib.error.URLError:
        return jsonify({"error": "Wake word service not running", "ready": False}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============ START ============
if __name__ == '__main__':
    port = config['port']
    has_key = config.get('anthropic_api_key', '') not in ['', 'YOUR_API_KEY_HERE']
    
    # Pre-warm Ollama: load model into memory for instant responses
    def _warm_ollama():
        try:
            import urllib.request
            model = config.get('ollama_model', 'qwen2.5:3b')
            print(f"🧠 Pre-loading Ollama model ({model}) into memory...")
            data = json.dumps({"model": model, "prompt": "Γεια!", "stream": False}).encode()
            req = urllib.request.Request("http://localhost:11434/api/generate", 
                                         data=data, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=30)
            # Keep model loaded with keep_alive
            keep = json.dumps({"model": model, "keep_alive": "60m"}).encode()
            req2 = urllib.request.Request("http://localhost:11434/api/generate",
                                          data=keep, headers={"Content-Type": "application/json"})
            try: urllib.request.urlopen(req2, timeout=5)
            except: pass
            print(f"✅ Ollama model loaded in memory (keep_alive=60m)")
        except Exception as e:
            print(f"⚠ Ollama pre-warm failed: {e}")
    threading.Thread(target=_warm_ollama, daemon=True).start()
    print(f"""
🤖 ═══════════════════════════════════════
   VIRON AI Companion
   ═══════════════════════════════════════
   📡 http://localhost:{port}
   📷 OpenCV: {'✓' if HAS_CV2 else '✗'}
   🧠 AI Proxy: {'✓' if HAS_REQUESTS else '✗'}
   🔑 API Key: {'✓' if has_key else '✗ Edit backend/config.json'}
   ═══════════════════════════════════════
""")
    if config.get("emotion_detection", True) and HAS_CV2:
        detector.start(config.get("camera_index", 0))
    if HAS_SOCKETIO:
        socketio = SocketIO(app, cors_allowed_origins="*")
        socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
    else:
        app.run(host='0.0.0.0', port=port, debug=False)
