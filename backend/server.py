#!/usr/bin/env python3
"""
VIRON AI Companion Backend
Works on: Ubuntu Desktop (dev) + Jetson Orin Nano (production)
Features: AI proxy, student emotion detection, hardware APIs
"""

from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
import subprocess, json, os, re, time, threading, sys

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
    print("⚠ requests not installed — AI proxy disabled")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")

DEFAULT_CONFIG = {
    "anthropic_api_key": "YOUR_API_KEY_HERE",
    "model": "claude-haiku-4-5-20251001",
    "camera_index": 0,
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
            "emotion_streak": 0, "yawn_count": 0
        }
        self.frame_count = 0
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
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"⚠ Cannot open camera {camera_index}")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        self.running = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        print(f"📷 Emotion detection started (camera {camera_index})")
        return True

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def _detection_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
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

# ============ AI PROXY ============
@app.route('/api/chat', methods=['POST'])
def chat_proxy():
    """Proxy to AI Router (port 8000) or direct to Anthropic as fallback."""
    if not HAS_REQUESTS:
        return jsonify({"error": "requests library not installed"}), 500
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    user_msg = data.get("messages", [{}])[-1].get("content", "")
    print(f"\n💬 CHAT REQUEST: '{user_msg}'")

    # Try AI Router first (port 8000)
    try:
        print(f"  → Trying AI Router (localhost:8000)...")
        resp = http_requests.post("http://localhost:8000/chat",
            json={
                "message": user_msg,
                "age_mode": data.get("age_mode", config.get("age_mode", "kids")),
                "conversation_id": data.get("conversation_id", "default"),
            }, timeout=12)  # 12s max for AI Router
        print(f"  → AI Router status: {resp.status_code}")
        if resp.status_code == 200:
            router_data = resp.json()
            reply = router_data.get("reply", "")
            provider = router_data.get("provider", "unknown")
            print(f"  → AI Router reply ({provider}): '{reply[:100]}'")
            # Check if the reply is just an echo or error
            is_echo = reply.lower().startswith("received:") or reply == user_msg
            is_error = provider == "error" or not reply or len(reply) < 3
            if not is_echo and not is_error:
                return jsonify({
                    "content": [{"type": "text", "text": reply}],
                    "provider": provider,
                    "complexity": router_data.get("complexity", "unknown"),
                    "emotion": router_data.get("emotion", "neutral"),
                })
            else:
                print(f"  ⚠ AI Router returned echo/error (provider={provider}), falling back")
        else:
            print(f"  ⚠ AI Router non-200: {resp.text[:200]}")
    except Exception as e:
        print(f"  ⚠ AI Router unavailable ({e}), falling back to direct API")

    # Fallback: direct to Anthropic
    api_key = config.get("anthropic_api_key", "")
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        # Try environment variable
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("  ❌ No API key available!")
        return jsonify({
            "content": [{"type": "text", "text": "[confused] I can't think right now. My AI brain isn't connected. Please check the API keys!"}],
            "provider": "none",
        })
    try:
        print(f"  → Trying direct Anthropic API...")
        resp = http_requests.post("https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"},
            json={"model": config.get("model", "claude-haiku-4-5-20251001"), "max_tokens": data.get("max_tokens", 1500),
                  "system": data.get("system", ""), "messages": data.get("messages", [])},
            timeout=15)  # 15s max for direct Anthropic
        print(f"  → Anthropic API status: {resp.status_code}")
        if resp.status_code != 200:
            print(f"  ❌ Anthropic error: {resp.text[:300]}")
        return Response(resp.content, status=resp.status_code, content_type="application/json")
    except Exception as e:
        print(f"  ❌ Anthropic API error: {e}")
        return jsonify({"error": str(e)}), 500

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
            communicate = edge_tts.Communicate("Γεια σου! Είμαι ο VIRON.", "el-GR-NestorasNeural", rate="+10%", pitch="-2st")
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            buf.seek(0)
            return buf
        loop = asyncio.new_event_loop()
        buf = loop.run_until_complete(gen())
        loop.close()
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
            json={"model": config.get("model", "claude-haiku-4-5-20251001"), "max_tokens": 100,
                  "messages": [{"role": "user", "content": "Say hello in 5 words"}]},
            timeout=15)
        return f"Status: {resp.status_code}\n{resp.text[:500]}", resp.status_code
    except Exception as e:
        return f"Error: {e}", 500

# ============ STUDENT EMOTION ============
@app.route('/api/student/emotion', methods=['GET'])
def student_emotion():
    return jsonify(detector.get_state())

@app.route('/api/student/start-detection', methods=['POST'])
def start_detection():
    cam = request.json.get('camera_index', config.get('camera_index', 0)) if request.json else config.get('camera_index', 0)
    return jsonify({"status": "started" if detector.start(cam) else "failed"})

@app.route('/api/student/stop-detection', methods=['POST'])
def stop_detection():
    detector.stop()
    return jsonify({"status": "stopped"})

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
                return jsonify({"connected": True, "ssid": p[2]})
        return jsonify({"connected": False})
    except:
        return jsonify({"connected": False})

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
    """Generate speech audio from text. Returns MP3. Uses edge-tts for male voice, gTTS fallback."""
    data = request.get_json()
    if not data or not data.get('text'):
        return jsonify({"error": "No text"}), 400
    text = data['text']
    lang = data.get('lang', 'el')
    
    # Try edge-tts first (has male Greek voice)
    try:
        import edge_tts, asyncio, io
        # Male voices: el-GR-NestorasNeural (Greek), en-GB-RyanNeural (soft British male)
        voice = "el-GR-NestorasNeural" if lang == "el" else "en-GB-RyanNeural"
        print(f"🎙️ edge-tts: voice={voice}, text='{text[:50]}'")
        
        async def gen():
            communicate = edge_tts.Communicate(text, voice, rate="+10%", pitch="-2st")
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            buf.seek(0)
            return buf
        
        loop = asyncio.new_event_loop()
        buf = loop.run_until_complete(gen())
        loop.close()
        audio_bytes = buf.read()
        print(f"✅ edge-tts OK: {len(audio_bytes)} bytes")
        return Response(audio_bytes, mimetype='audio/mpeg',
                       headers={'Content-Disposition': 'inline'})
    except ImportError:
        print("⚠ edge-tts not installed, using gTTS (pip install edge-tts)")
    except Exception as e:
        import traceback
        print(f"⚠ edge-tts error: {e}")
        traceback.print_exc()
        print("Falling back to gTTS")
    
    # Fallback: gTTS
    if not HAS_GTTS:
        return jsonify({"error": "No TTS engine available"}), 500
    try:
        import io
        tts = gTTS(text=text, lang=lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return Response(buf.read(), mimetype='audio/mpeg',
                       headers={'Content-Disposition': 'inline'})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

# ============ START ============
if __name__ == '__main__':
    port = config['port']
    has_key = config.get('anthropic_api_key', '') not in ['', 'YOUR_API_KEY_HERE']
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
        socketio.run(app, host='0.0.0.0', port=port, debug=False)
    else:
        app.run(host='0.0.0.0', port=port, debug=False)
