#!/usr/bin/env python3
"""
VIRON AI Tutor Backend - Jetson Orin Nano
Real-time student emotion detection + hardware APIs + WebSocket
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import subprocess, json, os, re, time, threading, sys, base64, mimetypes

# Optional: WebSocket for real-time emotion push
try:
    from flask_socketio import SocketIO, emit
    HAS_SOCKETIO = True
except ImportError:
    HAS_SOCKETIO = False
    print("⚠ flask-socketio not installed. Using polling mode.")

# Optional: OpenCV for face detection
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠ OpenCV not installed. Student emotion detection disabled.")

app = Flask(__name__, static_folder='.')
CORS(app)

# ============================================
# FILE UPLOAD CONFIG
# ============================================
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'md', 'csv', 'json'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if HAS_SOCKETIO:
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
else:
    socketio = None

# ============================================
# STUDENT EMOTION DETECTION
# ============================================

class StudentEmotionDetector:
    """
    Detects student facial expressions using OpenCV.
    Uses Haar cascades for face/eye/mouth detection and
    analyzes geometry to infer emotional state.
    """

    def __init__(self):
        self.running = False
        self.cap = None
        self.current_emotion = "neutral"
        self.confidence = 0.0
        self.face_detected = False
        self.engagement_score = 100  # 0-100
        self.attention_direction = "center"
        self.last_emotion_change = time.time()
        self.emotion_history = []  # Track patterns over time
        self.blink_rate = 0
        self.head_tilt = 0
        self.mouth_open = False

        # Engagement tracking
        self.no_face_start = None
        self.distracted_start = None
        self.yawn_count = 0
        self.last_yawn_time = 0

        # Load cascades
        if HAS_CV2:
            cascade_dir = cv2.data.haarcascades
            self.face_cascade = cv2.CascadeClassifier(cascade_dir + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cascade_dir + 'haarcascade_eye.xml')
            self.smile_cascade = cv2.CascadeClassifier(cascade_dir + 'haarcascade_smile.xml')
            self.mouth_cascade = cv2.CascadeClassifier(cascade_dir + 'haarcascade_mcs_mouth.xml')

    def start(self, camera_index=0):
        if not HAS_CV2:
            return False
        try:
            self.cap = cv2.VideoCapture(camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            if not self.cap.isOpened():
                return False
            self.running = True
            self.thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            print(f"Camera error: {e}")
            return False

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def _detection_loop(self):
        """Main detection loop - runs in background thread"""
        frame_count = 0
        eyes_history = []
        smile_history = []

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            frame_count += 1
            if frame_count % 3 != 0:  # Process every 3rd frame
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))

            if len(faces) == 0:
                self.face_detected = False
                if self.no_face_start is None:
                    self.no_face_start = time.time()
                elif time.time() - self.no_face_start > 5:
                    self.engagement_score = max(0, self.engagement_score - 2)
                    self.current_emotion = "absent"
                continue

            self.face_detected = True
            self.no_face_start = None

            # Use largest face
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            face_roi = gray[y:y+h, x:x+w]

            # Face position → attention direction
            frame_center_x = frame.shape[1] / 2
            face_center_x = x + w / 2
            offset = (face_center_x - frame_center_x) / frame_center_x
            if abs(offset) < 0.2:
                self.attention_direction = "center"
            elif offset < -0.2:
                self.attention_direction = "left"
            else:
                self.attention_direction = "right"

            # Head tilt (face height/width ratio)
            self.head_tilt = h / w if w > 0 else 1.0

            # Detect eyes
            upper_face = face_roi[0:int(h*0.6), :]
            eyes = self.eye_cascade.detectMultiScale(upper_face, 1.1, 5, minSize=(20, 20))
            num_eyes = min(len(eyes), 2)
            eyes_history.append(num_eyes)
            if len(eyes_history) > 15:
                eyes_history.pop(0)

            # Blink rate (eyes not detected = blink)
            if len(eyes_history) >= 10:
                blinks = sum(1 for e in eyes_history if e < 2)
                self.blink_rate = blinks / len(eyes_history)

            # Detect smile
            lower_face = face_roi[int(h*0.5):, :]
            smiles = self.smile_cascade.detectMultiScale(lower_face, 1.8, 20, minSize=(25, 15))
            has_smile = len(smiles) > 0
            smile_history.append(has_smile)
            if len(smile_history) > 10:
                smile_history.pop(0)
            smile_ratio = sum(smile_history) / len(smile_history) if smile_history else 0

            # Detect mouth open (for confusion/surprise/yawn)
            mouths = self.mouth_cascade.detectMultiScale(lower_face, 1.5, 11, minSize=(25, 15))
            mouth_open = False
            if len(mouths) > 0:
                (mx, my, mw, mh) = max(mouths, key=lambda m: m[2]*m[3])
                mouth_aspect = mh / mw if mw > 0 else 0
                mouth_open = mouth_aspect > 0.5  # Tall mouth = open
                if mouth_aspect > 0.7 and time.time() - self.last_yawn_time > 30:
                    self.yawn_count += 1
                    self.last_yawn_time = time.time()
            self.mouth_open = mouth_open

            # ---- EMOTION CLASSIFICATION ----
            prev_emotion = self.current_emotion
            confidence = 0.5

            # Eye openness analysis
            eye_area_ratio = 0
            if num_eyes >= 2 and len(eyes) >= 2:
                total_eye_area = sum(ew * eh for (ex, ey, ew, eh) in eyes[:2])
                eye_area_ratio = total_eye_area / (w * h) if w * h > 0 else 0

            # Determine emotion from features
            if smile_ratio > 0.6:
                self.current_emotion = "happy"
                confidence = 0.5 + smile_ratio * 0.4
                self.engagement_score = min(100, self.engagement_score + 1)

            elif self.blink_rate > 0.5 and not has_smile:
                # Eyes often closed = sleepy/bored
                self.current_emotion = "sleepy"
                confidence = 0.4 + self.blink_rate * 0.3
                self.engagement_score = max(0, self.engagement_score - 1)

            elif mouth_open and eye_area_ratio > 0.02:
                # Open mouth + wide eyes = surprised or confused
                self.current_emotion = "confused"
                confidence = 0.55

            elif mouth_open and self.blink_rate > 0.4:
                # Open mouth + frequent blinks = yawning/bored
                self.current_emotion = "bored"
                confidence = 0.5
                self.engagement_score = max(0, self.engagement_score - 2)

            elif not has_smile and num_eyes >= 2 and eye_area_ratio < 0.01:
                # Small/squinted eyes, no smile
                self.current_emotion = "frustrated"
                confidence = 0.45

            elif not has_smile and self.head_tilt > 1.3:
                # Head tilted, no smile = thinking or confused
                self.current_emotion = "thinking"
                confidence = 0.45

            elif not has_smile and self.attention_direction != "center":
                # Looking away = distracted
                if self.distracted_start is None:
                    self.distracted_start = time.time()
                elif time.time() - self.distracted_start > 3:
                    self.current_emotion = "distracted"
                    confidence = 0.5
                    self.engagement_score = max(0, self.engagement_score - 1)
            else:
                self.distracted_start = None
                if self.attention_direction == "center" and num_eyes >= 2:
                    self.current_emotion = "attentive"
                    confidence = 0.5
                    self.engagement_score = min(100, self.engagement_score + 0.5)
                else:
                    self.current_emotion = "neutral"
                    confidence = 0.4

            self.confidence = confidence

            # Track emotion history
            self.emotion_history.append({
                'emotion': self.current_emotion,
                'time': time.time(),
                'confidence': confidence
            })
            # Keep last 60 entries (~1 min)
            if len(self.emotion_history) > 60:
                self.emotion_history.pop(0)

            # Emit via WebSocket
            if HAS_SOCKETIO and socketio and prev_emotion != self.current_emotion:
                socketio.emit('student_emotion', self.get_state())

            time.sleep(0.05)

    def get_state(self):
        """Get full student state"""
        # Analyze patterns
        recent = self.emotion_history[-20:] if len(self.emotion_history) >= 20 else self.emotion_history
        emotion_counts = {}
        for e in recent:
            emotion_counts[e['emotion']] = emotion_counts.get(e['emotion'], 0) + 1

        dominant = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"
        streak = 0
        if self.emotion_history:
            for e in reversed(self.emotion_history):
                if e['emotion'] == self.current_emotion:
                    streak += 1
                else:
                    break

        return {
            'emotion': self.current_emotion,
            'confidence': round(self.confidence, 2),
            'face_detected': self.face_detected,
            'engagement_score': round(self.engagement_score),
            'attention': self.attention_direction,
            'blink_rate': round(self.blink_rate, 2),
            'mouth_open': self.mouth_open,
            'dominant_emotion': dominant,
            'emotion_streak': streak,
            'yawn_count': self.yawn_count,
        }


# Global detector
detector = StudentEmotionDetector()

# ============================================
# EMOTION API
# ============================================
@app.route('/api/student/emotion', methods=['GET'])
def student_emotion():
    return jsonify(detector.get_state())

@app.route('/api/student/start-detection', methods=['POST'])
def start_detection():
    cam_idx = request.json.get('camera', 0) if request.json else 0
    ok = detector.start(cam_idx)
    return jsonify({'success': ok, 'message': 'Detection started' if ok else 'Failed to open camera'})

@app.route('/api/student/stop-detection', methods=['POST'])
def stop_detection():
    detector.stop()
    return jsonify({'success': True})

# ============================================
# WIFI
# ============================================
@app.route('/api/wifi/list', methods=['GET'])
def wifi_list():
    try:
        result = subprocess.run(
            ['nmcli', '-t', '-f', 'SSID,SIGNAL,SECURITY,ACTIVE', 'dev', 'wifi', 'list'],
            capture_output=True, text=True, timeout=10
        )
        networks = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip(): continue
            parts = line.split(':')
            if len(parts) >= 4:
                networks.append({'ssid':parts[0],'signal':int(parts[1]) if parts[1].isdigit() else 0,'security':parts[2] or 'Open','connected':parts[3]=='yes'})
        seen = {}
        for n in networks:
            if n['ssid'] and (n['ssid'] not in seen or n['signal'] > seen[n['ssid']]['signal']):
                seen[n['ssid']] = n
        return jsonify({'networks': list(seen.values())})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wifi/connect', methods=['POST'])
def wifi_connect():
    data = request.json
    ssid, pwd = data.get('ssid',''), data.get('password','')
    try:
        cmd = ['nmcli','dev','wifi','connect',ssid] + (['password',pwd] if pwd else [])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return jsonify({'success':result.returncode==0,'message':result.stderr or 'OK'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wifi/status', methods=['GET'])
def wifi_status():
    try:
        r = subprocess.run(['nmcli','-t','-f','GENERAL.CONNECTION','dev','show','wlan0'], capture_output=True, text=True, timeout=10)
        ssid = ''; connected = False
        for line in r.stdout.strip().split('\n'):
            if 'CONNECTION' in line:
                ssid = line.split(':')[-1].strip()
                connected = bool(ssid and ssid != '--')
        return jsonify({'connected':connected,'ssid':ssid})
    except Exception as e:
        return jsonify({'error':str(e)}), 500

# ============================================
# BATTERY
# ============================================
@app.route('/api/battery', methods=['GET'])
def battery_status():
    try:
        bp = '/sys/class/power_supply/battery'
        if os.path.exists(bp):
            cap = int(open(f'{bp}/capacity').read().strip())
            st = open(f'{bp}/status').read().strip()
            return jsonify({'percent':cap,'charging':st in ('Charging','Full'),'status':st})
        # INA219 fallback
        r = subprocess.run(['i2cget','-y','1','0x42','0x02','w'], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            raw = int(r.stdout.strip(), 16)
            v = ((raw&0xFF)<<8|(raw>>8))*1.25/1000
            pct = max(0,min(100,int((v-3.0)/1.2*100)))
            return jsonify({'percent':pct,'charging':v>4.15,'voltage':round(v,2)})
        return jsonify({'percent':78,'charging':True,'status':'simulated'})
    except Exception as e:
        return jsonify({'error':str(e),'percent':-1}), 500

# ============================================
# DISPLAY / VOLUME
# ============================================
@app.route('/api/brightness', methods=['POST'])
def set_brightness():
    level = max(10,min(100,int(request.json.get('brightness',80))))
    try:
        r = subprocess.run(['xrandr','--listmonitors'], capture_output=True, text=True, timeout=5)
        mons = re.findall(r'\d+:\s+\+\*?(\S+)', r.stdout)
        if mons:
            subprocess.run(['xrandr','--output',mons[0],'--brightness',str(level/100)], timeout=5)
        return jsonify({'success':True,'brightness':level})
    except Exception as e:
        return jsonify({'error':str(e)}), 500

@app.route('/api/volume', methods=['POST'])
def set_volume():
    level = max(0,min(100,int(request.json.get('volume',75))))
    try:
        subprocess.run(['amixer','set','Master',f'{level}%'], capture_output=True, timeout=5)
        return jsonify({'success':True,'volume':level})
    except Exception as e:
        return jsonify({'error':str(e)}), 500

# ============================================
# SYSTEM
# ============================================
@app.route('/api/system/info', methods=['GET'])
def system_info():
    try:
        with open('/proc/uptime') as f: up = float(f.read().split()[0])
        h,m = int(up//3600), int((up%3600)//60)
        temp = 0
        if os.path.exists('/sys/class/thermal/thermal_zone0/temp'):
            temp = int(open('/sys/class/thermal/thermal_zone0/temp').read().strip())/1000
        return jsonify({'uptime':f'{h}h {m}m','cpu_temp':round(temp,1),'platform':'Jetson Orin Nano'})
    except Exception as e:
        return jsonify({'error':str(e)}), 500

@app.route('/api/system/shutdown', methods=['POST'])
def shutdown():
    subprocess.Popen(['sudo','shutdown','-h','now'])
    return jsonify({'success':True})

@app.route('/api/system/restart', methods=['POST'])
def restart():
    subprocess.Popen(['sudo','reboot'])
    return jsonify({'success':True})

@app.route('/api/ping', methods=['GET'])
def ping():
    return jsonify({'status':'ok','device':'VIRON','emotion_detection':HAS_CV2,'websocket':HAS_SOCKETIO})

# ============================================
# FILE UPLOAD / LOADING
# ============================================
@app.route('/api/files/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(f.filename):
        return jsonify({'error': f'File type not allowed. Allowed: {", ".join(sorted(ALLOWED_EXTENSIONS))}'}), 400
    filename = secure_filename(f.filename)
    # Avoid overwriting: append timestamp if file exists
    base, ext = os.path.splitext(filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(filepath):
        filename = f"{base}_{int(time.time())}{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
    f.save(filepath)
    # Read text content for text-based files
    content = None
    ext_lower = ext.lower()
    if ext_lower in ('.txt', '.md', '.csv', '.json'):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as tf:
                content = tf.read(50000)  # Cap at 50k chars
        except Exception:
            pass
    return jsonify({
        'success': True,
        'filename': filename,
        'size': os.path.getsize(filepath),
        'type': mimetypes.guess_type(filename)[0] or 'application/octet-stream',
        'content': content
    })

@app.route('/api/files', methods=['GET'])
def list_files():
    files = []
    for fname in sorted(os.listdir(UPLOAD_FOLDER)):
        fpath = os.path.join(UPLOAD_FOLDER, fname)
        if os.path.isfile(fpath):
            files.append({
                'filename': fname,
                'size': os.path.getsize(fpath),
                'type': mimetypes.guess_type(fname)[0] or 'application/octet-stream',
                'modified': os.path.getmtime(fpath)
            })
    return jsonify({'files': files})

@app.route('/api/files/<filename>', methods=['GET'])
def get_file(filename):
    filename = secure_filename(filename)
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/api/files/<filename>', methods=['DELETE'])
def delete_file(filename):
    filename = secure_filename(filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        return jsonify({'success': True})
    return jsonify({'error': 'File not found'}), 404

@app.route('/api/files/<filename>/content', methods=['GET'])
def get_file_content(filename):
    """Return text content of a file for AI context"""
    filename = secure_filename(filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    ext = os.path.splitext(filename)[1].lower()
    if ext in ('.txt', '.md', '.csv', '.json'):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read(50000)
            return jsonify({'filename': filename, 'content': content})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    elif ext in ('.png', '.jpg', '.jpeg', '.gif'):
        # Return base64 for images
        try:
            with open(filepath, 'rb') as f:
                data = base64.b64encode(f.read()).decode('utf-8')
            mime = mimetypes.guess_type(filename)[0] or 'image/png'
            return jsonify({'filename': filename, 'image': f'data:{mime};base64,{data}'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Cannot read content of this file type'}), 400

# Serve HTML
@app.route('/')
def index():
    return send_from_directory('.', 'boot.html')

@app.route('/viron-complete.html')
def main_face():
    return send_from_directory('.', 'viron-complete.html')

@app.route('/viron-logo.png')
def logo():
    return send_from_directory('.', 'viron-logo.png')

if __name__ == '__main__':
    # Auto-start emotion detection
    if HAS_CV2:
        print("🎥 Starting student emotion detection...")
        if detector.start(0):
            print("✅ Camera opened, detecting student emotions")
        else:
            print("⚠ Could not open camera")

    print("🤖 VIRON Backend starting on http://localhost:5000")
    if HAS_SOCKETIO:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
    else:
        app.run(host='0.0.0.0', port=5000, debug=False)
