"""
VIRON Face Recognition Module
Uses Haar Cascade for detection + SFace for recognition.
Compatible with OpenCV 4.5.4+
"""

import cv2
import numpy as np
import json
import os
import time
import base64
from pathlib import Path

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
FACES_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces_db.json")
RECOGNIZER_MODEL = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

COSINE_THRESHOLD = 0.363


class FaceRecognizer:
    def __init__(self):
        self.face_cascade = None
        self.recognizer = None
        self.known_faces = {}
        self.current_person = None
        self.current_confidence = 0.0
        self.last_recognition_time = 0
        self.recognition_interval = 1.5
        self.consecutive_matches = {}
        self.initialized = False

    def initialize(self):
        """Load cascade + SFace recognizer"""
        # Load Haar cascade (always available)
        try:
            cascade_path = None
            if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            else:
                for p in ['/usr/share/opencv4/haarcascades/',
                          '/usr/share/opencv/haarcascades/',
                          '/usr/local/share/opencv4/haarcascades/']:
                    if os.path.exists(p + 'haarcascade_frontalface_default.xml'):
                        cascade_path = p + 'haarcascade_frontalface_default.xml'
                        break
            
            if not cascade_path or not os.path.exists(cascade_path):
                print("⚠ Haar cascade not found")
                return False
            
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print(f"  ✅ Face cascade loaded")
        except Exception as e:
            print(f"⚠ Cascade error: {e}")
            return False

        # Load SFace recognizer
        if not os.path.exists(RECOGNIZER_MODEL) or os.path.getsize(RECOGNIZER_MODEL) < 100000:
            print(f"⚠ SFace model not found or too small: {RECOGNIZER_MODEL}")
            print("  Download from: https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx")
            return False

        try:
            self.recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_MODEL, "")
            print(f"  ✅ SFace recognizer loaded")
        except Exception as e:
            print(f"⚠ SFace init error: {e}")
            print("  OpenCV version:", cv2.__version__)
            return False

        self._load_faces()
        self.initialized = True
        print(f"👤 Face recognition initialized ({len(self.known_faces)} known faces)")
        return True

    def _load_faces(self):
        if not os.path.exists(FACES_DB):
            self.known_faces = {}
            return
        try:
            with open(FACES_DB, 'r') as f:
                data = json.load(f)
            self.known_faces = {}
            for name, encodings_b64 in data.items():
                self.known_faces[name] = [
                    np.frombuffer(base64.b64decode(enc), dtype=np.float32)
                    for enc in encodings_b64
                ]
            print(f"  Loaded faces: {list(self.known_faces.keys())}")
        except Exception as e:
            print(f"  ⚠ Error loading faces: {e}")
            self.known_faces = {}

    def _save_faces(self):
        data = {}
        for name, encodings in self.known_faces.items():
            data[name] = [
                base64.b64encode(enc.tobytes()).decode('ascii')
                for enc in encodings
            ]
        os.makedirs(os.path.dirname(FACES_DB), exist_ok=True)
        with open(FACES_DB, 'w') as f:
            json.dump(data, f)
        print(f"  💾 Saved {len(self.known_faces)} faces to {FACES_DB}")

    def _detect_face_bbox(self, frame):
        """Detect face using Haar cascade, return (x,y,w,h) or None"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        # Try with lenient params first
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(60, 60))
        if len(faces) == 0:
            # Retry with even more lenient params
            faces = self.face_cascade.detectMultiScale(gray, 1.05, 2, minSize=(40, 40))
        if len(faces) == 0:
            return None
        # Return largest face
        return max(faces, key=lambda f: f[2] * f[3])

    def _detect_all_face_bboxes(self, frame):
        """Detect ALL faces in frame, return list of (x,y,w,h)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(60, 60))
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(gray, 1.05, 2, minSize=(40, 40))
        return list(faces) if len(faces) > 0 else []

    def _get_face_crop(self, frame, bbox):
        """Crop and resize face to 112x112 for SFace"""
        x, y, w, h = bbox
        # Add padding (20%)
        pad = int(max(w, h) * 0.2)
        fh, fw = frame.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(fw, x + w + pad)
        y2 = min(fh, y + h + pad)
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None
        face_resized = cv2.resize(face_crop, (112, 112))
        return face_resized

    def _get_encoding(self, face_112):
        """Get SFace encoding from 112x112 face image"""
        try:
            blob = cv2.dnn.blobFromImage(face_112, 1.0, (112, 112),
                                          (0, 0, 0), swapRB=False, crop=False)
            self.recognizer.model.setInput(blob)
            encoding = self.recognizer.model.forward()
            return encoding.flatten()
        except AttributeError:
            # Newer OpenCV: use feature() with proper input
            try:
                encoding = self.recognizer.feature(face_112)
                return encoding.flatten()
            except Exception as e:
                print(f"  ⚠ Encoding error: {e}")
                return None
        except Exception as e:
            print(f"  ⚠ Encoding error: {e}")
            return None

    def register_face(self, frame, name):
        """Register a face from a camera frame"""
        if not self.initialized:
            return False, "Not initialized"

        bbox = self._detect_face_bbox(frame)
        if bbox is None:
            return False, f"No face detected ({frame.shape[1]}x{frame.shape[0]})"

        face_112 = self._get_face_crop(frame, bbox)
        if face_112 is None:
            return False, "Could not crop face"

        encoding = self._get_encoding(face_112)
        if encoding is None:
            return False, "Could not compute encoding"

        if name not in self.known_faces:
            self.known_faces[name] = []
        self.known_faces[name].append(encoding)
        self._save_faces()

        count = len(self.known_faces[name])
        print(f"  ✅ Registered face for {name} ({count} samples)")
        return True, f"Face registered for {name} ({count} samples)"

    def register_face_from_base64(self, image_b64, name):
        if not self.initialized:
            return False, "Not initialized"
        try:
            img_data = base64.b64decode(image_b64.split(',')[-1])
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return False, "Could not decode image"
            return self.register_face(frame, name)
        except Exception as e:
            return False, f"Error: {e}"

    def _cosine_similarity(self, a, b):
        """Compute cosine similarity between two vectors"""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0
        return dot / (norm_a * norm_b)

    def recognize(self, frame):
        """Recognize face in frame. Returns (name, confidence) or (None, 0)"""
        if not self.initialized or not self.known_faces:
            return None, 0.0

        now = time.time()
        if now - self.last_recognition_time < self.recognition_interval:
            return self.current_person, self.current_confidence
        self.last_recognition_time = now

        bbox = self._detect_face_bbox(frame)
        if bbox is None:
            self.consecutive_matches = {}
            return self.current_person, self.current_confidence

        face_112 = self._get_face_crop(frame, bbox)
        if face_112 is None:
            return self.current_person, self.current_confidence

        encoding = self._get_encoding(face_112)
        if encoding is None:
            return self.current_person, self.current_confidence

        best_name = None
        best_score = -1

        for name, encodings in self.known_faces.items():
            for known_enc in encodings:
                score = self._cosine_similarity(encoding, known_enc)
                if score > best_score:
                    best_score = score
                    best_name = name

        if best_score > COSINE_THRESHOLD and best_name:
            self.consecutive_matches[best_name] = self.consecutive_matches.get(best_name, 0) + 1
            for n in list(self.consecutive_matches.keys()):
                if n != best_name:
                    self.consecutive_matches[n] = 0
            if self.consecutive_matches[best_name] >= 2:
                self.current_person = best_name
                self.current_confidence = best_score
        else:
            self.consecutive_matches = {}
            self.current_person = None
            self.current_confidence = 0.0

        return self.current_person, self.current_confidence

    def recognize_all(self, frame):
        """Recognize ALL faces in frame. Returns list of {name, confidence} dicts."""
        if not self.initialized or not self.known_faces:
            return []

        bboxes = self._detect_all_face_bboxes(frame)
        if not bboxes:
            return []

        results = []
        seen_names = set()

        for bbox in bboxes:
            face_112 = self._get_face_crop(frame, bbox)
            if face_112 is None:
                continue
            encoding = self._get_encoding(face_112)
            if encoding is None:
                continue

            best_name = None
            best_score = -1
            for name, encodings in self.known_faces.items():
                for known_enc in encodings:
                    score = self._cosine_similarity(encoding, known_enc)
                    if score > best_score:
                        best_score = score
                        best_name = name

            if best_score > COSINE_THRESHOLD and best_name and best_name not in seen_names:
                results.append({"name": best_name, "confidence": round(best_score, 3)})
                seen_names.add(best_name)

        # Update current_person to the highest-confidence match
        if results:
            best = max(results, key=lambda r: r["confidence"])
            self.current_person = best["name"]
            self.current_confidence = best["confidence"]

        return results

    def delete_face(self, name):
        if name in self.known_faces:
            del self.known_faces[name]
            self._save_faces()
            return True, f"Deleted: {name}"
        return False, f"Not found: {name}"

    def list_faces(self):
        return {name: len(encs) for name, encs in self.known_faces.items()}

    def get_status(self):
        return {
            "initialized": self.initialized,
            "current_person": self.current_person,
            "confidence": round(self.current_confidence, 3),
            "known_faces": self.list_faces()
        }


face_recognizer = FaceRecognizer()
