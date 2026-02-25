"""
VIRON Face Recognition Module
Uses OpenCV FaceDetectorYN + FaceRecognizerSF for face detection and recognition.
Stores face encodings in a JSON file for persistence.
"""

import cv2
import numpy as np
import json
import os
import time
import base64
import urllib.request
from pathlib import Path

# Verify OpenCV has required features
if not hasattr(cv2, 'FaceDetectorYN'):
    raise ImportError(f"OpenCV {cv2.__version__} missing FaceDetectorYN. Need OpenCV 4.5.4+")
if not hasattr(cv2, 'FaceRecognizerSF'):
    raise ImportError(f"OpenCV {cv2.__version__} missing FaceRecognizerSF. Need OpenCV 4.5.4+")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
FACES_DB = os.path.join(os.path.dirname(__file__), "faces_db.json")

# Model files
DETECTOR_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
RECOGNIZER_MODEL = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

# Download URLs
DETECTOR_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
RECOGNIZER_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# Recognition thresholds
COSINE_THRESHOLD = 0.363  # OpenCV default for SFace cosine similarity
L2_THRESHOLD = 1.128      # OpenCV default for SFace L2 distance


class FaceRecognizer:
    def __init__(self):
        self.detector = None
        self.recognizer = None
        self.known_faces = {}  # name -> list of encodings (numpy arrays)
        self.current_person = None
        self.current_confidence = 0.0
        self.last_recognition_time = 0
        self.recognition_interval = 1.0  # Check every 1 second (not every frame)
        self.consecutive_matches = {}  # name -> count (need 3 consecutive matches)
        self.initialized = False
        
    def _download_model(self, url, dest):
        """Download a model file if missing"""
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"  📥 Downloading {os.path.basename(dest)}...")
        try:
            urllib.request.urlretrieve(url, dest)
            size = os.path.getsize(dest)
            if size > 100000:
                print(f"  ✅ Downloaded ({size // 1024}KB)")
                return True
            else:
                print(f"  ⚠ File too small ({size} bytes), download may have failed")
                os.remove(dest)
                return False
        except Exception as e:
            print(f"  ⚠ Download failed: {e}")
            return False
    
    def initialize(self):
        """Load models and face database"""
        # Auto-download models if missing
        if not os.path.exists(DETECTOR_MODEL) or os.path.getsize(DETECTOR_MODEL) < 100000:
            if not self._download_model(DETECTOR_URL, DETECTOR_MODEL):
                print(f"⚠ Face detector model not found. Run: bash backend/setup_models.sh")
                return False
        
        if not os.path.exists(RECOGNIZER_MODEL) or os.path.getsize(RECOGNIZER_MODEL) < 100000:
            if not self._download_model(RECOGNIZER_URL, RECOGNIZER_MODEL):
                print(f"⚠ Face recognizer model not found. Run: bash backend/setup_models.sh")
                return False
            
        try:
            # Initialize face detector (YuNet)
            self.detector = cv2.FaceDetectorYN.create(
                DETECTOR_MODEL,
                "",
                (640, 480),  # Will be updated per frame
                0.7,         # Score threshold
                0.3,         # NMS threshold
                5000         # Top K
            )
            
            # Initialize face recognizer (SFace)
            self.recognizer = cv2.FaceRecognizerSF.create(
                RECOGNIZER_MODEL, ""
            )
            
            # Load saved faces
            self._load_faces()
            
            self.initialized = True
            print(f"👤 Face recognition initialized ({len(self.known_faces)} known faces)")
            return True
            
        except Exception as e:
            print(f"⚠ Face recognition init error: {e}")
            return False
    
    def _load_faces(self):
        """Load face encodings from JSON file"""
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
        """Save face encodings to JSON file"""
        data = {}
        for name, encodings in self.known_faces.items():
            data[name] = [
                base64.b64encode(enc.tobytes()).decode('ascii')
                for enc in encodings
            ]
        with open(FACES_DB, 'w') as f:
            json.dump(data, f)
        print(f"  💾 Saved {len(self.known_faces)} faces to {FACES_DB}")
    
    def detect_faces(self, frame):
        """Detect faces in frame, return face data array"""
        if not self.initialized or self.detector is None:
            return None
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        return faces
    
    def get_encoding(self, frame, face):
        """Get face encoding (feature vector) for a detected face"""
        if not self.initialized or self.recognizer is None:
            return None
        try:
            aligned = self.recognizer.alignCrop(frame, face)
            encoding = self.recognizer.feature(aligned)
            return encoding.flatten()
        except Exception as e:
            print(f"  ⚠ Encoding error: {e}")
            return None
    
    def register_face(self, frame, name):
        """Register a face from a camera frame"""
        if not self.initialized:
            return False, "Face recognition not initialized"
        
        faces = self.detect_faces(frame)
        if faces is None or len(faces) == 0:
            return False, "No face detected in frame"
        
        if len(faces) > 1:
            return False, "Multiple faces detected — only one person should be in frame"
        
        face = faces[0]
        encoding = self.get_encoding(frame, face)
        if encoding is None:
            return False, "Could not generate face encoding"
        
        # Add to known faces
        if name not in self.known_faces:
            self.known_faces[name] = []
        
        self.known_faces[name].append(encoding)
        self._save_faces()
        
        count = len(self.known_faces[name])
        return True, f"Face registered for {name} ({count} sample{'s' if count > 1 else ''})"
    
    def register_face_from_base64(self, image_b64, name):
        """Register a face from a base64-encoded image"""
        if not self.initialized:
            return False, "Face recognition not initialized"
        
        try:
            # Decode base64 image
            img_data = base64.b64decode(image_b64.split(',')[-1])  # Handle data:image/... prefix
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return False, "Could not decode image"
            return self.register_face(frame, name)
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def recognize(self, frame):
        """Recognize faces in frame. Returns (name, confidence) or (None, 0)"""
        if not self.initialized or not self.known_faces:
            return None, 0.0
        
        # Rate limit recognition (expensive operation)
        now = time.time()
        if now - self.last_recognition_time < self.recognition_interval:
            return self.current_person, self.current_confidence
        self.last_recognition_time = now
        
        faces = self.detect_faces(frame)
        if faces is None or len(faces) == 0:
            # No face — clear after a few seconds
            self.consecutive_matches = {}
            if now - self.last_recognition_time > 3:
                self.current_person = None
                self.current_confidence = 0.0
            return self.current_person, self.current_confidence
        
        # Use largest face (closest person)
        face = max(faces, key=lambda f: f[2] * f[3])
        encoding = self.get_encoding(frame, face)
        if encoding is None:
            return self.current_person, self.current_confidence
        
        # Compare against all known faces
        best_name = None
        best_score = -1
        
        for name, encodings in self.known_faces.items():
            for known_enc in encodings:
                # Cosine similarity (higher = more similar, max 1.0)
                score = self.recognizer.match(
                    encoding.reshape(1, -1),
                    known_enc.reshape(1, -1),
                    cv2.FaceRecognizerSF_FR_COSINE
                )
                if score > best_score:
                    best_score = score
                    best_name = name
        
        # Check if match is good enough
        if best_score > COSINE_THRESHOLD and best_name:
            # Increment consecutive match counter
            self.consecutive_matches[best_name] = self.consecutive_matches.get(best_name, 0) + 1
            # Reset others
            for n in list(self.consecutive_matches.keys()):
                if n != best_name:
                    self.consecutive_matches[n] = 0
            
            # Need 2+ consecutive matches to confirm identity
            if self.consecutive_matches[best_name] >= 2:
                self.current_person = best_name
                self.current_confidence = best_score
        else:
            self.consecutive_matches = {}
            self.current_person = None
            self.current_confidence = 0.0
        
        return self.current_person, self.current_confidence
    
    def delete_face(self, name):
        """Remove a registered face"""
        if name in self.known_faces:
            del self.known_faces[name]
            self._save_faces()
            return True, f"Deleted face: {name}"
        return False, f"Face not found: {name}"
    
    def list_faces(self):
        """List all registered faces"""
        return {
            name: len(encodings) 
            for name, encodings in self.known_faces.items()
        }
    
    def get_status(self):
        """Get current recognition status"""
        return {
            "initialized": self.initialized,
            "current_person": self.current_person,
            "confidence": round(self.current_confidence, 3),
            "known_faces": self.list_faces()
        }


# Singleton instance
face_recognizer = FaceRecognizer()
