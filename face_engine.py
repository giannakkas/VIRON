"""
Face recognition engine — detection + matching.

Wraps the `face_recognition` library (built on dlib). Loads known faces from
FaceDB into memory at startup; on each frame, runs detection, encodes any
faces found, and matches against the in-memory set using L2 distance.

Distance threshold:
    distance < 0.45  → very confident (auto-update encoding)
    distance < 0.55  → confident (recognized as that person)
    distance < 0.60  → tentative (still recognized but no encoding update)
    distance >= 0.60 → unknown person

These values come from dlib's empirical face_recognition_model_v1 paper —
0.6 is the canonical threshold for "definitely a different person".

The engine is lazy: it imports `face_recognition` only on first use, so
the pipeline still boots if the library isn't installed yet (vision just
silently turns off for face matching, snapshot-to-Gemini still works).
"""
import io
import logging
import threading
import time
from typing import Optional

import numpy as np

from face_db import FaceDB, ENCODING_DIM

log = logging.getLogger("VIRON")

# Distance thresholds — see module docstring
THR_AUTO_UPDATE = 0.45
THR_CONFIDENT   = 0.55
THR_TENTATIVE   = 0.60


class FaceEngine:
    def __init__(self, db: FaceDB):
        self.db = db
        self._fr = None              # face_recognition module
        self._cv2 = None             # opencv module
        self._available = None       # tri-state: None/True/False
        self._known_ids: list[int] = []
        self._known_names: list[str] = []
        self._known_roles: list[str] = []
        self._known_enc: Optional[np.ndarray] = None  # (N, 128) stacked
        self._lock = threading.Lock()
        self._last_warm = 0.0

    # ─── library availability / warmup ───────────────────────

    def _ensure_loaded(self) -> bool:
        """Lazy-import face_recognition + cv2. Returns True if available."""
        if self._available is not None:
            return self._available
        try:
            import face_recognition  # type: ignore
            import cv2               # type: ignore
            self._fr = face_recognition
            self._cv2 = cv2
            self._available = True
            log.info("📸 face_recognition loaded — family recognition enabled")
            self.reload_known()
            return True
        except Exception as e:
            self._available = False
            log.warning(f"📸 face_recognition unavailable: {e} — "
                        "install with: pip3 install --break-system-packages face_recognition")
            return False

    def warm(self):
        """Pre-load the model so the first real recognition isn't slow.
        Call this once at startup from a background thread."""
        if not self._ensure_loaded():
            return
        # Trigger model download/load with a tiny dummy image
        try:
            t0 = time.time()
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self._fr.face_locations(dummy, model="hog")
            log.info(f"📸 Face engine warmed in {time.time()-t0:.1f}s")
        except Exception as e:
            log.warning(f"📸 Face engine warmup failed: {e}")

    def is_available(self) -> bool:
        return self._available is True

    # ─── known-face cache ────────────────────────────────────

    def reload_known(self):
        """Refresh the in-memory matrix of known encodings from the database.
        Call this after every enrollment / deletion."""
        with self._lock:
            entries = self.db.get_all_encodings()
            if not entries:
                self._known_ids, self._known_names, self._known_roles = [], [], []
                self._known_enc = None
                log.info("📸 No enrolled faces")
                return
            self._known_ids   = [e[0] for e in entries]
            self._known_names = [e[1] for e in entries]
            self._known_roles = [e[2] for e in entries]
            self._known_enc   = np.stack([e[3] for e in entries], axis=0)
            log.info(f"📸 Loaded {len(self._known_ids)} enrolled face(s): "
                     + ", ".join(self._known_names))

    # ─── helpers ─────────────────────────────────────────────

    def _decode_jpeg(self, jpeg_bytes: bytes) -> Optional[np.ndarray]:
        """Decode JPEG → RGB ndarray (face_recognition expects RGB, not BGR)."""
        if not self._ensure_loaded():
            return None
        cv2 = self._cv2
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None or bgr.size == 0:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _crop_thumbnail(self, rgb: np.ndarray, location: tuple,
                        size: int = 200) -> bytes:
        """Crop a face from the image, resize, encode as JPG. Used for the
        UI thumbnail shown next to the person's name in settings."""
        cv2 = self._cv2
        top, right, bottom, left = location
        # Pad 20% so we get some context, not just an extreme crop
        h = bottom - top
        w = right - left
        pad_y = int(h * 0.2)
        pad_x = int(w * 0.2)
        H, W = rgb.shape[:2]
        top    = max(0, top - pad_y)
        bottom = min(H, bottom + pad_y)
        left   = max(0, left - pad_x)
        right  = min(W, right + pad_x)
        crop_rgb = rgb[top:bottom, left:right]
        if crop_rgb.size == 0:
            return b""
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        # Resize to a square thumbnail
        crop_bgr = cv2.resize(crop_bgr, (size, size), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buf.tobytes() if ok else b""

    # ─── public API ──────────────────────────────────────────

    def detect_and_identify(self, jpeg_bytes: bytes) -> list[dict]:
        """Detect all faces in a JPEG; identify each against the known set.

        Returns a list of dicts:
            { "name": "Andreas" | "Unknown",
              "face_id": 3 | None,
              "role": "student",
              "confidence": 0.87,         # 1.0 - distance/THR_TENTATIVE
              "distance": 0.42,
              "location": [top, right, bottom, left] }

        Empty list = no faces detected.
        """
        if not self._ensure_loaded():
            return []

        rgb = self._decode_jpeg(jpeg_bytes)
        if rgb is None:
            return []

        # HOG model is CPU-only and ~5x faster than CNN on Jetson Orin Nano,
        # accuracy is fine for frontal faces in normal indoor lighting.
        try:
            locations = self._fr.face_locations(rgb, model="hog")
        except Exception as e:
            log.warning(f"📸 face_locations failed: {e}")
            return []
        if not locations:
            return []

        try:
            encodings = self._fr.face_encodings(rgb, known_face_locations=locations,
                                                 num_jitters=1, model="small")
        except Exception as e:
            log.warning(f"📸 face_encodings failed: {e}")
            return []

        results = []
        with self._lock:
            known = self._known_enc
            ids = list(self._known_ids)
            names = list(self._known_names)
            roles = list(self._known_roles)

        for loc, enc in zip(locations, encodings):
            if known is None or len(known) == 0:
                results.append({
                    "name": "Unknown", "face_id": None, "role": None,
                    "confidence": 0.0, "distance": None,
                    "location": list(loc),
                })
                continue
            # L2 distance to every known face
            distances = np.linalg.norm(known - enc, axis=1)
            best_idx = int(np.argmin(distances))
            best_dist = float(distances[best_idx])

            if best_dist < THR_TENTATIVE:
                face_id = ids[best_idx]
                name    = names[best_idx]
                role    = roles[best_idx]
                conf    = max(0.0, 1.0 - best_dist / THR_TENTATIVE)

                # Drift update: only when very confident, prevents bad frames
                # from corrupting the encoding
                if best_dist < THR_AUTO_UPDATE:
                    self.db.update_encoding(face_id, enc, averaging_weight=0.05)

                # Bump recognition stats
                self.db.touch_recognition(face_id)

                results.append({
                    "name": name, "face_id": face_id, "role": role,
                    "confidence": round(conf, 2),
                    "distance": round(best_dist, 3),
                    "location": list(loc),
                })
            else:
                results.append({
                    "name": "Unknown", "face_id": None, "role": None,
                    "confidence": 0.0,
                    "distance": round(best_dist, 3),
                    "location": list(loc),
                })

        return results

    def enroll(self, name: str, role: str, jpeg_bytes: bytes) -> dict:
        """Enroll a new face from a JPEG snapshot. Returns:
            { "success": bool, "face_id": int|None, "error": str|None,
              "name": str }
        Requires exactly one face in the frame. Multiple faces or no face
        is an error — caller should retake the snapshot."""
        if not self._ensure_loaded():
            return {"success": False, "error": "face_recognition not installed"}
        if not name or not name.strip():
            return {"success": False, "error": "Name required"}

        rgb = self._decode_jpeg(jpeg_bytes)
        if rgb is None:
            return {"success": False, "error": "Could not decode image"}

        locations = self._fr.face_locations(rgb, model="hog")
        if len(locations) == 0:
            return {"success": False, "error": "No face found — point camera at the person and try again"}
        if len(locations) > 1:
            return {"success": False, "error": f"Found {len(locations)} faces — only one person should be in frame"}

        encodings = self._fr.face_encodings(rgb, known_face_locations=locations,
                                             num_jitters=3, model="small")
        if not encodings:
            return {"success": False, "error": "Could not encode face — try again with better lighting"}

        encoding = encodings[0]
        thumbnail = self._crop_thumbnail(rgb, locations[0])

        face_id = self.db.add_face(name, role, encoding, thumbnail)
        self.reload_known()
        log.info(f"📸 Enrolled new face: '{name}' (role={role}) id={face_id}")
        return {"success": True, "face_id": face_id, "name": name, "error": None}

    def delete(self, face_id: int) -> bool:
        ok = self.db.delete_face(face_id)
        if ok:
            self.reload_known()
        return ok

    # ─── prompt helper ───────────────────────────────────────

    def vision_context_for_prompt(self, recognitions: list[dict]) -> str:
        """Build a short text snippet for the system prompt that tells the
        model who is in the camera right now. Returns '' if no useful info."""
        if not recognitions:
            return ""
        lines = []
        known = [r for r in recognitions if r["face_id"] is not None]
        unknown_count = sum(1 for r in recognitions if r["face_id"] is None)

        if known:
            people = ", ".join(
                f"{r['name']} ({r['role']})" if r["role"] and r["role"] != "other"
                else r["name"]
                for r in known
            )
            lines.append(f"Visible in camera right now: {people}.")
            # Identify primary student if exactly one student is present
            students = [r for r in known if r["role"] == "student"]
            if len(students) == 1:
                lines.append(f"You are tutoring {students[0]['name']}.")
        if unknown_count:
            lines.append(f"There {'is' if unknown_count == 1 else 'are'} also "
                         f"{unknown_count} unrecognized "
                         f"{'person' if unknown_count == 1 else 'people'} in frame.")
        if not known and unknown_count:
            lines.append("No registered family member is visible right now — "
                         "be polite and a little more guarded than usual.")
        return " ".join(lines)
