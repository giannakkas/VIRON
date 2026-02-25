"""
VIRON Voice Fingerprint — Speaker Verification
================================================
Uses Resemblyzer to create voice embeddings (fingerprints) for each student.
Before processing any speech, VIRON verifies the speaker is a registered student.

Enrollment: Student says 3+ phrases → embeddings stored
Verification: Compare incoming audio against stored embeddings
"""

import os
import json
import time
import numpy as np
import wave
import io
import tempfile
import base64
from typing import Optional, Dict, List, Tuple

# Paths
VOICE_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_prints.json")

# Thresholds
VERIFY_THRESHOLD = 0.75   # Cosine similarity threshold for accepting speaker
REJECT_THRESHOLD = 0.60   # Below this = definitely not the speaker

# Global state
_encoder = None
_voice_db = {}  # {name: {"embeddings": [list of embedding arrays], "enrolled_at": timestamp}}


def _load_encoder():
    """Load Resemblyzer encoder (lazy init — heavy model)."""
    global _encoder
    if _encoder is not None:
        return True
    try:
        from resemblyzer import VoiceEncoder
        _encoder = VoiceEncoder()
        print("  🎤 Voice encoder loaded (Resemblyzer)")
        return True
    except ImportError:
        print("  ⚠ Resemblyzer not installed. Install with: pip install resemblyzer")
        return False
    except Exception as e:
        print(f"  ⚠ Voice encoder error: {e}")
        return False


def _load_db():
    """Load voice prints database from disk."""
    global _voice_db
    if os.path.exists(VOICE_DB_PATH):
        try:
            with open(VOICE_DB_PATH, 'r') as f:
                data = json.load(f)
            # Convert embedding lists back to numpy arrays
            for name, info in data.items():
                info["embeddings"] = [np.array(e) for e in info["embeddings"]]
            _voice_db = data
            print(f"  🎤 Loaded {len(_voice_db)} voice prints")
        except Exception as e:
            print(f"  ⚠ Voice DB load error: {e}")
            _voice_db = {}
    else:
        _voice_db = {}


def _save_db():
    """Save voice prints database to disk."""
    try:
        # Convert numpy arrays to lists for JSON serialization
        data = {}
        for name, info in _voice_db.items():
            data[name] = {
                "embeddings": [e.tolist() if isinstance(e, np.ndarray) else e for e in info["embeddings"]],
                "enrolled_at": info.get("enrolled_at", ""),
                "sample_count": info.get("sample_count", 0),
            }
        with open(VOICE_DB_PATH, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"  ⚠ Voice DB save error: {e}")


def _audio_bytes_to_wav(audio_bytes: bytes, sample_rate: int = 16000) -> str:
    """Convert raw audio bytes to WAV file path for Resemblyzer."""
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    try:
        # If already a WAV file (starts with RIFF header)
        if audio_bytes[:4] == b'RIFF':
            tmp.write(audio_bytes)
        else:
            # Assume raw PCM 16-bit mono
            with wave.open(tmp.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)
        tmp.close()
        return tmp.name
    except Exception as e:
        tmp.close()
        os.unlink(tmp.name)
        raise e


def _get_embedding_from_audio(audio_bytes: bytes, sample_rate: int = 16000) -> Optional[np.ndarray]:
    """Extract voice embedding from audio bytes."""
    if not _load_encoder():
        return None
    
    from resemblyzer import preprocess_wav
    
    wav_path = None
    try:
        wav_path = _audio_bytes_to_wav(audio_bytes, sample_rate)
        wav = preprocess_wav(wav_path)
        
        if len(wav) < sample_rate * 0.5:  # Less than 0.5 seconds
            print("  ⚠ Audio too short for voice print (<0.5s)")
            return None
        
        embedding = _encoder.embed_utterance(wav)
        return embedding
    except Exception as e:
        print(f"  ⚠ Embedding extraction error: {e}")
        return None
    finally:
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)


def _get_embedding_from_base64(audio_b64: str, sample_rate: int = 16000) -> Optional[np.ndarray]:
    """Extract voice embedding from base64-encoded audio."""
    try:
        audio_bytes = base64.b64decode(audio_b64)
        return _get_embedding_from_audio(audio_bytes, sample_rate)
    except Exception as e:
        print(f"  ⚠ Base64 decode error: {e}")
        return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ═══════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════

def init():
    """Initialize voice fingerprint system."""
    _load_db()
    # Don't load encoder yet — it's heavy, load on first use
    print(f"  🎤 Voice fingerprint system ready ({len(_voice_db)} enrolled students)")


def enroll(name: str, audio_bytes: bytes, sample_rate: int = 16000) -> Dict:
    """
    Enroll a voice sample for a student.
    Call multiple times (3-5 samples) for better accuracy.
    
    Returns: {"success": bool, "message": str, "sample_count": int}
    """
    embedding = _get_embedding_from_audio(audio_bytes, sample_rate)
    if embedding is None:
        return {"success": False, "message": "Could not extract voice print. Try speaking louder.", "sample_count": 0}
    
    if name not in _voice_db:
        _voice_db[name] = {
            "embeddings": [],
            "enrolled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sample_count": 0,
        }
    
    _voice_db[name]["embeddings"].append(embedding)
    _voice_db[name]["sample_count"] = len(_voice_db[name]["embeddings"])
    _save_db()
    
    count = len(_voice_db[name]["embeddings"])
    quality = "basic" if count < 3 else "good" if count < 5 else "excellent"
    print(f"  🎤 Voice sample #{count} enrolled for {name} (quality: {quality})")
    
    return {
        "success": True,
        "message": f"Voice sample {count} recorded! ({quality} quality)",
        "sample_count": count,
        "quality": quality,
    }


def enroll_base64(name: str, audio_b64: str, sample_rate: int = 16000) -> Dict:
    """Enroll from base64-encoded audio (from browser MediaRecorder)."""
    try:
        audio_bytes = base64.b64decode(audio_b64)
        return enroll(name, audio_bytes, sample_rate)
    except Exception as e:
        return {"success": False, "message": f"Decode error: {e}", "sample_count": 0}


def verify(audio_bytes: bytes, sample_rate: int = 16000,
           expected_name: str = None) -> Dict:
    """
    Verify if the speaker matches a registered student.
    
    If expected_name is provided, only check against that student.
    Otherwise, check against all enrolled students.
    
    Returns: {
        "verified": bool,
        "name": str or None,
        "confidence": float,
        "message": str
    }
    """
    if not _voice_db:
        return {"verified": True, "name": None, "confidence": 0,
                "message": "No voice prints enrolled — allowing all"}
    
    embedding = _get_embedding_from_audio(audio_bytes, sample_rate)
    if embedding is None:
        # Can't verify — allow through (don't block on errors)
        return {"verified": True, "name": None, "confidence": 0,
                "message": "Could not extract voice print — allowing"}
    
    best_name = None
    best_score = -1.0
    
    candidates = {expected_name: _voice_db[expected_name]} if expected_name and expected_name in _voice_db else _voice_db
    
    for name, info in candidates.items():
        for stored_emb in info["embeddings"]:
            score = _cosine_similarity(embedding, stored_emb)
            if score > best_score:
                best_score = score
                best_name = name
    
    verified = best_score >= VERIFY_THRESHOLD
    
    if verified:
        print(f"  🎤 Voice verified: {best_name} (score={best_score:.3f})")
    elif best_score >= REJECT_THRESHOLD:
        print(f"  🎤 Voice uncertain: {best_name}? (score={best_score:.3f}) — allowing")
        verified = True  # Give benefit of the doubt in uncertain range
    else:
        print(f"  🎤 Voice REJECTED (best={best_name}, score={best_score:.3f})")
    
    return {
        "verified": verified,
        "name": best_name if verified else None,
        "confidence": round(best_score, 3),
        "message": f"Verified as {best_name}" if verified else "Speaker not recognized",
    }


def verify_base64(audio_b64: str, sample_rate: int = 16000,
                  expected_name: str = None) -> Dict:
    """Verify from base64-encoded audio."""
    try:
        audio_bytes = base64.b64decode(audio_b64)
        return verify(audio_bytes, sample_rate, expected_name)
    except Exception as e:
        return {"verified": True, "name": None, "confidence": 0,
                "message": f"Decode error: {e} — allowing"}


def delete_voice_print(name: str) -> bool:
    """Delete a student's voice print."""
    if name in _voice_db:
        del _voice_db[name]
        _save_db()
        print(f"  🎤 Voice print deleted: {name}")
        return True
    return False


def list_enrolled() -> Dict[str, int]:
    """List enrolled students and their sample counts."""
    return {name: info.get("sample_count", len(info.get("embeddings", [])))
            for name, info in _voice_db.items()}


def get_status() -> Dict:
    """Get voice fingerprint system status."""
    return {
        "initialized": _encoder is not None,
        "encoder_loaded": _encoder is not None,
        "enrolled_count": len(_voice_db),
        "enrolled_students": list_enrolled(),
        "verify_threshold": VERIFY_THRESHOLD,
    }


def is_enrolled(name: str) -> bool:
    """Check if a student has voice prints enrolled."""
    return name in _voice_db and len(_voice_db.get(name, {}).get("embeddings", [])) > 0


# Initialize on import
init()
