"""
VIRON Voice Fingerprint — Speaker Verification
================================================
Uses Resemblyzer to create voice embeddings ("fingerprints") for each student.
During conversation, verifies that the speaker matches a registered student.

Flow:
1. Setup: Student records 3+ voice samples → embeddings stored
2. Runtime: Audio captured → embedding extracted → compared to known voices
3. If match: process speech. If no match: ignore.

Requires: pip install resemblyzer numpy
"""

import os
import json
import time
import numpy as np
import tempfile
import wave
import struct
from typing import Optional, Dict, List, Tuple

# ═══════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════

VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_profiles")
SIMILARITY_THRESHOLD = 0.75  # Cosine similarity threshold (0.75 = good balance)
MIN_AUDIO_DURATION = 0.5     # Minimum audio duration in seconds
MAX_AUDIO_DURATION = 10.0    # Maximum audio duration in seconds

# ═══════════════════════════════════════════════════════
# Voice Verifier
# ═══════════════════════════════════════════════════════

class VoiceVerifier:
    """Speaker verification using voice embeddings."""
    
    def __init__(self):
        self.initialized = False
        self.encoder = None
        self.voice_profiles = {}  # {name: [embedding1, embedding2, ...]}
        self.last_verified = None
        self.last_confidence = 0.0
        self.enabled = True
        
    def initialize(self):
        """Load the Resemblyzer encoder model."""
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
            self.encoder = VoiceEncoder("cpu")  # Use CPU (works on Jetson too)
            self.preprocess_wav = preprocess_wav
            self.initialized = True
            print("🎙️ Voice verifier initialized (Resemblyzer)")
            
            # Load existing profiles
            self._load_profiles()
            if self.voice_profiles:
                print(f"  📁 Loaded {len(self.voice_profiles)} voice profiles: {', '.join(self.voice_profiles.keys())}")
            
            return True
        except ImportError:
            print("⚠ Resemblyzer not installed. Install with: pip install resemblyzer")
            print("  Voice verification disabled — will accept all speakers")
            return False
        except Exception as e:
            print(f"⚠ Voice verifier error: {e}")
            return False
    
    def register_voice(self, name: str, audio_data: bytes, sample_rate: int = 16000) -> Dict:
        """Register a voice sample for a student.
        
        Args:
            name: Student name (matches face recognition)
            audio_data: Raw audio bytes (WAV or raw PCM)
            sample_rate: Audio sample rate (default 16kHz)
        
        Returns: {success, message, samples_count}
        """
        if not self.initialized:
            return {"success": False, "message": "Voice verifier not initialized"}
        
        try:
            # Convert audio to wav format
            wav_path = self._save_temp_wav(audio_data, sample_rate)
            if not wav_path:
                return {"success": False, "message": "Invalid audio data"}
            
            # Preprocess and extract embedding
            wav = self.preprocess_wav(wav_path)
            
            # Check minimum duration
            duration = len(wav) / sample_rate
            if duration < MIN_AUDIO_DURATION:
                os.unlink(wav_path)
                return {"success": False, "message": f"Audio too short ({duration:.1f}s). Speak for at least 1 second."}
            
            # Extract embedding
            embedding = self.encoder.embed_utterance(wav)
            
            # Store
            if name not in self.voice_profiles:
                self.voice_profiles[name] = []
            self.voice_profiles[name].append(embedding.tolist())
            
            # Save to disk
            self._save_profiles()
            
            # Cleanup
            os.unlink(wav_path)
            
            count = len(self.voice_profiles[name])
            print(f"  🎙️ Voice sample {count} registered for {name} ({duration:.1f}s)")
            
            return {
                "success": True,
                "message": f"Voice sample {count} registered for {name}",
                "samples_count": count,
                "duration": round(duration, 1),
            }
        except Exception as e:
            print(f"  ⚠ Voice registration error: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "message": str(e)}
    
    def verify_speaker(self, audio_data: bytes, sample_rate: int = 16000) -> Dict:
        """Verify who is speaking from an audio sample.
        
        Returns: {verified, name, confidence, all_scores}
        """
        if not self.initialized or not self.voice_profiles:
            # No profiles = accept everyone
            return {"verified": True, "name": None, "confidence": 0.0, "reason": "no_profiles"}
        
        if not self.enabled:
            return {"verified": True, "name": None, "confidence": 0.0, "reason": "disabled"}
        
        try:
            # Convert audio
            wav_path = self._save_temp_wav(audio_data, sample_rate)
            if not wav_path:
                return {"verified": False, "name": None, "confidence": 0.0, "reason": "invalid_audio"}
            
            wav = self.preprocess_wav(wav_path)
            duration = len(wav) / sample_rate
            
            if duration < MIN_AUDIO_DURATION:
                os.unlink(wav_path)
                return {"verified": True, "name": None, "confidence": 0.0, "reason": "too_short"}
            
            # Extract embedding
            embedding = self.encoder.embed_utterance(wav)
            os.unlink(wav_path)
            
            # Compare against all profiles
            best_name = None
            best_score = -1
            all_scores = {}
            
            for name, embeddings in self.voice_profiles.items():
                scores = []
                for stored_emb in embeddings:
                    stored_arr = np.array(stored_emb)
                    score = np.dot(embedding, stored_arr) / (
                        np.linalg.norm(embedding) * np.linalg.norm(stored_arr)
                    )
                    scores.append(float(score))
                
                # Use average of top scores (more robust than single max)
                avg_score = np.mean(sorted(scores, reverse=True)[:3]) if scores else 0
                all_scores[name] = round(float(avg_score), 3)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_name = name
            
            verified = best_score >= SIMILARITY_THRESHOLD
            self.last_verified = best_name if verified else None
            self.last_confidence = best_score
            
            if verified:
                print(f"  🎙️ Speaker verified: {best_name} (confidence: {best_score:.3f})")
            else:
                print(f"  🔇 Speaker not recognized (best: {best_name} @ {best_score:.3f}, threshold: {SIMILARITY_THRESHOLD})")
            
            return {
                "verified": verified,
                "name": best_name if verified else None,
                "confidence": round(float(best_score), 3),
                "all_scores": all_scores,
                "threshold": SIMILARITY_THRESHOLD,
            }
        except Exception as e:
            print(f"  ⚠ Voice verification error: {e}")
            # On error, allow through (don't block the student)
            return {"verified": True, "name": None, "confidence": 0.0, "reason": f"error: {e}"}
    
    def delete_voice(self, name: str) -> bool:
        """Delete a voice profile."""
        if name in self.voice_profiles:
            del self.voice_profiles[name]
            self._save_profiles()
            print(f"  🗑️ Voice profile deleted: {name}")
            return True
        return False
    
    def list_profiles(self) -> Dict[str, int]:
        """List all voice profiles with sample counts."""
        return {name: len(embs) for name, embs in self.voice_profiles.items()}
    
    def get_status(self) -> Dict:
        """Get verifier status."""
        return {
            "initialized": self.initialized,
            "enabled": self.enabled,
            "profiles": self.list_profiles(),
            "profile_count": len(self.voice_profiles),
            "threshold": SIMILARITY_THRESHOLD,
            "last_verified": self.last_verified,
            "last_confidence": round(self.last_confidence, 3),
        }
    
    # ─── Internal Methods ───────────────────────────
    
    def _save_temp_wav(self, audio_data: bytes, sample_rate: int) -> Optional[str]:
        """Save audio data as a temporary WAV file."""
        try:
            tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            tmp_path = tmp.name
            
            # Check if it's already a WAV file
            if audio_data[:4] == b'RIFF':
                tmp.write(audio_data)
                tmp.close()
                return tmp_path
            
            # Check if it's WebM/Opus (from browser MediaRecorder)
            if audio_data[:4] == b'\x1aE\xdf\xa3' or b'webm' in audio_data[:20]:
                tmp.close()
                os.unlink(tmp_path)
                return self._convert_webm_to_wav(audio_data, sample_rate)
            
            # Assume raw PCM (16-bit signed, mono)
            tmp.close()
            with wave.open(tmp_path, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)
            return tmp_path
        except Exception as e:
            print(f"  ⚠ Audio save error: {e}")
            return None
    
    def _convert_webm_to_wav(self, webm_data: bytes, sample_rate: int) -> Optional[str]:
        """Convert WebM/Opus audio to WAV using ffmpeg."""
        try:
            import subprocess
            
            # Save webm to temp file
            webm_tmp = tempfile.NamedTemporaryFile(suffix='.webm', delete=False)
            webm_tmp.write(webm_data)
            webm_tmp.close()
            
            # Convert with ffmpeg
            wav_tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            wav_path = wav_tmp.name
            wav_tmp.close()
            
            result = subprocess.run([
                'ffmpeg', '-y', '-i', webm_tmp.name,
                '-ar', str(sample_rate), '-ac', '1', '-f', 'wav',
                wav_path
            ], capture_output=True, timeout=10)
            
            os.unlink(webm_tmp.name)
            
            if result.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 100:
                return wav_path
            else:
                if os.path.exists(wav_path):
                    os.unlink(wav_path)
                print(f"  ⚠ ffmpeg conversion failed: {result.stderr.decode()[:200]}")
                return None
        except FileNotFoundError:
            print("  ⚠ ffmpeg not found. Install with: sudo apt install ffmpeg")
            return None
        except Exception as e:
            print(f"  ⚠ WebM conversion error: {e}")
            return None
    
    def _save_profiles(self):
        """Save voice profiles to disk."""
        os.makedirs(VOICES_DIR, exist_ok=True)
        path = os.path.join(VOICES_DIR, "profiles.json")
        try:
            with open(path, 'w') as f:
                json.dump(self.voice_profiles, f)
        except Exception as e:
            print(f"  ⚠ Failed to save voice profiles: {e}")
    
    def _load_profiles(self):
        """Load voice profiles from disk."""
        path = os.path.join(VOICES_DIR, "profiles.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self.voice_profiles = json.load(f)
            except Exception as e:
                print(f"  ⚠ Failed to load voice profiles: {e}")
                self.voice_profiles = {}


# ═══════════════════════════════════════════════════════
# Singleton instance
# ═══════════════════════════════════════════════════════

voice_verifier = VoiceVerifier()
