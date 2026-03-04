"""
VIRON /api/listen — Combined Record + STT Endpoint
====================================================
This module adds the /api/listen endpoint to the Flask server.
It combines /api/record and /api/stt into a single call, eliminating
the browser roundtrip for WAV upload.

Import this in server.py and call register_listen_endpoint(app)
after the existing routes.
"""

import os
import re
import time
import wave
import json
import tempfile
import threading
import subprocess
import numpy as np


def register_listen_endpoint(app):
    """Register the /api/listen endpoint on the Flask app."""
    
    RECORD_ALSA_DEVICE = os.environ.get("VIRON_MIC_DEVICE", "plughw:2,0")
    
    def _clean_whisper_text(text):
        """Clean up Whisper hallucinations."""
        if not text:
            return text
        cleaned = re.sub(r'(.)\1{2,}', r'\1', text)
        cleaned = re.sub(r'\b(\w+)(\s+\1){2,}\b', r'\1', cleaned)
        return cleaned.strip()
    
    def _resume_wakeword():
        """Resume wakeword service after recording."""
        try:
            import urllib.request
            oww_port = int(os.environ.get('VIRON_WAKEWORD_PORT', '8085'))
            req = urllib.request.Request(
                f'http://127.0.0.1:{oww_port}/wakeword/resume',
                method='POST', data=b'{}',
                headers={'Content-Type': 'application/json'}
            )
            urllib.request.urlopen(req, timeout=2)
        except Exception:
            pass
    
    @app.route('/api/listen', methods=['POST'])
    def listen_and_transcribe():
        """Combined record + STT endpoint.
        
        Records from ReSpeaker mic with VAD, then immediately transcribes
        without returning to the browser. Saves ~500ms roundtrip.
        
        POST params (JSON):
          max_duration: max recording seconds (default 8)
          silence_duration: seconds of silence to stop (default 0.4)
          min_duration: minimum recording seconds (default 0.2)
          lang: language hint ('el', 'en', or '' for auto)
        
        Returns JSON:
          { text, language, duration, engine, record_ms }
        """
        from flask import request, jsonify
        
        params = request.get_json(silent=True) or {}
        max_duration = min(float(params.get('max_duration', 8)), 30)
        silence_duration = float(params.get('silence_duration', 0.4))
        min_duration = float(params.get('min_duration', 0.2))
        hint_lang = params.get('lang', '')
        
        sample_rate = 16000
        chunk_ms = 80
        chunk_samples = sample_rate * chunk_ms // 1000  # 1280
        bytes_per_chunk = chunk_samples * 4  # stereo int16
        
        # Language mapping
        whisper_lang = None
        if hint_lang in ('el', 'el-GR'):
            whisper_lang = 'el'
        elif hint_lang in ('en', 'en-US', 'en-GB'):
            whisper_lang = 'en'
        
        cmd = [
            "arecord", "-D", RECORD_ALSA_DEVICE,
            "-f", "S16_LE", "-r", str(sample_rate),
            "-c", "2", "-t", "raw",
        ]
        
        t_start = time.time()
        wav_path = None
        
        try:
            # Pause wakeword service
            oww_port = int(os.environ.get('VIRON_WAKEWORD_PORT', '8085'))
            try:
                import urllib.request
                req = urllib.request.Request(
                    f'http://127.0.0.1:{oww_port}/wakeword/pause',
                    method='POST', data=b'{}',
                    headers={'Content-Type': 'application/json'}
                )
                urllib.request.urlopen(req, timeout=2)
                time.sleep(0.2)  # Brief pause for device release
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
            
            # FAST calibration: 4 chunks = 320ms (down from 6 = 480ms)
            CALIBRATION_CHUNKS = 4
            SPEECH_THRESHOLD = 80
            SILENCE_THRESHOLD = 40
            
            noise_rms_values = []
            for _ in range(CALIBRATION_CHUNKS):
                data = proc.stdout.read(bytes_per_chunk)
                if not data or len(data) < bytes_per_chunk:
                    break
                stereo = np.frombuffer(data, dtype=np.int16)
                samples = stereo[1::2]  # right channel (ASR beam)
                audio_frames.append(samples.tobytes())
                total_chunks += 1
                rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
                noise_rms_values.append(rms)
            
            noise_floor = 0
            if noise_rms_values:
                noise_floor = np.mean(noise_rms_values)
                # ADJUSTED: 2.5x noise (down from 3x) for better sensitivity
                SPEECH_THRESHOLD = max(noise_floor * 2.5, 50)
                SILENCE_THRESHOLD = max(noise_floor * 1.5, 30)
                print(f"🎤 Listen: noise={noise_floor:.0f} speech>{SPEECH_THRESHOLD:.0f} silence<{SILENCE_THRESHOLD:.0f}")
            
            # Record with VAD
            while total_chunks < max_chunks:
                data = proc.stdout.read(bytes_per_chunk)
                if not data or len(data) < bytes_per_chunk:
                    break
                
                stereo = np.frombuffer(data, dtype=np.int16)
                samples = stereo[1::2]
                audio_frames.append(samples.tobytes())
                total_chunks += 1
                
                rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
                
                if not speech_started:
                    if rms > SPEECH_THRESHOLD:
                        speech_started = True
                        silence_chunks = 0
                        print(f"🎤 Speech START (RMS={rms:.0f}) at {total_chunks * chunk_ms}ms")
                    # REDUCED no-speech timeout: 2.5s (from 3s)
                    elif total_chunks > int(2500 / chunk_ms):
                        print(f"🎤 No speech after 2.5s (last RMS={rms:.0f})")
                        break
                else:
                    if rms < SILENCE_THRESHOLD:
                        silence_chunks += 1
                        if silence_chunks >= silence_chunks_needed:
                            print(f"🎤 Silence END ({total_chunks * chunk_ms}ms recorded)")
                            break
                    else:
                        silence_chunks = 0
            
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except:
                proc.kill()
            
            record_ms = int((time.time() - t_start) * 1000)
            
            if not audio_frames or not speech_started:
                _resume_wakeword()
                return jsonify({"text": "", "language": "", "duration": 0,
                               "engine": "none", "record_ms": record_ms,
                               "error": "no_speech"}), 204
            
            # Trim trailing silence
            while len(audio_frames) > 3:
                last = np.frombuffer(audio_frames[-1], dtype=np.int16)
                rms = np.sqrt(np.mean(last.astype(np.float32) ** 2))
                if rms < SILENCE_THRESHOLD:
                    audio_frames.pop()
                else:
                    break
            
            # SNR check
            if noise_floor > 0:
                peak_rms = max(
                    np.sqrt(np.mean(np.frombuffer(f, dtype=np.int16).astype(np.float32) ** 2))
                    for f in audio_frames
                )
                snr = peak_rms / noise_floor
                if snr < 2.0:
                    print(f"🎤 Listen: rejected low SNR ({snr:.1f})")
                    _resume_wakeword()
                    return jsonify({"text": "", "language": "",
                                   "error": "noise_rejected", "snr": round(snr, 1),
                                   "record_ms": record_ms}), 204
            
            # Package as WAV
            raw_audio = b''.join(audio_frames)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                wav_path = tmp.name
                with wave.open(tmp, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(raw_audio)
            
            duration_ms = len(audio_frames) * chunk_ms
            print(f"🎤 Listen: recorded {duration_ms}ms ({len(raw_audio) // 1024}KB), now transcribing...")
            
            # ── DIRECT STT (no browser roundtrip) ──
            t_stt = time.time()
            
            # Try OpenAI Whisper API first
            openai_key = os.environ.get("OPENAI_API_KEY", "")
            if openai_key and len(raw_audio) > 500:
                try:
                    import requests as req_lib
                    with open(wav_path, 'rb') as f:
                        resp = req_lib.post(
                            "https://api.openai.com/v1/audio/transcriptions",
                            headers={"Authorization": f"Bearer {openai_key}"},
                            files={"file": ("speech.wav", f, "audio/wav")},
                            data={
                                "model": "whisper-1",
                                **({"language": whisper_lang} if whisper_lang else {}),
                                "temperature": 0.0,
                                "prompt": "Αυτό είναι Ελληνικά." if whisper_lang == "el" else "",
                            },
                            timeout=10,
                        )
                    if resp.status_code == 200:
                        text = resp.json().get("text", "").strip()
                        text = _clean_whisper_text(text)
                        stt_ms = int((time.time() - t_stt) * 1000)
                        total_ms = int((time.time() - t_start) * 1000)
                        
                        # Hallucination filter
                        HALLUCINATION_BLACKLIST = [
                            "υπότιτλοι", "authorwave", "σας ευχαριστούμε",
                            "ευχαριστώ που παρακολουθήσατε", "εγγραφείτε στο κανάλι",
                            "like and subscribe", "thank you for watching",
                            "subtitles by", "translated by", "amara.org",
                            "music", "♪", "...", "…",
                        ]
                        if text and any(bl in text.lower() for bl in HALLUCINATION_BLACKLIST):
                            print(f"  ⚠ Filtered hallucination: '{text[:60]}'")
                            _resume_wakeword()
                            return jsonify({
                                "text": "", "language": whisper_lang or "auto",
                                "duration": round(stt_ms / 1000, 2),
                                "engine": "openai", "filtered": "hallucination",
                                "record_ms": record_ms, "stt_ms": stt_ms,
                                "total_ms": total_ms,
                            })
                        
                        if text and len(text) > 1:
                            print(f"  ✅ Listen: \"{text[:80]}\" (record={record_ms}ms, stt={stt_ms}ms, total={total_ms}ms)")
                            # Don't resume wakeword yet — browser handles that after conversation
                            return jsonify({
                                "text": text,
                                "language": whisper_lang or "auto",
                                "duration": round(stt_ms / 1000, 2),
                                "engine": "openai",
                                "record_ms": record_ms,
                                "stt_ms": stt_ms,
                                "total_ms": total_ms,
                            })
                        else:
                            print("  ⚠ OpenAI returned empty, trying local...")
                    else:
                        print(f"  ⚠ OpenAI API error {resp.status_code}: {resp.text[:100]}")
                except Exception as e:
                    print(f"  ⚠ OpenAI Whisper failed: {e}, trying local...")
            
            # Fallback to local faster-whisper
            try:
                # Import from the main server module's globals
                from __main__ import HAS_WHISPER, whisper_model, _whisper_lock
                if not HAS_WHISPER or whisper_model is None:
                    _resume_wakeword()
                    return jsonify({"text": "", "error": "No STT available"}), 503
                
                with _whisper_lock:
                    segments, info = whisper_model.transcribe(
                        wav_path,
                        language=whisper_lang,
                        beam_size=5,
                        vad_filter=True,
                        vad_parameters=dict(
                            min_silence_duration_ms=400,
                            speech_pad_ms=200,
                            threshold=0.35,
                        ),
                        no_speech_threshold=0.5,
                        condition_on_previous_text=False,
                    )
                    text_parts = []
                    for segment in segments:
                        t = segment.text.strip()
                        if t and not re.match(r'^[\s\.\,\!\?\;\:…]+$', t) and '[' not in t:
                            words = t.split()
                            if len(words) >= 3:
                                unique_words = set(w.lower() for w in words)
                                if len(unique_words) <= 2 and len(words) > 4:
                                    continue
                            text_parts.append(t)
                
                text = _clean_whisper_text(" ".join(text_parts).strip())
                stt_ms = int((time.time() - t_stt) * 1000)
                total_ms = int((time.time() - t_start) * 1000)
                lang = info.language if info else "unknown"
                
                print(f"  🏠 Listen (local): \"{text[:80]}\" (record={record_ms}ms, stt={stt_ms}ms, total={total_ms}ms)")
                return jsonify({
                    "text": text, "language": lang,
                    "duration": round(stt_ms / 1000, 2),
                    "engine": "local",
                    "record_ms": record_ms, "stt_ms": stt_ms,
                    "total_ms": total_ms,
                })
            except ImportError:
                # Can't access main module globals — use standalone approach
                print("  ⚠ Can't access whisper model from listen endpoint, falling back")
                _resume_wakeword()
                return jsonify({"text": "", "error": "local_stt_unavailable"}), 503
        
        except Exception as e:
            print(f"❌ Listen error: {e}")
            import traceback
            traceback.print_exc()
            _resume_wakeword()
            return jsonify({"text": "", "error": str(e)}), 500
        
        finally:
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)
    
    print("✅ Registered /api/listen endpoint (combined record+STT)")
