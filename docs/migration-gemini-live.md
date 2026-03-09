# VIRON — Gemini Live Native Audio Migration Guide

## Overview

This migration replaces the chained voice pipeline:
```
OLD: Mic → Porcupine → VAD → Whisper STT → LLM routing → edge-tts/OpenAI TTS → Speaker
NEW: Mic → Porcupine → [Gemini Live WebSocket: audio in ↔ audio out] → Speaker
```

## What Changed

### New Files
| File | Purpose |
|------|---------|
| `gemini_live_pipeline.py` | New main runtime — replaces `voice_pipeline.py` |
| `start_live.sh` | Startup script for Gemini Live mode |
| `requirements_live.txt` | Dependencies for Gemini Live |
| `docs/migration-gemini-live.md` | This file |

### What's Kept Unchanged
- `viron-complete.html` — Face UI (same HTTP API endpoints)
- `backend/server.py` — Flask backend for face serving
- `backend/student_profiles.py` — Student profiles
- `backend/viron_faces.py` — Face animation
- `gateway/` — Gateway (not used in live path, but preserved)
- `wakeword/` — Wake word models (Porcupine used directly)
- `voice_pipeline.py` — Old pipeline (kept as fallback)

### What's Replaced in the Live Path
- Deepgram STT → Gemini Native Audio (built-in speech recognition)
- Whisper STT → Gemini Native Audio
- OpenAI TTS → Gemini Native Audio (built-in speech synthesis)
- edge-tts → Gemini Native Audio (except startup greeting)
- LLM routing (Groq/Claude/ChatGPT) → Single Gemini model handles everything
- Manual VAD → Gemini's built-in VAD with barge-in
- Mic muting hacks → Not needed (audio is bidirectional)

## Environment Variables

Add to `.env`:
```bash
# Required
GEMINI_API_KEY=your_gemini_api_key_here
PICOVOICE_ACCESS_KEY=your_picovoice_key_here

# Optional (defaults shown)
GEMINI_LIVE_MODEL=gemini-2.5-flash-native-audio-preview-12-2025
VIRON_DEFAULT_LANGUAGE=el
VIRON_WAKE_KEYWORD=jarvis
VIRON_WAKE_SENSITIVITY=0.7
VIRON_MIC_DEVICE=plughw:0,0
VIRON_IDLE_TIMEOUT=30
VIRON_PIPELINE_PORT=8085
```

## Install & Run

```bash
# 1. Install dependencies
cd ~/VIRON
pip3 install -r requirements_live.txt

# 2. Stop old pipeline
bash start.sh stop 2>/dev/null
bash start_live.sh stop 2>/dev/null

# 3. Start Gemini Live mode
bash start_live.sh
```

## Test Plan

### 1. Wake Word Test
Say "Hey Jarvis" → Should see in logs:
```
🎯 WAKE WORD DETECTED!
🌐 Starting Gemini Live session...
✅ Gemini Live session connected!
```

### 2. Greek Conversation Test
After wake word, say "Τι κάνεις;" → VIRON should respond in Greek through the speaker.

### 3. Barge-In Test
While VIRON is speaking, say something → Response should stop and VIRON should listen.

### 4. Idle Timeout Test
After wake word, stay silent for 30s → Session should end:
```
⏰ Idle timeout (30s) — ending session
🔌 Gemini Live session ended
```

### 5. UI Sync Test
Open browser at `http://[jetson-ip]:5000` → Face should react to wake word, show listening/speaking states.

### 6. Audio Quality Test
Verify VIRON's voice sounds natural and Greek pronunciation is correct.

## Switching Between Modes

```bash
# Gemini Live mode (NEW — recommended)
bash start_live.sh

# Old STT/TTS mode (fallback)
bash start.sh
```

## Known Considerations

1. **Cost**: Gemini Live charges per audio token (~$0.50-1.50/hour). Monitor usage.
2. **Internet Required**: Gemini Live requires constant internet connection.
3. **Model**: `gemini-2.5-flash-native-audio-preview-12-2025` is a preview model. Monitor for updates.
4. **Whiteboard/News/Weather**: Not yet wired as function calls. The model can discuss topics but can't trigger UI overlays yet. To add: define function declarations in the Live config.
5. **YouTube Music**: Not yet wired. Can be added via function calling.
6. **Camera Input**: Ready to add — the Gemini Live API supports video frames in the same session.

## Next Steps

1. **Add Function Calling** for whiteboard, news, weather, music
2. **Add Camera 1** (face recognition) — stream frames to Gemini Live session
3. **Add Camera 2** (homework reader) — stream frames on demand
4. **Voice Selection** — test different Gemini voices for best Greek quality
5. **Session Resumption** — use Gemini's session handle for reconnection
