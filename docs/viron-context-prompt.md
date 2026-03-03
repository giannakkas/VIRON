# VIRON — AI Study Buddy Robot: Complete Context Prompt

Paste this into a new Claude chat to give full context about the VIRON project.

---

## PROJECT OVERVIEW

VIRON is an AI companion robot / study buddy for kids and teens. It runs on a **Jetson Orin Nano 8GB** (aarch64, JetPack 6.2.2, Ubuntu 22.04, CUDA 12.6). The hardware is a physical robot with a screen (showing animated face), microphone, and speaker. All interaction is **voice-only** — no keyboard/mouse.

**GitHub repo:** https://github.com/giannakkas/VIRON

---

## ARCHITECTURE

```
Student speaks → Browser (Silero VAD) → WAV → Flask STT (/api/stt)
                                                    ↓
                                         OpenAI Whisper API (primary, accurate Greek)
                                         or local faster-whisper (fallback)
                                                    ↓
                                         Transcribed text
                                                    ↓
                                    Gateway (/v1/chat, port 8080)
                                         ┌──────┴──────┐
                                    Gemma 2B Router    (intent classification, GPU, <1s)
                                         ┌──────┴──────┐
                                    LOCAL              CLOUD
                                  (Gemma 2B)    ┌──────┼──────┐
                                               ChatGPT  Claude  Gemini
                                                    ↓
                                         Response with [emotion] tag
                                                    ↓
                                    Flask TTS (edge-tts, Greek/English)
                                                    ↓
                                    Browser plays audio + animates face
```

### Routing Logic
- **Simple/casual chat** ("τι κάνεις", greetings) → **LOCAL** Gemma 2B on GPU (~1-2s)
- **Math, programming** → **Cloud ChatGPT**
- **English, writing, history, emotional support** → **Cloud Claude**
- **Science, STEM, geography** → **Cloud Gemini**
- **Weather** → fetches wttr.in data → **Cloud** summarizes
- **News** → fetches Google News RSS → **Cloud** summarizes
- **Music** → **Cloud** returns [YOUTUBE:videoId:Title]

### Memory Constraint
Jetson Orin Nano has 8GB shared RAM. **Only Gemma 2B fits on GPU** (Mistral 7B causes OOM). The `call_tutor()` function uses the router model (port 8081) for local responses. Complex questions route to cloud.

---

## KEY FILES

### `gateway/main.py` (668 lines) — FastAPI Hybrid Gateway
- Port 8080
- `call_router()` — Gemma 2B intent classification via llama.cpp
- `call_tutor()` — Local response generation (uses Gemma 2B on ROUTER_URL since Mistral OOM)
- `call_cloud()` → `call_chatgpt()` / `call_claude()` / `call_gemini()`
- `enrich_message()` — injects weather data (wttr.in) and news (Google News RSS)
- `apply_keyword_overrides()` — forces cloud for weather/news/music/educational keywords
- `FORCE_CLOUD` env var — set to "1" to bypass all local routing
- Safety filter blocks unsafe content locally

### `gateway/config.py` (71 lines) — Configuration
- `ROUTER_URL` = localhost:8081 (Gemma 2B)
- `TUTOR_URL` = localhost:8082 (Mistral 7B — NOT USED, OOM on 8GB)
- Cloud models: gpt-4o-mini, claude-sonnet-4-20250514, gemini-2.0-flash
- `VIRON_SYSTEM_PROMPT` — master prompt for cloud providers, includes:
  - Greek/English language matching
  - [emotion] tags (MUST be English: [happy], [sad], etc.)
  - [YOUTUBE:videoId:Title] format for music
  - [WHITEBOARD:Title] format for educational content
  - Weather/news summarization instructions

### `backend/server.py` (1836 lines) — Flask Backend
- Port 5000
- `/api/stt` — Speech-to-text: OpenAI Whisper API primary, local faster-whisper fallback
- `/api/tts` — Text-to-speech via edge-tts (Greek: el-GR-AthinaNeural, English: en-US)
- `/api/student/emotion` — returns current robot emotion for face animation
- Silero VAD tuning on server side (min_silence_duration_ms=400, threshold=0.35)
- Hallucination filter for repeated-word patterns from Whisper

### `viron-complete.html` (2690 lines) — Single-page Face UI
- Animated robot face with eyes, eyebrows, mouth (CSS/JS)
- Silero VAD (client-side neural speech detection via ONNX)
- Wake word detection ("Hey VIRON")
- Voice enrollment (optional, can skip)
- `parseEmotionAndStrip()` — extracts [emotion] tag, triggers face animation
- `EMOTION_MAP` — maps Greek emotion words to English
- YouTube player integration ([YOUTUBE:videoId:Title] parser)
- No subtitle text display (voice-only)
- `say()` → `sayServerTTS()` sends text to Flask /api/tts
- `sendMessage()` → POST to gateway /v1/chat

### `gateway/db.py` (99 lines) — SQLite persistence
- Student profiles, message history, session tracking

### `gateway/safety.py` (64 lines) — Safety filter
- Blocks unsafe content locally before routing

---

## SERVICES ON JETSON (100.66.223.46)

| Service | Port | Status | Notes |
|---------|------|--------|-------|
| Gemma 2B (llama-server) | 8081 | ✅ Running on GPU | Router + local chat, -ngl 99, 25 layers offloaded |
| Mistral 7B | 8082 | ❌ OOM | Doesn't fit in 8GB with router |
| Gateway (FastAPI) | 8080 | ✅ Running | Hybrid routing |
| Flask (backend) | 5000 | ✅ Running | STT/TTS/Face UI |

### Start Commands
```bash
cd ~/VIRON
source .env
export OPENAI_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY DB_PATH ROUTER_URL TUTOR_URL GATEWAY_PORT FORCE_CLOUD

# Router (Gemma 2B on GPU)
llama-server -m models/gemma-2-2b-it-Q4_K_M.gguf --port 8081 -ngl 99 -c 2048 --threads 4 > /tmp/viron_router.log 2>&1 &

# Gateway
cd ~/VIRON/gateway && python3 main.py > /tmp/viron_gateway.log 2>&1 &

# Flask
cd ~/VIRON && python3 backend/server.py > /tmp/viron_flask.log 2>&1 &
```

### .env File
```
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
ROUTER_URL=http://localhost:8081
TUTOR_URL=http://localhost:8082
DB_PATH=/home/test/VIRON/data/viron.db
GATEWAY_PORT=8080
FORCE_CLOUD=0
```

---

## EMOTION SYSTEM

Cloud responses MUST start with English emotion tags: `[happy] Γεια σου!`

The client-side `EMOTION_MAP` maps Greek→English:
```
χαρούμενος→happy, λυπημένος→sad, σκεπτικός→thinking, ενθουσιασμένος→excited,
ήρεμος→calm, εκπληκτικός→surprised, μπερδεμένος→confused, περήφανος→proud,
ανήσυχος→worried, πονηρός→cheeky
```

`parseEmotionAndStrip()` extracts the tag, calls `cE()` to animate the face, and strips it from spoken text.

---

## KNOWN ISSUES / CURRENT STATUS

1. **Greek STT accuracy** — Local Whisper (small model) is terrible for Greek. OpenAI Whisper API (large-v3) is much better. The server tries OpenAI first, falls back to local. Check OPENAI_API_KEY is exported.

2. **8GB RAM limit** — Only one LLM fits on GPU. Gemma 2B handles both routing and simple chat. Complex stuff goes to cloud.

3. **Language detection** — If Whisper hears English instead of Greek, the whole pipeline responds in English. The client should send `lang=el` hint. The system prompt tells cloud to match the student's language.

4. **TTS voices** — Greek: el-GR-AthinaNeural, English: en-US-GuyNeural. Edge-tts is free and good quality.

5. **Chrome mic access** — Remote browsers need `chrome://flags/#unsafely-treat-insecure-origin-as-secure` set to `http://JETSON-IP:5000` because it's not HTTPS.

---

## HARDWARE

- **Jetson Orin Nano 8GB Developer Kit** (aarch64)
- JetPack 6.2.2 (Ubuntu 22.04, CUDA 12.6, Linux 5.15)
- NVMe SSD
- CUDA arch: SM 8.7 (Ampere)
- llama.cpp built with: `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87`
- Super Mode enabled: `sudo nvpmodel -m 0 && sudo jetson_clocks`
- Tailscale IP: 100.66.223.46
- SSH: `ssh test@100.66.223.46`

---

## DEVELOPMENT PC

- Ubuntu 22.04 PC at 192.168.100.173 (original dev machine)
- Also has VIRON running (CPU-only, with FORCE_CLOUD=1)
- Used for development, then `git push` → `git pull` on Jetson

---

## PENDING WORK

- Fix Greek language detection (STT should default to Greek, response should be Greek)
- Age-mode personality switching (from Buddy Tutor spec)
- Memory injection from student profile
- Session summary generation
- Points/achievements system
- systemd auto-start service on Jetson boot
- Production: HTTPS with self-signed cert for mic access
- Physical robot housing design

---

## BUDDY TUTOR SPEC (Future)

Full spec in `docs/buddy-tutor-spec.md`. Key features:
- Age modes: kid (3-7), tween (8-12), teen (13-18)
- Student profiles with learning progress
- Session summaries for parents
- Points/achievements system
- Memory injection (remembers previous conversations)
- Whiteboard for visual explanations
