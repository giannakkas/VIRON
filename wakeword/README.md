# VIRON Wake Word Service (openWakeWord)

Server-side wake word detection using [openWakeWord](https://github.com/dscripka/openWakeWord).
Replaces unreliable Chrome Speech API wake word detection with neural network-based detection.

## Architecture

```
Microphone → PyAudio → openWakeWord (CPU) → Detection event
                                                    ↓
Browser ← HTTP poll (250ms) ← Flask API ← Detection queue
```

The service captures audio directly from the Jetson's microphone using PyAudio,
processes it through openWakeWord's neural network, and exposes detections via HTTP.
The browser polls every 250ms for wake word events.

## Setup

```bash
cd ~/VIRON
bash wakeword/setup.sh
```

## Start

```bash
# Start wake word service (runs on port 8085)
python3 wakeword/service.py &

# Start main VIRON services as usual
python3 backend/server.py &
python3 gateway/main.py &
```

## Custom "Hey VIRON" Model

The default uses "hey jarvis" as a stand-in. For best results, train a custom model:

1. **Quick way**: Use [wakeword_trainer](https://github.com/bbarrick/wakeword_trainer)
2. **Full way**: Use the [openWakeWord Colab notebook](https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb)
3. **Generate samples**: `python3 wakeword/train_hey_viron.py` (creates edge-tts samples)

Place the trained model at `wakeword/models/hey_viron.onnx` and restart the service.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/wakeword/poll` | GET | Check for wake word (returns `{wake: true/false}`) |
| `/wakeword/pause` | POST | Pause detection (during VIRON speech) |
| `/wakeword/resume` | POST | Resume detection |
| `/wakeword/status` | GET | Service status, models, threshold |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VIRON_WAKEWORD_THRESHOLD` | `0.65` | Detection sensitivity (0-1) |
| `VIRON_WAKEWORD_PORT` | `8085` | Service HTTP port |
| `VIRON_WAKEWORD_MODEL` | | Path to custom .onnx model |
| `VIRON_MIC_DEVICE` | | PyAudio device index |

## Fallback

If the wake word service is not running, VIRON automatically falls back to
Chrome Speech API + Silero VAD wake word detection (previous behavior).
