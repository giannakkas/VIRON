# VIRON Mic Setup — ReSpeaker XVF3800

## Quick Start

```bash
# 1. Find your device
arecord -l

# 2. Set values in .env
VIRON_MIC_DEVICE="plughw:0,0"
VIRON_MIC_CHANNEL=0

# 3. Kill PulseAudio (steals ALSA device)
pulseaudio --kill; pkill -f arecord; sleep 2

# 4. Verify services
curl http://127.0.0.1:8085/wakeword/status
curl -s -X POST http://127.0.0.1:5000/api/listen \
  -H 'Content-Type: application/json' \
  -d '{"max_duration":6,"silence_duration":0.25,"lang":"el"}'
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VIRON_MIC_DEVICE` | `plughw:0,0` | ALSA capture device. Run `arecord -l` to find card:device. |
| `VIRON_MIC_CHANNEL` | `0` | Mono channel: `0`=beamformed (XVF3800 outputs mono on ch0). |
| `VIRON_WAKEWORD_THRESHOLD` | `0.28` | OWW detection sensitivity (0-1). See tuning guide below. |
| `VIRON_WAKEWORD_ADAPTIVE` | `1` | Enable adaptive threshold (auto-adjusts based on noise level). |
| `VIRON_LISTEN_CHUNK_MS` | `40` | Audio chunk size for VAD. Lower = faster detection, higher CPU. |
| `VIRON_NO_SPEECH_MS` | `1200` | Max wait time before giving up if no speech detected. |

## Finding a Stable Device Name

The XVF3800 USB device index can change across reboots. To find a stable name:

```bash
# List all ALSA playback names
arecord -L | grep -A1 "plughw\|hw:"

# The card number (e.g., card 2) may change. Use either:
# a) The plughw format (works if card index is stable):
VIRON_MIC_DEVICE="plughw:2,0"

# b) Or create a udev rule for stability:
# /etc/udev/rules.d/99-respeaker.rules
# SUBSYSTEM=="sound", ATTRS{idVendor}=="2886", ATTRS{idProduct}=="0018", SYMLINK+="respeaker"
```

## XVF3800 Audio Output

**IMPORTANT:** The XVF3800 outputs **mono** beamformed audio on channel 0 (not stereo as some docs suggest). The voice pipeline uses `arecord -c 1` to capture this directly.

- **Channel 0**: Beamformed + Acoustic Echo Cancellation (AEC) mono output
- The pipeline reads mono int16 directly: `bytes_needed = frame_length * 2`

### PulseAudio Conflict

PulseAudio grabs the ALSA device exclusively. **Must kill PulseAudio before starting the pipeline:**

```bash
pulseaudio --kill; pkill -f arecord; sleep 2
```

To permanently disable PulseAudio on Jetson:
```bash
# Mask systemd user services
systemctl --user mask pulseaudio.service pulseaudio.socket
# Disable autospawn
mkdir -p ~/.config/pulse
echo "autospawn = no" > ~/.config/pulse/client.conf
# Reboot and verify
reboot
# After reboot:
pgrep pulseaudio  # should return nothing
```

Use the audio probe to verify capture:

```bash
# Speak "Hey Jarvis" during the recording
python3 scripts/audio_probe.py --speak --save

# Listen to the saved file:
# /tmp/viron_probe_ch0.wav
```

## Wake Word Threshold Tuning

| Environment | Recommended Threshold | Notes |
|---|---|---|
| Quiet room, close (< 1m) | `0.22-0.26` | Lower = more sensitive |
| Normal room, 1-2m distance | `0.26-0.32` | Default range |
| Noisy (TV, music playing) | `0.32-0.40` | Raise to reduce false triggers |

With `VIRON_WAKEWORD_ADAPTIVE=1` (default), the threshold auto-adjusts:
- Quiet room (noise < 50 RMS): threshold drops by ~0.06
- Noisy room (noise > 300 RMS): threshold raises by ~0.08
- Hard caps: min 0.18, max 0.45

## Training a Custom "Hey VIRON" Model

The default `hey_jarvis_v0.1` model only reliably detects "Hey Jarvis". For "Hey VIRON", the Whisper fallback handles it, but a custom model would be faster (~50ms vs ~1s).

```bash
# 1. Generate synthetic training samples
python3 wakeword/train_hey_viron.py

# 2. The script creates wakeword/models/hey_viron.onnx

# 3. Restart the wakeword service — it auto-detects the custom model
pkill -f "wakeword/service.py"
python3 wakeword/service.py > /tmp/viron_wakeword.log 2>&1 &

# 4. Verify it loaded
curl http://127.0.0.1:8085/wakeword/status
# Should show "models": ["hey_viron"] instead of ["hey_jarvis_v0.1"]
```

Training requirements: `pip install openwakeword[full] edge-tts scipy`
Training time: ~1-4 hours depending on hardware.

## One-Command Verification

```bash
# Full system check
echo "=== Wake Word ===" && \
curl -s http://127.0.0.1:8085/wakeword/status | python3 -m json.tool && \
echo "=== Listen Test ===" && \
echo "Speak now..." && \
curl -s -X POST http://127.0.0.1:5000/api/listen \
  -H 'Content-Type: application/json' \
  -d '{"max_duration":5,"silence_duration":0.25,"lang":"en"}' | python3 -m json.tool
```
