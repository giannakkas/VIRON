#!/usr/bin/env python3
"""
VIRON Wake Word — Record Your Voice Samples
=============================================
Records your voice saying "Hey VIRON" multiple times for custom model training.

Usage:
  python3 scripts/record_wake_samples.py                    # 30 samples
  python3 scripts/record_wake_samples.py --count 50         # 50 samples
  python3 scripts/record_wake_samples.py --device plughw:3,0 --channel 0

Outputs:  wakeword/training_data/personal/*.wav
"""

import os
import sys
import time
import wave
import subprocess
import argparse
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "wakeword"
OUTPUT_DIR = SCRIPT_DIR / "training_data" / "personal"
RATE = 16000


def record_clip(device, channel, duration=2.0):
    """Record a short clip from the mic."""
    cmd = ["arecord", "-D", device, "-f", "S16_LE", "-r", str(RATE),
           "-c", "2", "-t", "raw", "-d", str(int(duration + 0.5))]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=int(duration) + 5)
        if result.returncode != 0:
            return None
        stereo = np.frombuffer(result.stdout, dtype=np.int16)
        mono = stereo[channel::2]
        # Trim to exact duration
        max_samples = int(duration * RATE)
        return mono[:max_samples]
    except Exception:
        return None


def find_speech(audio, threshold_mult=2.5):
    """Find speech region in audio, return trimmed clip with small padding."""
    win = RATE // 20  # 50ms windows
    rms_vals = []
    for i in range(0, len(audio) - win, win):
        w = audio[i:i + win].astype(np.float32)
        rms_vals.append(np.sqrt(np.mean(w ** 2)))
    rms_vals = np.array(rms_vals)

    if len(rms_vals) == 0:
        return audio

    noise = np.percentile(rms_vals, 20)
    thresh = max(noise * threshold_mult, 30)

    # Find first and last window above threshold
    above = np.where(rms_vals > thresh)[0]
    if len(above) == 0:
        return None  # no speech found

    start_win = max(0, above[0] - 2)  # 100ms padding before
    end_win = min(len(rms_vals), above[-1] + 3)  # 150ms padding after

    start_sample = start_win * win
    end_sample = min(end_win * win, len(audio))
    return audio[start_sample:end_sample]


def save_wav(audio, path):
    """Save numpy int16 array as 16kHz mono WAV."""
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(audio.tobytes())


def play_beep():
    """Play a short beep to signal recording start."""
    try:
        # Generate a 200ms 880Hz beep
        t = np.linspace(0, 0.2, int(RATE * 0.2))
        beep = (np.sin(2 * np.pi * 880 * t) * 8000).astype(np.int16)
        with open("/tmp/beep.raw", "wb") as f:
            f.write(beep.tobytes())
        subprocess.run(
            ["aplay", "-f", "S16_LE", "-r", str(RATE), "-c", "1", "/tmp/beep.raw"],
            capture_output=True, timeout=3
        )
    except Exception:
        pass  # beep is optional


def main():
    parser = argparse.ArgumentParser(description="Record 'Hey VIRON' voice samples")
    parser.add_argument("--count", "-n", type=int, default=30, help="Number of samples (default 30)")
    parser.add_argument("--device", "-d", default=os.environ.get("VIRON_MIC_DEVICE", "plughw:2,0"))
    parser.add_argument("--channel", "-c", type=int, default=int(os.environ.get("VIRON_MIC_CHANNEL", "1")))
    parser.add_argument("--duration", type=float, default=2.0, help="Seconds per recording (default 2)")
    parser.add_argument("--no-trim", action="store_true", help="Don't auto-trim silence")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Count existing samples
    existing = list(OUTPUT_DIR.glob("*.wav"))
    start_idx = len(existing)

    print()
    print("=" * 60)
    print("  VIRON Wake Word — Voice Recording")
    print("=" * 60)
    print(f"  Device: {args.device}  Channel: {args.channel}")
    print(f"  Target: {args.count} samples")
    print(f"  Existing: {start_idx} samples")
    print(f"  Output: {OUTPUT_DIR}")
    print()
    print("  INSTRUCTIONS:")
    print("  - Say 'Hey VIRON' clearly after each beep")
    print("  - Vary your tone, distance, and volume slightly")
    print("  - Some whispered, some normal, some louder")
    print("  - Press Enter to record, 's' to skip/redo, 'q' to quit")
    print("=" * 60)
    print()

    saved = 0
    idx = start_idx
    skipped = 0
    
    while saved < args.count:
        remaining = args.count - saved
        prompt = f"[{saved + 1}/{args.count}] Press Enter to record (s=skip, q=quit): "
        
        try:
            user_input = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        
        if user_input == 'q':
            break
        if user_input == 's':
            skipped += 1
            continue

        # Beep then record
        play_beep()
        time.sleep(0.1)
        
        sys.stdout.write("  🎤 Recording... say 'Hey VIRON'! ")
        sys.stdout.flush()
        
        audio = record_clip(args.device, args.channel, args.duration)
        
        if audio is None:
            print("❌ Recording failed!")
            continue

        rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
        peak = np.max(np.abs(audio.astype(np.float32)))
        
        if rms < 15:
            print(f"⚠ Too quiet (RMS={rms:.0f}). Try again, speak louder.")
            continue

        # Auto-trim silence
        if not args.no_trim:
            trimmed = find_speech(audio)
            if trimmed is None:
                print(f"⚠ No speech detected (RMS={rms:.0f}). Try again.")
                continue
            audio = trimmed

        duration_ms = len(audio) / RATE * 1000
        rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
        
        # Save
        outfile = OUTPUT_DIR / f"hey_viron_personal_{idx:04d}.wav"
        save_wav(audio, outfile)
        idx += 1
        saved += 1
        
        bar_len = min(int(rms / 20), 30)
        bar = "█" * bar_len
        print(f"✅ {duration_ms:.0f}ms RMS={rms:.0f} {bar}")
    
    print()
    print("=" * 60)
    total = start_idx + saved
    print(f"  ✅ Recorded {saved} new samples ({total} total)")
    print(f"     Skipped: {skipped}")
    print(f"     Location: {OUTPUT_DIR}")
    print()
    if total >= 20:
        print("  Next step — train the model:")
        print("  python3 wakeword/train_hey_viron.py")
    else:
        print(f"  ⚠ Need at least 20 samples. You have {total}.")
        print("  Run this script again to add more.")
    print("=" * 60)


if __name__ == "__main__":
    main()
