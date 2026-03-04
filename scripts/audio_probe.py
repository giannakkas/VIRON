#!/usr/bin/env python3
"""
VIRON Audio Probe — Find and test the ReSpeaker XVF3800 mic.
=============================================================
Lists ALSA devices, records a short clip, compares left/right channels,
and outputs recommended env vars.

Usage:
  python3 scripts/audio_probe.py                         # auto-detect XVF3800
  python3 scripts/audio_probe.py --device "plughw:2,0"   # test specific device
  python3 scripts/audio_probe.py --seconds 3             # record 3 seconds
  python3 scripts/audio_probe.py --speak                 # prompt to speak during test
"""

import argparse
import os
import re
import subprocess
import sys
import wave
import tempfile
import numpy as np

RATE = 16000


def run(cmd, timeout=10):
    """Run a command, return stdout or None on failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, text=True)
        return r.stdout if r.returncode == 0 else None
    except Exception:
        return None


def list_capture_devices():
    """List ALSA capture devices and return parsed entries."""
    raw = run(["arecord", "-l"])
    if not raw:
        print("❌ arecord -l failed — is ALSA installed?")
        return []
    print("═" * 60)
    print("  ALSA Capture Devices")
    print("═" * 60)
    print(raw)

    devices = []
    for line in raw.splitlines():
        m = re.match(r"card (\d+): .+\[(.+?)\].+device (\d+): .+\[(.+?)\]", line)
        if m:
            card, card_name, dev, dev_name = m.group(1), m.group(2), m.group(3), m.group(4)
            hw = f"plughw:{card},{dev}"
            is_xvf = any(k in (card_name + dev_name).lower() for k in ["xvf", "respeaker", "xmos", "usb audio"])
            devices.append({
                "card": card, "dev": dev, "hw": hw,
                "card_name": card_name, "dev_name": dev_name,
                "likely_xvf": is_xvf,
            })
            tag = " ← likely XVF3800" if is_xvf else ""
            print(f"  {hw} — {card_name} / {dev_name}{tag}")
    print()
    return devices


def auto_detect_xvf(devices):
    """Pick the most likely XVF3800 device."""
    for d in devices:
        if d["likely_xvf"]:
            return d["hw"]
    # If nothing obvious, prefer higher card numbers (USB audio tends to be last)
    usb = [d for d in devices if "usb" in (d["card_name"] + d["dev_name"]).lower()]
    if usb:
        return usb[-1]["hw"]
    return None


def record_stereo(device, seconds):
    """Record stereo from device, return raw bytes or None."""
    cmd = ["arecord", "-D", device, "-f", "S16_LE", "-r", str(RATE),
           "-c", "2", "-t", "raw", "-d", str(seconds)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=seconds + 5)
        if r.returncode != 0:
            err = r.stderr.decode(errors="replace")
            print(f"❌ arecord failed for {device}: {err[:200]}")
            return None
        return r.stdout
    except subprocess.TimeoutExpired:
        print(f"❌ arecord timed out for {device}")
        return None


def analyze_channels(raw_stereo, seconds):
    """Analyze left and right channels, return dict with stats."""
    stereo = np.frombuffer(raw_stereo, dtype=np.int16)
    if len(stereo) < RATE:
        print("❌ Recording too short")
        return None

    ch0 = stereo[0::2].astype(np.float32)  # left
    ch1 = stereo[1::2].astype(np.float32)  # right

    # Split into 200ms windows
    win = RATE // 5
    results = {}
    for i, (ch, name) in enumerate([(ch0, "ch0 (left)"), (ch1, "ch1 (right)")]):
        rms_vals = []
        for start in range(0, len(ch) - win, win):
            w = ch[start:start + win]
            rms_vals.append(np.sqrt(np.mean(w ** 2)))
        rms_vals = np.array(rms_vals)
        results[i] = {
            "name": name,
            "rms_mean": float(np.mean(rms_vals)),
            "rms_min": float(np.min(rms_vals)),
            "rms_max": float(np.max(rms_vals)),
            "rms_std": float(np.std(rms_vals)),
            "peak": float(np.max(np.abs(ch))),
            "dynamic_range": float(np.max(rms_vals) / max(np.min(rms_vals), 1)),
        }
    return results


def save_channels(raw_stereo, prefix="/tmp/viron_probe"):
    """Save individual channel WAVs for listening."""
    stereo = np.frombuffer(raw_stereo, dtype=np.int16)
    ch0 = stereo[0::2]
    ch1 = stereo[1::2]
    for ch, suffix in [(ch0, "ch0_left"), (ch1, "ch1_right")]:
        path = f"{prefix}_{suffix}.wav"
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(ch.tobytes())
        print(f"  💾 {path}")


def main():
    parser = argparse.ArgumentParser(description="VIRON Audio Probe — find and test XVF3800 mic")
    parser.add_argument("--device", "-d", help="ALSA device (e.g. plughw:2,0). Auto-detects if omitted.")
    parser.add_argument("--seconds", "-s", type=int, default=2, help="Recording duration (default 2)")
    parser.add_argument("--speak", action="store_true", help="Prompt user to speak during recording")
    parser.add_argument("--save", action="store_true", help="Save WAV files for each channel")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only output env vars")
    args = parser.parse_args()

    if not args.quiet:
        print()
        print("🔊 VIRON Audio Probe")
        print("=" * 60)
        print()

    # List devices
    devices = list_capture_devices()

    # Pick device
    device = args.device
    if not device:
        device = os.environ.get("VIRON_MIC_DEVICE")
    if not device:
        device = auto_detect_xvf(devices)
    if not device:
        print("❌ No suitable capture device found. Specify --device manually.")
        sys.exit(1)

    if not args.quiet:
        print(f"📍 Testing device: {device}")
        print(f"   Recording {args.seconds}s stereo at {RATE}Hz...")
        if args.speak:
            print()
            print("   🗣️  SAY SOMETHING NOW! (e.g., 'Hey VIRON, hello!')")
        print()

    raw = record_stereo(device, args.seconds)
    if not raw:
        sys.exit(1)

    stats = analyze_channels(raw, args.seconds)
    if not stats:
        sys.exit(1)

    # Determine best channel
    # The "better" channel for STT has higher dynamic range (speech stands out from noise)
    # and ideally lower baseline noise (rms_min)
    ch0, ch1 = stats[0], stats[1]

    if not args.quiet:
        print("═" * 60)
        print("  Channel Analysis")
        print("═" * 60)
        for i in [0, 1]:
            s = stats[i]
            print(f"  {s['name']}:")
            print(f"    RMS mean={s['rms_mean']:.0f}  min={s['rms_min']:.0f}  max={s['rms_max']:.0f}  std={s['rms_std']:.0f}")
            print(f"    Peak={s['peak']:.0f}  Dynamic range={s['dynamic_range']:.1f}x")
        print()

    # Decision logic
    # Prefer the channel with:
    # 1. Higher dynamic range (speech pops out of noise)
    # 2. Lower minimum RMS (quieter floor)
    score0 = ch0["dynamic_range"] * 2 + (1000 / max(ch0["rms_min"], 1))
    score1 = ch1["dynamic_range"] * 2 + (1000 / max(ch1["rms_min"], 1))

    best = 1 if score1 >= score0 else 0
    best_stats = stats[best]

    if not args.quiet:
        print(f"  🏆 Best channel: ch{best} ({best_stats['name']})")
        if best == 1:
            print("     (ch1 = ASR-optimized beam on XVF3800 — expected winner)")
        else:
            print("     (ch0 = beamformed+AEC on XVF3800 — unusual, verify manually)")
        print()

    if args.save:
        print("  Saving WAV files:")
        save_channels(raw)
        print()

    # Output recommended env vars
    print("═" * 60)
    print("  Recommended .env settings")
    print("═" * 60)
    print(f'  VIRON_MIC_DEVICE="{device}"')
    print(f"  VIRON_MIC_CHANNEL={best}")
    print()

    # Threshold recommendation based on noise
    noise = best_stats["rms_min"]
    if noise < 50:
        thresh = "0.22-0.26"
        note = "quiet room — lower threshold OK"
    elif noise < 200:
        thresh = "0.26-0.32"
        note = "moderate noise — default range"
    else:
        thresh = "0.32-0.40"
        note = "noisy (TV/music) — raise threshold to avoid false triggers"
    print(f"  VIRON_WAKEWORD_THRESHOLD=0.28  # range {thresh} ({note})")
    print()

    # Quick test commands
    if not args.quiet:
        print("═" * 60)
        print("  Quick verification commands")
        print("═" * 60)
        print(f"  # Test wake word service:")
        print(f"  curl http://127.0.0.1:8085/wakeword/status")
        print()
        print(f"  # Test listen endpoint:")
        print(f"  curl -s -X POST http://127.0.0.1:5000/api/listen \\")
        print(f"    -H 'Content-Type: application/json' \\")
        print(f"    -d '{{\"max_duration\":6,\"silence_duration\":0.25,\"lang\":\"en\"}}'")
        print()

    if args.quiet:
        # Machine-readable output
        print(f"VIRON_MIC_DEVICE={device}")
        print(f"VIRON_MIC_CHANNEL={best}")


if __name__ == "__main__":
    main()
