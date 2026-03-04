#!/usr/bin/env python3
"""
Test XVF3800 beamforming — compares raw mic vs beamformed channel noise.
Run: python3 scripts/test_beamforming.py
Leave TV ON during test to see the difference.
"""
import subprocess, wave, tempfile, os
import numpy as np

DEVICE = os.environ.get("VIRON_MIC_DEVICE", "plughw:2,0")
RATE   = 16000
DURATION = 3  # seconds per test
CHUNK  = RATE * DURATION  # total samples per channel

def record_stereo():
    print(f"🎤 Recording 3s stereo from {DEVICE} ...")
    cmd = ["arecord", "-D", DEVICE, "-f", "S16_LE", "-r", str(RATE),
           "-c", "2", "-t", "raw", "-d", str(DURATION)]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"❌ arecord failed: {result.stderr.decode()}")
        return None
    return result.stdout

def record_mono():
    print(f"🎤 Recording 3s mono from {DEVICE} ...")
    cmd = ["arecord", "-D", DEVICE, "-f", "S16_LE", "-r", str(RATE),
           "-c", "1", "-t", "raw", "-d", str(DURATION)]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"❌ arecord failed: {result.stderr.decode()}")
        return None
    return result.stdout

def rms(data):
    a = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return np.sqrt(np.mean(a**2))

def peak(data):
    a = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return np.max(np.abs(a))

def save_wav(data, path, channels=1):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(data)
    print(f"   💾 Saved: {path}")

print("=" * 55)
print("  VIRON XVF3800 Beamforming Test")
print("  Leave the TV ON — test shows noise rejection")
print("=" * 55)
print()

# --- Test 1: Raw mono (old way) ---
print("TEST 1: Raw mono (old method — no beamforming)")
mono_raw = record_mono()
if mono_raw:
    r = rms(mono_raw)
    p = peak(mono_raw)
    print(f"   RMS noise : {r:.0f}")
    print(f"   Peak      : {p:.0f}")
    save_wav(mono_raw, "/tmp/test_mono_raw.wav", channels=1)

print()

# --- Test 2: Stereo — extract both channels ---
print("TEST 2: Stereo — ch0 (beamformed+AEC) vs ch1 (ASR beam)")
stereo_raw = record_stereo()
if stereo_raw:
    stereo = np.frombuffer(stereo_raw, dtype=np.int16)
    ch0 = stereo[0::2]  # left  = beamformed + AEC
    ch1 = stereo[1::2]  # right = ASR-optimized beam

    r0, p0 = np.sqrt(np.mean(ch0.astype(np.float32)**2)), np.max(np.abs(ch0.astype(np.float32)))
    r1, p1 = np.sqrt(np.mean(ch1.astype(np.float32)**2)), np.max(np.abs(ch1.astype(np.float32)))

    print(f"   ch0 (AEC beam)  — RMS: {r0:.0f}  Peak: {p0:.0f}")
    print(f"   ch1 (ASR beam)  — RMS: {r1:.0f}  Peak: {p1:.0f}")

    save_wav(ch0.tobytes(), "/tmp/test_ch0_aec.wav")
    save_wav(ch1.tobytes(), "/tmp/test_ch1_asr.wav")

    print()
    print("=" * 55)
    print("  RESULTS")
    print("=" * 55)

    if mono_raw:
        r_mono = rms(mono_raw)
        improvement0 = r_mono / r0 if r0 > 0 else 0
        improvement1 = r_mono / r1 if r1 > 0 else 0
        print(f"  Raw mono RMS    : {r_mono:.0f}")
        print(f"  ch0 (AEC) RMS   : {r0:.0f}  ({improvement0:.1f}x quieter)")
        print(f"  ch1 (ASR) RMS   : {r1:.0f}  ({improvement1:.1f}x quieter)")
        print()
        if improvement1 > 1.5:
            print("  ✅ BEAMFORMING WORKS — ch1 has less noise than raw mono")
            print("     Voice wake word should now work with TV on!")
        elif improvement1 > 1.0:
            print("  ⚠  Mild improvement — beamforming active but limited")
            print("     Try pointing the ReSpeaker toward your voice")
        else:
            print("  ❌ No improvement — XVF3800 may need xvf_host initialization")
            print("     Run: sudo /tmp/xvf3800/host_control/jetson/xvf_host VERSION")

    print()
    print("  Listen to the saved WAV files to compare:")
    print("  /tmp/test_mono_raw.wav  ← old (raw, TV noise)")
    print("  /tmp/test_ch0_aec.wav   ← ch0 beamformed+AEC")
    print("  /tmp/test_ch1_asr.wav   ← ch1 ASR beam (used by VIRON)")
    print()
    print("  Copy to your PC:")
    print("  scp test@100.66.223.46:/tmp/test_*.wav .")
