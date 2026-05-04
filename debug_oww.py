#!/usr/bin/env python3
"""
OWW live score monitor. Stop VIRON first, then run this to see
real-time wake word scores while you speak.
"""
import subprocess
import numpy as np
import time
import sys
from openwakeword.model import Model

print("Loading OpenWakeWord model...")
oww = Model(wakeword_models=["hey_jarvis_v0.1"], inference_framework="onnx")
print("Model loaded. Listening for 60 seconds.")
print("Say 'Hey Jarvis' clearly, multiple times.")
print()
print(f"{'Time':>6}  {'Audio_Max':>10}  {'Audio_RMS':>10}  {'OWW_Score':>10}  Status")
print("-" * 70)

ALSA = "plughw:CARD=Array,DEV=0"
proc = subprocess.Popen(
    ["arecord", "-D", ALSA, "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw", "-q"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
)

start = time.time()
max_seen = 0.0
last_print = 0
detections = 0

try:
    while time.time() - start < 60:
        raw = proc.stdout.read(2560)  # 1280 samples * 2 bytes
        if len(raw) < 2560:
            print("MIC READ FAILED — arecord died")
            break
        samples = np.frombuffer(raw, dtype=np.int16)
        audio_max = abs(samples).max()
        audio_rms = int(np.sqrt((samples.astype(float) ** 2).mean()))
        scores = oww.predict(samples)
        score = max(scores.values()) if scores else 0.0
        max_seen = max(max_seen, score)

        flag = ""
        if score >= 0.5:
            flag = "  *** WAKE DETECTED ***"
            detections += 1
            oww.reset()
        elif score >= 0.2:
            flag = "  (close)"

        # Print every 0.5 seconds OR when score is interesting
        now = time.time()
        if score >= 0.2 or now - last_print > 0.5:
            elapsed = now - start
            print(f"{elapsed:6.1f}  {audio_max:>10}  {audio_rms:>10}  {score:>10.3f}{flag}")
            last_print = now
finally:
    proc.terminate()
    print()
    print(f"Done. Max OWW score seen: {max_seen:.3f}")
    print(f"Wake detections: {detections}")
    if max_seen < 0.05:
        print("VERDICT: OWW is seeing nothing. Mic likely silent or hardware issue.")
    elif max_seen < 0.2:
        print("VERDICT: OWW barely registers anything. Mic too quiet, or wake phrase not pronounced clearly.")
    elif max_seen < 0.5:
        print("VERDICT: Close but below threshold. Lower VIRON_WAKE_SENSITIVITY in .env or speak louder.")
    else:
        print("VERDICT: OWW is detecting wake word fine — issue is in pipeline integration.")
