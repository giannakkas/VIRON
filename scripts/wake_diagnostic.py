#!/usr/bin/env python3
"""
VIRON Wake Word Diagnostic — Test OWW + mic audio
===================================================
Tests whether openWakeWord is receiving and scoring audio correctly.

Usage:
  python3 scripts/wake_diagnostic.py                    # full diagnostic
  python3 scripts/wake_diagnostic.py --test-synth       # test OWW with synthetic audio
  python3 scripts/wake_diagnostic.py --test-mic 5       # record 5s and score it
"""

import os
import sys
import time
import argparse
import subprocess
import tempfile
import wave
import numpy as np

ALSA_DEVICE = os.environ.get("VIRON_MIC_DEVICE", "plughw:2,0")
MIC_CHANNEL = int(os.environ.get("VIRON_MIC_CHANNEL", "1"))
RATE = 16000
CHUNK = 1280  # 80ms


def test_oww_loads():
    """Test 1: Does openWakeWord load?"""
    print("TEST 1: Loading openWakeWord...")
    try:
        import openwakeword
        from openwakeword.model import Model
        openwakeword.utils.download_models()
        model = Model(wakeword_models=["hey_jarvis_v0.1"], vad_threshold=0.5)
        names = list(model.models.keys())
        print(f"  ✅ OWW loaded: {names}")
        return model, names
    except Exception as e:
        print(f"  ❌ OWW failed: {e}")
        return None, []


def test_oww_with_silence(model, names):
    """Test 2: Feed silence to OWW — should score near 0."""
    print("\nTEST 2: Feeding silence to OWW...")
    silence = np.zeros(CHUNK, dtype=np.int16)
    scores = []
    for _ in range(20):
        preds = model.predict(silence)
        for name in names:
            scores.append(preds[name])
    avg = np.mean(scores)
    mx = np.max(scores)
    print(f"  Silence scores: avg={avg:.4f} max={mx:.4f}")
    if mx < 0.01:
        print(f"  ✅ OWW correctly ignores silence")
    else:
        print(f"  ⚠ OWW gives non-zero scores for silence — unusual")
    return True


def test_oww_with_synthetic(model, names):
    """Test 3: Generate 'Hey Jarvis' with edge-tts and feed to OWW."""
    print("\nTEST 3: Generating synthetic 'Hey Jarvis' and scoring...")
    try:
        import edge_tts
        import asyncio

        async def gen():
            mp3 = tempfile.mktemp(suffix=".mp3")
            wav = tempfile.mktemp(suffix=".wav")
            comm = edge_tts.Communicate("Hey Jarvis", "en-US-GuyNeural")
            await comm.save(mp3)
            subprocess.run(["ffmpeg", "-y", "-i", mp3, "-ar", "16000", "-ac", "1", wav],
                           capture_output=True, timeout=10)
            os.unlink(mp3)
            return wav

        wav_path = asyncio.run(gen())
        with wave.open(wav_path, 'rb') as wf:
            raw = wf.readframes(wf.getnframes())
        os.unlink(wav_path)

        audio = np.frombuffer(raw, dtype=np.int16)
        print(f"  Synthetic clip: {len(audio)} samples ({len(audio)/RATE:.2f}s), RMS={np.sqrt(np.mean(audio.astype(np.float32)**2)):.0f}")

        # Reset model and feed in chunks
        model.reset()
        max_score = 0
        scores_log = []
        for i in range(0, len(audio) - CHUNK, CHUNK):
            chunk = audio[i:i + CHUNK]
            preds = model.predict(chunk)
            for name in names:
                s = preds[name]
                scores_log.append(s)
                if s > max_score:
                    max_score = s

        print(f"  Peak OWW score: {max_score:.4f}")
        top5 = sorted(scores_log, reverse=True)[:5]
        print(f"  Top 5 scores: {[f'{s:.3f}' for s in top5]}")

        if max_score > 0.3:
            print(f"  ✅ OWW correctly detects synthetic 'Hey Jarvis'")
        elif max_score > 0.1:
            print(f"  ⚠ OWW gives low score — model works but sensitivity is marginal")
        else:
            print(f"  ❌ OWW gives near-zero — model may be broken or audio format wrong")
        return max_score
    except ImportError:
        print("  ⚠ edge-tts not installed, skipping synthetic test")
        return 0
    except Exception as e:
        print(f"  ❌ Synthetic test failed: {e}")
        return 0


def test_mic_audio(model, names, seconds=5):
    """Test 4: Record from mic and score with OWW."""
    print(f"\nTEST 4: Recording {seconds}s from mic — say 'Hey Jarvis'...")
    print(f"  Device: {ALSA_DEVICE}, Channel: {MIC_CHANNEL}")
    print(f"  🗣️  SAY 'HEY JARVIS' NOW!")

    cmd = ["arecord", "-D", ALSA_DEVICE, "-f", "S16_LE", "-r", str(RATE),
           "-c", "2", "-t", "raw", "-d", str(seconds)]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=seconds + 5)
        if result.returncode != 0:
            print(f"  ❌ arecord failed: {result.stderr.decode()[:200]}")
            return

        raw = result.stdout
        stereo = np.frombuffer(raw, dtype=np.int16)
        audio = stereo[MIC_CHANNEL::2]  # extract selected channel

        rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
        peak = np.max(np.abs(audio.astype(np.float32)))
        print(f"  Recorded: {len(audio)} samples ({len(audio)/RATE:.2f}s)")
        print(f"  RMS={rms:.0f}  Peak={peak:.0f}")

        if rms < 10:
            print(f"  ❌ Audio is nearly silent — wrong device/channel?")
            print(f"     Try: VIRON_MIC_CHANNEL={'0' if MIC_CHANNEL == 1 else '1'}")
            return

        # Feed to OWW in chunks
        model.reset()
        max_score = 0
        scores_log = []
        for i in range(0, len(audio) - CHUNK, CHUNK):
            chunk = audio[i:i + CHUNK]
            preds = model.predict(chunk)
            for name in names:
                s = preds[name]
                scores_log.append((i / RATE, s))
                if s > max_score:
                    max_score = s

        print(f"\n  Peak OWW score from mic: {max_score:.4f}")
        top5 = sorted(scores_log, key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top 5 scores: {[f't={t:.1f}s score={s:.3f}' for t, s in top5]}")

        # Also save the mono audio for inspection
        wav_path = "/tmp/viron_mic_test.wav"
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(audio.tobytes())
        print(f"\n  💾 Saved to {wav_path}")

        if max_score > 0.3:
            print(f"  ✅ OWW detects 'Hey Jarvis' from mic!")
        elif max_score > 0.08:
            print(f"  ⚠ OWW gives some score but below threshold (0.28)")
            print(f"     Try lowering VIRON_WAKEWORD_THRESHOLD to {max_score - 0.05:.2f}")
        else:
            print(f"  ❌ OWW gives near-zero for live mic audio")
            print(f"     Possible causes:")
            print(f"     - Audio too quiet (RMS={rms:.0f}) — try speaking louder/closer")
            print(f"     - Wrong channel — try VIRON_MIC_CHANNEL={'0' if MIC_CHANNEL == 1 else '1'}")
            print(f"     - Greek accent too different from training data")
            print(f"     → SOLUTION: Train custom 'Hey VIRON' model with your voice")

    except subprocess.TimeoutExpired:
        print(f"  ❌ Recording timed out")
    except Exception as e:
        print(f"  ❌ Mic test failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="VIRON Wake Word Diagnostic")
    parser.add_argument("--test-synth", action="store_true", help="Test OWW with synthetic audio only")
    parser.add_argument("--test-mic", type=int, nargs="?", const=5, help="Record N seconds and test (default 5)")
    args = parser.parse_args()

    print("=" * 60)
    print("  VIRON Wake Word Diagnostic")
    print("=" * 60)
    print(f"  Device: {ALSA_DEVICE}  Channel: {MIC_CHANNEL}")
    print()

    model, names = test_oww_loads()
    if not model:
        sys.exit(1)

    test_oww_with_silence(model, names)

    if args.test_synth or not args.test_mic:
        test_oww_with_synthetic(model, names)

    if args.test_mic or (not args.test_synth):
        test_mic_audio(model, names, args.test_mic or 5)

    print("\n" + "=" * 60)
    print("  Diagnostic complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
