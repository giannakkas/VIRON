#!/usr/bin/env python3
"""
VIRON Custom Wake Word Training — "Hey VIRON"
=============================================
Complete pipeline: synthetic samples + personal recordings → ONNX model.

Prerequisites:
  pip install openwakeword edge-tts scipy onnxruntime numpy

Usage:
  # Step 1: Record your voice (30+ samples)
  python3 scripts/record_wake_samples.py --count 30

  # Step 2: Generate synthetic + train
  python3 wakeword/train_hey_viron.py

  # Step 3: Restart wakeword service (auto-detects new model)
  pkill -f "wakeword/service.py" && python3 wakeword/service.py &

Options:
  python3 wakeword/train_hey_viron.py --skip-synth    # only train (if samples exist)
  python3 wakeword/train_hey_viron.py --synth-only     # only generate synthetic data
  python3 wakeword/train_hey_viron.py --test            # test trained model
"""

import os
import sys
import asyncio
import random
import subprocess
import tempfile
import time
import json
import argparse
import wave
import shutil
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"
DATA_DIR = SCRIPT_DIR / "training_data"
POSITIVE_DIR = DATA_DIR / "positive"
PERSONAL_DIR = DATA_DIR / "personal"
NEGATIVE_DIR = DATA_DIR / "negative"

RATE = 16000

# Edge-TTS voices for diverse training data
VOICES = [
    "en-US-GuyNeural", "en-US-JennyNeural", "en-US-AriaNeural",
    "en-US-DavisNeural", "en-US-AmberNeural", "en-US-BrandonNeural",
    "en-US-ChristopherNeural", "en-US-CoraNeural", "en-US-EricNeural",
    "en-US-JacobNeural", "en-US-MichelleNeural",
    "en-GB-RyanNeural", "en-GB-SoniaNeural", "en-GB-LibbyNeural",
    "en-AU-NatashaNeural", "en-AU-WilliamNeural",
    "en-IN-NeerjaNeural", "en-IN-PrabhatNeural",
    "en-IE-ConnorNeural", "en-IE-EmilyNeural",
    "el-GR-AthinaNeural", "el-GR-NestorasNeural",
]

PHRASES = [
    "Hey VIRON", "Hey Viron", "hey viron", "Hey viron",
    "Hey VIRON!", "Hey, VIRON", "hey Viron",
    # Common mispronunciations (helps model generalize)
    "Hey Byron", "Hey Veron", "Hey Viro",
]

RATES = ["-15%", "-5%", "+0%", "+10%", "+20%"]
PITCHES = ["-3Hz", "+0Hz", "+3Hz"]


async def generate_synthetic_samples(count_target=1500):
    """Generate synthetic 'Hey VIRON' clips using edge-tts."""
    import edge_tts
    
    POSITIVE_DIR.mkdir(parents=True, exist_ok=True)
    
    existing = len(list(POSITIVE_DIR.glob("synth_*.wav")))
    if existing >= count_target:
        print(f"  Already have {existing} synthetic samples (target {count_target}), skipping")
        return existing
    
    count = existing
    combos = [(v, p, r, pt) for v in VOICES for p in PHRASES for r in RATES for pt in PITCHES]
    random.shuffle(combos)
    combos = combos[:count_target - existing]
    
    total = len(combos)
    print(f"  Generating {total} synthetic samples...")
    
    errors = 0
    for v, phrase, rate, pitch in combos:
        outfile = POSITIVE_DIR / f"synth_{count:04d}.wav"
        mp3file = POSITIVE_DIR / f"synth_{count:04d}.mp3"
        
        try:
            comm = edge_tts.Communicate(phrase, v, rate=rate, pitch=pitch)
            await comm.save(str(mp3file))
            
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3file),
                "-ar", "16000", "-ac", "1", str(outfile)
            ], capture_output=True, timeout=10)
            
            mp3file.unlink(missing_ok=True)
            count += 1
            
            if (count - existing) % 100 == 0:
                print(f"    {count - existing}/{total} generated...")
                
        except Exception as e:
            errors += 1
            mp3file.unlink(missing_ok=True)
            if errors > 20:
                print(f"  ⚠ Too many errors ({errors}), stopping synthesis")
                break
    
    print(f"  ✅ {count} total synthetic samples")
    return count


def generate_negative_samples(count_target=500):
    """Generate negative samples — silence, noise, non-wake-word speech."""
    NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)
    
    existing = len(list(NEGATIVE_DIR.glob("neg_*.wav")))
    if existing >= count_target:
        print(f"  Already have {existing} negative samples, skipping")
        return existing
    
    count = existing
    print(f"  Generating {count_target - existing} negative samples...")
    
    # Generate various noise types
    for i in range(count, count_target):
        outfile = NEGATIVE_DIR / f"neg_{i:04d}.wav"
        duration = random.uniform(0.5, 2.0)
        samples = int(RATE * duration)
        
        noise_type = random.choice(["silence", "white", "pink", "hum"])
        
        if noise_type == "silence":
            audio = np.zeros(samples, dtype=np.int16)
        elif noise_type == "white":
            level = random.uniform(10, 200)
            audio = (np.random.randn(samples) * level).astype(np.int16)
        elif noise_type == "pink":
            level = random.uniform(10, 150)
            white = np.random.randn(samples)
            # Simple pink noise approximation
            b = [0.049922035, -0.095993537, 0.050612699, -0.004709510]
            a = [1, -2.494956002, 2.017265875, -0.522189400]
            from scipy.signal import lfilter
            pink = lfilter(b, a, white)
            audio = (pink * level).astype(np.int16)
        elif noise_type == "hum":
            t = np.linspace(0, duration, samples)
            freq = random.choice([50, 60, 100, 120])
            level = random.uniform(20, 100)
            audio = (np.sin(2 * np.pi * freq * t) * level).astype(np.int16)
        
        with wave.open(str(outfile), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(audio.tobytes())
        count += 1
    
    print(f"  ✅ {count} negative samples")
    return count


def load_clips_as_features(clip_dir, max_clips=None):
    """Load WAV clips and convert to mel-spectrogram features."""
    clips = sorted(Path(clip_dir).glob("*.wav"))
    if max_clips:
        clips = clips[:max_clips]
    
    features = []
    for clip_path in clips:
        try:
            with wave.open(str(clip_path), 'rb') as wf:
                raw = wf.readframes(wf.getnframes())
                sr = wf.getframerate()
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Resample if needed
            if sr != RATE:
                from scipy.signal import resample
                audio = resample(audio, int(len(audio) * RATE / sr))
            
            # Pad/trim to 1.5s
            target = int(RATE * 1.5)
            if len(audio) > target:
                audio = audio[:target]
            elif len(audio) < target:
                audio = np.pad(audio, (0, target - len(audio)))
            
            features.append(audio)
        except Exception:
            continue
    
    return np.array(features) if features else np.array([])


def train_model_oww():
    """Try training with openWakeWord's built-in training pipeline."""
    try:
        # Check if training module is available
        from openwakeword import train_custom_models
        
        print("  Using openWakeWord's built-in training pipeline")
        
        # Merge personal + synthetic into one directory
        all_positive_dir = DATA_DIR / "all_positive"
        all_positive_dir.mkdir(exist_ok=True)
        
        for src in [POSITIVE_DIR, PERSONAL_DIR]:
            if src.exists():
                for f in src.glob("*.wav"):
                    dst = all_positive_dir / f.name
                    if not dst.exists():
                        shutil.copy2(f, dst)
        
        positive_count = len(list(all_positive_dir.glob("*.wav")))
        print(f"  Total positive clips: {positive_count}")
        
        if positive_count < 50:
            print(f"  ⚠ Need at least 50 positive clips. Have {positive_count}.")
            return False
        
        # Train
        output = MODELS_DIR / "hey_viron"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        train_custom_models.train(
            positive_clips_dir=str(all_positive_dir),
            output_dir=str(output),
            model_name="hey_viron",
        )
        
        onnx_path = output / "hey_viron.onnx"
        if onnx_path.exists():
            final = MODELS_DIR / "hey_viron.onnx"
            shutil.copy2(onnx_path, final)
            print(f"  ✅ Model saved: {final}")
            return True
        
        # Check alternative output paths
        for f in output.rglob("*.onnx"):
            final = MODELS_DIR / "hey_viron.onnx"
            shutil.copy2(f, final)
            print(f"  ✅ Model saved: {final}")
            return True
        
        print("  ⚠ Training completed but no ONNX file found")
        return False
        
    except ImportError:
        print("  openWakeWord training module not available")
        return False
    except Exception as e:
        print(f"  ⚠ OWW training failed: {e}")
        return False


def train_model_simple():
    """Simple training: build a small ONNX classifier from mel features."""
    try:
        import onnx
        from onnx import helper, TensorProto
        import onnxruntime as ort
    except ImportError:
        print("  ⚠ Need onnx + onnxruntime for simple training")
        return False
    
    print("  Using simple mel-spectrogram classifier")
    
    # Load positive clips
    pos_clips = []
    for src in [POSITIVE_DIR, PERSONAL_DIR]:
        if src.exists():
            clips = load_clips_as_features(src)
            if len(clips) > 0:
                pos_clips.append(clips)
    
    if not pos_clips:
        print("  ❌ No positive clips found!")
        return False
    
    positive = np.concatenate(pos_clips)
    
    # Load negative clips
    negative = load_clips_as_features(NEGATIVE_DIR) if NEGATIVE_DIR.exists() else np.array([])
    
    # If not enough negatives, generate some
    if len(negative) < 200:
        print("  Generating quick negative samples...")
        generate_negative_samples(300)
        negative = load_clips_as_features(NEGATIVE_DIR)
    
    print(f"  Positive: {len(positive)} clips")
    print(f"  Negative: {len(negative)} clips")
    
    if len(positive) < 20:
        print("  ❌ Need at least 20 positive clips")
        return False
    
    # Compute mel spectrograms
    def audio_to_mel(audio, n_mels=40, n_fft=512, hop=160):
        """Simple mel spectrogram."""
        from scipy.signal import stft
        _, _, Zxx = stft(audio, fs=RATE, nperseg=n_fft, noverlap=n_fft - hop)
        power = np.abs(Zxx) ** 2
        
        # Simple mel filterbank
        mel_freqs = np.linspace(0, 2595 * np.log10(1 + RATE / 2 / 700), n_mels + 2)
        mel_freqs = 700 * (10 ** (mel_freqs / 2595) - 1)
        bin_freqs = np.floor((n_fft + 1) * mel_freqs / RATE).astype(int)
        
        fb = np.zeros((n_mels, power.shape[0]))
        for i in range(n_mels):
            start, center, end = bin_freqs[i], bin_freqs[i + 1], bin_freqs[i + 2]
            if start < power.shape[0] and end < power.shape[0]:
                for j in range(start, center):
                    if center > start:
                        fb[i, j] = (j - start) / (center - start)
                for j in range(center, end):
                    if end > center:
                        fb[i, j] = (end - j) / (end - center)
        
        mel = np.dot(fb, power)
        mel = np.log(mel + 1e-10)
        return mel.mean(axis=1)  # Average over time → fixed-size feature
    
    print("  Computing features...")
    X_pos = np.array([audio_to_mel(a) for a in positive])
    X_neg = np.array([audio_to_mel(a) for a in negative])
    
    X = np.concatenate([X_pos, X_neg])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    # Normalize
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X = (X - mean) / std
    
    # Simple logistic regression trained with numpy
    n_features = X.shape[1]
    W = np.zeros(n_features)
    b = 0.0
    lr = 0.1
    
    print("  Training classifier...")
    for epoch in range(200):
        logits = X @ W + b
        probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
        loss = -np.mean(y * np.log(probs + 1e-10) + (1 - y) * np.log(1 - probs + 1e-10))
        
        grad_w = X.T @ (probs - y) / len(y) + 0.01 * W
        grad_b = np.mean(probs - y)
        W -= lr * grad_w
        b -= lr * grad_b
        
        if epoch % 50 == 0:
            preds = (probs > 0.5).astype(float)
            acc = np.mean(preds == y)
            print(f"    Epoch {epoch}: loss={loss:.4f} acc={acc:.3f}")
    
    # Final accuracy
    logits = X @ W + b
    probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
    preds = (probs > 0.5).astype(float)
    acc = np.mean(preds == y)
    
    # Per-class accuracy
    pos_acc = np.mean(preds[y == 1] == 1)
    neg_acc = np.mean(preds[y == 0] == 0)
    print(f"  Final: acc={acc:.3f} pos_acc={pos_acc:.3f} neg_acc={neg_acc:.3f}")
    
    # Save model parameters + normalization as JSON (for custom inference)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_data = {
        "type": "logistic_mel",
        "weights": W.tolist(),
        "bias": float(b),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_mels": 40,
        "n_fft": 512,
        "hop": 160,
        "rate": RATE,
        "accuracy": float(acc),
        "pos_accuracy": float(pos_acc),
        "neg_accuracy": float(neg_acc),
        "n_positive": len(X_pos),
        "n_negative": len(X_neg),
    }
    
    model_json = MODELS_DIR / "hey_viron_simple.json"
    with open(model_json, 'w') as f:
        json.dump(model_data, f, indent=2)
    print(f"  ✅ Simple model saved: {model_json}")
    
    # Also try exporting as ONNX for compatibility
    try:
        # Create a simple ONNX model (linear classifier)
        from onnx import numpy_helper
        
        W_init = numpy_helper.from_array(W.astype(np.float32).reshape(1, -1), "W")
        b_init = numpy_helper.from_array(np.array([b], dtype=np.float32), "b")
        mean_init = numpy_helper.from_array(mean.astype(np.float32), "mean")
        std_init = numpy_helper.from_array(std.astype(np.float32), "std")
        
        X_in = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, n_features])
        Y_out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])
        
        # Normalize → matmul → add → sigmoid
        norm_sub = helper.make_node("Sub", ["input", "mean"], ["normed_sub"])
        norm_div = helper.make_node("Div", ["normed_sub", "std"], ["normed"])
        matmul = helper.make_node("MatMul", ["normed", "W"], ["logits_raw"])
        # Reshape for add
        reshape_shape = numpy_helper.from_array(np.array([1, 1], dtype=np.int64), "reshape_shape")
        reshape = helper.make_node("Reshape", ["logits_raw", "reshape_shape"], ["logits_reshaped"])
        add = helper.make_node("Add", ["logits_reshaped", "b"], ["logits"])
        sigmoid = helper.make_node("Sigmoid", ["logits"], ["output"])
        
        graph = helper.make_graph(
            [norm_sub, norm_div, matmul, reshape, add, sigmoid],
            "hey_viron",
            [X_in], [Y_out],
            [W_init, b_init, mean_init, std_init, reshape_shape]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        
        onnx_path = MODELS_DIR / "hey_viron_simple.onnx"
        onnx.save(model, str(onnx_path))
        print(f"  ✅ ONNX model saved: {onnx_path}")
        
        # Quick test
        sess = ort.InferenceSession(str(onnx_path))
        test_input = X_pos[0:1].astype(np.float32)
        # Don't normalize — model does it internally
        test_input_raw = np.concatenate([X_pos[:1] * std + mean], axis=0).astype(np.float32)
        result = sess.run(None, {"input": test_input_raw})
        print(f"  Test: positive sample → score={result[0][0][0]:.3f}")
        
    except Exception as e:
        print(f"  ⚠ ONNX export failed ({e}), JSON model still available")
    
    return True


def test_model():
    """Test the trained model with live mic audio."""
    model_json = MODELS_DIR / "hey_viron_simple.json"
    if not model_json.exists():
        print("❌ No trained model found. Run training first.")
        return
    
    with open(model_json) as f:
        model = json.load(f)
    
    W = np.array(model["weights"])
    b = model["bias"]
    mean = np.array(model["mean"])
    std = np.array(model["std"])
    
    print(f"Model accuracy: {model['accuracy']:.3f}")
    print(f"Trained on: {model['n_positive']} positive, {model['n_negative']} negative clips")
    print()
    
    device = os.environ.get("VIRON_MIC_DEVICE", "plughw:2,0")
    channel = int(os.environ.get("VIRON_MIC_CHANNEL", "1"))
    
    print(f"Recording 5s from {device} ch{channel} — say 'Hey VIRON'...")
    
    cmd = ["arecord", "-D", device, "-f", "S16_LE", "-r", str(RATE),
           "-c", "2", "-t", "raw", "-d", "5"]
    result = subprocess.run(cmd, capture_output=True, timeout=10)
    if result.returncode != 0:
        print(f"❌ arecord failed: {result.stderr.decode()[:200]}")
        return
    
    stereo = np.frombuffer(result.stdout, dtype=np.int16)
    audio = stereo[channel::2].astype(np.float32) / 32768.0
    
    # Score with sliding window
    from scipy.signal import stft
    
    window_samples = int(RATE * 1.5)
    hop_samples = int(RATE * 0.2)  # 200ms steps
    
    print("Scores (time → score):")
    for start in range(0, len(audio) - window_samples, hop_samples):
        chunk = audio[start:start + window_samples]
        
        # Compute mel
        _, _, Zxx = stft(chunk, fs=RATE, nperseg=512, noverlap=512 - 160)
        power = np.abs(Zxx) ** 2
        n_mels = 40
        mel_freqs = np.linspace(0, 2595 * np.log10(1 + RATE / 2 / 700), n_mels + 2)
        mel_freqs = 700 * (10 ** (mel_freqs / 2595) - 1)
        bin_freqs = np.floor((513) * mel_freqs / RATE).astype(int)
        fb = np.zeros((n_mels, power.shape[0]))
        for i in range(n_mels):
            s, c, e = bin_freqs[i], bin_freqs[i + 1], bin_freqs[i + 2]
            if s < power.shape[0] and e < power.shape[0]:
                for j in range(s, c):
                    if c > s: fb[i, j] = (j - s) / (c - s)
                for j in range(c, e):
                    if e > c: fb[i, j] = (e - j) / (e - c)
        mel = np.dot(fb, power)
        mel = np.log(mel + 1e-10).mean(axis=1)
        
        # Normalize + score
        normed = (mel - mean) / std
        logit = normed @ W + b
        score = 1 / (1 + np.exp(-logit))
        
        t = start / RATE
        bar = "█" * int(score * 40)
        marker = " ← DETECTED!" if score > 0.7 else ""
        print(f"  {t:5.1f}s: {score:.3f} {bar}{marker}")


async def main():
    parser = argparse.ArgumentParser(description="Train custom 'Hey VIRON' wake word model")
    parser.add_argument("--skip-synth", action="store_true", help="Skip synthetic sample generation")
    parser.add_argument("--synth-only", action="store_true", help="Only generate synthetic data")
    parser.add_argument("--test", action="store_true", help="Test trained model with live mic")
    parser.add_argument("--synth-count", type=int, default=1500, help="Number of synthetic samples")
    args = parser.parse_args()
    
    if args.test:
        test_model()
        return
    
    print("=" * 60)
    print("  VIRON Custom Wake Word Training")
    print("=" * 60)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check personal samples
    personal_count = len(list(PERSONAL_DIR.glob("*.wav"))) if PERSONAL_DIR.exists() else 0
    print(f"\n  Personal voice samples: {personal_count}")
    if personal_count < 20:
        print(f"  ⚠ Recommend at least 20 personal samples for best results")
        print(f"  Run: python3 scripts/record_wake_samples.py --count 30")
        if personal_count == 0:
            print(f"  Continuing with synthetic-only (less accurate)...")
    
    # Step 1: Generate synthetic samples
    if not args.skip_synth:
        print(f"\n📌 Step 1: Generate synthetic training samples")
        try:
            synth_count = await generate_synthetic_samples(args.synth_count)
        except ImportError:
            print("  ⚠ edge-tts not installed: pip install edge-tts")
            synth_count = 0
    else:
        synth_count = len(list(POSITIVE_DIR.glob("*.wav"))) if POSITIVE_DIR.exists() else 0
        print(f"\n📌 Step 1: Skipped (have {synth_count} synthetic samples)")
    
    if args.synth_only:
        print(f"\n✅ Synthetic generation complete. Total: {synth_count}")
        return
    
    # Step 2: Generate negative samples
    print(f"\n📌 Step 2: Generate negative samples")
    try:
        neg_count = generate_negative_samples(500)
    except ImportError:
        print("  ⚠ scipy not available for pink noise, using basic negatives")
        neg_count = generate_negative_samples(300)
    
    total_positive = synth_count + personal_count
    print(f"\n  Total training data:")
    print(f"    Synthetic positive: {synth_count}")
    print(f"    Personal positive:  {personal_count}")
    print(f"    Negative:           {neg_count}")
    
    if total_positive < 50:
        print(f"\n  ❌ Need at least 50 positive samples total. Have {total_positive}.")
        return
    
    # Step 3: Train model
    print(f"\n📌 Step 3: Train model")
    
    # Try openWakeWord's built-in training first
    success = train_model_oww()
    
    if not success:
        print("  Falling back to simple classifier...")
        success = train_model_simple()
    
    if success:
        print(f"\n" + "=" * 60)
        print(f"  ✅ Training complete!")
        print(f"  Model: {MODELS_DIR}")
        print(f"")
        print(f"  Test: python3 wakeword/train_hey_viron.py --test")
        print(f"  Deploy: pkill -f 'wakeword/service.py'")
        print(f"          python3 wakeword/service.py &")
        print(f"=" * 60)
    else:
        print(f"\n  ❌ Training failed. Try:")
        print(f"  1. Install deps: pip install onnx onnxruntime scipy")
        print(f"  2. Use Colab notebook: https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb")
        print(f"  3. Copy trained model to: {MODELS_DIR / 'hey_viron.onnx'}")


if __name__ == "__main__":
    asyncio.run(main())
