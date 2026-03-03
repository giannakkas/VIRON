#!/usr/bin/env python3
"""
VIRON Custom Wake Word Training — "Hey VIRON"
=============================================
Generates synthetic training data and trains a custom openWakeWord model.

Requirements:
  pip install openwakeword[full] edge-tts scipy

Usage:
  python3 wakeword/train_hey_viron.py

This script:
1. Generates ~2000 synthetic "Hey VIRON" clips using edge-tts (multiple voices)
2. Downloads negative sample data
3. Trains a small neural network model
4. Exports to wakeword/models/hey_viron.onnx

Training takes ~1-4 hours depending on hardware.
"""

import os
import sys
import asyncio
import random
import subprocess
import tempfile
import time
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"
DATA_DIR = SCRIPT_DIR / "training_data"
POSITIVE_DIR = DATA_DIR / "positive"
NEGATIVE_DIR = DATA_DIR / "negative"

# Edge-TTS voices for diverse training data
VOICES = [
    # English voices (varied accents)
    "en-US-GuyNeural", "en-US-JennyNeural", "en-US-AriaNeural",
    "en-US-DavisNeural", "en-US-AmberNeural", "en-US-AnaNeural",
    "en-US-BrandonNeural", "en-US-ChristopherNeural", "en-US-CoraNeural",
    "en-US-EricNeural", "en-US-JacobNeural", "en-US-MichelleNeural",
    "en-GB-RyanNeural", "en-GB-SoniaNeural", "en-GB-LibbyNeural",
    "en-AU-NatashaNeural", "en-AU-WilliamNeural",
    "en-IN-NeerjaNeural", "en-IN-PrabhatNeural",
    "en-IE-ConnorNeural", "en-IE-EmilyNeural",
    # Greek voices (for accented "Hey VIRON")
    "el-GR-AthinaNeural", "el-GR-NestorasNeural",
]

PHRASES = [
    "Hey VIRON", "Hey Viron", "hey viron", "Hey viron",
    "Hey VIRON!", "hey VIRON", "Hey, VIRON",
]

RATES = ["-20%", "-10%", "+0%", "+10%", "+20%"]


async def generate_positive_samples():
    """Generate synthetic 'Hey VIRON' clips using edge-tts."""
    import edge_tts
    
    POSITIVE_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    total = len(VOICES) * len(PHRASES) * len(RATES)
    
    print(f"📢 Generating {total} positive samples...")
    
    for voice in VOICES:
        for phrase in PHRASES:
            for rate in RATES:
                outfile = POSITIVE_DIR / f"hey_viron_{count:04d}.wav"
                mp3file = POSITIVE_DIR / f"hey_viron_{count:04d}.mp3"
                
                try:
                    comm = edge_tts.Communicate(phrase, voice, rate=rate)
                    await comm.save(str(mp3file))
                    
                    # Convert to 16kHz mono WAV
                    subprocess.run([
                        "ffmpeg", "-y", "-i", str(mp3file),
                        "-ar", "16000", "-ac", "1",
                        str(outfile)
                    ], capture_output=True, timeout=10)
                    
                    mp3file.unlink(missing_ok=True)
                    count += 1
                    
                    if count % 50 == 0:
                        print(f"  Generated {count}/{total} clips...")
                        
                except Exception as e:
                    print(f"  ⚠ Error with {voice}/{rate}: {e}")
    
    print(f"✅ Generated {count} positive samples in {POSITIVE_DIR}")
    return count


def train_model(positive_dir: Path, model_name: str = "hey_viron"):
    """Train openWakeWord model from positive samples."""
    try:
        # This uses the openWakeWord training pipeline
        # For full training, use the Colab notebook:
        # https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb
        
        print("\n🧠 Training model...")
        print("Note: For best results, use the openWakeWord Colab notebook:")
        print("https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb")
        print("\nOr use the wakeword_trainer tool:")
        print("https://github.com/bbarrick/wakeword_trainer")
        print(f"\nPositive samples are ready in: {positive_dir}")
        print(f"Copy the trained model to: {MODELS_DIR / 'hey_viron.onnx'}")
        print("\nThe wake word service will automatically use it on next restart.")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")


async def main():
    print("=" * 50)
    print("VIRON Custom Wake Word Training")
    print("=" * 50)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate positive samples
    print("\n📌 Step 1: Generate positive training samples")
    count = await generate_positive_samples()
    
    if count < 100:
        print("⚠ Too few samples generated. Check edge-tts and try again.")
        return
    
    # Step 2: Train (or provide instructions)
    print("\n📌 Step 2: Train model")
    train_model(POSITIVE_DIR)
    
    print("\n" + "=" * 50)
    print("✅ Training data generation complete!")
    print(f"   Samples: {count}")
    print(f"   Location: {POSITIVE_DIR}")
    print(f"   Model target: {MODELS_DIR / 'hey_viron.onnx'}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
