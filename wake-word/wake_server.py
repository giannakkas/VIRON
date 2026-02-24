#!/usr/bin/env python3
"""
VIRON Wake Word Server
Uses openWakeWord with WebSocket streaming from browser.
Browser captures mic audio → sends PCM via WebSocket → server detects wake word → sends event back.
"""

import asyncio
import json
import numpy as np
import websockets
import argparse
import sys
import os

# openWakeWord
try:
    import openwakeword
    from openwakeword.model import Model
    HAS_OWW = True
except ImportError:
    HAS_OWW = False
    print("❌ openwakeword not installed. Run: pip3 install openwakeword")
    print("   Then download models: python3 -c \"import openwakeword; openwakeword.utils.download_models()\"")

# Resampling
try:
    import resampy
    HAS_RESAMPY = True
except ImportError:
    HAS_RESAMPY = False
    try:
        from scipy import signal
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

# Global model instance
oww_model = None
THRESHOLD = 0.5  # Wake word detection threshold

def init_model():
    """Initialize openWakeWord model."""
    global oww_model
    if not HAS_OWW:
        return False
    
    try:
        # Download models if not present
        openwakeword.utils.download_models()
        
        # Use "hey jarvis" as closest match to "hey viron"
        # Later we can train a custom model
        oww_model = Model(inference_framework="onnx")
        
        print("✅ openWakeWord models loaded:")
        for name in oww_model.models.keys():
            print(f"   - {name}")
        return True
    except Exception as e:
        print(f"❌ Failed to load openWakeWord: {e}")
        return False

def resample_audio(audio_data, orig_sr, target_sr=16000):
    """Resample audio to 16kHz."""
    if orig_sr == target_sr:
        return audio_data
    
    if HAS_RESAMPY:
        return resampy.resample(audio_data.astype(np.float32), orig_sr, target_sr).astype(np.int16)
    elif HAS_SCIPY:
        num_samples = int(len(audio_data) * target_sr / orig_sr)
        return signal.resample(audio_data, num_samples).astype(np.int16)
    else:
        # Simple decimation (not ideal but works)
        ratio = orig_sr / target_sr
        indices = np.arange(0, len(audio_data), ratio).astype(int)
        indices = indices[indices < len(audio_data)]
        return audio_data[indices]

async def handle_client(websocket, path=None):
    """Handle a WebSocket client connection for wake word detection."""
    global oww_model
    
    client_sr = 16000  # Will be updated by client
    print(f"🔌 Wake word client connected")
    
    try:
        async for message in websocket:
            # Handle JSON messages (config)
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                    if "sample_rate" in data:
                        client_sr = data["sample_rate"]
                        print(f"   Client sample rate: {client_sr}Hz")
                    if "threshold" in data:
                        THRESHOLD = data["threshold"]
                        print(f"   Detection threshold: {THRESHOLD}")
                    continue
                except json.JSONDecodeError:
                    continue
            
            # Handle binary audio data
            if oww_model is None:
                continue
            
            # Convert bytes to numpy array (16-bit PCM)
            audio_data = np.frombuffer(message, dtype=np.int16)
            
            # Resample if needed
            if client_sr != 16000:
                audio_data = resample_audio(audio_data, client_sr, 16000)
            
            # Feed to openWakeWord
            prediction = oww_model.predict(audio_data)
            
            # Check all models for activation
            for model_name, score in prediction.items():
                if score > THRESHOLD:
                    print(f"🎤 Wake word detected: {model_name} (score: {score:.3f})")
                    # Reset model to avoid repeated triggers
                    oww_model.reset()
                    
                    await websocket.send(json.dumps({
                        "type": "wake",
                        "model": model_name,
                        "score": float(score)
                    }))
                    break
    
    except websockets.exceptions.ConnectionClosed:
        print("🔌 Wake word client disconnected")
    except Exception as e:
        print(f"❌ Wake word error: {e}")

async def main(host="0.0.0.0", port=9000):
    """Start the wake word WebSocket server."""
    if not init_model():
        print("❌ Cannot start without openWakeWord. Install it first.")
        sys.exit(1)
    
    print(f"""
╔══════════════════════════════════════╗
║     VIRON Wake Word Server           ║
╠══════════════════════════════════════╣
║  🌐 WebSocket: ws://0.0.0.0:{port}    ║
║  🎤 Models: {len(oww_model.models)} loaded               ║
║  📊 Threshold: {THRESHOLD}                  ║
║                                      ║
║  Using 'hey jarvis' until custom     ║
║  'hey viron' model is trained        ║
╚══════════════════════════════════════╝
""")
    
    async with websockets.serve(handle_client, host, port):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VIRON Wake Word Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9000, help="WebSocket port")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    args = parser.parse_args()
    
    THRESHOLD = args.threshold
    asyncio.run(main(args.host, args.port))
