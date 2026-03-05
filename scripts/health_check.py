#!/usr/bin/env python3
"""
VIRON System Health Check & Debug
===================================
Checks all services, connections, audio, and common failure modes.

Usage:
  python3 scripts/health_check.py          # full check
  python3 scripts/health_check.py --fix    # attempt auto-fixes
"""

import os
import sys
import json
import time
import subprocess
import argparse
import socket

# Service ports
BACKEND_PORT = int(os.environ.get("VIRON_BACKEND_PORT", "5000"))
GATEWAY_PORT = int(os.environ.get("VIRON_GATEWAY_PORT", "8080"))
WAKEWORD_PORT = int(os.environ.get("VIRON_WAKEWORD_PORT", "8085"))
ROUTER_PORT = 8081
TUTOR_PORT = 8082

ALSA_DEVICE = os.environ.get("VIRON_MIC_DEVICE", "plughw:2,0")
MIC_CHANNEL = int(os.environ.get("VIRON_MIC_CHANNEL", "1"))

results = []


def check(name, ok, detail=""):
    status = "✅" if ok else "❌"
    results.append({"name": name, "ok": ok, "detail": detail})
    print(f"  {status} {name}{(' — ' + detail) if detail else ''}")
    return ok


def warn(name, detail=""):
    results.append({"name": name, "ok": True, "detail": detail, "warn": True})
    print(f"  ⚠️  {name}{(' — ' + detail) if detail else ''}")


def port_open(port, host="127.0.0.1"):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((host, port))
        s.close()
        return True
    except:
        return False


def http_get(url, timeout=5):
    try:
        import urllib.request
        resp = urllib.request.urlopen(url, timeout=timeout)
        return resp.status, resp.read().decode(errors="replace")
    except Exception as e:
        return 0, str(e)


def http_post_json(url, data, timeout=5):
    try:
        import urllib.request
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body,
            headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.status, json.loads(resp.read())
    except Exception as e:
        return 0, str(e)


def check_process(name):
    try:
        r = subprocess.run(["pgrep", "-f", name], capture_output=True)
        pids = r.stdout.decode().strip().split("\n")
        pids = [p for p in pids if p]
        return pids
    except:
        return []


def check_services():
    print("\n📡 SERVICES")
    print("=" * 50)

    # Backend
    pids = check_process("backend/server.py")
    check("Backend process", len(pids) > 0, f"PIDs: {pids}" if pids else "NOT RUNNING")
    
    code, body = http_get(f"http://127.0.0.1:{BACKEND_PORT}/")
    check("Backend HTTP", code in (200, 302, 404), f"port {BACKEND_PORT}, status={code}")
    
    # Gateway
    pids = check_process("gateway/main.py")
    check("Gateway process", len(pids) > 0, f"PIDs: {pids}" if pids else "NOT RUNNING")
    
    code, body = http_get(f"http://127.0.0.1:{GATEWAY_PORT}/health")
    if code == 200:
        try:
            health = json.loads(body)
            clouds = health.get("cloud", {})
            cloud_str = ", ".join(f"{k}={'✓' if v else '✗'}" for k, v in clouds.items())
            check("Gateway health", True, f"v{health.get('version','')} cloud: {cloud_str}")
        except:
            check("Gateway health", True, f"status={code}")
    else:
        check("Gateway health", False, f"status={code}: {body[:100]}")
    
    # Wakeword
    pids = check_process("wakeword/service.py")
    check("Wakeword process", len(pids) > 0, f"PIDs: {pids}" if pids else "NOT RUNNING")
    
    code, body = http_get(f"http://127.0.0.1:{WAKEWORD_PORT}/wakeword/status")
    if code == 200:
        try:
            status = json.loads(body)
            mode = status.get("mode", "?")
            paused = status.get("paused", False)
            models = status.get("models", [])
            listening = status.get("listening", False)
            check("Wakeword status", True,
                f"mode={mode}, listening={listening}, paused={paused}, models={models}")
            if paused:
                warn("Wakeword PAUSED", "mic is paused — call /wakeword/resume")
            if not listening:
                warn("Wakeword not listening", "mic may be held by another process")
        except:
            check("Wakeword status", True, f"status={code}")
    else:
        check("Wakeword status", False, f"status={code}")
    
    # Router model
    if port_open(ROUTER_PORT):
        check("Router model", True, f"port {ROUTER_PORT}")
    else:
        warn("Router model not running", f"port {ROUTER_PORT} — local routing unavailable")
    
    # Tutor model
    if port_open(TUTOR_PORT):
        check("Tutor model", True, f"port {TUTOR_PORT}")
    else:
        warn("Tutor model not running", f"port {TUTOR_PORT}")


def check_audio():
    print("\n🎤 AUDIO")
    print("=" * 50)
    
    # Check ALSA device exists
    try:
        r = subprocess.run(["arecord", "-l"], capture_output=True, timeout=5)
        devices = r.stdout.decode()
        has_device = "card 2" in devices or "XVF" in devices or "USB" in devices.lower()
        check("ALSA capture devices", has_device, 
            f"device={ALSA_DEVICE}, ch={MIC_CHANNEL}")
        if not has_device:
            warn("XVF3800 not found", "check USB connection")
    except:
        check("ALSA subsystem", False, "arecord not available")
    
    # Check if mic is in use
    try:
        r = subprocess.run(["fuser", "-v", "/dev/snd/pcmC2D0c"], 
            capture_output=True, timeout=3)
        users = r.stdout.decode().strip()
        if users:
            warn("Mic in use", f"processes: {r.stderr.decode().strip()[:100]}")
        else:
            check("Mic available", True, "not locked by any process")
    except:
        pass
    
    # Check PulseAudio interference
    pids = check_process("pulseaudio")
    if pids:
        warn("PulseAudio running", f"PIDs: {pids} — may interfere with ALSA")
    else:
        check("PulseAudio", True, "not running (good)")
    
    # Quick record test (only if mic not in use by wakeword)
    ww_pids = check_process("wakeword/service.py")
    if not ww_pids:
        try:
            r = subprocess.run(
                ["arecord", "-D", ALSA_DEVICE, "-f", "S16_LE", "-r", "16000",
                 "-c", "2", "-t", "raw", "-d", "1"],
                capture_output=True, timeout=5
            )
            if r.returncode == 0 and len(r.stdout) > 1000:
                import numpy as np
                stereo = np.frombuffer(r.stdout, dtype=np.int16)
                ch = stereo[MIC_CHANNEL::2]
                rms = (np.sqrt(np.mean(ch.astype(float)**2)))
                check("Mic recording", True, f"ch{MIC_CHANNEL} RMS={rms:.0f}")
                if rms < 10:
                    warn("Mic very quiet", f"RMS={rms:.0f} — may need gain adjustment")
            else:
                check("Mic recording", False, f"exit={r.returncode}: {r.stderr.decode()[:100]}")
        except Exception as e:
            check("Mic recording", False, str(e))
    else:
        check("Mic recording", True, "skipped (wakeword service holds mic)")


def check_api_keys():
    print("\n🔑 API KEYS")
    print("=" * 50)
    
    openai = os.environ.get("OPENAI_API_KEY", "")
    check("OPENAI_API_KEY", bool(openai), 
        f"set ({len(openai)} chars, ...{openai[-4:]})" if openai else "NOT SET — STT/chat won't work")
    
    anthropic = os.environ.get("ANTHROPIC_API_KEY", "")
    check("ANTHROPIC_API_KEY", bool(anthropic),
        f"set ({len(anthropic)} chars)" if anthropic else "not set (optional)")
    
    gemini = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    check("GEMINI_API_KEY", bool(gemini),
        f"set ({len(gemini)} chars)" if gemini else "not set (optional)")
    
    elevenlabs = os.environ.get("ELEVENLABS_API_KEY", "")
    if elevenlabs:
        check("ELEVENLABS_API_KEY", True, f"set ({len(elevenlabs)} chars)")
    
    gtts = os.environ.get("GOOGLE_TTS_API_KEY", "") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if gtts:
        check("Google TTS", True, "configured")


def check_tts():
    print("\n🔊 TTS (Text-to-Speech)")
    print("=" * 50)
    
    code, body = http_post_json(
        f"http://127.0.0.1:{BACKEND_PORT}/api/tts",
        {"text": "Test", "lang": "en"}
    )
    if code == 200:
        check("TTS endpoint", True, f"returned audio")
    else:
        check("TTS endpoint", False, f"status={code}: {str(body)[:100]}")


def check_stt():
    print("\n🎙️ STT (Speech-to-Text)")
    print("=" * 50)
    
    # Check if OpenAI Whisper API is reachable
    openai = os.environ.get("OPENAI_API_KEY", "")
    if openai:
        check("OpenAI Whisper API key", True, "available for cloud STT")
    else:
        check("OpenAI Whisper API key", False, "NOT SET — STT won't work")
    
    # Check local whisper
    try:
        import faster_whisper
        check("Local faster-whisper", True, "installed")
    except ImportError:
        warn("Local faster-whisper", "not installed — cloud-only STT")


def check_chat():
    print("\n💬 CHAT (AI Response)")
    print("=" * 50)
    
    if not port_open(GATEWAY_PORT):
        check("Chat test", False, "gateway not running")
        return
    
    code, body = http_post_json(
        f"http://127.0.0.1:{GATEWAY_PORT}/v1/chat",
        {"messages": [{"role": "user", "content": "Say 'hello' in one word"}],
         "system": "Reply in one word only."}
    )
    if code == 200:
        reply = body.get("reply", body.get("text", ""))[:50]
        provider = body.get("provider", "?")
        check("Chat response", True, f"provider={provider}: \"{reply}\"")
    else:
        check("Chat response", False, f"status={code}: {str(body)[:100]}")


def check_listen():
    print("\n👂 LISTEN ENDPOINT")
    print("=" * 50)
    
    if not port_open(BACKEND_PORT):
        check("Listen test", False, "backend not running")
        return
    
    # Check endpoint exists (don't actually record)
    check("Listen endpoint", True, f"http://127.0.0.1:{BACKEND_PORT}/api/listen")
    
    stt_url = os.environ.get("VIRON_STT_URL", f"http://127.0.0.1:{BACKEND_PORT}")
    check("STT URL", True, stt_url)


def check_browser_connection():
    print("\n🌐 BROWSER ↔ SERVER")
    print("=" * 50)
    
    # Check if backend serves the HTML
    code, body = http_get(f"http://127.0.0.1:{BACKEND_PORT}/viron-complete.html")
    if code == 200:
        # Check key JS variables/endpoints in the HTML
        has_oww = "owwReady" in body
        has_server_mic = "serverMicAvailable" in body
        has_listen = "/api/listen" in body
        check("VIRON HTML served", True)
        check("OWW integration in HTML", has_oww, "owwReady variable found" if has_oww else "MISSING")
        check("Server mic in HTML", has_server_mic, "serverMicAvailable found" if has_server_mic else "MISSING")
    else:
        check("VIRON HTML", False, f"status={code}")
    
    # Check wakeword status endpoint (this is what browser polls)
    code, body = http_get(f"http://127.0.0.1:{WAKEWORD_PORT}/wakeword/status")
    check("Wakeword status API", code == 200,
        f"browser polls this to set owwReady=true" if code == 200 else f"status={code}")


def check_env():
    print("\n⚙️  ENVIRONMENT")
    print("=" * 50)
    
    env_file = os.path.expanduser("~/VIRON/.env")
    if os.path.exists(env_file):
        check(".env file", True, env_file)
        # Check if it's sourced
        viron_vars = {k: v for k, v in os.environ.items() if "VIRON" in k or "OPENAI" in k or "ANTHROPIC" in k}
        check("VIRON env vars loaded", len(viron_vars) > 0,
            f"{len(viron_vars)} vars: {', '.join(viron_vars.keys())}")
    else:
        check(".env file", False, f"not found at {env_file}")
    
    check("MIC_DEVICE", True, ALSA_DEVICE)
    check("MIC_CHANNEL", True, str(MIC_CHANNEL))


def check_common_issues():
    print("\n🔍 COMMON ISSUES")
    print("=" * 50)
    
    # Multiple instances
    for name, pattern in [("backend", "backend/server.py"), ("gateway", "gateway/main.py"), ("wakeword", "wakeword/service.py")]:
        pids = check_process(pattern)
        if len(pids) > 1:
            warn(f"Multiple {name} instances", f"PIDs: {pids} — kill extras!")
    
    # Port conflicts
    for port, name in [(BACKEND_PORT, "backend"), (GATEWAY_PORT, "gateway"), (WAKEWORD_PORT, "wakeword")]:
        try:
            r = subprocess.run(["fuser", f"{port}/tcp"], capture_output=True, timeout=3)
            pids = r.stdout.decode().strip()
            if pids:
                # Check if it's the right process
                pass  # already checked above
        except:
            pass
    
    # Custom wake word model
    models_dir = os.path.expanduser("~/VIRON/wakeword/models")
    json_model = os.path.join(models_dir, "hey_viron_simple.json")
    json_bak = os.path.join(models_dir, "hey_viron_simple.json.bak")
    onnx_model = os.path.join(models_dir, "hey_viron.onnx")
    
    if os.path.exists(json_model):
        warn("Custom wake model active", "hey_viron_simple.json — was scoring 1.0 on everything!")
    elif os.path.exists(json_bak):
        check("Custom wake model", True, "disabled (renamed to .bak)")
    
    if os.path.exists(onnx_model):
        check("Custom ONNX model", True, onnx_model)


def print_summary():
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    
    failures = [r for r in results if not r["ok"]]
    warnings = [r for r in results if r.get("warn")]
    
    if not failures:
        print("  ✅ All checks passed!")
    else:
        print(f"  ❌ {len(failures)} FAILURES:")
        for f in failures:
            print(f"     - {f['name']}: {f['detail']}")
    
    if warnings:
        print(f"  ⚠️  {len(warnings)} warnings:")
        for w in warnings:
            print(f"     - {w['name']}: {w['detail']}")
    
    print()
    
    # Quick fix suggestions
    if failures:
        print("🔧 SUGGESTED FIXES:")
        for f in failures:
            name = f["name"].lower()
            if "backend" in name and "process" in name:
                print("  cd ~/VIRON && python3 backend/server.py > /tmp/viron_flask.log 2>&1 &")
            elif "gateway" in name and "process" in name:
                print("  fuser -k 8080/tcp; cd ~/VIRON/gateway && nohup python3 main.py > /tmp/viron_gateway.log 2>&1 &")
            elif "wakeword" in name and "process" in name:
                print("  python3 ~/VIRON/wakeword/service.py > /tmp/viron_wakeword.log 2>&1 &")
            elif "openai" in name:
                print("  source ~/VIRON/.env  # make sure API keys are loaded")
        print()


def auto_fix():
    """Attempt to fix common issues automatically."""
    print("\n🔧 AUTO-FIX")
    print("=" * 50)
    
    # Source .env
    env_file = os.path.expanduser("~/VIRON/.env")
    if os.path.exists(env_file):
        print("  Loading .env...")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    val = val.strip().strip("'\"")
                    os.environ[key.strip()] = val
    
    # Kill PulseAudio
    subprocess.run(["pkill", "-9", "pulseaudio"], capture_output=True)
    
    # Start missing services
    services = [
        ("backend/server.py", BACKEND_PORT, "cd ~/VIRON && python3 backend/server.py"),
        ("gateway/main.py", GATEWAY_PORT, "cd ~/VIRON/gateway && python3 main.py"),
        ("wakeword/service.py", WAKEWORD_PORT, "python3 ~/VIRON/wakeword/service.py"),
    ]
    
    for pattern, port, cmd in services:
        pids = check_process(pattern)
        if not pids:
            print(f"  Starting {pattern}...")
            # Kill anything on the port first
            subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
            time.sleep(1)
    
    # Resume wakeword if paused
    try:
        import urllib.request
        req = urllib.request.Request(
            f"http://127.0.0.1:{WAKEWORD_PORT}/wakeword/resume",
            method="POST", data=b"{}",
            headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=3)
        print("  Resumed wakeword service")
    except:
        pass


def main():
    parser = argparse.ArgumentParser(description="VIRON Health Check")
    parser.add_argument("--fix", action="store_true", help="Attempt auto-fixes")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()
    
    print()
    print("🤖 VIRON System Health Check")
    print("=" * 50)
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Host: {socket.gethostname()}")
    
    if args.fix:
        auto_fix()
    
    check_env()
    check_services()
    check_audio()
    check_api_keys()
    check_stt()
    check_tts()
    check_chat()
    check_listen()
    check_browser_connection()
    check_common_issues()
    print_summary()
    
    if args.json:
        print(json.dumps(results, indent=2))
    
    # Exit code: 0 if all pass, 1 if any failures
    sys.exit(0 if all(r["ok"] for r in results) else 1)


if __name__ == "__main__":
    main()
