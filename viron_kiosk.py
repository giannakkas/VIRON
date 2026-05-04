#!/usr/bin/env python3
"""
VIRON Kiosk — GPU-accelerated Chromium for Jetson Orin Nano
Launches Chromium in fullscreen kiosk mode with GPU compositing enabled.
"""
import subprocess
import os
import time
import sys

URL = "http://localhost:5000"
DISPLAY = os.environ.get("DISPLAY", ":0")

# GPU-accelerated Chromium flags for Jetson
CHROME_FLAGS = [
    "--kiosk",
    "--noerrdialogs",
    "--disable-translate",
    "--no-first-run",
    "--fast",
    "--fast-start",
    "--disable-features=TranslateUI",
    "--autoplay-policy=no-user-gesture-required",
    "--use-fake-ui-for-media-stream",
    # Critical: feed a synthetic silent audio stream instead of grabbing the real mic.
    # The voice pipeline owns the mic via arecord; the face UI must not touch it.
    "--use-fake-device-for-media-stream",
    # Belt and suspenders: deny the page mic permission entirely
    "--deny-permission-prompts",
    "--disable-pinch",
    "--overscroll-history-navigation=0",
    "--disable-session-crashed-bubble",
    "--check-for-update-interval=31536000",
    # GPU acceleration (critical for smooth face/whiteboard animations)
    "--enable-gpu-rasterization",
    "--enable-zero-copy",
    "--ignore-gpu-blocklist",
    "--enable-accelerated-video-decode",
    "--enable-native-gpu-memory-buffers",
    "--force-gpu-rasterization",
    "--enable-oop-rasterization",
    # Performance
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows",
    "--disable-component-update",
    "--disable-hang-monitor",
]

def find_chrome():
    for cmd in ["chromium-browser", "chromium", "google-chrome"]:
        try:
            subprocess.run(["which", cmd], capture_output=True, check=True)
            return cmd
        except subprocess.CalledProcessError:
            continue
    return "chromium-browser"


def wait_for_flask(url=URL, timeout_sec=60, poll_interval=0.5):
    """Block until the face server is actually serving requests, or until
    timeout_sec elapses. Without this, Chromium loads while Flask is still
    starting and shows 'site can't be reached', which only refreshes if
    the user manually reloads."""
    import urllib.request
    deadline = time.time() + timeout_sec
    last_err = None
    attempt = 0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status < 500:
                    print(f"✅ Flask ready ({attempt + 1} attempts)")
                    return True
        except Exception as e:
            last_err = e
        attempt += 1
        if attempt % 10 == 0:
            print(f"   ...still waiting for Flask ({attempt * poll_interval:.0f}s elapsed)")
        time.sleep(poll_interval)
    print(f"⚠️  Flask did not respond after {timeout_sec}s (last error: {last_err}). Launching anyway.")
    return False


def main():
    os.environ["DISPLAY"] = DISPLAY
    
    # Kill existing browser
    subprocess.run(["pkill", "-f", "chromium"], capture_output=True)
    time.sleep(1)
    
    # Clear crash flags
    profile = os.path.expanduser("~/.config/chromium/Default/Preferences")
    if os.path.exists(profile):
        try:
            import json
            with open(profile, "r") as f:
                prefs = json.load(f)
            prefs.get("profile", {}).pop("exit_type", None)
            prefs.get("profile", {}).pop("exited_cleanly", None)
            with open(profile, "w") as f:
                json.dump(prefs, f)
        except:
            pass
    
    # Block until Flask is up so Chromium doesn't get a "site can't be reached"
    print(f"⏳ Waiting for Flask at {URL}...")
    wait_for_flask()
    
    chrome = find_chrome()
    cmd = [chrome] + CHROME_FLAGS + [URL]
    print(f"🖥️  Launching: {chrome} → {URL}")
    print(f"   GPU flags: rasterization, zero-copy, native buffers")
    
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


if __name__ == "__main__":
    main()
