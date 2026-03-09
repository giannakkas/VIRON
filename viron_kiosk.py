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
