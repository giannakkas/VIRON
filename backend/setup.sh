#!/bin/bash
# ============================================
# VIRON AI Tutor - Complete Setup for Jetson Orin Nano
# ============================================

echo "🤖 VIRON AI Tutor Setup"
echo "========================"
echo ""

# 1. Directories
echo "[1/7] Creating directories..."
mkdir -p ~/viron
cd ~/viron

# 2. System packages
echo "[2/7] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pip chromium-browser pulseaudio alsa-utils \
    python3-opencv libopencv-dev \
    plymouth plymouth-themes

# 3. Python packages
echo "[3/7] Installing Python packages..."
pip3 install flask flask-cors flask-socketio

# 4. Backend service
echo "[4/7] Creating backend service..."
sudo tee /etc/systemd/system/viron-backend.service > /dev/null <<EOF
[Unit]
Description=VIRON AI Tutor Backend
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/viron
ExecStart=/usr/bin/python3 /home/$USER/viron/server.py
Restart=always
RestartSec=3
Environment=DISPLAY=:0

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable viron-backend
sudo systemctl start viron-backend

# 5. Kiosk autostart (hides desktop, goes straight to VIRON)
echo "[5/7] Setting up kiosk autostart..."
mkdir -p ~/.config/autostart

# Hide desktop completely
tee ~/.config/autostart/viron-kiosk.desktop > /dev/null <<EOF
[Desktop Entry]
Type=Application
Name=VIRON Kiosk
Exec=bash -c 'sleep 3 && chromium-browser --kiosk --noerrdialogs --disable-translate --no-first-run --fast --fast-start --disable-features=TranslateUI --autoplay-policy=no-user-gesture-required --use-fake-ui-for-media-stream --disable-pinch --overscroll-history-navigation=0 --disable-session-crashed-bubble --check-for-update-interval=31536000 http://localhost:5000'
X-GNOME-Autostart-enabled=true
EOF

# Disable desktop icons and panels
tee ~/.config/autostart/viron-hide-desktop.desktop > /dev/null <<EOF
[Desktop Entry]
Type=Application
Name=Hide Desktop
Exec=bash -c 'sleep 1 && xdotool key super+d 2>/dev/null; gsettings set org.gnome.desktop.background show-desktop-icons false 2>/dev/null'
X-GNOME-Autostart-enabled=true
Hidden=false
EOF

# 6. Permissions
echo "[6/7] Configuring permissions..."
sudo tee /etc/sudoers.d/viron-shutdown > /dev/null <<EOF
$USER ALL=(ALL) NOPASSWD: /sbin/shutdown, /sbin/reboot
EOF

# 7. Boot splash
echo "[7/7] Setting up boot splash..."
if [ -f ./setup-bootsplash.sh ]; then
    chmod +x ./setup-bootsplash.sh
    ./setup-bootsplash.sh
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║    ✅ VIRON AI TUTOR SETUP COMPLETE!     ║"
echo "╠══════════════════════════════════════════╣"
echo "║                                          ║"
echo "║  Files in ~/viron/:                      ║"
echo "║    viron-complete.html  (Face UI)        ║"
echo "║    server.py            (Backend)        ║"
echo "║                                          ║"
echo "║  Boot sequence:                          ║"
echo "║    1. VIRON splash logo (not Ubuntu!)    ║"
echo "║    2. Auto-login (no desktop shown)      ║"
echo "║    3. VIRON face in fullscreen            ║"
echo "║    4. Camera starts detecting student    ║"
echo "║    5. Ready to teach!                    ║"
echo "║                                          ║"
echo "║  Commands:                               ║"
echo "║    sudo systemctl status viron-backend   ║"
echo "║    journalctl -u viron-backend -f        ║"
echo "║                                          ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "🔄 Reboot to start: sudo reboot"
