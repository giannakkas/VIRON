#!/bin/bash
# VIRON Auto-Start Setup
# Run once: sudo bash scripts/setup_autostart.sh

set -e
VIRON_DIR="/home/test/VIRON"
USER="test"

echo "═══════════════════════════════════════"
echo "  VIRON Auto-Start Setup"
echo "═══════════════════════════════════════"

# 1. Create .env if not exists
if [ ! -f "$VIRON_DIR/.env" ]; then
    echo "⚠ Create $VIRON_DIR/.env with your API keys first!"
    exit 1
fi

# 2. Create systemd service for backend
cat > /etc/systemd/system/viron-backend.service << EOF
[Unit]
Description=VIRON Backend (Flask)
After=network.target
Wants=viron-gateway.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$VIRON_DIR
EnvironmentFile=$VIRON_DIR/.env
ExecStart=/usr/bin/python3 $VIRON_DIR/backend/server.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# 3. Create systemd service for gateway
cat > /etc/systemd/system/viron-gateway.service << EOF
[Unit]
Description=VIRON Gateway (FastAPI)
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$VIRON_DIR/gateway
EnvironmentFile=$VIRON_DIR/.env
ExecStart=/usr/bin/python3 $VIRON_DIR/gateway/main.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# 4. Create systemd service for voice pipeline
cat > /etc/systemd/system/viron-pipeline.service << EOF
[Unit]
Description=VIRON Voice Pipeline (Porcupine + Deepgram + Groq)
After=viron-backend.service viron-gateway.service sound.target
Wants=viron-backend.service viron-gateway.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$VIRON_DIR
EnvironmentFile=$VIRON_DIR/.env
ExecStartPre=/bin/sleep 5
ExecStart=/usr/bin/python3 $VIRON_DIR/voice_pipeline.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 5. Create kiosk browser autostart for face display
mkdir -p /home/$USER/.config/autostart

cat > /home/$USER/.config/autostart/viron-face.desktop << EOF
[Desktop Entry]
Type=Application
Name=VIRON Face
Comment=VIRON AI Face Display
Exec=bash -c "sleep 10 && chromium-browser --kiosk --noerrdialogs --disable-infobars --disable-session-crashed-bubble --disable-translate --no-first-run --start-fullscreen --autoplay-policy=no-user-gesture-required http://localhost:5000/viron-complete.html"
X-GNOME-Autostart-enabled=true
EOF

chown $USER:$USER /home/$USER/.config/autostart/viron-face.desktop

# 6. Disable screen blanking/sleep
mkdir -p /home/$USER/.config/autostart
cat > /home/$USER/.config/autostart/viron-nosleep.desktop << EOF
[Desktop Entry]
Type=Application
Name=VIRON No Sleep
Exec=bash -c "xset s off; xset -dpms; xset s noblank"
X-GNOME-Autostart-enabled=true
EOF

chown $USER:$USER /home/$USER/.config/autostart/viron-nosleep.desktop

# 7. Enable services
systemctl daemon-reload
systemctl enable viron-backend.service
systemctl enable viron-gateway.service
systemctl enable viron-pipeline.service

echo ""
echo "✅ Auto-start configured!"
echo ""
echo "Services:"
echo "  viron-backend   - Flask backend (port 5000)"
echo "  viron-gateway   - AI Gateway (port 8080)"
echo "  viron-pipeline  - Voice pipeline (Porcupine + STT + LLM)"
echo ""
echo "Browser kiosk will auto-launch on desktop login"
echo ""
echo "Commands:"
echo "  sudo systemctl start viron-backend viron-gateway viron-pipeline"
echo "  sudo systemctl status viron-pipeline"
echo "  sudo journalctl -u viron-pipeline -f"
echo ""
echo "⚠ Make sure ~/VIRON/.env has ALL keys:"
echo "  OPENAI_API_KEY, PICOVOICE_ACCESS_KEY, GROQ_API_KEY, DEEPGRAM_API_KEY"
echo "  VIRON_MIC_DEVICE, VIRON_MIC_CHANNEL, VIRON_WAKE_KEYWORD"
