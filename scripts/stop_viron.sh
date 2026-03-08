#!/bin/bash
# VIRON Stop All Services

echo "🛑 Stopping VIRON..."

pkill -f viron_kiosk 2>/dev/null
pkill -f WebKitWebProcess 2>/dev/null
pkill -f ffplay 2>/dev/null
sudo systemctl stop viron-pipeline
sudo systemctl stop viron-backend

echo "✅ VIRON stopped."
