#!/usr/bin/env bash
# VIRON diagnostic dump — run this when something is broken and paste the
# output. Captures everything Claude needs to debug without round-trips:
# git state, env config, mic/audio state, face DB, and the most recent
# pipeline log lines.
#
# Usage:
#   bash ~/VIRON/diag.sh                   # print to stdout
#   bash ~/VIRON/diag.sh > /tmp/diag.txt   # capture to file
#   bash ~/VIRON/diag.sh | xclip -sel clip # copy to clipboard (if xclip)

set -uo pipefail

VIRON_DIR="${VIRON_DIR:-$HOME/VIRON}"
LOG="${VIRON_LOG:-/tmp/viron_pipeline.log}"
FLASK_LOG="${VIRON_FLASK_LOG:-/tmp/viron_flask.log}"
START_LOG="${VIRON_START_LOG:-/tmp/viron_start.log}"
N_LOG="${N_LOG:-100}"

hr() { printf '\n=== %s ===\n' "$1"; }

hr "TIMESTAMP"
date -u +'%Y-%m-%d %H:%M:%S UTC'
date +'%Y-%m-%d %H:%M:%S %Z (local)'

hr "GIT"
if [ -d "$VIRON_DIR/.git" ]; then
    cd "$VIRON_DIR" && git log --oneline -5 && \
        echo "--- working tree status ---" && git status --short
else
    echo "No git repo at $VIRON_DIR"
fi

hr "VIRON ENV (.env, secrets redacted)"
if [ -f "$VIRON_DIR/.env" ]; then
    # Redact anything that looks like an API key, token, or password
    sed -E 's/(KEY|TOKEN|PASSWORD|SECRET|PAT)([^=]*)=.*/\1\2=***REDACTED***/Ig' "$VIRON_DIR/.env"
else
    echo "No .env at $VIRON_DIR/.env"
fi

hr "RUNNING PROCESSES"
pgrep -af 'gemini_live_pipeline|backend/server|viron_kiosk|chromium' 2>/dev/null || echo "(none matching)"

hr "AUDIO DEVICES"
echo "--- ALSA capture cards ---"
arecord -l 2>&1 | head -10
echo "--- ALSA playback cards ---"
aplay -l 2>&1 | head -10
echo "--- mic device holders (sudo, password may be needed) ---"
sudo -n fuser -v /dev/snd/pcmC0D0c 2>&1 | head -5 || echo "(could not run sudo fuser non-interactively)"

hr "MIXER STATE"
amixer -c 0 2>&1 | grep -E "Simple mixer|Capture |Playback" | head -20

hr "FACE DATABASE"
if [ -f "$VIRON_DIR/faces.db" ]; then
    sqlite3 "$VIRON_DIR/faces.db" \
        "SELECT id, name, role, datetime(created_at, 'unixepoch') AS created, recognized_count FROM faces ORDER BY id;" \
        2>&1 || echo "(sqlite3 not available or DB read failed)"
else
    echo "No faces.db at $VIRON_DIR/faces.db"
fi

hr "DISK / TEMP"
df -h / /tmp 2>/dev/null | head -5

hr "PIPELINE LOG (last $N_LOG lines)"
if [ -f "$LOG" ]; then
    tail -n "$N_LOG" "$LOG"
else
    echo "No log at $LOG"
fi

hr "STARTUP LOG (last 30 lines)"
if [ -f "$START_LOG" ]; then
    tail -n 30 "$START_LOG"
else
    echo "No log at $START_LOG"
fi

hr "FLASK LOG (last 20 lines)"
if [ -f "$FLASK_LOG" ]; then
    tail -n 20 "$FLASK_LOG"
else
    echo "No log at $FLASK_LOG"
fi

hr "DONE"
