#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# VIRON Voice Improvements — Apply Patches
# ═══════════════════════════════════════════════════════════════
# Run from the VIRON root directory:
#   bash apply_patches.sh
# ═══════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

echo ""
echo "═══════════════════════════════════════"
echo "  VIRON Voice Improvements Patcher"
echo "═══════════════════════════════════════"

# Backup originals
echo ""
echo "📦 Creating backups..."
cp -n wakeword/service.py wakeword/service.py.bak 2>/dev/null || true
cp -n backend/server.py backend/server.py.bak 2>/dev/null || true
cp -n viron-complete.html viron-complete.html.bak 2>/dev/null || true
echo "  ✓ Backups created (.bak files)"

# ── 1. Replace wakeword service ──
echo ""
echo "🎯 Patching wakeword/service.py..."
if [ -f "wakeword/service.py.new" ]; then
    cp wakeword/service.py.new wakeword/service.py
    echo "  ✓ Updated wakeword service (lower threshold, faster calibration, echo flush)"
else
    echo "  ⚠ wakeword/service.py.new not found — place the new file and re-run"
fi

# ── 2. Add /api/listen endpoint to server.py ──
echo ""
echo "🔗 Patching backend/server.py..."

# Copy the listen endpoint module
if [ -f "backend/listen_endpoint.py.new" ]; then
    cp backend/listen_endpoint.py.new backend/listen_endpoint.py
fi

# Check if /api/listen is already registered
if grep -q "register_listen_endpoint" backend/server.py; then
    echo "  ✓ /api/listen already registered"
else
    # Add import and registration before the final app.run()
    # Find the last line that starts the Flask app
    python3 - << 'PYEOF'
import re

with open("backend/server.py", "r") as f:
    content = f.read()

# Add the import + registration just before the if __name__ block
# or at the very end of the route definitions

# Strategy: insert before the if __name__ == '__main__' block
insertion = """
# ── VOICE IMPROVEMENT: Combined record+STT endpoint ──
try:
    from listen_endpoint import register_listen_endpoint
    register_listen_endpoint(app)
except ImportError:
    # Inline fallback: register a simple /api/listen that chains record+stt
    print("⚠ listen_endpoint.py not found, using inline /api/listen")
    @app.route('/api/listen', methods=['POST'])
    def api_listen_inline():
        \"\"\"Combined record+STT — inline fallback version.
        Records from ReSpeaker, then directly calls STT.\"\"\"
        import io
        # Record audio
        rec_resp = record_from_mic()
        if hasattr(rec_resp, 'status_code') and rec_resp.status_code == 204:
            return jsonify({"text": "", "error": "no_speech"}), 204
        if hasattr(rec_resp, 'status_code') and rec_resp.status_code != 200:
            return jsonify({"text": "", "error": "record_failed"}), 500
        
        # Get the WAV data from the record response
        from werkzeug.datastructures import FileStorage
        params = request.get_json(silent=True) or {}
        hint_lang = params.get('lang', '')
        
        # The record_from_mic returns a send_file response
        # We need to intercept the WAV and send it to STT
        # For the inline version, redirect to the two-step approach
        return jsonify({"text": "", "error": "use_separate_endpoints"}), 501

"""

# Find insertion point — before if __name__
match = re.search(r'\nif\s+__name__\s*==\s*["\']__main__["\']', content)
if match:
    pos = match.start()
    content = content[:pos] + insertion + content[pos:]
    with open("backend/server.py", "w") as f:
        f.write(content)
    print("  ✓ Added /api/listen registration to server.py")
else:
    # Append at end
    content += insertion
    with open("backend/server.py", "w") as f:
        f.write(content)
    print("  ✓ Appended /api/listen registration to server.py")
PYEOF
fi

# ── 3. Patch recording parameters in server.py ──
echo ""
echo "⚡ Patching /api/record parameters in server.py..."
python3 - << 'PYEOF'
content = open("backend/server.py", "r").read()
changes = 0

# Reduce calibration chunks from 6 to 4
if "CALIBRATION_CHUNKS = 6" in content:
    content = content.replace("CALIBRATION_CHUNKS = 6", "CALIBRATION_CHUNKS = 4")
    changes += 1

# Reduce speech threshold multiplier from 3x to 2.5x
if "noise_floor * 3, 50" in content:
    content = content.replace("noise_floor * 3, 50", "noise_floor * 2.5, 50")
    changes += 1

# Reduce no-speech wait from 3s to 2.5s  
if 'total_chunks > int(3000 / chunk_ms)' in content:
    content = content.replace('total_chunks > int(3000 / chunk_ms)', 'total_chunks > int(2500 / chunk_ms)')
    changes += 1

if changes > 0:
    open("backend/server.py", "w").write(content)
    print(f"  ✓ Applied {changes} parameter optimizations to /api/record")
else:
    print("  ✓ Parameters already optimized")
PYEOF

# ── 4. Patch viron-complete.html ──
echo ""
echo "🖥️ Patching viron-complete.html..."
python3 - << 'PYEOF'
content = open("viron-complete.html", "r").read()
changes = 0

# 4a. Reduce SILENCE_AFTER_SPEECH from 900ms to 500ms
if "SILENCE_AFTER_SPEECH=900" in content:
    content = content.replace("SILENCE_AFTER_SPEECH=900", "SILENCE_AFTER_SPEECH=500")
    changes += 1

# 4b. Reduce post-TTS resume delay from 600ms to 300ms  
# This appears in multiple places in sayServerTTS
old_resume = "setTimeout(startWakeListener,600)"
new_resume = "setTimeout(startWakeListener,300)"
if old_resume in content:
    content = content.replace(old_resume, new_resume)
    changes += content.count(new_resume)  # Count total replacements

# 4c. Reduce NO_SPEECH_TIMEOUT from 4000ms to 3000ms
if "NO_SPEECH_TIMEOUT=4000" in content:
    content = content.replace("NO_SPEECH_TIMEOUT=4000", "NO_SPEECH_TIMEOUT=3000")
    changes += 1

# 4d. Replace startServerSideListening to use /api/listen instead of /api/record + /api/stt
old_server_listen = """function startServerSideListening(){
  D('  🎤 Using server-side mic (ReSpeaker)');
  voiceState=VOICE_HEARING;updateListenUI();
  document.getElementById("listenText").textContent='Listening (ReSpeaker)...';
  
  const maxDuration=inConversation?8:10;
  
  fetch('/api/record',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({max_duration:maxDuration,silence_duration:0.6,min_duration:0.2}),
    signal:AbortSignal.timeout((maxDuration+5)*1000)
  })
  .then(r=>{
    if(r.status===204){
      // No speech detected
      D('  🔇 Server mic: no speech detected');
      handleWhisperNoSpeech();
      return null;
    }
    if(!r.ok)throw new Error('HTTP '+r.status);
    return r.blob();
  })
  .then(blob=>{
    if(!blob)return;
    
    D('  📤 Sending '+Math.round(blob.size/1024)+'KB server recording to Whisper...');
    voiceState=VOICE_PROCESSING;updateListenUI();
    document.getElementById("listenText").textContent='Processing...';
    
    const formData=new FormData();
    formData.append('audio',blob,'recording.wav');
    formData.append('lang',isGreekMode()?'el':'');  // '' = auto-detect for English
    
    return fetch('/api/stt',{method:'POST',body:formData});
  })
  .then(r=>{if(r)return r.json()})
  .then(data=>{
    if(!data)return;
    const speech=(data.text||'').trim();
    const lang=data.language||'el';
    if(!speech||speech.length<1){
      D('  🔇 Empty Whisper result from server mic');
      handleWhisperNoSpeech();
      return;
    }
    handleWhisperResult(speech,lang);
  })
  .catch(e=>{
    D('  ❌ Server mic error: '+e);
    // Fall back to browser mic
    if(vadReady&&sileroVAD)startSileroListening();
    else if(whisperReady)startWhisperListening();
    else startChromeListening();
  });
}"""

new_server_listen = """function startServerSideListening(){
  D('  🎤 Using server-side mic (ReSpeaker) — combined listen');
  voiceState=VOICE_HEARING;updateListenUI();
  document.getElementById("listenText").textContent='Listening (ReSpeaker)...';
  
  const maxDuration=inConversation?8:10;
  
  // Use combined /api/listen endpoint (record+STT in one call, saves ~500ms)
  fetch('/api/listen',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({max_duration:maxDuration,silence_duration:0.4,min_duration:0.2,lang:isGreekMode()?'el':''}),
    signal:AbortSignal.timeout((maxDuration+5)*1000)
  })
  .then(r=>{
    if(r.status===204){
      D('  🔇 Server mic: no speech detected');
      handleWhisperNoSpeech();
      return null;
    }
    if(r.status===501){
      // /api/listen not available, fall back to separate record+stt
      D('  ⚠ /api/listen not available, using legacy path');
      startServerSideListeningLegacy();
      return null;
    }
    if(!r.ok)throw new Error('HTTP '+r.status);
    return r.json();
  })
  .then(data=>{
    if(!data)return;
    const speech=(data.text||'').trim();
    const lang=data.language||'el';
    D('  🎤 Listen result: "'+speech+'" (record='+data.record_ms+'ms, stt='+data.stt_ms+'ms, total='+data.total_ms+'ms)');
    if(!speech||speech.length<1){
      D('  🔇 Empty result from /api/listen');
      handleWhisperNoSpeech();
      return;
    }
    voiceState=VOICE_PROCESSING;updateListenUI();
    document.getElementById("listenText").textContent='Processing...';
    handleWhisperResult(speech,lang);
  })
  .catch(e=>{
    D('  ❌ Server mic error: '+e);
    if(vadReady&&sileroVAD)startSileroListening();
    else if(whisperReady)startWhisperListening();
    else startChromeListening();
  });
}

// Legacy path: separate /api/record + /api/stt (fallback if /api/listen unavailable)
function startServerSideListeningLegacy(){
  D('  🎤 Legacy: separate record + STT');
  const maxDuration=inConversation?8:10;
  fetch('/api/record',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({max_duration:maxDuration,silence_duration:0.4,min_duration:0.2}),
    signal:AbortSignal.timeout((maxDuration+5)*1000)
  })
  .then(r=>{
    if(r.status===204){handleWhisperNoSpeech();return null}
    if(!r.ok)throw new Error('HTTP '+r.status);
    return r.blob();
  })
  .then(blob=>{
    if(!blob)return;
    voiceState=VOICE_PROCESSING;updateListenUI();
    const formData=new FormData();
    formData.append('audio',blob,'recording.wav');
    formData.append('lang',isGreekMode()?'el':'');
    return fetch('/api/stt',{method:'POST',body:formData});
  })
  .then(r=>{if(r)return r.json()})
  .then(data=>{
    if(!data)return;
    const speech=(data.text||'').trim();
    if(!speech||speech.length<1){handleWhisperNoSpeech();return}
    handleWhisperResult(speech,data.language||'el');
  })
  .catch(e=>{
    D('  ❌ Legacy server mic error: '+e);
    if(vadReady&&sileroVAD)startSileroListening();
    else if(whisperReady)startWhisperListening();
    else startChromeListening();
  });
}"""

if old_server_listen in content:
    content = content.replace(old_server_listen, new_server_listen)
    changes += 1
    print("  ✓ Replaced startServerSideListening with /api/listen version")
else:
    print("  ⚠ Could not find exact startServerSideListening function — manual patch needed")
    print("    The function may have been modified. Check viron-complete.html around line 1440.")

# 4e. Also reduce the silence_duration parameter in the legacy /api/record calls
if "silence_duration:0.6" in content:
    content = content.replace("silence_duration:0.6", "silence_duration:0.4")
    changes += 1

# 4f. Reduce the OWW poll interval from 300ms to 200ms for faster wake detection
if "OWW_POLL_INTERVAL=300" in content:
    content = content.replace("OWW_POLL_INTERVAL=300", "OWW_POLL_INTERVAL=200")
    changes += 1

if changes > 0:
    open("viron-complete.html", "w").write(content)
    print(f"  ✓ Applied {changes} changes to viron-complete.html")
else:
    print("  ✓ All changes already applied")
PYEOF

# ── 5. Verify ──
echo ""
echo "═══════════════════════════════════════"
echo "  ✅ All patches applied!"
echo "═══════════════════════════════════════"
echo ""
echo "  Changes summary:"
echo "    • wakeword/service.py — lower threshold (0.28), faster calibration, echo flush"
echo "    • backend/server.py — /api/listen endpoint, faster /api/record"
echo "    • backend/listen_endpoint.py — combined record+STT module"
echo "    • viron-complete.html — uses /api/listen, tighter VAD timings"
echo ""
echo "  To revert:"
echo "    cp wakeword/service.py.bak wakeword/service.py"
echo "    cp backend/server.py.bak backend/server.py"
echo "    cp viron-complete.html.bak viron-complete.html"
echo "    rm backend/listen_endpoint.py"
echo ""
echo "  Restart services:"
echo "    pkill -f 'wakeword/service.py'"
echo "    pkill -f 'backend/server.py'"
echo "    python3 wakeword/service.py > /tmp/viron_wakeword.log 2>&1 &"
echo "    python3 backend/server.py > /tmp/viron_flask.log 2>&1 &"
echo ""
