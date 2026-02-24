#!/usr/bin/env python3
"""
VIRON Debug Logger
Collects logs from browser and all services into a single file.
Run alongside other VIRON services.
"""

import json
import time
import os
from datetime import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

LOG_FILE = os.path.expanduser("~/VIRON/debug.log")

def log(source, message, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] [{level}] [{source}] {message}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# Clear log on start
with open(LOG_FILE, "w") as f:
    f.write(f"=== VIRON Debug Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")

@app.route('/debug/log', methods=['POST'])
def receive_log():
    """Receive log entries from browser."""
    data = request.get_json()
    if data:
        log(
            data.get('source', 'browser'),
            data.get('message', ''),
            data.get('level', 'INFO')
        )
    return jsonify({"ok": True})

@app.route('/debug/status', methods=['GET'])
def status():
    """Check all services status."""
    import subprocess
    
    checks = {}
    
    # Check Flask server (port 5000)
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:5000/api/battery", timeout=2)
        checks["flask_5000"] = "✅ Running"
    except:
        checks["flask_5000"] = "❌ Down"
    
    # Check AI Router (port 8000)
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:8000/health", timeout=2)
        checks["ai_router_8000"] = "✅ Running"
    except:
        checks["ai_router_8000"] = "❌ Down"
    
    # Check Wake Word Server (port 9000)
    try:
        result = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True, timeout=3)
        if ":9000" in result.stdout:
            checks["wake_word_9000"] = "✅ Running"
        else:
            checks["wake_word_9000"] = "❌ Down"
    except:
        checks["wake_word_9000"] = "❓ Unknown"
    
    # Check Ollama
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        checks["ollama"] = "✅ Running"
    except:
        checks["ollama"] = "❌ Down"
    
    return jsonify(checks)

@app.route('/debug/viewlog', methods=['GET'])
def get_log():
    """Return the debug log contents."""
    try:
        with open(LOG_FILE, "r") as f:
            return f.read(), 200, {'Content-Type': 'text/plain'}
    except:
        return "No log file yet", 200, {'Content-Type': 'text/plain'}

@app.route('/debug', methods=['GET'])
def debug_page():
    """Serve a diagnostic page."""
    return """<!DOCTYPE html>
<html><head><title>VIRON Debug</title>
<style>
body{background:#111;color:#0f0;font-family:monospace;padding:20px}
h2{color:#0ff}
.ok{color:#0f0} .err{color:#f00} .warn{color:#ff0}
#log{background:#000;padding:10px;height:400px;overflow-y:auto;border:1px solid #333;white-space:pre-wrap;font-size:12px}
button{background:#333;color:#fff;border:1px solid #555;padding:8px 16px;margin:5px;cursor:pointer}
button:hover{background:#555}
.status{margin:10px 0}
</style></head><body>
<h2>🤖 VIRON Debug Console</h2>

<div class="status" id="status">Checking services...</div>

<h3>Quick Tests</h3>
<button onclick="testMic()">🎤 Test Microphone</button>
<button onclick="testWakeWs()">🔌 Test Wake Word WS</button>
<button onclick="testTTS()">🗣️ Test Greek TTS</button>
<button onclick="testChat()">💬 Test AI Chat</button>
<button onclick="testSpeechRec()">👂 Test Speech Recognition</button>
<button onclick="clearLog()">🗑️ Clear Log</button>
<button onclick="refreshLog()">🔄 Refresh Log</button>

<h3>Live Log</h3>
<div id="log"></div>

<script>
function addLog(msg,cls='ok'){
    const d=document.getElementById('log');
    const t=new Date().toTimeString().split(' ')[0];
    d.innerHTML+=`<span class="${cls}">[${t}] ${msg}</span>\n`;
    d.scrollTop=d.scrollHeight;
    // Also send to server
    fetch('/debug/log',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({source:'debug-page',message:msg,level:cls==='err'?'ERROR':cls==='warn'?'WARN':'INFO'})}).catch(()=>{});
}

// Check services
async function checkStatus(){
    try{
        const r=await fetch('/debug/status');
        const d=await r.json();
        let html='<b>Service Status:</b><br>';
        for(const[k,v]of Object.entries(d)){
            html+=`  ${k}: ${v}<br>`;
        }
        document.getElementById('status').innerHTML=html;
    }catch(e){
        document.getElementById('status').innerHTML='<span class="err">Cannot reach debug server</span>';
    }
}
checkStatus();

// Test Microphone
async function testMic(){
    addLog('Testing microphone access...');
    try{
        if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia){
            addLog('❌ getUserMedia NOT available (need HTTPS or localhost)','err');
            return;
        }
        const stream=await navigator.mediaDevices.getUserMedia({audio:true});
        const tracks=stream.getAudioTracks();
        addLog(`✅ Mic access OK: ${tracks[0].label}`);
        addLog(`   Settings: ${JSON.stringify(tracks[0].getSettings())}`);
        
        // Check if we get actual audio data
        const ctx=new AudioContext();
        const src=ctx.createMediaStreamSource(stream);
        const analyser=ctx.createAnalyser();
        src.connect(analyser);
        const data=new Float32Array(analyser.fftSize);
        
        setTimeout(()=>{
            analyser.getFloatTimeDomainData(data);
            const max=Math.max(...data.map(Math.abs));
            if(max>0.01){
                addLog(`✅ Audio data flowing (level: ${max.toFixed(4)})`);
            }else{
                addLog(`⚠️ Mic connected but very low/no audio (level: ${max.toFixed(6)})`,'warn');
                addLog('   → Is your mic muted? Try speaking loudly.','warn');
            }
            stream.getTracks().forEach(t=>t.stop());
            ctx.close();
        },1000);
        
    }catch(e){
        addLog(`❌ Mic error: ${e.message}`,'err');
    }
}

// Test Wake Word WebSocket
function testWakeWs(){
    addLog('Testing Wake Word WebSocket on ws://localhost:9000...');
    try{
        const ws=new WebSocket('ws://localhost:9000');
        ws.onopen=()=>{
            addLog('✅ Wake Word WebSocket connected!');
            ws.send(JSON.stringify({sample_rate:16000}));
            addLog('   Sent sample rate config');
            setTimeout(()=>ws.close(),2000);
        };
        ws.onerror=(e)=>{
            addLog(`❌ Wake Word WebSocket error — server not running?`,'err');
            addLog('   Run: python3 wake-word/wake_server.py &','warn');
        };
        ws.onclose=()=>{
            addLog('   WebSocket closed');
        };
        ws.onmessage=(e)=>{
            addLog(`   Received: ${e.data}`);
        };
    }catch(e){
        addLog(`❌ WebSocket error: ${e.message}`,'err');
    }
}

// Test Greek TTS
async function testTTS(){
    addLog('Testing Greek TTS via /api/tts...');
    try{
        const r=await fetch('/api/tts',{method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({text:'Γεια σου, είμαι ο Βίρον',lang:'el'})});
        if(!r.ok){
            addLog(`❌ TTS error: HTTP ${r.status}`,'err');
            const t=await r.text();
            addLog(`   Response: ${t}`,'err');
            return;
        }
        const blob=await r.blob();
        addLog(`✅ TTS OK: got ${blob.size} bytes of audio`);
        const url=URL.createObjectURL(blob);
        const a=new Audio(url);
        a.play();
        addLog('   Playing audio...');
    }catch(e){
        addLog(`❌ TTS error: ${e.message}`,'err');
    }
}

// Test AI Chat
async function testChat(){
    addLog('Testing AI Chat via /api/chat...');
    try{
        const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({max_tokens:100,system:'Reply briefly.',
                messages:[{role:'user',content:'Say hello in one sentence'}]})});
        if(!r.ok){
            addLog(`❌ Chat error: HTTP ${r.status}`,'err');
            return;
        }
        const d=await r.json();
        const text=d.content?.map(c=>c.text).join('')||JSON.stringify(d);
        addLog(`✅ AI Response: ${text.substring(0,200)}`);
    }catch(e){
        addLog(`❌ Chat error: ${e.message}`,'err');
    }
}

// Test Speech Recognition
function testSpeechRec(){
    addLog('Testing Chrome Speech Recognition...');
    const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
    if(!SR){
        addLog('❌ SpeechRecognition not available','err');
        return;
    }
    addLog('   SpeechRecognition API available ✓');
    
    const rec=new SR();
    rec.continuous=false;
    rec.interimResults=true;
    rec.lang='en-US';
    rec.maxAlternatives=5;
    
    rec.onstart=()=>addLog('   🎤 Listening... speak now! (5 sec timeout)');
    rec.onresult=(e)=>{
        let results=[];
        for(let i=0;i<e.results.length;i++){
            for(let j=0;j<e.results[i].length;j++){
                results.push({
                    text:e.results[i][j].transcript,
                    confidence:e.results[i][j].confidence?.toFixed(3),
                    final:e.results[i].isFinal
                });
            }
        }
        addLog(`   ✅ Heard: ${JSON.stringify(results)}`);
    };
    rec.onerror=(e)=>addLog(`   ❌ Error: ${e.error}`,'err');
    rec.onend=()=>addLog('   Speech recognition ended');
    
    try{
        rec.start();
        setTimeout(()=>{try{rec.stop()}catch{}},5000);
    }catch(e){
        addLog(`   ❌ Start error: ${e.message}`,'err');
    }
}

function clearLog(){document.getElementById('log').innerHTML='';addLog('Log cleared');}

async function refreshLog(){
    try{
        const r=await fetch('/debug/viewlog');
        const t=await r.text();
        addLog('=== Server Log ===\\n'+t);
    }catch(e){
        addLog('Cannot fetch server log','err');
    }
}

addLog('Debug console ready. Run tests above to diagnose issues.');
</script>
</body></html>"""

if __name__ == "__main__":
    log("debug", "Debug server starting on port 5001")
    app.run(host="0.0.0.0", port=5001, debug=False)
