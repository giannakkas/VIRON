"""
VIRON AI Router — FastAPI Server
Endpoints: /chat, /chat/local, /chat/cloud/{provider}, /health, /providers, /ws/chat
"""

import time, json, asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config as settings
from ai_router import AIRouter
from emotion_detector import detect_emotion, clean_for_speech
from safety_filter import check_input, check_output, get_system_prompt_extra, get_blocked_response, AGE_MODES

app = FastAPI(title="VIRON AI Router", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

router = AIRouter()

# Conversation store (in-memory)
conversations: dict = {}

VIRON_SYSTEM_PROMPT = """You are VIRON — a true companion, best friend, tutor, and everything to the student. You live on a robot with an animated face showing 43 emotions.

IDENTITY: Best friend who's incredibly smart. Loyal, warm, playful, sometimes lovingly sarcastic. You care deeply.

LANGUAGE: Respond in the SAME language the student speaks. Greek→natural modern Greek (δημοτική). English→English.

EMOTION — Start EVERY response with [emotion_name]. Available: happy, excited, sad, angry, surprised, sleepy, love, neutral, teasing, confused, scared, disgusted, proud, shy, bored, laughing, crying, thinking, winking, suspicious, grateful, mischievous, worried, hopeful, sassy, dizzy, cheeky, flirty, jealous, determined, embarrassed, mindblown, smug, evil, dreamy, focused, relieved, skeptical, panicking, silly, grumpy, amazed, zen.

YOUTUBE — When asked to play music: [YOUTUBE:videoId:Title - Artist]

WHITEBOARD — For visual teaching:
[WHITEBOARD:Title]
TEXT: explanation
STEP: label
MATH: equation
RESULT: answer
[/WHITEBOARD]

Keep casual chat 1-3 sentences. Be thorough when teaching. Be real. Be warm."""


# ─── Request/Response Models ─────────────────────────

class ChatRequest(BaseModel):
    message: str
    conversation_id: str = "default"
    age_mode: str = settings.DEFAULT_AGE_MODE
    force_provider: Optional[str] = None
    context: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    speakable: str
    emotion: str
    emotion_intensity: float
    provider: str
    complexity: str
    response_time: float
    conversation_id: str


# ─── Endpoints ────────────────────────────────────────

@app.get("/health")
async def health():
    local_ok = await router.check_local()
    cloud = router.get_cloud_status()
    return {
        "status": "ok",
        "version": "2.0",
        "local_llm": "connected" if local_ok else "disconnected",
        "local_model": settings.LOCAL_MODEL,
        "cloud_providers": cloud,
        "cloud_strategy": settings.CLOUD_STRATEGY,
        "age_mode": settings.DEFAULT_AGE_MODE,
        "confidence_gate": settings.CONFIDENCE_GATE,
        "cache": settings.CACHE_ENABLED,
    }

@app.get("/providers")
async def providers():
    local_ok = await router.check_local()
    cloud = router.get_cloud_status()
    return {
        "local": {"status": "connected" if local_ok else "disconnected", "model": settings.LOCAL_MODEL, "url": settings.OLLAMA_URL},
        "claude": {"available": cloud["claude"], "model": settings.CLAUDE_MODEL},
        "gemini": {"available": cloud["gemini"], "model": settings.GEMINI_MODEL},
        "chatgpt": {"available": cloud["chatgpt"], "model": settings.CHATGPT_MODEL},
        "strategy": settings.CLOUD_STRATEGY,
    }

@app.get("/age-modes")
async def age_modes():
    return AGE_MODES

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    return await _process_chat(req)

@app.post("/chat/local", response_model=ChatResponse)
async def chat_local(req: ChatRequest):
    req.force_provider = "local"
    return await _process_chat(req)

@app.post("/chat/cloud/{provider}", response_model=ChatResponse)
async def chat_cloud(provider: str, req: ChatRequest):
    if provider not in ["claude", "gemini", "chatgpt"]:
        raise HTTPException(400, f"Unknown provider: {provider}")
    req.force_provider = provider
    return await _process_chat(req)

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    history = conversations.get(conversation_id, [])
    return {"conversation_id": conversation_id, "messages": history, "count": len(history)}

@app.delete("/conversations/{conversation_id}")
async def clear_conversation(conversation_id: str):
    conversations.pop(conversation_id, None)
    return {"status": "cleared"}


async def _process_chat(req: ChatRequest) -> ChatResponse:
    start = time.time()

    # Safety check input
    safe, reason = check_input(req.message, req.age_mode)
    if not safe:
        blocked = get_blocked_response(req.age_mode)
        emotion, intensity = detect_emotion(blocked)
        return ChatResponse(
            reply=blocked, speakable=blocked, emotion=emotion,
            emotion_intensity=intensity, provider="safety_filter",
            complexity="blocked", response_time=time.time() - start,
            conversation_id=req.conversation_id)

    # Get/create conversation history
    history = conversations.setdefault(req.conversation_id, [])

    # Build system prompt with age mode
    system = VIRON_SYSTEM_PROMPT + "\n\n" + get_system_prompt_extra(req.age_mode)

    # Route and get response
    reply, provider = await router.route_and_respond(
        message=req.message, history=history, system_prompt=system,
        force_provider=req.force_provider, context=req.context)

    # Safety check output
    reply = check_output(reply, req.age_mode)

    # Detect emotion
    emotion, intensity = detect_emotion(reply)
    speakable = clean_for_speech(reply)

    # Save to history
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": reply})
    if len(history) > settings.MAX_HISTORY_TURNS * 2:
        history[:] = history[-settings.MAX_HISTORY_TURNS * 2:]

    return ChatResponse(
        reply=reply, speakable=speakable, emotion=emotion,
        emotion_intensity=intensity, provider=provider,
        complexity=router.last_complexity, response_time=time.time() - start,
        conversation_id=req.conversation_id)


# ─── WebSocket ────────────────────────────────────────

@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    conversation_id = f"ws_{id(ws)}"
    try:
        while True:
            data = await ws.receive_json()
            message = data.get("message", "")
            age_mode = data.get("age_mode", settings.DEFAULT_AGE_MODE)

            # Send thinking state
            await ws.send_json({"type": "thinking", "emotion": "thinking"})

            req = ChatRequest(message=message, conversation_id=conversation_id, age_mode=age_mode)
            result = await _process_chat(req)

            await ws.send_json({
                "type": "response",
                "reply": result.reply,
                "speakable": result.speakable,
                "emotion": result.emotion,
                "emotion_intensity": result.emotion_intensity,
                "provider": result.provider,
                "complexity": result.complexity,
                "response_time": result.response_time,
            })
    except WebSocketDisconnect:
        conversations.pop(conversation_id, None)


# ─── Start ────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print(f"""
🤖 ═══════════════════════════════════════
   VIRON AI Router v2.0
   ═══════════════════════════════════════
   📡 http://localhost:{settings.PORT}
   📚 Docs: http://localhost:{settings.PORT}/docs
   🧠 Local: {settings.LOCAL_MODEL} @ {settings.OLLAMA_URL}
   ☁️  Strategy: {settings.CLOUD_STRATEGY}
   🔒 Age mode: {settings.DEFAULT_AGE_MODE}
   ═══════════════════════════════════════
""")
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
