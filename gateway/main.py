"""
VIRON Hybrid Gateway — FastAPI Server
=====================================
Unified /v1/chat endpoint that:
1. Runs safety check locally (never forwards unsafe content)
2. Calls local Gemma 2B router (llama.cpp) for intent classification
3. Routes to local Mistral 8B tutor OR cloud provider based on router decision
4. Falls back gracefully: cloud → local tutor → error message

Architecture:
  Student → /v1/chat → Safety Filter → Gemma Router (local)
                                            │
                               ┌────────────┴────────────┐
                               ▼                         ▼
                         local tutor              cloud provider
                        (Mistral 8B)         ┌────┼────┐
                                             ▼    ▼    ▼
                                          ChatGPT Claude Gemini
"""

import time
import json
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx

import config as cfg
from safety import check_safety, get_blocked_response, age_mode_from_age
from db import init_db, ensure_student, log_message, get_recent_messages

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("viron-gateway")

app = FastAPI(title="VIRON Hybrid Gateway", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = httpx.AsyncClient(timeout=60.0)


# ─── Request/Response Models ─────────────────────────

class ChatRequest(BaseModel):
    student_id: str = "default"
    age: int = Field(default=10, ge=3, le=18)
    message: str
    language: str = "en"
    context: Optional[dict] = None

class RouterResult(BaseModel):
    intent_type: str = "casual_chat"
    subject: str = "general"
    complexity_level: str = "simple"
    mode: str = "local"
    cloud_provider: str = "none"
    safety_flag: str = "safe"

class ChatResponse(BaseModel):
    reply: str
    mode: str
    cloud_provider: str
    router: RouterResult
    latency_ms: float


# ─── Router (Gemma 2B via llama.cpp) ─────────────────

ROUTER_SYSTEM_PROMPT = """You are an intent classification router for a children's AI tutor robot.
Analyze the student's message and return ONLY a JSON object with these exact fields:
{
  "intent_type": "command|short_question|explanation_request|homework_help|casual_chat|emotional_support|unsafe_request",
  "subject": "math|english|science|history|programming|general|unknown",
  "complexity_level": "very_simple|simple|moderate|complex",
  "mode": "local|cloud",
  "cloud_provider": "chatgpt|claude|gemini|none",
  "safety_flag": "safe|sensitive|unsafe"
}

ROUTING RULES:
- math, programming, logic, code → cloud_provider: "chatgpt"
- english, writing, literature, emotional support → cloud_provider: "claude"
- science, STEM, multimodal, geography, translation → cloud_provider: "gemini"
- greetings, very simple questions, casual chat → mode: "local", cloud_provider: "none"
- If complexity is "very_simple" or "simple" AND subject is "general" → mode: "local"
- If complexity is "moderate" or "complex" → mode: "cloud"
- Any unsafe or inappropriate content → safety_flag: "unsafe", mode: "local"

Return ONLY valid JSON. No explanations, no markdown, no extra text."""


async def call_router(message: str, age: int, language: str) -> Optional[RouterResult]:
    """Call the local Gemma router via llama.cpp server."""
    prompt = f"Student age: {age}, language: {language}\nStudent message: {message}"
    try:
        resp = await client.post(
            f"{cfg.ROUTER_URL}/v1/chat/completions",
            json={
                "model": "gemma-router",
                "messages": [
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 256,
                "stream": False,
            },
            timeout=cfg.ROUTER_TIMEOUT,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]

        # Parse JSON from response (handle markdown fences)
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0]
        result = json.loads(content)
        return RouterResult(**result)

    except (httpx.HTTPError, json.JSONDecodeError, KeyError, Exception) as e:
        logger.warning(f"Router call failed: {e}")
        return None


def default_routing(message: str) -> RouterResult:
    """Fallback routing when Gemma router is unavailable."""
    words = len(message.split())
    if words <= 6:
        return RouterResult(
            intent_type="casual_chat", subject="general",
            complexity_level="very_simple", mode="local",
            cloud_provider="none", safety_flag="safe"
        )
    # Complex → cloud chatgpt as default
    return RouterResult(
        intent_type="explanation_request", subject="general",
        complexity_level="moderate", mode="cloud",
        cloud_provider="chatgpt", safety_flag="safe"
    )


# Keywords that MUST go to cloud (Greek + English)
_CLOUD_KEYWORDS = {
    "math": {
        "chatgpt": [
            "πυθαγόρ", "pythag", "εξίσωση", "equation", "θεώρημα", "theorem",
            "αλγόριθμ", "algorithm", "γεωμετρ", "geometry", "άλγεβρ", "algebra",
            "τριγωνομ", "trigon", "παράγωγ", "derivative", "ολοκλήρωμα", "integral",
            "κλάσμ", "fraction", "ποσοστό", "percent", "solve", "λύσε", "υπολόγισε",
            "calculate", "squared", "τετράγωνο", "formula", "τύπος", "μαθηματ",
        ]
    },
    "science": {
        "gemini": [
            "φωτοσύνθ", "photosynth", "βαρύτ", "gravity", "πλανήτ", "planet",
            "ηλιακ", "solar", "κύτταρ", "cell", "dna", "ατομ", "atom", "μόρι",
            "molecule", "ενέργ", "energy", "θερμ", "therm", "ηλεκτρ", "electr",
            "μαγνητ", "magnet", "δύναμ", "force", "φυσικ", "physics", "χημε",
            "chemistry", "βιολογ", "biology", "οικοσύστημ", "ecosystem", "εξέλιξ",
            "evolution", "σύμπαν", "universe", "γαλαξ", "galaxy",
        ]
    },
    "history": {
        "claude": [
            "ιστορ", "history", "πόλεμ", "war", "επανάστ", "revolution",
            "αρχαί", "ancient", "μεσαίων", "medieval", "αυτοκρατ", "empire",
            "δημοκρατ", "democra", "φιλόσοφ", "philosoph", "αναγέννηση", "renaissance",
        ]
    },
    "english": {
        "claude": [
            "ποίημ", "poem", "essay", "story", "write", "γράψε",
            "λογοτεχ", "literature", "μετάφρα", "translat", "explain.*word",
            "τι σημαίνει", "what does.*mean", "what is the meaning",
        ]
    },
}

# Words that signal educational intent
_EXPLAIN_WORDS = [
    "εξήγησ", "explain", "πώς", "how does", "how do", "what is", "τι είναι",
    "γιατί", "why", "πες μου", "tell me about", "teach", "μάθε", "learn",
    "describe", "περίγραψ", "define", "ορισμός",
]


def override_routing(router_result: RouterResult, message: str) -> RouterResult:
    """Override Gemma router if it misclassifies known educational topics."""
    msg_lower = message.lower()

    # Check if message contains educational explain-words
    has_explain = any(w in msg_lower for w in _EXPLAIN_WORDS)

    # Check for subject keywords
    for subject, providers in _CLOUD_KEYWORDS.items():
        for provider, keywords in providers.items():
            if any(kw in msg_lower for kw in keywords):
                if has_explain or router_result.mode == "local":
                    logger.info(f"  🔄 Override: {router_result.mode}/{router_result.subject} → cloud/{subject}/{provider}")
                    router_result.mode = "cloud"
                    router_result.subject = subject
                    router_result.cloud_provider = provider
                    router_result.complexity_level = "moderate"
                    if has_explain:
                        router_result.intent_type = "explanation_request"
                    return router_result

    # If explain words + non-general subject → force cloud
    if has_explain and router_result.subject in ("math", "science", "history", "english", "programming"):
        if router_result.mode == "local":
            provider = {"math": "chatgpt", "science": "gemini", "history": "claude",
                        "english": "claude", "programming": "chatgpt"}.get(router_result.subject, "chatgpt")
            logger.info(f"  🔄 Override: explain + {router_result.subject} → cloud/{provider}")
            router_result.mode = "cloud"
            router_result.cloud_provider = provider
            router_result.complexity_level = "moderate"

    return router_result


# ─── Local Tutor (Mistral 8B via llama.cpp) ──────────

def _tutor_system_prompt(age: int, language: str) -> str:
    return f"""You are VIRON, a friendly AI tutor for a {age}-year-old.
Reply in {"Greek" if language == "el" else "English"}. Max 4 sentences. Be warm and encouraging.
Start with [emotion] tag like [happy] or [thinking]. No emojis."""


async def call_tutor(message: str, age: int, language: str, history: list) -> str:
    """Call the local Mistral tutor via llama.cpp server."""
    messages = [{"role": "system", "content": _tutor_system_prompt(age, language)}]
    # Add last few turns of history
    for h in history[-6:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    try:
        resp = await client.post(
            f"{cfg.TUTOR_URL}/v1/chat/completions",
            json={
                "model": "mistral-tutor",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 150,
                "stream": False,
            },
            timeout=cfg.TUTOR_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Tutor call failed: {e}")
        return "[confused] I'm having trouble thinking right now. Try again in a moment!"


# ─── Cloud Providers ─────────────────────────────────

async def call_chatgpt(message: str, history: list, system_prompt: str) -> str:
    """Call OpenAI ChatGPT."""
    if not cfg.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    msgs = [{"role": "system", "content": system_prompt}]
    for h in history[-10:]:
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": message})
    resp = await client.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {cfg.OPENAI_API_KEY}"},
        json={"model": cfg.OPENAI_MODEL, "messages": msgs, "max_tokens": 1500, "temperature": 0.7},
        timeout=cfg.CLOUD_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def call_claude(message: str, history: list, system_prompt: str) -> str:
    """Call Anthropic Claude."""
    if not cfg.ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set")
    msgs = []
    for h in history[-10:]:
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": message})
    resp = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": cfg.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={"model": cfg.ANTHROPIC_MODEL, "max_tokens": 1500, "system": system_prompt, "messages": msgs},
        timeout=cfg.CLOUD_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return "".join(c["text"] for c in data["content"] if c["type"] == "text")


async def call_gemini(message: str, history: list, system_prompt: str) -> str:
    """Call Google Gemini."""
    if not cfg.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set")
    contents = [
        {"role": "user", "parts": [{"text": f"System: {system_prompt}"}]},
        {"role": "model", "parts": [{"text": "Understood."}]},
    ]
    for h in history[-10:]:
        role = "user" if h["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": h["content"]}]})
    contents.append({"role": "user", "parts": [{"text": message}]})
    resp = await client.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{cfg.GEMINI_MODEL}:generateContent?key={cfg.GEMINI_API_KEY}",
        json={"contents": contents, "generationConfig": {"maxOutputTokens": 1500, "temperature": 0.7}},
        timeout=cfg.CLOUD_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


CLOUD_DISPATCH = {
    "chatgpt": call_chatgpt,
    "claude": call_claude,
    "gemini": call_gemini,
}

# Fallback order if primary cloud provider fails
CLOUD_FALLBACK = {
    "chatgpt": ["claude", "gemini"],
    "claude": ["gemini", "chatgpt"],
    "gemini": ["claude", "chatgpt"],
}


async def call_cloud(provider: str, message: str, history: list, age: int, language: str) -> tuple[str, str]:
    """
    Try the primary cloud provider, then fallbacks. Returns (reply, actual_provider).
    If all cloud fails, returns None.
    """
    age_mode = age_mode_from_age(age)
    lang_hint = "Greek" if language == "el" else "English"
    system = cfg.VIRON_SYSTEM_PROMPT + f"\nStudent age: {age} ({age_mode}). Respond in {lang_hint}."

    # Try primary provider
    providers_to_try = [provider] + CLOUD_FALLBACK.get(provider, [])
    for p in providers_to_try:
        fn = CLOUD_DISPATCH.get(p)
        if fn is None:
            continue
        try:
            reply = await fn(message, history, system)
            if reply and len(reply.strip()) > 2:
                return reply, p
        except Exception as e:
            logger.warning(f"Cloud {p} failed: {e}")
            continue

    return None, "none"


# ─── Main Chat Endpoint ──────────────────────────────

@app.on_event("startup")
async def startup():
    init_db()
    logger.info(f"VIRON Hybrid Gateway starting on port {cfg.GATEWAY_PORT}")
    logger.info(f"  Router: {cfg.ROUTER_URL}")
    logger.info(f"  Tutor:  {cfg.TUTOR_URL}")
    logger.info(f"  Cloud:  ChatGPT={'✓' if cfg.OPENAI_API_KEY else '✗'} | Claude={'✓' if cfg.ANTHROPIC_API_KEY else '✗'} | Gemini={'✓' if cfg.GEMINI_API_KEY else '✗'}")


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    start = time.time()

    # 1. Ensure student in DB
    ensure_student(req.student_id, req.age, req.language)

    # 2. Safety check (LOCAL — never lets unsafe content reach cloud)
    is_safe, reason = check_safety(req.message, req.age)
    if not is_safe:
        blocked_reply = get_blocked_response(req.age)
        router_result = RouterResult(
            intent_type="unsafe_request", safety_flag="unsafe",
            mode="local", cloud_provider="none"
        )
        log_message(req.student_id, "user", req.message, "blocked", "", router_result.dict())
        log_message(req.student_id, "assistant", blocked_reply, "blocked", "")
        return ChatResponse(
            reply=blocked_reply, mode="local", cloud_provider="none",
            router=router_result, latency_ms=_ms(start)
        )

    # 3. Call local router (Gemma 2B) for intent classification
    router_result = await call_router(req.message, req.age, req.language)
    retried = False
    if router_result is None:
        # Retry once
        logger.info("Router failed, retrying...")
        router_result = await call_router(req.message, req.age, req.language)
        retried = True
    if router_result is None:
        # Apply default routing
        logger.info("Router unavailable, using default routing")
        router_result = default_routing(req.message)

    # Override safety: if router says unsafe, block it
    if router_result.safety_flag == "unsafe":
        blocked_reply = get_blocked_response(req.age)
        log_message(req.student_id, "user", req.message, "blocked", "", router_result.dict())
        log_message(req.student_id, "assistant", blocked_reply, "blocked", "")
        return ChatResponse(
            reply=blocked_reply, mode="local", cloud_provider="none",
            router=router_result, latency_ms=_ms(start)
        )

    # Smart override: catch misclassified educational questions
    router_result = override_routing(router_result, req.message)

    # 4. Get conversation history for context (BEFORE logging current message)
    history = get_recent_messages(req.student_id, limit=12)

    # 5. Log user message
    log_message(req.student_id, "user", req.message,
                router_result.mode, router_result.cloud_provider, router_result.dict())

    # 6. Route to appropriate backend
    reply = ""
    actual_mode = router_result.mode
    actual_provider = router_result.cloud_provider

    if router_result.mode == "cloud" and router_result.cloud_provider != "none":
        # Try cloud
        cloud_reply, used_provider = await call_cloud(
            router_result.cloud_provider, req.message, history, req.age, req.language
        )
        if cloud_reply:
            reply = cloud_reply
            actual_provider = used_provider
            actual_mode = "cloud"
        else:
            # Cloud failed → fallback to local tutor
            logger.warning("All cloud providers failed, falling back to local tutor")
            reply = await call_tutor(req.message, req.age, req.language, history)
            actual_mode = "local"
            actual_provider = "none"
    else:
        # Local tutor
        reply = await call_tutor(req.message, req.age, req.language, history)
        actual_mode = "local"
        actual_provider = "none"

    # 7. Log assistant response
    latency = _ms(start)
    log_message(req.student_id, "assistant", reply, actual_mode, actual_provider, latency_ms=latency)

    logger.info(f"[{req.student_id}] {actual_mode}/{actual_provider} | {router_result.intent_type}/{router_result.subject} | {latency:.0f}ms | '{req.message[:60]}'")

    return ChatResponse(
        reply=reply,
        mode=actual_mode,
        cloud_provider=actual_provider if actual_provider else "none",
        router=router_result,
        latency_ms=latency,
    )


@app.get("/health")
async def health():
    """Health check — reports status of local models and cloud keys."""
    router_ok = False
    tutor_ok = False
    try:
        r = await client.get(f"{cfg.ROUTER_URL}/health", timeout=3)
        router_ok = r.status_code == 200
    except:
        pass
    try:
        r = await client.get(f"{cfg.TUTOR_URL}/health", timeout=3)
        tutor_ok = r.status_code == 200
    except:
        pass
    return {
        "status": "ok",
        "version": "3.0-hybrid",
        "router_model": {"url": cfg.ROUTER_URL, "connected": router_ok},
        "tutor_model": {"url": cfg.TUTOR_URL, "connected": tutor_ok},
        "cloud": {
            "chatgpt": bool(cfg.OPENAI_API_KEY),
            "claude": bool(cfg.ANTHROPIC_API_KEY),
            "gemini": bool(cfg.GEMINI_API_KEY),
        },
    }


def _ms(start: float) -> float:
    return round((time.time() - start) * 1000, 1)


# ─── Startup ─────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print(f"""
🤖 ═══════════════════════════════════════
   VIRON Hybrid Gateway v3.0
   ═══════════════════════════════════════
   📡 http://0.0.0.0:{cfg.GATEWAY_PORT}
   📚 Docs: http://0.0.0.0:{cfg.GATEWAY_PORT}/docs
   🧠 Router: {cfg.ROUTER_URL} (Gemma 2B)
   🎓 Tutor:  {cfg.TUTOR_URL} (Mistral 8B)
   ☁️  Cloud:  ChatGPT | Claude | Gemini
   ═══════════════════════════════════════
""")
    uvicorn.run(app, host="0.0.0.0", port=cfg.GATEWAY_PORT)
