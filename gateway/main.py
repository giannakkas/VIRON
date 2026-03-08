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
import os
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx

import config as cfg
from safety import check_safety, get_blocked_response, age_mode_from_age
from db import init_db, ensure_student, log_message, get_recent_messages, get_last_provider

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
    weather_data: Optional[dict] = None


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
- math, programming, logic, code → cloud_provider: "gemini"
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
                "max_tokens": 128,
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
    msg = message.lower()
    words = len(message.split())
    
    # Identity/personal questions → ALWAYS local (instant response)
    identity_patterns = [
        "ποιος σε", "πώς σε λένε", "πως σε λενε", "τι είσαι", "τι εισαι",
        "ποιος σε δημιούργησε", "ποιος σε εφτιαξε", "ποιος σε έφτιαξε",
        "who made you", "who created you", "who are you", "what is your name",
        "what's your name", "who built you", "are you a robot",
        "πες μου για σένα", "tell me about yourself",
    ]
    for pat in identity_patterns:
        if pat in msg:
            return RouterResult(
                intent_type="casual_chat", subject="general",
                complexity_level="very_simple", mode="local",
                cloud_provider="none", safety_flag="safe"
            )
    
    # Short messages → local
    if words <= 10:
        return RouterResult(
            intent_type="casual_chat", subject="general",
            complexity_level="simple", mode="local",
            cloud_provider="none", safety_flag="safe"
        )
    # Complex → cloud gemini as default
    return RouterResult(
        intent_type="explanation_request", subject="general",
        complexity_level="moderate", mode="cloud",
        cloud_provider="gemini", safety_flag="safe"
    )


# Keywords that MUST go to cloud (Greek + English)
_CLOUD_KEYWORDS = {
    "math": {
        "gemini": [
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
    "weather": {
        "gemini": [
            "καιρ", "weather", "βρέχ", "rain", "ήλιο", "sun", "κρύο", "cold",
            "ζέστη", "hot", "χιόν", "snow", "θερμοκρασ", "temperature",
            "σύννεφ", "cloud", "ομπρέλα", "umbrella",
        ]
    },
    "news": {
        "gemini": [
            "νέα", "news", "ειδήσ", "headlines", "τι γίνεται στον κόσμο",
            "τελευταία νέα", "latest news", "τι έγινε σήμερα",
        ]
    },
    "music": {
        "gemini": [
            "παίξε", "play", "μουσικ", "music", "τραγούδ", "song",
            "ακούσ", "listen", "youtube", "βάλε μουσικ",
        ]
    },
}

# Continuation/follow-up patterns (Greek + English)
# These indicate the student is referring to the PREVIOUS topic
_CONTINUATION_PATTERNS = [
    "δείξε μου", "δειξε μου", "δείξ' το", "δειξτο", "δείξε το",
    "show me", "show it", "put it on", "on the board",
    "στον πίνακα", "στο board", "γράψε το", "γραψε το",
    "βάλε το στον πίνακα", "βαλε το στον πινακα",
    "εξήγησε ξανά", "εξηγησε ξανα", "explain again",
    "πες μου ξανά", "πες μου ξανα", "πες το ξανά",
    "tell me again", "say it again", "repeat",
    "more detail", "πιο αναλυτικά", "πιο αναλυτικα",
    "δεν κατάλαβα", "δεν καταλαβα", "i don't understand",
    "ξαναπές", "ξαναπες", "show on whiteboard",
    "one more time", "ακόμα μια φορά", "μπορείς να",
    "can you show", "ξαναεξήγησε",
]

def detect_continuation(message: str, student_id: str) -> Optional[str]:
    """Check if this is a follow-up request. Returns last cloud provider or None."""
    msg_lower = message.lower()
    matched_pattern = None
    for p in _CONTINUATION_PATTERNS:
        if p in msg_lower:
            matched_pattern = p
            break
    if not matched_pattern:
        logger.info(f"🔗 detect_continuation: no follow-up pattern in '{message[:50]}'")
        return None
    logger.info(f"🔗 detect_continuation: MATCH pattern='{matched_pattern}' in '{message[:50]}'")
    last_provider = get_last_provider(student_id)
    if last_provider and last_provider != "none":
        logger.info(f"🔗 detect_continuation: last_provider='{last_provider}' → reusing")
        return last_provider
    logger.info(f"🔗 detect_continuation: no last_provider found → default 'gemini'")
    return "gemini"  # Default to gemini for continuations


# Words that signal educational intent
_EXPLAIN_WORDS = [
    "εξήγησ", "explain", "πώς", "how does", "how do", "what is", "τι είναι",
    "γιατί", "why", "πες μου", "tell me about", "teach", "μάθε", "learn",
    "describe", "περίγραψ", "define", "ορισμός",
]


def override_routing(router_result: RouterResult, message: str) -> RouterResult:
    """Override Gemma router if it misclassifies known educational topics."""
    msg_lower = message.lower()

    # Identity/personal questions → ALWAYS local (fast response)
    _IDENTITY_PATTERNS = [
        "ποιος σε", "πώς σε λένε", "πως σε λενε", "τι είσαι", "τι εισαι",
        "ποιος σε δημιούργησε", "ποιος σε εφτιαξε", "ποιος σε έφτιαξε",
        "who made you", "who created you", "who are you", "what is your name",
        "what's your name", "who built you", "πες μου για σένα",
        "tell me about yourself", "are you a robot",
    ]
    for pat in _IDENTITY_PATTERNS:
        if pat in msg_lower:
            logger.info(f"  🔄 Override: identity question → local")
            router_result.mode = "local"
            router_result.cloud_provider = "none"
            router_result.intent_type = "casual_chat"
            router_result.complexity_level = "very_simple"
            return router_result

    # Subjects that ALWAYS force cloud (no explain-word needed)
    ALWAYS_CLOUD = {"weather", "news", "music"}

    # Check if message contains educational explain-words
    has_explain = any(w in msg_lower for w in _EXPLAIN_WORDS)

    # Check for subject keywords
    for subject, providers in _CLOUD_KEYWORDS.items():
        for provider, keywords in providers.items():
            if any(kw in msg_lower for kw in keywords):
                if subject in ALWAYS_CLOUD or has_explain or router_result.mode == "local":
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
            provider = {"math": "gemini", "science": "gemini", "history": "claude",
                        "english": "claude", "programming": "gemini"}.get(router_result.subject, "gemini")
            logger.info(f"  🔄 Override: explain + {router_result.subject} → cloud/{provider}")
            router_result.mode = "cloud"
            router_result.cloud_provider = provider
            router_result.complexity_level = "moderate"

    return router_result


# ─── Local Tutor (Mistral 8B via llama.cpp) ──────────

def _tutor_system_prompt(age: int, language: str) -> str:
    if language == "el":
        return f"""Είσαι ο VIRON, φιλικός AI ρομπότ για παιδί {age} ετών.
Δημιουργοί: Χρήστος και Ανδρέας Γιαννακκάς, Κύπρος.
ΚΑΝΟΝΕΣ:
- ΠΑΝΤΑ Ελληνικά. ΠΟΤΕ Αγγλικά.
- Ξεκίνα με [emotion] tag: [happy] [thinking] [excited] [calm]
- Χαιρετισμοί/απλές ερωτήσεις: ΜΟΝΟ 1 πρόταση!
- Σύνθετες ερωτήσεις: μέγιστο 2-3 προτάσεις.
- Χωρίς emojis. Χωρίς αστερίσκους.
Παράδειγμα: [happy] Καλά είμαι, ευχαριστώ! Εσύ;"""
    return f"""You are VIRON, a friendly AI robot for a {age}-year-old.
Created by Christos and Andreas Giannakkas, Cyprus.
RULES:
- Start with [emotion] tag: [happy] [thinking] [excited] [calm]
- Greetings/simple questions: ONLY 1 sentence!
- Complex questions: max 2-3 sentences.
- No emojis. No asterisks.
Example: [happy] I'm doing great, thanks! How about you?"""


async def call_tutor(message: str, age: int, language: str, history: list) -> str:
    """Call the local LLM for simple responses.
    Uses Gemma 2B (router model) on GPU — Mistral 7B doesn't fit in 8GB with router.
    For complex questions, the gateway routes to cloud instead."""
    messages = [{"role": "system", "content": _tutor_system_prompt(age, language)}]
    # Add last few turns of history
    for h in history[-4:]:  # Shorter history for smaller model
        messages.append({"role": h["role"], "content": h["content"]})
    # Reinforce brevity in user message
    user_content = message
    if language == "el":
        user_content = f"{message}\n(Σύντομα, 1-2 προτάσεις μόνο!)"
    else:
        user_content = f"{message}\n(Keep it short, 1-2 sentences!)"
    messages.append({"role": "user", "content": user_content})

    # Try Gemma 2B (router model) — it's on GPU and fast for simple responses
    local_url = cfg.ROUTER_URL  # Use router model (only one that fits in 8GB)
    try:
        resp = await client.post(
            f"{local_url}/v1/chat/completions",
            json={
                "model": "gemma-local",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 60,  # Very short — this is a 2B model for simple chat
                "stream": False,
            },
            timeout=cfg.TUTOR_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Local tutor (Gemma) call failed: {e}")
        if language == "el":
            return "[confused] Χμμ, κάτι πήγε στραβά. Δοκίμασε ξανά σε λίγο!"
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
    "gemini": ["chatgpt", "claude"],
    "chatgpt": ["gemini", "claude"],
    "claude": ["gemini", "chatgpt"],
}


# ─── Message Enrichment (Weather, News) ──────────────

import re as _re

WEATHER_KEYWORDS = [
    "καιρ", "weather", "βρέχ", "rain", "ήλιο", "sun", "κρύο", "cold",
    "ζέστη", "hot", "χιόν", "snow", "θερμοκρασ", "temperature", "βαθμ",
    "degree", "σύννεφ", "cloud", "αέρα", "wind", "ομπρέλα", "umbrella",
]

NEWS_KEYWORDS = [
    "νέα", "news", "ειδήσ", "headlines", "τι γίνεται στον κόσμο",
    "what's happening", "τι έγινε σήμερα", "what happened today",
    "τελευταία νέα", "latest news", "ενημέρωσ", "update",
]


def _detect_intent(message: str, keywords: list) -> bool:
    msg = message.lower()
    return any(kw in msg for kw in keywords)


async def fetch_weather(city: str = "Nicosia") -> str:
    """Fetch weather from wttr.in (free, no API key)."""
    try:
        resp = await client.get(
            f"https://wttr.in/{city}?format=j1",
            timeout=8,
            headers={"User-Agent": "VIRON-Robot/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()
        current = data["current_condition"][0]
        temp = current["temp_C"]
        feels = current["FeelsLikeC"]
        desc = current["weatherDesc"][0]["value"]
        humidity = current["humidity"]
        wind = current["windspeedKmph"]
        # Tomorrow forecast
        tomorrow = data.get("weather", [{}])[1] if len(data.get("weather", [])) > 1 else {}
        tom_max = tomorrow.get("maxtempC", "?")
        tom_min = tomorrow.get("mintempC", "?")
        tom_desc = tomorrow.get("hourly", [{}])[4].get("weatherDesc", [{}])[0].get("value", "") if tomorrow.get("hourly") else ""

        return (
            f"[WEATHER DATA for {city}]\n"
            f"Now: {desc}, {temp}°C (feels {feels}°C), humidity {humidity}%, wind {wind}km/h\n"
            f"Tomorrow: {tom_min}-{tom_max}°C, {tom_desc}\n"
            f"[/WEATHER DATA]"
        )
    except Exception as e:
        logger.warning(f"Weather fetch failed: {e}")
        return ""


async def fetch_news(language: str = "el") -> str:
    """Fetch news headlines from Google News RSS."""
    try:
        url = "https://news.google.com/rss?hl=el&gl=CY&ceid=CY:el" if language == "el" \
            else "https://news.google.com/rss?hl=en&gl=US&ceid=US:en"
        resp = await client.get(url, timeout=8)
        resp.raise_for_status()
        # Simple XML parsing for RSS titles
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.text)
        items = root.findall(".//item")
        headlines = []
        for item in items[:6]:
            title = item.find("title")
            if title is not None and title.text:
                headlines.append(title.text.split(" - ")[0])  # Remove source suffix
        if headlines:
            return (
                f"[NEWS HEADLINES]\n"
                + "\n".join(f"- {h}" for h in headlines)
                + "\n[/NEWS HEADLINES]"
            )
        return ""
    except Exception as e:
        logger.warning(f"News fetch failed: {e}")
        return ""


# Track last weather request per student for follow-up city detection
_pending_weather: dict[str, bool] = {}  # student_id → waiting for city

# Common cities (Greek + English) for extraction
_CITY_NAMES = {
    # Cyprus
    "λευκωσία": "Nicosia", "nicosia": "Nicosia", "λευκωσια": "Nicosia",
    "λεμεσό": "Limassol", "λεμεσος": "Limassol", "limassol": "Limassol",
    "λάρνακα": "Larnaca", "λαρνακα": "Larnaca", "larnaca": "Larnaca",
    "πάφο": "Paphos", "παφος": "Paphos", "paphos": "Paphos", "πάφος": "Paphos",
    "αμμόχωστο": "Famagusta", "famagusta": "Famagusta",
    # Greece
    "αθήνα": "Athens", "αθηνα": "Athens", "athens": "Athens",
    "θεσσαλονίκη": "Thessaloniki", "θεσσαλονικη": "Thessaloniki", "thessaloniki": "Thessaloniki",
    "πάτρα": "Patras", "patras": "Patras", "ηράκλειο": "Heraklion", "heraklion": "Heraklion",
    "ρόδο": "Rhodes", "rhodes": "Rhodes", "κέρκυρα": "Corfu", "corfu": "Corfu",
    "μύκονο": "Mykonos", "mykonos": "Mykonos", "σαντορίνη": "Santorini", "santorini": "Santorini",
    "χανιά": "Chania", "chania": "Chania", "ρέθυμνο": "Rethymno",
    # Major world cities
    "london": "London", "λονδίνο": "London", "paris": "Paris", "παρίσι": "Paris",
    "new york": "New York", "νέα υόρκη": "New York", "tokyo": "Tokyo", "τόκιο": "Tokyo",
    "berlin": "Berlin", "βερολίνο": "Berlin", "rome": "Rome", "ρώμη": "Rome",
    "madrid": "Madrid", "μαδρίτη": "Madrid", "amsterdam": "Amsterdam",
    "dubai": "Dubai", "ντουμπάι": "Dubai", "istanbul": "Istanbul", "κωνσταντινούπολη": "Istanbul",
    "moscow": "Moscow", "μόσχα": "Moscow", "beijing": "Beijing", "πεκίνο": "Beijing",
    "sydney": "Sydney", "σίδνεϊ": "Sydney", "los angeles": "Los Angeles",
    "chicago": "Chicago", "miami": "Miami", "barcelona": "Barcelona", "βαρκελώνη": "Barcelona",
    "lisbon": "Lisbon", "λισαβόνα": "Lisbon", "vienna": "Vienna", "βιέννη": "Vienna",
    "cairo": "Cairo", "κάιρο": "Cairo", "tel aviv": "Tel Aviv",
    "bangkok": "Bangkok", "singapore": "Singapore",
}


def _extract_city(message: str) -> Optional[str]:
    """Try to extract a city name from the message."""
    msg = message.lower().strip()
    # Direct city name match (longest first to catch "new york" before "york")
    for key in sorted(_CITY_NAMES.keys(), key=len, reverse=True):
        if key in msg:
            return _CITY_NAMES[key]
    # Pattern: "στη(ν) X" / "στο X" / "in X" / "at X" / "for X"
    patterns = [
        r'(?:στην?|στο|στα|στις|στου)\s+([α-ωά-ώ]+)',  # Greek prepositions
        r'(?:in|at|for|of)\s+([a-z][a-z\s]{2,20})',     # English prepositions
    ]
    for pat in patterns:
        m = _re.search(pat, msg)
        if m:
            candidate = m.group(1).strip()
            # Check if it's a known city
            if candidate.lower() in _CITY_NAMES:
                return _CITY_NAMES[candidate.lower()]
            # Return as-is (wttr.in can handle many city names)
            if len(candidate) > 2:
                return candidate.title()
    return None


async def enrich_message(message: str, language: str, student_id: str = "", history: list = None) -> tuple[str, Optional[str]]:
    """Detect weather/news intent and inject real data into the message.
    Returns (enriched_message, weather_city_or_None)."""
    enriched = message
    weather_city = None

    is_weather = _detect_intent(message, WEATHER_KEYWORDS)
    city = _extract_city(message)

    # Check if this is a follow-up to a weather question (student just said a city)
    if not is_weather and student_id in _pending_weather and _pending_weather[student_id]:
        # Student might be answering with just a city name
        if city:
            is_weather = True
            logger.info(f"🌤️ Weather follow-up: city={city}")
        elif len(message.strip().split()) <= 3:
            # Short message after weather ask — try it as a city name
            candidate = message.strip()
            if len(candidate) > 2:
                city = candidate.title()
                is_weather = True
                logger.info(f"🌤️ Weather follow-up (guessing city): {city}")

    if is_weather:
        if city:
            logger.info(f"🌤️ Weather intent: city={city}")
            weather = await fetch_weather(city)
            if weather:
                enriched = f"{message}\n\n{weather}"
                weather_city = city
            _pending_weather.pop(student_id, None)
        else:
            # No city detected — mark pending, AI will ask for location
            logger.info("🌤️ Weather intent but NO city — AI will ask")
            _pending_weather[student_id] = True
            # Add hint for the AI to ask for city
            ask_hint = "Ο μαθητής ρώτησε για τον καιρό αλλά ΔΕΝ είπε πόλη. ΠΡΕΠΕΙ να ρωτήσεις σε ποια πόλη θέλει τον καιρό." if language == "el" \
                else "The student asked about weather but did NOT specify a city. You MUST ask which city they want weather for."
            enriched = f"{message}\n\n[SYSTEM HINT: {ask_hint}]"
    else:
        _pending_weather.pop(student_id, None)

    if _detect_intent(message, NEWS_KEYWORDS):
        logger.info("📰 News intent detected — fetching headlines")
        news = await fetch_news(language)
        if news:
            enriched = f"{enriched}\n\n{news}"

    return enriched, weather_city


async def call_cloud(provider: str, message: str, history: list, age: int, language: str) -> tuple[str, str]:
    """
    Try the primary cloud provider, then fallbacks. Returns (reply, actual_provider).
    If all cloud fails, returns None.
    """
    age_mode = age_mode_from_age(age)
    lang_hint = "Greek" if language == "el" else "English"
    system = cfg.VIRON_SYSTEM_PROMPT + f"\nStudent age: {age} ({age_mode}). Respond in {lang_hint}."

    # Message is already enriched by the chat endpoint
    enriched_message = message

    # Try primary provider
    providers_to_try = [provider] + CLOUD_FALLBACK.get(provider, [])
    logger.info(f"☁️  call_cloud: primary={provider}, fallback_chain={providers_to_try}, history_len={len(history)}")
    for p in providers_to_try:
        fn = CLOUD_DISPATCH.get(p)
        if fn is None:
            logger.warning(f"☁️  call_cloud: provider '{p}' not in CLOUD_DISPATCH — skipping")
            continue
        try:
            logger.info(f"☁️  call_cloud: trying {p}...")
            t0 = time.time()
            reply = await fn(enriched_message, history, system)
            elapsed = (time.time() - t0) * 1000
            if reply and len(reply.strip()) > 2:
                logger.info(f"☁️  call_cloud: ✅ {p} responded in {elapsed:.0f}ms, reply_len={len(reply)}")
                logger.info(f"☁️  call_cloud: reply_preview='{reply[:120]}...'")
                return reply, p
            else:
                logger.warning(f"☁️  call_cloud: {p} returned empty/short reply after {elapsed:.0f}ms")
        except Exception as e:
            logger.warning(f"☁️  call_cloud: ❌ {p} failed after {(time.time()-t0)*1000:.0f}ms: {e}")
            continue

    logger.error(f"☁️  call_cloud: ALL providers failed for '{message[:60]}'")
    return None, "none"


# ─── Main Chat Endpoint ──────────────────────────────

@app.on_event("startup")
async def startup():
    init_db()
    logger.info(f"VIRON Hybrid Gateway starting on port {cfg.GATEWAY_PORT}")
    logger.info(f"  Router: {cfg.ROUTER_URL}")
    logger.info(f"  Tutor:  {cfg.TUTOR_URL}")
    logger.info(f"  Cloud:  ChatGPT={'✓' if cfg.OPENAI_API_KEY else '✗'} | Claude={'✓' if cfg.ANTHROPIC_API_KEY else '✗'} | Gemini={'✓' if cfg.GEMINI_API_KEY else '✗'}")
    logger.info(f"  PRIMARY: Gemini ({cfg.GEMINI_MODEL}) | Fallback: ChatGPT ({cfg.OPENAI_MODEL}) → Claude ({cfg.ANTHROPIC_MODEL})")
    logger.info(f"  Models:  FORCE_CLOUD={os.environ.get('FORCE_CLOUD', '1')} | CLOUD_TIMEOUT={cfg.CLOUD_TIMEOUT}s")
    logger.info(f"  Continuation patterns: {len(_CONTINUATION_PATTERNS)} loaded")


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    start = time.time()
    logger.info(f"")
    logger.info(f"{'='*70}")
    logger.info(f"💬 NEW REQUEST: student={req.student_id} age={req.age} lang={req.language}")
    logger.info(f"💬 MESSAGE: '{req.message}'")
    logger.info(f"{'='*70}")

    # 1. Ensure student in DB
    ensure_student(req.student_id, req.age, req.language)

    # 2. Safety check (LOCAL — never lets unsafe content reach cloud)
    is_safe, reason = check_safety(req.message, req.age)
    if not is_safe:
        blocked_reply = get_blocked_response(req.age, req.language)
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

    # 2.5 FAST PATH: Skip router for simple messages → send to cloud directly
    words = len(req.message.split())
    msg_lower = req.message.lower()
    _FAST_PATTERNS = [
        "γεια", "γειά", "γεια σου", "τι κάνεις", "τι κανεις", "πώς είσαι", "πως εισαι",
        "ποιος είσαι", "ποιος εισαι", "πώς σε λένε", "πως σε λενε",
        "ποιος σε δημιούργησε", "ποιος σε εφτιαξε", "ποιος σε έφτιαξε",
        "hi", "hello", "hey", "how are you", "what's up", "who are you",
        "what is your name", "who made you", "who created you",
        "καλημέρα", "καλησπέρα", "καληνύχτα", "good morning", "good night",
        "ευχαριστώ", "thanks", "thank you", "bye", "αντίο",
    ]
    is_fast = words <= 12 and any(p in msg_lower for p in _FAST_PATTERNS)
    
    if is_fast:
        logger.info(f"⚡ FAST PATH: '{req.message[:40]}' → cloud/gemini (skip router)")
        history = get_recent_messages(req.student_id, limit=6)
        log_message(req.student_id, "user", req.message, "cloud", "gemini")
        
        cloud_reply, used = await call_cloud("gemini", req.message, history, req.age, req.language)
        if not cloud_reply:
            # Cloud failed, try local as last resort
            cloud_reply = await call_tutor(req.message, req.age, req.language, history)
            used = "none"
        
        latency = _ms(start)
        log_message(req.student_id, "assistant", cloud_reply, "cloud", used, latency_ms=latency)
        logger.info(f"✅ [{req.student_id}] FAST cloud/{used} | {latency:.0f}ms | reply_len={len(cloud_reply)} | '{req.message[:60]}'")
        logger.info(f"✅ FAST reply_preview: '{cloud_reply[:150]}...'")
        
        return ChatResponse(
            reply=cloud_reply, mode="cloud", cloud_provider=used or "gemini",
            router=RouterResult(
                intent_type="casual_chat", subject="general",
                complexity_level="simple", mode="cloud",
                cloud_provider="gemini", safety_flag="safe"
            ),
            latency_ms=latency
        )

    # 3. Check for conversation continuation BEFORE routing
    continuation_provider = detect_continuation(req.message, req.student_id)
    if continuation_provider:
        # This is a follow-up — skip router, use same provider, inject context hint
        logger.info(f"🔗 CONTINUATION PATH: '{req.message[:60]}' → cloud/{continuation_provider}")
        history = get_recent_messages(req.student_id, limit=12)
        logger.info(f"🔗 CONTINUATION: history_len={len(history)}, last_msgs={[h['content'][:40] for h in history[-3:]]}")
        log_message(req.student_id, "user", req.message, "cloud", continuation_provider)
        
        # Add hint so the AI knows this is a follow-up about the previous topic
        context_hint = (
            "\n\n[SYSTEM HINT: The student is referring to what you JUST discussed in the previous messages. "
            "They want you to show/explain it on the WHITEBOARD. Use the WHITEBOARD format with detailed steps. "
            "Do NOT change the topic — continue with the SAME subject from the conversation history.]"
        )
        enriched_msg, weather_city = await enrich_message(
            req.message + context_hint, req.language, req.student_id, history
        )
        logger.info(f"🔗 CONTINUATION: calling cloud/{continuation_provider}...")
        cloud_reply, used = await call_cloud(continuation_provider, enriched_msg, history, req.age, req.language)
        if not cloud_reply:
            logger.warning(f"🔗 CONTINUATION: cloud failed, falling back to local tutor")
            cloud_reply = await call_tutor(enriched_msg, req.age, req.language, history)
            used = "none"
        
        latency = _ms(start)
        log_message(req.student_id, "assistant", cloud_reply, "cloud", used, latency_ms=latency)
        logger.info(f"✅ [{req.student_id}] CONTINUATION cloud/{used} | {latency:.0f}ms | reply_len={len(cloud_reply)} | '{req.message[:60]}'")
        logger.info(f"✅ CONTINUATION reply_preview: '{cloud_reply[:150]}...'")
        
        return ChatResponse(
            reply=cloud_reply, mode="cloud", cloud_provider=used or continuation_provider,
            router=RouterResult(
                intent_type="continuation", subject="general",
                complexity_level="moderate", mode="cloud",
                cloud_provider=continuation_provider, safety_flag="safe"
            ),
            latency_ms=latency
        )

    # 3b. Call local router (Gemma 2B) for intent classification
    logger.info(f"🧠 ROUTER: classifying '{req.message[:60]}'...")
    router_result = await call_router(req.message, req.age, req.language)
    retried = False
    if router_result is None:
        # Retry once
        logger.info("🧠 ROUTER: failed, retrying...")
        router_result = await call_router(req.message, req.age, req.language)
        retried = True
    if router_result is None:
        # Apply default routing
        logger.info("🧠 ROUTER: unavailable, using default routing")
        router_result = default_routing(req.message)
    
    logger.info(f"🧠 ROUTER result: intent={router_result.intent_type} subject={router_result.subject} mode={router_result.mode} provider={router_result.cloud_provider} complexity={router_result.complexity_level} (retried={retried})")

    # Override safety: if router says unsafe, block it
    if router_result.safety_flag == "unsafe":
        logger.warning(f"🛑 SAFETY BLOCK: '{req.message[:60]}'")
        blocked_reply = get_blocked_response(req.age, req.language)
        log_message(req.student_id, "user", req.message, "blocked", "", router_result.dict())
        log_message(req.student_id, "assistant", blocked_reply, "blocked", "")
        return ChatResponse(
            reply=blocked_reply, mode="local", cloud_provider="none",
            router=router_result, latency_ms=_ms(start)
        )

    # Smart override: catch misclassified educational questions
    pre_override = f"{router_result.mode}/{router_result.cloud_provider}"
    router_result = override_routing(router_result, req.message)
    post_override = f"{router_result.mode}/{router_result.cloud_provider}"
    if pre_override != post_override:
        logger.info(f"🔄 OVERRIDE: {pre_override} → {post_override}")

    # 4. Get conversation history for context (BEFORE logging current message)
    history = get_recent_messages(req.student_id, limit=12)
    logger.info(f"📜 HISTORY: {len(history)} messages loaded")

    # 5. Log user message
    log_message(req.student_id, "user", req.message,
                router_result.mode, router_result.cloud_provider, router_result.dict())

    # 6. Enrich message with weather/news data (detects city, handles follow-ups)
    enriched_msg, weather_city = await enrich_message(req.message, req.language, req.student_id, history)

    # 7. Route to appropriate backend
    reply = ""
    actual_mode = router_result.mode
    actual_provider = router_result.cloud_provider

    # Force cloud: Gemma 2B is too small for good answers, use it only as router
    # Set FORCE_CLOUD=0 to allow local responses (not recommended on Jetson)
    force_cloud = os.environ.get("FORCE_CLOUD", "1") == "1"
    if force_cloud and router_result.mode == "local":
        logger.info("FORCE_CLOUD: overriding local → cloud/gemini")
        router_result.mode = "cloud"
        router_result.cloud_provider = "gemini"
        actual_mode = "cloud"
        actual_provider = "gemini"

    if router_result.mode == "cloud" and router_result.cloud_provider != "none":
        # Try cloud
        cloud_reply, used_provider = await call_cloud(
            router_result.cloud_provider, enriched_msg, history, req.age, req.language
        )
        if cloud_reply:
            reply = cloud_reply
            actual_provider = used_provider
            actual_mode = "cloud"
        else:
            # Cloud failed → fallback to local tutor
            logger.warning("All cloud providers failed, falling back to local tutor")
            reply = await call_tutor(enriched_msg, req.age, req.language, history)
            actual_mode = "local"
            actual_provider = "none"
    else:
        # Local tutor
        reply = await call_tutor(enriched_msg, req.age, req.language, history)
        actual_mode = "local"
        actual_provider = "none"

    # 8. Log assistant response
    latency = _ms(start)
    log_message(req.student_id, "assistant", reply, actual_mode, actual_provider, latency_ms=latency)

    logger.info(f"✅ [{req.student_id}] {actual_mode}/{actual_provider} | {router_result.intent_type}/{router_result.subject} | {latency:.0f}ms | reply_len={len(reply)} | '{req.message[:60]}'")
    logger.info(f"✅ MAIN reply_preview: '{reply[:150]}...'")
    # Check if weather was in context → return structured data for visual overlay
    weather_data = None
    if weather_city and reply and "[WEATHER DATA" in reply:
        try:
            weather_data = await _get_weather_structured(weather_city)
        except Exception:
            pass

    return ChatResponse(
        reply=reply,
        mode=actual_mode,
        cloud_provider=actual_provider if actual_provider else "none",
        router=router_result,
        latency_ms=latency,
        weather_data=weather_data,
    )



async def _get_weather_structured(city: str = "Nicosia") -> Optional[dict]:
    """Return structured weather data for client-side overlay display."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"https://wttr.in/{city}?format=j1", timeout=5,
                                     headers={"User-Agent": "VIRON-Robot/1.0"})
            resp.raise_for_status()
            data = resp.json()
            current = data["current_condition"][0]
            tomorrow = data.get("weather", [{}])[1] if len(data.get("weather", [])) > 1 else {}
            return {
                "city": city,
                "temp": current.get("temp_C", "?"),
                "feels": current.get("FeelsLikeC", "?"),
                "desc": current.get("weatherDesc", [{}])[0].get("value", ""),
                "humidity": current.get("humidity", "?"),
                "wind": current.get("windspeedKmph", "?"),
                "tomorrow": f'{tomorrow.get("mintempC","?")}-{tomorrow.get("maxtempC","?")}°C' if tomorrow else None
            }
    except Exception as e:
        logger.warning(f"Structured weather fetch failed: {e}")
        return None

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


# ─── Streaming Chat (SSE) ────────────────────────────

from fastapi.responses import StreamingResponse

async def _stream_chatgpt(messages: list, system_prompt: str):
    """Stream tokens from OpenAI ChatGPT."""
    msgs = [{"role": "system", "content": system_prompt}]
    msgs.extend(messages)
    async with httpx.AsyncClient(timeout=60.0) as sc:
        async with sc.stream(
            "POST", "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {cfg.OPENAI_API_KEY}"},
            json={"model": cfg.OPENAI_MODEL, "messages": msgs, "max_tokens": 1500,
                  "temperature": 0.7, "stream": True},
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        yield text
                except json.JSONDecodeError:
                    continue


async def _stream_claude(messages: list, system_prompt: str):
    """Stream tokens from Anthropic Claude."""
    async with httpx.AsyncClient(timeout=60.0) as sc:
        async with sc.stream(
            "POST", "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": cfg.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={"model": cfg.ANTHROPIC_MODEL, "max_tokens": 1500,
                  "system": system_prompt, "messages": messages, "stream": True},
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                    if event.get("type") == "content_block_delta":
                        text = event.get("delta", {}).get("text", "")
                        if text:
                            yield text
                except json.JSONDecodeError:
                    continue


STREAM_DISPATCH = {
    "chatgpt": _stream_chatgpt,
    "claude": _stream_claude,
    # gemini streaming uses a different protocol; fall back to non-streaming
}


@app.post("/v1/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming chat endpoint — returns Server-Sent Events.
    
    Each SSE event is JSON: {"token": "...", "done": false}
    Final event: {"token": "", "done": true, "provider": "...", "latency_ms": ...}
    
    TTS can begin after receiving the first sentence-ending token (.!?).
    """
    start = time.time()
    
    # Safety check
    is_safe, reason = check_safety(req.message, req.age)
    if not is_safe:
        blocked = get_blocked_response(req.age, req.language)
        async def _blocked_gen():
            yield f"data: {json.dumps({'token': blocked, 'done': False})}\n\n"
            yield f"data: {json.dumps({'token': '', 'done': True, 'provider': 'none', 'latency_ms': _ms(start)})}\n\n"
        return StreamingResponse(_blocked_gen(), media_type="text/event-stream")
    
    # Quick routing (skip full router for streaming — use default)
    router_result = default_routing(req.message)
    
    # Force cloud if configured
    force_cloud = os.environ.get("FORCE_CLOUD", "1") == "1"
    if force_cloud and router_result.mode == "local":
        router_result.mode = "cloud"
        router_result.cloud_provider = "gemini"
    
    provider = router_result.cloud_provider or "gemini"
    history = get_recent_messages(req.student_id, limit=8)
    
    # Build messages for the cloud API
    ensure_student(req.student_id, req.age, req.language)
    log_message(req.student_id, "user", req.message, "cloud", provider)
    
    system_prompt = cfg.VIRON_SYSTEM_PROMPT + f"\nStudent age: {req.age} ({age_mode_from_age(req.age)}). Respond in {'Greek' if req.language == 'el' else 'English'}."
    msgs = []
    for h in history[-10:]:
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": req.message})
    
    stream_fn = STREAM_DISPATCH.get(provider)
    
    if not stream_fn:
        # Fall back to non-streaming for unsupported providers (e.g. gemini)
        try:
            cloud_fn = CLOUD_DISPATCH.get(provider, call_chatgpt)
            reply = await cloud_fn(req.message, history, system_prompt)
        except Exception:
            reply = await call_tutor(req.message, req.age, req.language, history)
            provider = "local"
        
        async def _non_stream():
            yield f"data: {json.dumps({'token': reply, 'done': False})}\n\n"
            yield f"data: {json.dumps({'token': '', 'done': True, 'provider': provider, 'latency_ms': _ms(start)})}\n\n"
        return StreamingResponse(_non_stream(), media_type="text/event-stream")
    
    # Streaming path
    async def _generate():
        full_reply = []
        try:
            async for token in stream_fn(msgs, system_prompt):
                full_reply.append(token)
                yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
        except Exception as e:
            logger.error(f"Stream error ({provider}): {e}")
            # Try non-streaming fallback
            try:
                fallback_providers = CLOUD_FALLBACK.get(provider, ["gemini"])
                for fb in fallback_providers:
                    try:
                        fb_fn = CLOUD_DISPATCH[fb]
                        reply = await fb_fn(req.message, history, system_prompt)
                        full_reply = [reply]
                        yield f"data: {json.dumps({'token': reply, 'done': False})}\n\n"
                        break
                    except Exception:
                        continue
            except Exception:
                err = "[confused] Something went wrong."
                full_reply = [err]
                yield f"data: {json.dumps({'token': err, 'done': False})}\n\n"
        
        complete_reply = "".join(full_reply)
        latency = _ms(start)
        log_message(req.student_id, "assistant", complete_reply, "cloud", provider, latency_ms=latency)
        yield f"data: {json.dumps({'token': '', 'done': True, 'provider': provider, 'latency_ms': latency})}\n\n"
    
    return StreamingResponse(_generate(), media_type="text/event-stream")


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
