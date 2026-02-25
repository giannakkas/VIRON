"""
VIRON AI BRAIN — Multi-LLM Orchestrator
=========================================
Routes student questions to the BEST AI based on subject:
  • Claude  → Reasoning, explanations, essays, literature, Greek history
  • ChatGPT → Math, code, science, step-by-step solutions
  • Gemini  → Greek language, translation, multilingual, geography
  • Ollama  → Offline fallback, simple greetings, zero cost

4 Routing Strategies (switchable by voice command):
  ⚡ Turbo (best_one)  — #1 ranked AI only           (~$3-5/mo)
  🏎️ Race              — Top 2, use fastest            (~$5-8/mo)
  ✅ Check (verify)    — Answer + verify with 2nd AI   (~$6-10/mo)
  🧠 Smart (consensus) — All 3, Claude merges best     (~$15-25/mo)

Architecture:
  Student speaks → Classify subject → Route to best AI(s)
  → Confidence gate → Merge/verify → Speak response
"""

import asyncio
import os
import re
import time
import json
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor

# Try httpx for async HTTP
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# Try requests as fallback
try:
    import requests as sync_requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger("viron_brain")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RouterConfig:
    """API keys and model settings. Load from environment or pass directly."""
    # API Keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    ollama_url: str = "http://localhost:11434"

    # Models
    claude_model: str = "claude-opus-4-20250514"
    chatgpt_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-2.0-flash"
    ollama_model: str = "phi3"

    # Timeouts
    cloud_timeout: int = 60
    local_timeout: int = 30
    max_retries: int = 3

    # Strategy
    strategy: str = "best_one"  # best_one, race, verify, consensus

    # Confidence gate
    confidence_gate: bool = True

    @classmethod
    def from_env(cls) -> "RouterConfig":
        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            google_api_key=os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", "")),
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            claude_model=os.getenv("CLAUDE_MODEL", "claude-opus-4-20250514"),
            chatgpt_model=os.getenv("CHATGPT_MODEL", "gpt-4o-mini"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            ollama_model=os.getenv("OLLAMA_MODEL", "phi3"),
            cloud_timeout=int(os.getenv("CLOUD_TIMEOUT", "60")),
            local_timeout=int(os.getenv("LOCAL_TIMEOUT", "30")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            strategy=os.getenv("ROUTING_STRATEGY", "best_one"),
            confidence_gate=os.getenv("CONFIDENCE_GATE", "true").lower() == "true",
        )


# ═══════════════════════════════════════════════════════════════════
# SUBJECT CLASSIFICATION & LANGUAGE DETECTION
# ═══════════════════════════════════════════════════════════════════

class Subject(Enum):
    MATH = "math"
    SCIENCE = "science"
    CODING = "coding"
    GREEK_LANG = "greek_language"
    ENGLISH = "english"
    HISTORY = "history"
    LITERATURE = "literature"
    TRANSLATION = "translation"
    GEOGRAPHY = "geography"
    CREATIVE = "creative_writing"
    GENERAL = "general"
    GREETING = "greeting"


# Subject → [1st choice, 2nd choice, 3rd choice] provider
ROUTING_TABLE: Dict[Subject, List[str]] = {
    Subject.MATH:       ["chatgpt", "claude", "gemini"],
    Subject.SCIENCE:    ["chatgpt", "claude", "gemini"],
    Subject.CODING:     ["chatgpt", "claude", "gemini"],
    Subject.GREEK_LANG: ["gemini", "claude", "chatgpt"],
    Subject.ENGLISH:    ["claude", "chatgpt", "gemini"],
    Subject.HISTORY:    ["claude", "gemini", "chatgpt"],
    Subject.LITERATURE: ["claude", "gemini", "chatgpt"],
    Subject.TRANSLATION:["gemini", "claude", "chatgpt"],
    Subject.GEOGRAPHY:  ["gemini", "chatgpt", "claude"],
    Subject.CREATIVE:   ["claude", "chatgpt", "gemini"],
    Subject.GENERAL:    ["claude", "gemini", "chatgpt"],
    Subject.GREETING:   ["ollama"],  # Greetings always local
}


def detect_language(text: str) -> str:
    """Detect if text is Greek or English."""
    greek_chars = len(re.findall(r'[\u0370-\u03FF\u1F00-\u1FFF]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    return "el" if greek_chars > latin_chars else "en"


def classify_subject(text: str) -> Subject:
    """Classify student's question into a subject for optimal routing."""
    lower = text.lower().strip()

    # Greetings — always local/Ollama
    greeting_patterns = [
        r'^(hi|hello|hey|γεια|γεια σου|τι κάνεις|how are you|what\'s up|good morning|καλημέρα|καλησπέρα|ευχαριστώ|thanks|thank you|bye|ok|okay|yes|no|ναι|όχι|cool|nice|wow|haha)\b',
        r'^.{0,15}$',  # Very short messages are usually greetings/acknowledgements
    ]
    for p in greeting_patterns:
        if re.match(p, lower):
            return Subject.GREETING

    # Math
    math_patterns = [
        r'\b(math|μαθηματικ|algebra|γεωμετρ|equation|εξίσωσ|solve|λύσε|calculate|υπολόγισ|derivative|παράγωγ|integral|ολοκλήρωμα|fraction|κλάσμ|percent|ποσοστό|triangle|τρίγωνο|pythagoras|πυθαγόρ|formula|τύπος|multiply|πολλαπλ|divide|διαίρεσ|square root|ρίζα|exponent|δύναμη|logarithm|λογάριθμ|probability|πιθανότητ|statistics|στατιστικ|factorial|matrix|πίνακ|function|συνάρτησ)',
        r'^\s*\d+\s*[\+\-\*\/\^]\s*\d+',  # "5 + 3", "12 * 7"
        r'\b\d+\s*[\%x×÷]\s*\d+',
    ]
    for p in math_patterns:
        if re.search(p, lower):
            return Subject.MATH

    # Science
    science_patterns = [
        r'\b(physics|φυσικ|chemistry|χημεί|biology|βιολογ|photosynthesis|φωτοσύνθεσ|gravity|βαρύτητ|atom|άτομ|molecule|μόρι|energy|ενέργει|force|δύναμη|velocity|ταχύτητ|acceleration|επιτάχυνσ|cell|κύτταρ|dna|rna|evolution|εξέλιξ|ecosystem|οικοσύστημα|planet|πλανήτ|chemical|element|στοιχείο|electron|ηλεκτρόνι|proton|neutron|newton|magnetic|μαγνητ|electric|ηλεκτρ|temperature|θερμοκρασ|experiment|πείραμα|hypothesis|υπόθεσ|organism|οργανισμ|volcano|ηφαίστει|earthquake|σεισμ|photon|φωτόνι|quantum|κβαντ|relativity|σχετικότητ)\b',
    ]
    for p in science_patterns:
        if re.search(p, lower):
            return Subject.SCIENCE

    # Coding
    coding_patterns = [
        r'\b(code|coding|program|προγραμμ|python|javascript|html|css|java|algorithm|αλγόριθμ|function|loop|variable|μεταβλητ|debug|error|bug|compile|syntax|array|list|class|object|database|api|server|git|terminal|console|command line)\b',
    ]
    for p in coding_patterns:
        if re.search(p, lower):
            return Subject.CODING

    # Translation
    translation_patterns = [
        r'\b(translate|μετάφρασ|how do you say|πώς (λέ(με|γεται|νε)|λέ(ω|ς)|είναι)\b.*\b(στα|in)\b|what does .+ mean|τι σημαίνει|in english|in greek|στα αγγλικά|στα ελληνικά)\b',
    ]
    for p in translation_patterns:
        if re.search(p, lower):
            return Subject.TRANSLATION

    # Greek Language (grammar, conjugation, syntax)
    greek_lang_patterns = [
        r'\b(κλίν[εέη]ται|κλίση|ρήμα|ουσιαστικό|επίθετο|σύνταξ|γραμματικ|ορθογραφ|πτώσ[εη]|ενεστώτα|αόριστο|παρατατικ|μετοχ|αντωνυμ|πρόθεσ|σύνδεσμ|ενεργητικ|παθητικ|υποτακτικ|μελλοντ|greek grammar|conjugat|declension|accent)\b',
    ]
    for p in greek_lang_patterns:
        if re.search(p, lower):
            return Subject.GREEK_LANG

    # English Language
    english_patterns = [
        r'\b(english grammar|past tense|present tense|future tense|irregular verb|preposition|adjective|adverb|noun|pronoun|article|conjunction|vocabulary|spelling|pronunciation)\b',
    ]
    for p in english_patterns:
        if re.search(p, lower):
            return Subject.ENGLISH

    # History
    history_patterns = [
        r'\b(history|ιστορ|war|πόλεμ|revolution|επανάστασ|ancient|αρχαί|byzantine|βυζαντ|ottoman|οθωμαν|world war|παγκόσμι|civilization|πολιτισμ|emperor|αυτοκράτορ|king|βασιλι|queen|βασίλισσα|dynasty|δυναστεί|century|αιών|independence|ανεξαρτησί|battle|μάχ|treaty|συνθήκ|colony|αποικ|1821|1940|1453|democracy|δημοκρατ)\b',
    ]
    for p in history_patterns:
        if re.search(p, lower):
            return Subject.HISTORY

    # Literature
    literature_patterns = [
        r'\b(literature|λογοτεχν|poem|ποίημα|novel|μυθιστόρημα|author|συγγραφ|odyssey|οδύσσεια|iliad|ιλιάδα|homer|όμηρος|shakespeare|kafka|poetry|ποίηση|story|ιστορία|book|βιβλίο|character|χαρακτήρα|plot|πλοκή|theme|θέμα|metaphor|μεταφορ|symbolism|συμβολισμ|essay|δοκίμι|rhetoric|ρητορικ|myth|μύθο)\b',
    ]
    for p in literature_patterns:
        if re.search(p, lower):
            return Subject.LITERATURE

    # Geography
    geography_patterns = [
        r'\b(geography|γεωγραφ|country|χώρα|capital|πρωτεύουσα|continent|ήπειρο|mountain|βουνό|river|ποτάμι|ocean|ωκεανό|island|νησί|population|πληθυσμ|climate|κλίμα|map|χάρτη|border|σύνορ|city|πόλη|region|περιοχ|lake|λίμν)\b',
    ]
    for p in geography_patterns:
        if re.search(p, lower):
            return Subject.GEOGRAPHY

    # Creative Writing
    creative_patterns = [
        r'\b(write|γράψε|poem|ποίημα|story|ιστορία|compose|create|δημιούργησε|imagine|φαντάσου|creative|δημιουργικ|song|τραγούδι|lyrics|στίχ|dialogue|διάλογ|screenplay|σενάρι|describe|περίγραψ|invent|εφεύρ)\b',
    ]
    for p in creative_patterns:
        if re.search(p, lower):
            return Subject.CREATIVE

    # Complexity-based fallback: long or complex → GENERAL, short → GREETING
    words = lower.split()
    if len(words) <= 4:
        return Subject.GREETING
    
    # Check for explanation requests
    explain_patterns = [
        r'\b(explain|εξήγησ|why|γιατί|how does|πώς λειτουργ|teach me|δίδαξέ|what is|τι είναι|compare|σύγκριν|analyze|αναλ|step by step|βήμα βήμα|describe|tell me about|πες μου)\b',
    ]
    for p in explain_patterns:
        if re.search(p, lower):
            return Subject.GENERAL

    return Subject.GENERAL


# ═══════════════════════════════════════════════════════════════════
# VOICE COMMAND DETECTION (strategy switching)
# ═══════════════════════════════════════════════════════════════════

class RoutingStrategy(Enum):
    BEST_ONE = "best_one"     # ⚡ Turbo — #1 ranked AI only
    RACE = "race"             # 🏎️ Race — top 2, use fastest
    VERIFY = "verify"         # ✅ Check — answer + verify with 2nd AI
    CONSENSUS = "consensus"   # 🧠 Smart — all 3, Claude merges best


def detect_voice_command(text: str) -> Optional[Tuple[str, RoutingStrategy]]:
    """
    Detect if the student is switching routing mode via voice.
    Returns (response_message, new_strategy) or None.
    """
    lower = text.lower().strip()
    lang = detect_language(text)

    commands = {
        RoutingStrategy.BEST_ONE: {
            "patterns": [r'\bturbo mode\b', r'\bγρήγορα\b', r'\bfast mode\b', r'\bturbo\b'],
            "en": "[excited] Turbo mode! I'll use the fastest AI for each question.",
            "el": "[excited] Γρήγορη λειτουργία! Θα χρησιμοποιώ τον καλύτερο AI!",
        },
        RoutingStrategy.RACE: {
            "patterns": [r'\brace mode\b', r'\bκούρσα\b', r'\brace\b'],
            "en": "[excited] Race mode! I'll ask two AIs and use the fastest answer.",
            "el": "[excited] Λειτουργία κούρσας! Θα ρωτάω δύο AI και θα χρησιμοποιώ τον πιο γρήγορο!",
        },
        RoutingStrategy.VERIFY: {
            "patterns": [r'\bcheck mode\b', r'\bέλεγξε\b', r'\bverify mode\b', r'\bcheck\b'],
            "en": "[thinking] Check mode! I'll double-check math and science answers with a second AI.",
            "el": "[thinking] Λειτουργία ελέγχου! Θα ελέγχω τις απαντήσεις με δεύτερο AI!",
        },
        RoutingStrategy.CONSENSUS: {
            "patterns": [r'\bsmart mode\b', r'\bσκέψου καλά\b', r'\bconsensus\b', r'\bsmart\b'],
            "en": "[thinking] Smart mode! I'll ask all three AIs and combine the best answer for you.",
            "el": "[thinking] Έξυπνη λειτουργία! Θα ρωτήσω και τους τρεις AI και θα συνδυάσω την καλύτερη απάντηση!",
        },
    }

    for strategy, cmd in commands.items():
        for pattern in cmd["patterns"]:
            if re.search(pattern, lower):
                msg = cmd["el"] if lang == "el" else cmd["en"]
                return (msg, strategy)

    return None


# ═══════════════════════════════════════════════════════════════════
# CONFIDENCE GATE — Detect if Ollama is uncertain
# ═══════════════════════════════════════════════════════════════════

class ConfidenceGate:
    """Check if local LLM response is confident enough or needs cloud escalation."""

    UNCERTAINTY_PHRASES = [
        # English
        "i'm not sure", "i don't know", "i'm uncertain", "i cannot", "i can't",
        "i'm not able", "beyond my", "not confident", "i think maybe",
        "i may be wrong", "don't quote me", "hard to say", "it's unclear",
        "might be", "could be wrong", "not certain", "i believe",
        # Greek
        "δεν ξέρω", "δεν είμαι σίγουρ", "ίσως", "μπορεί", "νομίζω",
        "δεν μπορώ", "δεν γνωρίζω", "πιθανόν", "υποθέτω",
    ]

    REFUSAL_PHRASES = [
        "i can't help", "beyond my ability", "i don't have",
        "δεν μπορώ να βοηθήσω", "δεν έχω πληροφορίες",
    ]

    @staticmethod
    def check(response: str) -> Tuple[bool, float]:
        """Returns (is_confident, confidence_score 0-1)."""
        if not response or len(response.strip()) < 10:
            return False, 0.0

        lower = response.lower()
        score = 1.0

        # Check refusals (immediate escalation)
        for phrase in ConfidenceGate.REFUSAL_PHRASES:
            if phrase in lower:
                return False, 0.1

        # Check uncertainty phrases
        uncertainties = sum(1 for p in ConfidenceGate.UNCERTAINTY_PHRASES if p in lower)
        score -= uncertainties * 0.25

        # Very short responses are suspicious
        if len(response.strip()) < 30:
            score -= 0.2

        # Excessive hedging
        hedge_count = len(re.findall(r'\b(maybe|perhaps|possibly|might|could|ίσως|μπορεί|πιθανόν)\b', lower))
        score -= hedge_count * 0.15

        score = max(0.0, min(1.0, score))
        return score >= 0.6, score


# ═══════════════════════════════════════════════════════════════════
# AI PROVIDER CLIENTS
# ═══════════════════════════════════════════════════════════════════

async def query_ollama(message: str, history: list, system_prompt: str, config: RouterConfig) -> Tuple[str, bool]:
    """Query local Ollama. Returns (response, success)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history[-6:])  # Local model gets less context
    messages.append({"role": "user", "content": message})

    payload = {
        "model": config.ollama_model,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": 500, "temperature": 0.4},
    }

    try:
        async with httpx.AsyncClient(timeout=config.local_timeout) as client:
            resp = await client.post(f"{config.ollama_url}/api/chat", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                text = data.get("message", {}).get("content", "").strip()
                return text, bool(text)
            return "", False
    except Exception as e:
        logger.warning(f"Ollama error: {e}")
        return "", False


async def query_claude(message: str, history: list, system_prompt: str, config: RouterConfig) -> Tuple[str, bool]:
    """Query Anthropic Claude with retry on 529."""
    if not config.anthropic_api_key:
        return "", False

    messages = []
    messages.extend(history[-10:])  # Cloud gets more context
    messages.append({"role": "user", "content": message})

    payload = {
        "model": config.claude_model,
        "max_tokens": 2000,
        "messages": messages,
    }
    if system_prompt:
        payload["system"] = system_prompt

    headers = {
        "Content-Type": "application/json",
        "x-api-key": config.anthropic_api_key,
        "anthropic-version": "2023-06-01",
    }

    for attempt in range(config.max_retries):
        try:
            async with httpx.AsyncClient(timeout=config.cloud_timeout) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    json=payload, headers=headers,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    text = "".join(
                        c.get("text", "") for c in data.get("content", [])
                        if c.get("type") == "text"
                    ).strip()
                    return text, bool(text)
                elif resp.status_code in (429, 529):
                    wait = (2 ** attempt) * 1
                    logger.warning(f"Claude {resp.status_code}, retry in {wait}s (attempt {attempt+1})")
                    await asyncio.sleep(wait)
                    continue
                else:
                    logger.warning(f"Claude HTTP {resp.status_code}: {resp.text[:200]}")
                    return "", False
        except Exception as e:
            logger.warning(f"Claude error (attempt {attempt+1}): {e}")
            if attempt < config.max_retries - 1:
                await asyncio.sleep(2 ** attempt)

    return "", False


async def query_gemini(message: str, history: list, system_prompt: str, config: RouterConfig) -> Tuple[str, bool]:
    """Query Google Gemini."""
    if not config.google_api_key:
        return "", False

    # Build Gemini message format
    contents = []
    if system_prompt:
        contents.append({"role": "user", "parts": [{"text": f"System instruction: {system_prompt}"}]})
        contents.append({"role": "model", "parts": [{"text": "Understood. I'll follow these instructions."}]})

    for msg in history[-10:]:
        role = "model" if msg.get("role") == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})
    contents.append({"role": "user", "parts": [{"text": message}]})

    payload = {
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 2000, "temperature": 0.4},
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.gemini_model}:generateContent?key={config.google_api_key}"

    try:
        async with httpx.AsyncClient(timeout=config.cloud_timeout) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts).strip()
                return text, bool(text)
            else:
                logger.warning(f"Gemini HTTP {resp.status_code}: {resp.text[:200]}")
                return "", False
    except Exception as e:
        logger.warning(f"Gemini error: {e}")
        return "", False


async def query_chatgpt(message: str, history: list, system_prompt: str, config: RouterConfig) -> Tuple[str, bool]:
    """Query OpenAI ChatGPT."""
    if not config.openai_api_key:
        return "", False

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": message})

    payload = {
        "model": config.chatgpt_model,
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0.4,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.openai_api_key}",
    }

    try:
        async with httpx.AsyncClient(timeout=config.cloud_timeout) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload, headers=headers,
            )
            if resp.status_code == 200:
                data = resp.json()
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                return text, bool(text)
            else:
                logger.warning(f"ChatGPT HTTP {resp.status_code}: {resp.text[:200]}")
                return "", False
    except Exception as e:
        logger.warning(f"ChatGPT error: {e}")
        return "", False


# Provider name → query function mapping
PROVIDER_FNS = {
    "claude": query_claude,
    "gemini": query_gemini,
    "chatgpt": query_chatgpt,
    "ollama": query_ollama,
}


# ═══════════════════════════════════════════════════════════════════
# ROUTING STRATEGIES
# ═══════════════════════════════════════════════════════════════════

async def strategy_best_one(
    message: str, history: list, system_prompt: str,
    subject: Subject, config: RouterConfig, available: List[str],
) -> Tuple[str, str]:
    """⚡ Turbo: Send to the #1 ranked AI for this subject only."""
    ranking = ROUTING_TABLE.get(subject, ["claude", "gemini", "chatgpt"])

    for provider in ranking:
        if provider not in available:
            continue
        fn = PROVIDER_FNS[provider]
        text, ok = await fn(message, history, system_prompt, config)
        if ok and text:
            return text, provider
        logger.warning(f"{provider} failed for {subject.value}, trying next...")

    # All cloud failed → Ollama fallback
    text, ok = await query_ollama(message, history, system_prompt, config)
    if ok:
        return text, "ollama"

    return "", "none"


async def strategy_race(
    message: str, history: list, system_prompt: str,
    subject: Subject, config: RouterConfig, available: List[str],
) -> Tuple[str, str]:
    """🏎️ Race: Send to top 2 AIs simultaneously, use fastest response."""
    ranking = ROUTING_TABLE.get(subject, ["claude", "gemini", "chatgpt"])
    top_two = [p for p in ranking if p in available][:2]

    if len(top_two) < 2:
        return await strategy_best_one(message, history, system_prompt, subject, config, available)

    async def race_query(provider: str) -> Tuple[str, str, bool]:
        fn = PROVIDER_FNS[provider]
        text, ok = await fn(message, history, system_prompt, config)
        return text, provider, ok

    tasks = [asyncio.create_task(race_query(p)) for p in top_two]

    # Wait for first successful result
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    # Cancel slower task
    for task in pending:
        task.cancel()

    for task in done:
        try:
            text, provider, ok = task.result()
            if ok and text:
                logger.info(f"🏎️ Race winner: {provider}")
                return text, provider
        except Exception:
            pass

    # Both failed, wait for pending
    for task in pending:
        try:
            text, provider, ok = await task
            if ok and text:
                return text, provider
        except Exception:
            pass

    # All failed → Ollama
    text, ok = await query_ollama(message, history, system_prompt, config)
    return (text, "ollama") if ok else ("", "none")


async def strategy_verify(
    message: str, history: list, system_prompt: str,
    subject: Subject, config: RouterConfig, available: List[str],
) -> Tuple[str, str]:
    """✅ Check: Get answer from #1, then verify with #2 for math/science."""
    ranking = ROUTING_TABLE.get(subject, ["claude", "gemini", "chatgpt"])
    ranked_available = [p for p in ranking if p in available]

    if not ranked_available:
        text, ok = await query_ollama(message, history, system_prompt, config)
        return (text, "ollama") if ok else ("", "none")

    # Get primary answer
    primary_provider = ranked_available[0]
    fn = PROVIDER_FNS[primary_provider]
    primary_text, ok = await fn(message, history, system_prompt, config)

    if not ok or not primary_text:
        return await strategy_best_one(message, history, system_prompt, subject, config, available)

    # Only verify math and science
    if subject not in (Subject.MATH, Subject.SCIENCE) or len(ranked_available) < 2:
        return primary_text, primary_provider

    # Verify with secondary AI
    verify_provider = ranked_available[1]
    lang = detect_language(message)

    verify_prompt = f"""A student asked: "{message}"

Another AI tutor answered: "{primary_text}"

Is this answer correct? If yes, say VERIFIED and give a clean version.
If there are errors, provide the CORRECT answer instead.
Keep the same language ({'Greek' if lang == 'el' else 'English'}) as the original."""

    verify_fn = PROVIDER_FNS[verify_provider]
    verify_text, v_ok = await verify_fn(verify_prompt, [], system_prompt, config)

    if v_ok and verify_text:
        if "VERIFIED" in verify_text.upper():
            # Primary was correct — use it
            logger.info(f"✅ {verify_provider} verified {primary_provider}'s answer")
            return primary_text, f"{primary_provider}+verified"
        else:
            # Verifier disagreed — use their corrected answer
            logger.info(f"✅ {verify_provider} corrected {primary_provider}'s answer")
            return verify_text, f"{verify_provider}+corrected"

    # Verification failed, return primary anyway
    return primary_text, primary_provider


async def strategy_consensus(
    message: str, history: list, system_prompt: str,
    subject: Subject, config: RouterConfig, available: List[str],
) -> Tuple[str, str]:
    """🧠 Smart: Ask ALL 3 AIs, then Claude merges the best answer."""
    cloud_available = [p for p in available if p != "ollama"]

    if len(cloud_available) < 2:
        return await strategy_best_one(message, history, system_prompt, subject, config, available)

    # Query all available cloud providers simultaneously
    async def run_query(provider: str) -> Tuple[str, str, bool]:
        fn = PROVIDER_FNS[provider]
        text, ok = await fn(message, history, system_prompt, config)
        return text, provider, ok

    tasks = [asyncio.create_task(run_query(p)) for p in cloud_available]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = []
    for r in results:
        if isinstance(r, tuple):
            text, provider, ok = r
            if ok and text:
                successful.append((text, provider))

    if len(successful) == 0:
        text, ok = await query_ollama(message, history, system_prompt, config)
        return (text, "ollama") if ok else ("", "none")

    if len(successful) == 1:
        return successful[0]

    # Multiple answers — merge with Claude
    lang = detect_language(message)
    lang_name = "Greek" if lang == "el" else "English"

    merge_parts = []
    for text, provider in successful:
        merge_parts.append(f"[{provider.upper()} answer]:\n{text}")
    all_answers = "\n\n".join(merge_parts)

    merge_prompt = f"""A student asked: "{message}"

Multiple AI tutors gave these answers:

{all_answers}

Your job: Create the BEST possible answer by combining the strongest parts from each.
- Use the most accurate facts and clearest explanations
- If answers disagree, use the most correct one
- Keep the answer in {lang_name}
- Follow the same format rules as the original system prompt
- Do NOT mention that multiple AIs were consulted"""

    # Try Claude for merging (it's best at synthesis)
    if "claude" in available:
        merged, ok = await query_claude(merge_prompt, [], system_prompt, config)
        if ok and merged:
            providers_used = "+".join(p for _, p in successful)
            logger.info(f"🧠 Consensus merged from: {providers_used}")
            return merged, f"consensus({providers_used})"

    # Merge with whatever is available
    for provider in available:
        if provider != "ollama":
            fn = PROVIDER_FNS[provider]
            merged, ok = await fn(merge_prompt, [], system_prompt, config)
            if ok and merged:
                providers_used = "+".join(p for _, p in successful)
                return merged, f"consensus({providers_used})"

    # Merge failed — return the first successful answer (from best-ranked provider)
    ranking = ROUTING_TABLE.get(subject, ["claude", "gemini", "chatgpt"])
    for preferred in ranking:
        for text, provider in successful:
            if provider == preferred:
                return text, provider

    return successful[0]


STRATEGY_FNS = {
    RoutingStrategy.BEST_ONE: strategy_best_one,
    RoutingStrategy.RACE: strategy_race,
    RoutingStrategy.VERIFY: strategy_verify,
    RoutingStrategy.CONSENSUS: strategy_consensus,
}


# ═══════════════════════════════════════════════════════════════════
# MAIN ROUTER (ASYNC)
# ═══════════════════════════════════════════════════════════════════

class VironAIRouter:
    """
    The main orchestrator. Routes questions to the best AI(s) based on
    subject classification and selected strategy.
    """

    def __init__(self, config: RouterConfig):
        self.config = config
        self.strategy = RoutingStrategy(config.strategy)
        self.stats = {
            "total": 0,
            "by_provider": {},
            "by_subject": {},
            "by_strategy": {},
            "escalations": 0,
            "ollama_calls": 0,
            "claude_calls": 0,
            "gemini_calls": 0,
            "chatgpt_calls": 0,
        }
        # Last request info (for debug endpoint)
        self.last_subject = ""
        self.last_strategy = ""
        self.last_provider = ""
        self.last_confidence = 1.0
        self.last_language = ""

    def _get_available_providers(self) -> List[str]:
        """List providers that have API keys configured."""
        available = []
        if self.config.anthropic_api_key:
            available.append("claude")
        if self.config.google_api_key:
            available.append("gemini")
        if self.config.openai_api_key:
            available.append("chatgpt")
        # Ollama is always "available" as fallback
        return available

    async def route(
        self,
        message: str,
        history: list = None,
        system_prompt: str = "",
        force_provider: str = None,
    ) -> Tuple[str, str]:
        """
        Main entry point. Returns (response_text, provider_name).

        Flow:
        1. Check for voice commands (strategy switching)
        2. Classify subject
        3. Greetings → Ollama (always local)
        4. Complex → route via selected strategy
        5. Confidence gate on local responses
        """
        history = history or []
        self.stats["total"] += 1

        # ── Step 1: Voice command detection ──
        voice_cmd = detect_voice_command(message)
        if voice_cmd:
            response_msg, new_strategy = voice_cmd
            self.strategy = new_strategy
            self.last_strategy = new_strategy.value
            self.last_provider = "system"
            self.last_subject = "command"
            logger.info(f"🎛️ Strategy switched to: {new_strategy.value}")
            return response_msg, "system"

        # ── Step 2: Classify ──
        subject = classify_subject(message)
        language = detect_language(message)
        self.last_subject = subject.value
        self.last_language = language
        self.last_strategy = self.strategy.value

        available = self._get_available_providers()
        logger.info(f"📚 Subject: {subject.value} | 🌐 {language} | 🎛️ {self.strategy.value} | Available: {available}")

        # ── Step 3: Force provider if requested ──
        if force_provider:
            fn = PROVIDER_FNS.get(force_provider)
            if fn:
                text, ok = await fn(message, history, system_prompt, self.config)
                provider = force_provider if ok else "none"
                self._update_stats(provider, subject)
                return text if ok else "", provider

        # ── Step 4: Greetings always go to Ollama ──
        if subject == Subject.GREETING:
            text, ok = await query_ollama(message, history, system_prompt, self.config)
            if ok:
                # Confidence gate on local response
                if self.config.confidence_gate:
                    confident, score = ConfidenceGate.check(text)
                    self.last_confidence = score
                    if not confident and available:
                        logger.info(f"⬆️ Escalating greeting (confidence={score:.2f}) to cloud")
                        self.stats["escalations"] += 1
                        text, provider = await strategy_best_one(
                            message, history, system_prompt, subject, self.config, available
                        )
                        self._update_stats(provider, subject)
                        return text, provider

                self._update_stats("ollama", subject)
                return text, "ollama"
            # Ollama failed — try cloud even for greetings
            if available:
                text, provider = await strategy_best_one(
                    message, history, system_prompt, subject, self.config, available
                )
                self._update_stats(provider, subject)
                return text, provider
            return "", "none"

        # ── Step 5: Route via strategy ──
        if not available:
            # No cloud keys → Ollama only
            text, ok = await query_ollama(message, history, system_prompt, self.config)
            self._update_stats("ollama" if ok else "none", subject)
            self.last_confidence = 0.5
            return text if ok else "", "ollama" if ok else "none"

        strategy_fn = STRATEGY_FNS[self.strategy]
        text, provider = await strategy_fn(
            message, history, system_prompt, subject, self.config, available
        )

        self.last_confidence = 0.9 if text else 0.0
        self._update_stats(provider, subject)

        # Final fallback
        if not text:
            text, ok = await query_ollama(message, history, system_prompt, self.config)
            if ok:
                self._update_stats("ollama", subject)
                return text, "ollama"

        return text, provider

    def _update_stats(self, provider: str, subject: Subject):
        self.last_provider = provider

        # Count by provider
        base_provider = provider.split("+")[0].split("(")[0]  # "claude+verified" → "claude"
        self.stats["by_provider"][base_provider] = self.stats["by_provider"].get(base_provider, 0) + 1

        # Count calls per AI
        for ai in ["ollama", "claude", "gemini", "chatgpt"]:
            if ai in provider:
                self.stats[f"{ai}_calls"] = self.stats.get(f"{ai}_calls", 0) + 1

        # Count by subject
        self.stats["by_subject"][subject.value] = self.stats["by_subject"].get(subject.value, 0) + 1

        # Count by strategy
        self.stats["by_strategy"][self.strategy.value] = self.stats["by_strategy"].get(self.strategy.value, 0) + 1

    def get_status(self) -> dict:
        available = self._get_available_providers()
        return {
            "strategy": self.strategy.value,
            "strategy_name": {
                "best_one": "⚡ Turbo",
                "race": "🏎️ Race",
                "verify": "✅ Check",
                "consensus": "🧠 Smart",
            }.get(self.strategy.value, self.strategy.value),
            "last_request": {
                "subject": self.last_subject,
                "language": self.last_language,
                "strategy": self.last_strategy,
                "provider": self.last_provider,
                "confidence": round(self.last_confidence, 2),
            },
            "providers": {
                "ollama": {"configured": True, "model": self.config.ollama_model, "url": self.config.ollama_url},
                "claude": {"configured": bool(self.config.anthropic_api_key), "model": self.config.claude_model},
                "gemini": {"configured": bool(self.config.google_api_key), "model": self.config.gemini_model},
                "chatgpt": {"configured": bool(self.config.openai_api_key), "model": self.config.chatgpt_model},
            },
            "available_cloud": available,
            "stats": self.stats,
            "routing_table": {
                s.value: ROUTING_TABLE[s] for s in Subject
            },
        }


# ═══════════════════════════════════════════════════════════════════
# SYNC WRAPPER (for Flask — server.py uses this)
# ═══════════════════════════════════════════════════════════════════

class VironAIRouterSync:
    """Synchronous wrapper for Flask (non-async) backend."""

    def __init__(self, config: RouterConfig):
        self.router = VironAIRouter(config)
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._loop = None

    def _get_loop(self):
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def chat(
        self,
        message: str,
        history: list = None,
        system_prompt: str = "",
        force_provider: str = None,
    ) -> Tuple[str, str]:
        """Sync wrapper around async route()."""
        loop = self._get_loop()
        try:
            return loop.run_until_complete(
                self.router.route(message, history, system_prompt, force_provider)
            )
        except RuntimeError:
            # If loop is running in another thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.router.route(message, history, system_prompt, force_provider),
                )
                return future.result(timeout=90)

    # Expose router properties
    @property
    def last_subject(self): return self.router.last_subject
    @property
    def last_strategy(self): return self.router.last_strategy
    @property
    def last_provider(self): return self.router.last_provider
    @property
    def last_confidence(self): return self.router.last_confidence
    @property
    def last_language(self): return self.router.last_language
    @property
    def strategy(self): return self.router.strategy

    def get_status(self): return self.router.get_status()
