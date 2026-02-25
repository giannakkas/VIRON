"""
VIRON AI BRAIN — Multi-LLM Orchestrator (Sync for Flask)
==========================================================
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

ALL HTTP calls are SYNCHRONOUS — works perfectly with Flask.
"""

import os
import re
import time
import json
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger("viron_brain")


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RouterConfig:
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    ollama_url: str = "http://localhost:11434"

    claude_model: str = "claude-opus-4-20250514"
    chatgpt_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-2.0-flash"
    ollama_model: str = "qwen2.5:3b"

    cloud_timeout: int = 45
    local_timeout: int = 45  # Needs to be high for first model load (then instant)
    max_retries: int = 2
    strategy: str = "best_one"
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
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
            cloud_timeout=int(os.getenv("CLOUD_TIMEOUT", "45")),
            local_timeout=int(os.getenv("LOCAL_TIMEOUT", "45")),
            max_retries=int(os.getenv("MAX_RETRIES", "2")),
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


# Subject → [1st choice, 2nd, 3rd] provider
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
    Subject.GREETING:   ["gemini", "claude", "chatgpt"],  # Cloud fallback if Ollama fails
}


def detect_language(text: str) -> str:
    greek = len(re.findall(r'[\u0370-\u03FF\u1F00-\u1FFF]', text))
    latin = len(re.findall(r'[a-zA-Z]', text))
    return "el" if greek > latin else "en"


def classify_subject(text: str) -> Subject:
    lower = text.lower().strip()

    # Greetings
    if re.match(r'^(hi|hello|hey|γεια|γεια σου|τι κάνεις|how are you|what\'s up|good morning|καλημέρα|καλησπέρα|ευχαριστώ|thanks|thank you|bye|ok|okay|yes|no|ναι|όχι|cool|nice|wow|haha|play|παίξε|σταμάτα|stop)\b', lower):
        return Subject.GREETING
    if len(lower) <= 15 and not re.search(r'\b(explain|what is|τι είναι|γιατί|why|how)\b', lower):
        return Subject.GREETING

    # Math
    if re.search(r'\b(math|μαθηματικ|algebra|γεωμετρ|equation|εξίσωσ|solve|λύσε|calculate|υπολόγισ|derivative|παράγωγ|integral|ολοκλήρωμα|fraction|κλάσμ|percent|ποσοστό|triangle|τρίγωνο|pythagoras|πυθαγόρ|formula|τύπος|multiply|πολλαπλ|divide|διαίρεσ|square root|ρίζα|logarithm|λογάριθμ|probability|πιθανότητ|factorial|matrix|function|συνάρτησ)\b', lower):
        return Subject.MATH
    if re.match(r'^\s*\d+\s*[\+\-\*\/\^x×÷]\s*\d+', lower):
        return Subject.MATH

    # Science
    if re.search(r'\b(physics|φυσικ|chemistry|χημεί|biology|βιολογ|photosynthesis|φωτοσύνθεσ|gravity|βαρύτητ|atom|άτομ|molecule|μόρι|energy|ενέργει|force|δύναμη|velocity|ταχύτητ|cell|κύτταρ|dna|evolution|εξέλιξ|planet|πλανήτ|element|στοιχείο|electron|proton|newton|magnetic|μαγνητ|temperature|θερμοκρασ|experiment|πείραμα|organism|volcano|ηφαίστει|earthquake|σεισμ|quantum|κβαντ)\b', lower):
        return Subject.SCIENCE

    # Coding
    if re.search(r'\b(code|coding|program|προγραμμ|python|javascript|html|java|algorithm|αλγόριθμ|loop|variable|μεταβλητ|debug|error|bug|syntax|array|class|database|api|git|terminal)\b', lower):
        return Subject.CODING

    # Translation
    if re.search(r'\b(translate|μετάφρασ|how do you say|πώς λέ(με|γεται|νε)|what does .+ mean|τι σημαίνει|in english|in greek|στα αγγλικά|στα ελληνικά)\b', lower):
        return Subject.TRANSLATION

    # Greek Language
    if re.search(r'\b(κλίν[εέη]ται|κλίση|ρήμα|ουσιαστικό|επίθετο|σύνταξ|γραμματικ|ορθογραφ|πτώσ[εη]|ενεστώτα|αόριστο|παρατατικ|μετοχ|αντωνυμ|greek grammar|conjugat|declension)\b', lower):
        return Subject.GREEK_LANG

    # English Language
    if re.search(r'\b(english grammar|past tense|present tense|irregular verb|preposition|adjective|adverb|vocabulary|spelling|pronunciation)\b', lower):
        return Subject.ENGLISH

    # History
    if re.search(r'\b(history|ιστορ|war|πόλεμ|revolution|επανάστασ|ancient|αρχαί|byzantine|βυζαντ|ottoman|οθωμαν|world war|civilization|πολιτισμ|emperor|αυτοκράτορ|king|βασιλι|dynasty|independence|ανεξαρτησί|battle|μάχ|1821|1940|1453|democracy|δημοκρατ)\b', lower):
        return Subject.HISTORY

    # Literature
    if re.search(r'\b(literature|λογοτεχν|poem|ποίημα|novel|μυθιστόρημα|author|συγγραφ|odyssey|οδύσσεια|iliad|ιλιάδα|homer|όμηρος|shakespeare|poetry|ποίηση|book|βιβλίο|metaphor|μεταφορ|symbolism|essay|δοκίμι|myth|μύθο)\b', lower):
        return Subject.LITERATURE

    # Geography
    if re.search(r'\b(geography|γεωγραφ|country|χώρα|capital|πρωτεύουσα|continent|ήπειρο|mountain|βουνό|river|ποτάμι|ocean|ωκεανό|island|νησί|population|πληθυσμ|climate|κλίμα|map|χάρτη|city|πόλη|lake|λίμν)\b', lower):
        return Subject.GEOGRAPHY

    # Creative Writing
    if re.search(r'\b(write|γράψε|compose|create|δημιούργησε|imagine|φαντάσου|creative|δημιουργικ|song|τραγούδι|lyrics|dialogue|διάλογ|describe|περίγραψ)\b', lower):
        return Subject.CREATIVE

    # Explanation requests → GENERAL (cloud)
    if re.search(r'\b(explain|εξήγησ|why|γιατί|how does|πώς λειτουργ|teach me|δίδαξέ|what is|τι είναι|compare|σύγκριν|analyze|step by step|βήμα βήμα|tell me about|πες μου)\b', lower):
        return Subject.GENERAL

    # Short/simple → GREETING, long → GENERAL
    if len(lower.split()) <= 4:
        return Subject.GREETING

    return Subject.GENERAL


# ═══════════════════════════════════════════════════════════════════
# VOICE COMMAND DETECTION
# ═══════════════════════════════════════════════════════════════════

class RoutingStrategy(Enum):
    BEST_ONE = "best_one"
    RACE = "race"
    VERIFY = "verify"
    CONSENSUS = "consensus"


def detect_voice_command(text: str) -> Optional[Tuple[str, RoutingStrategy]]:
    lower = text.lower().strip()
    lang = detect_language(text)

    commands = {
        RoutingStrategy.BEST_ONE: {
            "en": [r'\bturbo mode\b', r'\bfast mode\b'],
            "el": [r'\bγρήγορα\b', r'\bturbo\b'],
            "msg_en": "[excited] Turbo mode! I'll use the fastest AI.",
            "msg_el": "[excited] Γρήγορη λειτουργία!",
        },
        RoutingStrategy.RACE: {
            "en": [r'\brace mode\b'],
            "el": [r'\bκούρσα\b'],
            "msg_en": "[excited] Race mode! Two AIs race, fastest wins.",
            "msg_el": "[excited] Κούρσα! Δύο AI τρέχουν, κερδίζει ο πιο γρήγορος!",
        },
        RoutingStrategy.VERIFY: {
            "en": [r'\bcheck mode\b', r'\bverify mode\b'],
            "el": [r'\bέλεγξε\b'],
            "msg_en": "[thinking] Check mode! I'll double-check answers.",
            "msg_el": "[thinking] Λειτουργία ελέγχου! Θα ελέγχω τις απαντήσεις!",
        },
        RoutingStrategy.CONSENSUS: {
            "en": [r'\bsmart mode\b', r'\bconsensus\b'],
            "el": [r'\bσκέψου καλά\b', r'\bsmart\b'],
            "msg_en": "[thinking] Smart mode! All three AIs combine the best answer.",
            "msg_el": "[thinking] Έξυπνη λειτουργία! Και οι τρεις AI μαζί!",
        },
    }

    for strategy, cmd in commands.items():
        patterns = cmd.get("en", []) + cmd.get("el", [])
        for pattern in patterns:
            if re.search(pattern, lower):
                msg = cmd["msg_el"] if lang == "el" else cmd["msg_en"]
                return (msg, strategy)
    return None


# ═══════════════════════════════════════════════════════════════════
# CONFIDENCE GATE
# ═══════════════════════════════════════════════════════════════════

class ConfidenceGate:
    UNCERTAINTY = [
        "i'm not sure", "i don't know", "i cannot", "i can't",
        "might be", "could be wrong", "not certain",
        "δεν ξέρω", "δεν είμαι σίγουρ", "ίσως", "μπορεί", "δεν μπορώ",
    ]
    REFUSAL = [
        "i can't help", "beyond my ability",
        "δεν μπορώ να βοηθήσω",
    ]

    @staticmethod
    def check(response: str) -> Tuple[bool, float]:
        if not response or len(response.strip()) < 10:
            return False, 0.0
        lower = response.lower()
        for p in ConfidenceGate.REFUSAL:
            if p in lower:
                return False, 0.1
        score = 1.0
        for p in ConfidenceGate.UNCERTAINTY:
            if p in lower:
                score -= 0.25
        if len(response.strip()) < 30:
            score -= 0.2
        score = max(0.0, min(1.0, score))
        return score >= 0.6, score


# ═══════════════════════════════════════════════════════════════════
# SYNC AI PROVIDER QUERIES (using requests — no asyncio!)
# ═══════════════════════════════════════════════════════════════════

# Short system prompt for Ollama (small models can't handle the massive frontend prompt)
OLLAMA_SYSTEM_PROMPT = """You are VIRON, a friendly male AI companion robot for students. 
RULES: If the student speaks Greek, reply in Greek. If English, reply in English.
Start every response with [emotion] tag like [happy] or [excited].
Keep responses SHORT (1-3 sentences for chat, longer for explanations).
Be warm, friendly, and helpful. You're their best friend."""


# ── Local Model Strategy ──
# SPEED IS KING. Ollama can only keep 1 model in RAM at a time.
# Swapping models takes 3-8 seconds — unacceptable for a tutor.
#
# Strategy: ONE primary model handles all local queries (stays in RAM = instant).
# Cloud AIs handle complex questions (they're smarter anyway).
#
# Recommended models (pick ONE as primary):
#   qwen2.5:3b   — Best overall: great Greek, good English, decent math (~2GB)
#   phi3          — Best math/logic, weaker Greek (~2.2GB)  
#   llama3.2:3b   — Best English reasoning, OK Greek (~2GB)
#   gemma2:2b     — Fastest loading, smallest, decent quality (~1.6GB)
#
# The PRIMARY model is what's set in OLLAMA_MODEL env var (default: qwen2.5:3b)
# It stays loaded in RAM → responses in 1-3 seconds instead of 8-12.


def query_ollama(message: str, history: list, system_prompt: str, config: RouterConfig,
                 subject: Subject = None, timeout_override: int = None) -> Tuple[str, bool]:
    """Query local Ollama — uses the primary model (stays in RAM for speed)."""
    model = config.ollama_model
    timeout = timeout_override or config.local_timeout  # ONE model, always loaded, instant response

    messages = [{"role": "system", "content": OLLAMA_SYSTEM_PROMPT}]
    messages.extend(history[-4:])
    messages.append({"role": "user", "content": message})

    logger.info(f"🦙 Ollama: {model}")

    try:
        start = time.time()
        resp = requests.post(
            f"{config.ollama_url}/api/chat",
            json={"model": model, "messages": messages, "stream": False,
                  "options": {"num_predict": 200, "temperature": 0.5}},
            timeout=timeout,
        )
        elapsed = time.time() - start
        if resp.status_code == 200:
            text = resp.json().get("message", {}).get("content", "").strip()
            logger.info(f"  🦙 {model} responded in {elapsed:.1f}s ({len(text)} chars)")
            return text, bool(text)
        logger.warning(f"Ollama HTTP {resp.status_code}")
        return "", False
    except requests.exceptions.Timeout:
        logger.warning(f"Ollama timeout ({timeout}s)")
        return "", False
    except Exception as e:
        logger.warning(f"Ollama error: {e}")
        return "", False


def query_claude(message: str, history: list, system_prompt: str, config: RouterConfig) -> Tuple[str, bool]:
    """Query Anthropic Claude with retry on 529. Sync."""
    if not config.anthropic_api_key:
        return "", False

    messages = list(history[-10:])
    messages.append({"role": "user", "content": message})
    payload = {"model": config.claude_model, "max_tokens": 2000, "messages": messages}
    if system_prompt:
        payload["system"] = system_prompt
    headers = {
        "Content-Type": "application/json",
        "x-api-key": config.anthropic_api_key,
        "anthropic-version": "2023-06-01",
    }

    for attempt in range(config.max_retries):
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                json=payload, headers=headers, timeout=config.cloud_timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                text = "".join(
                    c.get("text", "") for c in data.get("content", []) if c.get("type") == "text"
                ).strip()
                return text, bool(text)
            elif resp.status_code in (429, 529):
                wait = (2 ** attempt)
                logger.warning(f"Claude {resp.status_code}, retry in {wait}s ({attempt+1}/{config.max_retries})")
                time.sleep(wait)
                continue
            else:
                logger.warning(f"Claude HTTP {resp.status_code}: {resp.text[:200]}")
                return "", False
        except Exception as e:
            logger.warning(f"Claude error ({attempt+1}): {e}")
            if attempt < config.max_retries - 1:
                time.sleep(2 ** attempt)

    return "", False


def query_gemini(message: str, history: list, system_prompt: str, config: RouterConfig) -> Tuple[str, bool]:
    """Query Google Gemini. Sync."""
    if not config.google_api_key:
        return "", False

    contents = []
    if system_prompt:
        contents.append({"role": "user", "parts": [{"text": f"System instruction: {system_prompt}"}]})
        contents.append({"role": "model", "parts": [{"text": "Understood."}]})
    for msg in history[-10:]:
        role = "model" if msg.get("role") == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})
    contents.append({"role": "user", "parts": [{"text": message}]})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.gemini_model}:generateContent?key={config.google_api_key}"

    try:
        resp = requests.post(
            url,
            json={"contents": contents, "generationConfig": {"maxOutputTokens": 2000, "temperature": 0.4}},
            timeout=config.cloud_timeout,
        )
        if resp.status_code == 200:
            parts = resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts).strip()
            return text, bool(text)
        logger.warning(f"Gemini HTTP {resp.status_code}: {resp.text[:200]}")
        return "", False
    except Exception as e:
        logger.warning(f"Gemini error: {e}")
        return "", False


def query_chatgpt(message: str, history: list, system_prompt: str, config: RouterConfig) -> Tuple[str, bool]:
    """Query OpenAI ChatGPT. Sync."""
    if not config.openai_api_key:
        return "", False

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": message})

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            json={"model": config.chatgpt_model, "messages": messages, "max_tokens": 2000, "temperature": 0.4},
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {config.openai_api_key}"},
            timeout=config.cloud_timeout,
        )
        if resp.status_code == 200:
            text = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            return text, bool(text)
        logger.warning(f"ChatGPT HTTP {resp.status_code}: {resp.text[:200]}")
        return "", False
    except Exception as e:
        logger.warning(f"ChatGPT error: {e}")
        return "", False


PROVIDER_FNS = {
    "claude": query_claude,
    "gemini": query_gemini,
    "chatgpt": query_chatgpt,
    "ollama": query_ollama,
}


# ═══════════════════════════════════════════════════════════════════
# ROUTING STRATEGIES (all sync, use ThreadPool for parallel)
# ═══════════════════════════════════════════════════════════════════

def strategy_best_one(msg, hist, sys_p, subject, config, available):
    """⚡ Turbo: #1 ranked AI for this subject."""
    ranking = ROUTING_TABLE.get(subject, ["claude", "gemini", "chatgpt"])
    for provider in ranking:
        if provider not in available:
            continue
        text, ok = PROVIDER_FNS[provider](msg, hist, sys_p, config)
        if ok and text:
            return text, provider
        logger.warning(f"{provider} failed for {subject.value}")

    # Fallback to Ollama
    text, ok = query_ollama(msg, hist, sys_p, config, subject)
    return (text, "ollama") if ok else ("", "none")


def strategy_race(msg, hist, sys_p, subject, config, available):
    """🏎️ Race: Top 2 in parallel, use fastest."""
    ranking = ROUTING_TABLE.get(subject, ["claude", "gemini", "chatgpt"])
    top_two = [p for p in ranking if p in available][:2]
    if len(top_two) < 2:
        return strategy_best_one(msg, hist, sys_p, subject, config, available)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(PROVIDER_FNS[p], msg, hist, sys_p, config): p
            for p in top_two
        }
        for future in as_completed(futures, timeout=config.cloud_timeout):
            provider = futures[future]
            try:
                text, ok = future.result()
                if ok and text:
                    logger.info(f"🏎️ Race winner: {provider}")
                    return text, provider
            except Exception:
                pass

    return strategy_best_one(msg, hist, sys_p, subject, config, available)


def strategy_verify(msg, hist, sys_p, subject, config, available):
    """✅ Check: Answer + verify with 2nd AI for math/science."""
    ranking = ROUTING_TABLE.get(subject, ["claude", "gemini", "chatgpt"])
    ranked = [p for p in ranking if p in available]
    if not ranked:
        text, ok = query_ollama(msg, hist, sys_p, config, subject)
        return (text, "ollama") if ok else ("", "none")

    # Primary answer
    primary_p = ranked[0]
    primary_text, ok = PROVIDER_FNS[primary_p](msg, hist, sys_p, config)
    if not ok or not primary_text:
        return strategy_best_one(msg, hist, sys_p, subject, config, available)

    # Only verify math/science
    if subject not in (Subject.MATH, Subject.SCIENCE) or len(ranked) < 2:
        return primary_text, primary_p

    # Verify
    verify_p = ranked[1]
    lang = detect_language(msg)
    verify_q = f'A student asked: "{msg}"\nAnother AI answered: "{primary_text}"\nIs this correct? If yes say VERIFIED and give a clean version. If wrong, give the CORRECT answer. Answer in {"Greek" if lang == "el" else "English"}.'

    verify_text, v_ok = PROVIDER_FNS[verify_p](verify_q, [], sys_p, config)
    if v_ok and verify_text:
        if "VERIFIED" in verify_text.upper():
            logger.info(f"✅ {verify_p} verified {primary_p}")
            return primary_text, f"{primary_p}+verified"
        else:
            logger.info(f"✅ {verify_p} corrected {primary_p}")
            return verify_text, f"{verify_p}+corrected"

    return primary_text, primary_p


def strategy_consensus(msg, hist, sys_p, subject, config, available):
    """🧠 Smart: All 3 in parallel, Claude merges best answer."""
    cloud = [p for p in available if p != "ollama"]
    if len(cloud) < 2:
        return strategy_best_one(msg, hist, sys_p, subject, config, available)

    # Query all in parallel
    successful = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(PROVIDER_FNS[p], msg, hist, sys_p, config): p
            for p in cloud
        }
        for future in as_completed(futures, timeout=config.cloud_timeout + 5):
            provider = futures[future]
            try:
                text, ok = future.result()
                if ok and text:
                    successful.append((text, provider))
            except Exception:
                pass

    if not successful:
        text, ok = query_ollama(msg, hist, sys_p, config, subject)
        return (text, "ollama") if ok else ("", "none")

    if len(successful) == 1:
        return successful[0]

    # Merge with Claude (or first available)
    lang = detect_language(msg)
    parts = "\n\n".join(f"[{p.upper()}]:\n{t}" for t, p in successful)
    merge_q = f'Student asked: "{msg}"\n\nMultiple AIs answered:\n\n{parts}\n\nCreate the BEST answer combining the strongest parts. Use {"Greek" if lang == "el" else "English"}. Do NOT mention multiple AIs were consulted.'

    # Try Claude first for merging, then others
    merge_order = ["claude", "gemini", "chatgpt"]
    for mp in merge_order:
        if mp in available:
            merged, ok = PROVIDER_FNS[mp](merge_q, [], sys_p, config)
            if ok and merged:
                used = "+".join(p for _, p in successful)
                logger.info(f"🧠 Consensus merged from: {used}")
                return merged, f"consensus({used})"

    # Merge failed — return best ranked
    ranking = ROUTING_TABLE.get(subject, ["claude", "gemini", "chatgpt"])
    for pref in ranking:
        for text, provider in successful:
            if provider == pref:
                return text, provider
    return successful[0]


STRATEGY_FNS = {
    RoutingStrategy.BEST_ONE: strategy_best_one,
    RoutingStrategy.RACE: strategy_race,
    RoutingStrategy.VERIFY: strategy_verify,
    RoutingStrategy.CONSENSUS: strategy_consensus,
}


# ═══════════════════════════════════════════════════════════════════
# MAIN ROUTER (SYNC — used directly by Flask)
# ═══════════════════════════════════════════════════════════════════

class VironAIRouterSync:
    """
    The main orchestrator. 100% synchronous — works perfectly with Flask.
    No asyncio, no event loops, no threading issues.
    """

    def __init__(self, config: RouterConfig):
        self.config = config
        self.strategy = RoutingStrategy(config.strategy)
        self.stats = {"total": 0, "by_provider": {}, "by_subject": {}, "escalations": 0}
        self.last_subject = ""
        self.last_strategy = ""
        self.last_provider = ""
        self.last_confidence = 1.0
        self.last_language = ""

    def _available(self) -> List[str]:
        out = []
        if self.config.anthropic_api_key: out.append("claude")
        if self.config.google_api_key: out.append("gemini")
        if self.config.openai_api_key: out.append("chatgpt")
        return out

    def chat(self, message: str, history: list = None, system_prompt: str = "",
             force_provider: str = None) -> Tuple[str, str]:
        """Main entry. Returns (response, provider_name)."""
        history = history or []
        self.stats["total"] += 1

        # Voice command?
        voice_cmd = detect_voice_command(message)
        if voice_cmd:
            msg, new_strat = voice_cmd
            self.strategy = new_strat
            self.last_strategy = new_strat.value
            self.last_provider = "system"
            self.last_subject = "command"
            logger.info(f"🎛️ Strategy → {new_strat.value}")
            return msg, "system"

        # Classify
        subject = classify_subject(message)
        language = detect_language(message)
        self.last_subject = subject.value
        self.last_language = language
        self.last_strategy = self.strategy.value

        available = self._available()
        logger.info(f"📚 {subject.value} | 🌐 {language} | 🎛️ {self.strategy.value} | 🔌 {available}")

        # Force provider
        if force_provider and force_provider in PROVIDER_FNS:
            text, ok = PROVIDER_FNS[force_provider](message, history, system_prompt, self.config)
            self._stat(force_provider if ok else "none", subject)
            return (text, force_provider) if ok else ("", "none")

        # Greetings → Ollama FAST (short timeout, cloud fallback if slow)
        if subject == Subject.GREETING:
            # Try Ollama with SHORT timeout — greetings must be instant
            try:
                start = time.time()
                text, ok = query_ollama(message, history, system_prompt, self.config,
                                        subject, timeout_override=8)  # 8s max for greetings
                elapsed = time.time() - start
                if ok and text:
                    self._stat("ollama", subject)
                    self.last_confidence = 0.95
                    logger.info(f"  ✅ Greeting via Ollama in {elapsed:.1f}s")
                    return text, "ollama"
                logger.info(f"  ⚠ Ollama failed for greeting ({elapsed:.1f}s), trying cloud")
            except Exception as e:
                logger.warning(f"  ⚠ Ollama error: {e}, trying cloud")

            # Ollama failed/slow → cloud fallback (Gemini is fastest)
            if available:
                for provider in ["gemini", "claude", "chatgpt"]:
                    if provider in available:
                        try:
                            text, ok = PROVIDER_FNS[provider](message, history, system_prompt, self.config)
                            if ok and text:
                                self._stat(provider, subject)
                                logger.info(f"  ✅ Greeting via {provider} (Ollama was down)")
                                return text, provider
                        except Exception:
                            continue
            return "[happy] Hey!", "fallback"

        # Route via strategy
        if not available:
            text, ok = query_ollama(message, history, system_prompt, self.config, subject)
            self._stat("ollama" if ok else "none", subject)
            return (text, "ollama") if ok else ("", "none")

        strategy_fn = STRATEGY_FNS[self.strategy]
        text, provider = strategy_fn(message, history, system_prompt, subject, self.config, available)
        self.last_confidence = 0.9 if text else 0.0
        self._stat(provider, subject)

        if not text:
            text, ok = query_ollama(message, history, system_prompt, self.config, subject)
            if ok:
                self._stat("ollama", subject)
                return text, "ollama"

        return text, provider

    def _stat(self, provider, subject):
        self.last_provider = provider
        base = provider.split("+")[0].split("(")[0]
        self.stats["by_provider"][base] = self.stats["by_provider"].get(base, 0) + 1
        self.stats["by_subject"][subject.value] = self.stats["by_subject"].get(subject.value, 0) + 1

    def get_status(self):
        return {
            "strategy": self.strategy.value,
            "last": {
                "subject": self.last_subject, "language": self.last_language,
                "provider": self.last_provider, "confidence": round(self.last_confidence, 2),
            },
            "providers": {
                "ollama": {"configured": True, "model": self.config.ollama_model},
                "claude": {"configured": bool(self.config.anthropic_api_key), "model": self.config.claude_model},
                "gemini": {"configured": bool(self.config.google_api_key), "model": self.config.gemini_model},
                "chatgpt": {"configured": bool(self.config.openai_api_key), "model": self.config.chatgpt_model},
            },
            "available_cloud": self._available(),
            "stats": self.stats,
            "routing_table": {s.value: ROUTING_TABLE[s] for s in Subject},
        }
