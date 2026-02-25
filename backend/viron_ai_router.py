"""
VIRON AI Router - Integrated for Flask Backend (port 5000)
===========================================================
Replaces direct Anthropic API calls with smart routing:
  - Ollama handles: greetings, simple Q&A, casual chat, factual lookups
  - Claude Opus handles: explanations, analysis, tutoring, complex topics
  - Fallback chain: Claude → Gemini → ChatGPT → Ollama
  - Retry with exponential backoff on 529/overloaded errors
  - Confidence gating on local responses

Drop-in replacement: import and use in your server.py
"""

import re
import time
import asyncio
import logging
import httpx
from typing import Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("viron.router")

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION - Edit these or load from environment
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RouterConfig:
    # Ollama (local)
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "phi3"  # or llama3.2, mistral, etc.
    local_timeout: float = 30.0
    
    # Claude (primary cloud - for explanations)
    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-4-0-20250514"  # Opus for quality explanations
    
    # Gemini (fallback 1)
    google_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    
    # ChatGPT (fallback 2)
    openai_api_key: str = ""
    chatgpt_model: str = "gpt-4o-mini"
    
    # Routing
    cloud_timeout: float = 60.0
    max_retries: int = 3
    retry_base_delay: float = 1.0  # exponential backoff base
    confidence_gate_enabled: bool = True


def load_config_from_env() -> RouterConfig:
    """Load config from environment variables."""
    import os
    return RouterConfig(
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "phi3"),
        local_timeout=float(os.getenv("LOCAL_TIMEOUT", "30")),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        claude_model=os.getenv("CLAUDE_MODEL", "claude-opus-4-0-20250514"),
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        chatgpt_model=os.getenv("CHATGPT_MODEL", "gpt-4o-mini"),
        cloud_timeout=float(os.getenv("CLOUD_TIMEOUT", "60")),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        retry_base_delay=float(os.getenv("RETRY_BASE_DELAY", "1.0")),
        confidence_gate_enabled=os.getenv("CONFIDENCE_GATE", "true").lower() == "true",
    )


# ═══════════════════════════════════════════════════════════════════
# COMPLEXITY ANALYZER - Decides Ollama vs Cloud
# ═══════════════════════════════════════════════════════════════════

class ComplexityAnalyzer:
    """
    Determines if a message needs Ollama (simple) or Cloud AI (complex).
    
    SIMPLE (→ Ollama):
      - Greetings, small talk, casual chat
      - Simple factual questions (who, what, when, where)
      - Yes/no questions
      - Definitions, translations, conversions
      - Short commands ("play music", "what time is it")
    
    COMPLEX (→ Claude Opus):
      - "Explain", "why", "how does X work"
      - Step-by-step problem solving
      - Math beyond basic arithmetic
      - Essay writing, analysis, comparison
      - Debugging, code help
      - Multi-part questions
      - Any tutoring/teaching request
    """
    
    # Patterns that DEFINITELY need cloud AI (explanations, teaching)
    COMPLEX_PATTERNS = [
        # Explanation requests - THE key trigger for Opus
        r'\b(explain|εξήγησ[εέη]|εξηγ[ήη]σ[εέ])\b',
        r'\b(why does|why is|why do|why are|why can\'t|γιατί)\b',
        r'\b(how does|how do|how is|how are|how can|πώς)\b',
        r'\b(what causes|what happens when|what would happen)\b',
        r'\b(can you teach|teach me|help me understand|βοήθησ[εέ])\b',
        r'\b(μάθε μου|δίδαξ[εέ] μου|πες μου γιατί)\b',
        
        # Analysis & comparison
        r'\b(analyze|analyse|compare|contrast|evaluate|αναλυ|σύγκρι)\b',
        r'\b(advantages|disadvantages|pros and cons|πλεονεκτ|μειονεκτ)\b',
        r'\b(difference between|similarities between|διαφορ[αά])\b',
        
        # Problem solving & math
        r'\b(solve|calculate|prove|derive|λύσε|υπολόγισε)\b',
        r'\b(step by step|βήμα βήμα|αναλυτικά)\b',
        r'\b(equation|formula|theorem|integral|derivative)\b',
        r'\b(εξίσωση|τύπος|θεώρημα|ολοκλήρωμα|παράγωγος)\b',
        
        # Writing & composition
        r'\b(write an essay|write a report|compose|draft|γράψε έκθεση)\b',
        r'\b(summarize|summarise|συνόψισε|περίληψη)\b',
        
        # Code & technical
        r'\b(debug|refactor|write code|implement|program)\b',
        r'\b(algorithm|recursion|data structure)\b',
        
        # Exam prep
        r'\b(exam|test prep|quiz me|εξετάσ|διαγώνισμα)\b',
        
        # Deep questions
        r'\b(what is the meaning|what is the purpose|what is the significance)\b',
        r'\b(in detail|in depth|thoroughly|elaborate|αναλυτικά|λεπτομερ)\b',
    ]
    
    # Patterns that are DEFINITELY simple (→ Ollama)
    SIMPLE_PATTERNS = [
        # Greetings & small talk
        r'^(hi|hello|hey|γεια|γεια σου|γεια σας|καλημέρα|καλησπέρα|yo|sup)[\s!.?]*$',
        r'^(thanks|thank you|ευχαριστώ|bye|goodbye|αντίο|goodnight|καληνύχτα)[\s!.?]*$',
        r'\b(how are you|τι κάνεις|what\'s up|τι γίνεται|τι νέα)\b',
        r'\b(what\'s your name|πώς σε λένε|who are you|ποιος είσαι)\b',
        
        # Simple factual (who/what/when/where)
        r'^(who is|who was|ποιος είναι|ποιος ήταν)\s',
        r'^(what is|what are|what was|τι είναι|τι σημαίνει)\s\S+\s?\S*\??$',  # short "what is X?"
        r'^(when was|when did|when is|πότε)\s',
        r'^(where is|where was|πού είναι|πού βρίσκεται)\s',
        r'^(how many|how much|πόσ[αοε])\s',
        
        # Yes/no
        r'\b(is it true|is it correct|σωστά|αλήθεια)\b',
        r'^(can|could|does|do|is|are|was|were|did|has|have|will|would|should)\s\w+',
        
        # Commands
        r'\b(play|παίξε|stop|σταμάτα|pause|repeat|επανάλαβε)\b',
        r'\b(play music|play a song|βάλε μουσική|βάλε τραγούδι)\b',
        
        # Definitions (simple)
        r'^(define|definition of|ορισμός|τι σημαίνει)\s\S+\s?\S*\??$',
        
        # Translate/convert
        r'\b(translate|μετάφρασε|convert|μετατρεψε)\b.*\b(to|σε)\b',
        
        # Basic math (2+2, not "explain calculus")
        r'^\d+\s*[\+\-\*\/\×\÷]\s*\d+\s*[=\?]?\s*$',
        
        # Time/weather
        r'\b(what time|τι ώρα|what day|τι μέρα|weather|καιρ[οό]ς)\b',
    ]
    
    def analyze(self, message: str, context: Optional[str] = None) -> Tuple[str, float]:
        """
        Returns: ("simple"|"complex", confidence_score)
        """
        msg = message.strip()
        msg_lower = msg.lower()
        word_count = len(msg_lower.split())
        
        # Very short messages (1-3 words) are almost always simple
        if word_count <= 3:
            # Unless it's "explain photosynthesis" etc.
            for pattern in self.COMPLEX_PATTERNS:
                if re.search(pattern, msg_lower):
                    return "complex", 0.8
            return "simple", 0.9
        
        # Check explicit complex patterns first (these override everything)
        complex_hits = 0
        for pattern in self.COMPLEX_PATTERNS:
            if re.search(pattern, msg_lower):
                complex_hits += 1
        
        if complex_hits >= 2:
            return "complex", 0.95
        if complex_hits == 1:
            # One complex indicator + long message = complex
            if word_count > 8:
                return "complex", 0.85
            # Short message with one complex word - still complex
            return "complex", 0.75
        
        # Check explicit simple patterns
        simple_hits = 0
        for pattern in self.SIMPLE_PATTERNS:
            if re.search(pattern, msg_lower):
                simple_hits += 1
        
        if simple_hits >= 1:
            return "simple", 0.85
        
        # Heuristic: longer messages tend to be more complex
        if word_count > 20:
            return "complex", 0.7
        elif word_count > 12:
            return "complex", 0.6
        
        # Default: if no clear signal, use local (cheaper, faster)
        # The confidence gate will catch bad answers
        return "simple", 0.5


# ═══════════════════════════════════════════════════════════════════
# CONFIDENCE GATE - Checks if Ollama's answer is trustworthy
# ═══════════════════════════════════════════════════════════════════

class ConfidenceGate:
    """Checks if a local LLM response is confident enough to use."""
    
    UNCERTAINTY_PHRASES = [
        r"\bi('m| am) not (sure|certain|confident)\b",
        r"\bi don'?t (know|think|believe)\b",
        r"\b(might|could|may) be\b",
        r"\b(possibly|perhaps|maybe|probably)\b",
        r"\bi('m| am) (unsure|uncertain)\b",
        r"\bδεν (ξέρω|είμαι σίγουρος|γνωρίζω)\b",
        r"\b(ίσως|πιθανόν|μπορεί)\b",
        r"\b(i can'?t|i cannot) (help|answer|assist)\b",
        r"\b(sorry|apologi[sz]e|forgive)\b.*\b(can'?t|unable|don'?t)\b",
        r"\bas an ai\b",
        r"\bi would (suggest|recommend) (asking|consulting)\b",
    ]
    
    REFUSAL_PHRASES = [
        r"\bi (can'?t|cannot|won'?t|will not) (help|assist|provide)\b",
        r"\b(inappropriate|not appropriate)\b",
        r"\b(beyond my|outside my) (ability|capabilities|scope)\b",
    ]
    
    def check(self, response: str) -> Tuple[bool, float]:
        """
        Returns: (is_confident, confidence_score)
        confidence_score: 0.0 (no confidence) to 1.0 (fully confident)
        """
        if not response or len(response.strip()) < 10:
            return False, 0.1
        
        resp_lower = response.lower()
        
        # Check for refusals
        for pattern in self.REFUSAL_PHRASES:
            if re.search(pattern, resp_lower):
                return False, 0.05
        
        # Count uncertainty markers
        uncertainty_count = 0
        for pattern in self.UNCERTAINTY_PHRASES:
            if re.search(pattern, resp_lower):
                uncertainty_count += 1
        
        # Score based on uncertainty density
        words = len(resp_lower.split())
        if words < 5:
            # Very short response - suspicious
            confidence = 0.3
        elif uncertainty_count == 0:
            confidence = 0.95
        elif uncertainty_count == 1 and words > 30:
            # One hedge in a long response is normal
            confidence = 0.75
        elif uncertainty_count == 1:
            confidence = 0.5
        else:
            # Multiple uncertainty markers = not confident
            confidence = max(0.1, 0.5 - uncertainty_count * 0.15)
        
        return confidence >= 0.6, confidence


# ═══════════════════════════════════════════════════════════════════
# AI ROUTER - Main routing logic
# ═══════════════════════════════════════════════════════════════════

class VironAIRouter:
    """
    Smart AI router for VIRON.
    
    Routing flow:
    1. Analyze message complexity
    2. SIMPLE → Ollama → confidence gate → escalate if uncertain
    3. COMPLEX → Claude Opus → retry on 529 → fallback to Gemini → ChatGPT → Ollama
    """
    
    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or load_config_from_env()
        self.analyzer = ComplexityAnalyzer()
        self.gate = ConfidenceGate()
        self.client = httpx.AsyncClient(timeout=self.config.cloud_timeout)
        self.last_complexity = "unknown"
        self.last_provider = "none"
        self.last_confidence = 0.0
        
        # Stats
        self.stats = {
            "ollama_calls": 0,
            "claude_calls": 0,
            "gemini_calls": 0,
            "chatgpt_calls": 0,
            "escalations": 0,
            "retries": 0,
            "fallbacks": 0,
        }
    
    # ─── Provider queries ──────────────────────────────────────
    
    async def check_ollama(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            resp = await self.client.get(
                f"{self.config.ollama_url}/api/tags",
                timeout=5.0
            )
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                return any(self.config.ollama_model in m.get("name", "") for m in models)
            return False
        except Exception:
            return False
    
    async def query_ollama(self, message: str, history: list, system_prompt: str) -> str:
        """Query local Ollama model."""
        messages = [{"role": "system", "content": system_prompt}]
        # Only send last 6 turns to local (smaller context window)
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": message})
        
        resp = await self.client.post(
            f"{self.config.ollama_url}/api/chat",
            json={
                "model": self.config.ollama_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 512,  # keep responses concise
                }
            },
            timeout=self.config.local_timeout,
        )
        data = resp.json()
        self.stats["ollama_calls"] += 1
        return data.get("message", {}).get("content", "")
    
    async def query_claude(self, message: str, history: list, system_prompt: str) -> str:
        """Query Claude with retry on 529."""
        if not self.config.anthropic_api_key:
            raise ValueError("No Anthropic API key")
        
        messages = []
        for msg in history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": message})
        
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                resp = await self.client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.config.anthropic_api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": self.config.claude_model,
                        "max_tokens": 1024,
                        "system": system_prompt,
                        "messages": messages,
                    },
                    timeout=self.config.cloud_timeout,
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    self.stats["claude_calls"] += 1
                    return "".join(
                        block["text"]
                        for block in data.get("content", [])
                        if block.get("type") == "text"
                    )
                elif resp.status_code == 529:
                    # Overloaded - retry with backoff
                    self.stats["retries"] += 1
                    delay = self.config.retry_base_delay * (2 ** attempt)
                    logger.warning(f"Claude 529 overloaded, retry {attempt+1}/{self.config.max_retries} in {delay}s")
                    await asyncio.sleep(delay)
                    last_error = f"529 Overloaded (attempt {attempt+1})"
                    continue
                elif resp.status_code == 429:
                    # Rate limited
                    delay = self.config.retry_base_delay * (2 ** attempt) * 2
                    logger.warning(f"Claude 429 rate limited, retry in {delay}s")
                    await asyncio.sleep(delay)
                    last_error = f"429 Rate limited"
                    continue
                else:
                    error_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                    last_error = f"Claude HTTP {resp.status_code}: {error_data.get('error', {}).get('message', 'Unknown')}"
                    raise ValueError(last_error)
                    
            except httpx.TimeoutException:
                last_error = "Claude timeout"
                self.stats["retries"] += 1
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_base_delay * (2 ** attempt))
                    continue
                raise
        
        raise ValueError(f"Claude failed after {self.config.max_retries} retries: {last_error}")
    
    async def query_gemini(self, message: str, history: list, system_prompt: str) -> str:
        """Query Google Gemini as fallback."""
        if not self.config.google_api_key:
            raise ValueError("No Google API key")
        
        # Build Gemini format
        contents = []
        for msg in history[-10:]:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        contents.append({"role": "user", "parts": [{"text": message}]})
        
        resp = await self.client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.gemini_model}:generateContent",
            params={"key": self.config.google_api_key},
            json={
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": contents,
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 1024,
                },
            },
            timeout=self.config.cloud_timeout,
        )
        
        if resp.status_code != 200:
            raise ValueError(f"Gemini error {resp.status_code}")
        
        data = resp.json()
        self.stats["gemini_calls"] += 1
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts)
        return ""
    
    async def query_chatgpt(self, message: str, history: list, system_prompt: str) -> str:
        """Query OpenAI ChatGPT as fallback."""
        if not self.config.openai_api_key:
            raise ValueError("No OpenAI API key")
        
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": message})
        
        resp = await self.client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.config.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.config.chatgpt_model,
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.7,
            },
            timeout=self.config.cloud_timeout,
        )
        
        data = resp.json()
        if "choices" in data and len(data["choices"]) > 0:
            self.stats["chatgpt_calls"] += 1
            return data["choices"][0]["message"]["content"]
        if "error" in data:
            raise ValueError(f"ChatGPT: {data['error'].get('message', 'Unknown')}")
        return ""
    
    # ─── Cloud fallback chain ──────────────────────────────────
    
    async def query_cloud_with_fallback(
        self, message: str, history: list, system_prompt: str
    ) -> Tuple[str, str]:
        """
        Try cloud providers in order: Claude → Gemini → ChatGPT
        Returns: (response_text, provider_name)
        """
        providers = []
        if self.config.anthropic_api_key:
            providers.append(("claude", self.query_claude))
        if self.config.google_api_key:
            providers.append(("gemini", self.query_gemini))
        if self.config.openai_api_key:
            providers.append(("chatgpt", self.query_chatgpt))
        
        if not providers:
            raise ValueError("No cloud providers configured")
        
        last_error = None
        for name, query_fn in providers:
            try:
                logger.info(f"Trying cloud provider: {name}")
                reply = await query_fn(message, history, system_prompt)
                if reply and len(reply.strip()) > 0:
                    return reply, name
            except Exception as e:
                last_error = f"{name}: {e}"
                self.stats["fallbacks"] += 1
                logger.warning(f"Cloud provider {name} failed: {e}, trying next...")
                continue
        
        raise ValueError(f"All cloud providers failed. Last error: {last_error}")
    
    # ─── Main routing ──────────────────────────────────────────
    
    async def route(
        self,
        message: str,
        history: list,
        system_prompt: str,
        force_provider: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Main routing logic.
        
        Returns: (response_text, provider_name)
        
        Flow:
        1. Analyze complexity
        2. SIMPLE → Ollama → confidence gate → escalate if needed
        3. COMPLEX → Cloud (Claude Opus) → fallback chain → Ollama last resort
        """
        start = time.time()
        
        # ── Forced provider ──
        if force_provider:
            if force_provider == "local":
                reply = await self.query_ollama(message, history, system_prompt)
                self.last_complexity = "forced_local"
                self.last_provider = "ollama"
                return reply, "ollama"
            elif force_provider == "claude":
                reply = await self.query_claude(message, history, system_prompt)
                self.last_complexity = "forced_cloud"
                self.last_provider = "claude"
                return reply, "claude"
            elif force_provider == "gemini":
                reply = await self.query_gemini(message, history, system_prompt)
                self.last_complexity = "forced_cloud"
                self.last_provider = "gemini"
                return reply, "gemini"
            elif force_provider == "chatgpt":
                reply = await self.query_chatgpt(message, history, system_prompt)
                self.last_complexity = "forced_cloud"
                self.last_provider = "chatgpt"
                return reply, "chatgpt"
        
        # ── Analyze complexity ──
        complexity, conf = self.analyzer.analyze(message)
        self.last_complexity = complexity
        logger.info(f"Complexity: {complexity} (confidence: {conf:.2f}) for: {message[:60]}...")
        
        # ── SIMPLE → Ollama first ──
        if complexity == "simple":
            try:
                ollama_ok = await self.check_ollama()
                if ollama_ok:
                    reply = await self.query_ollama(message, history, system_prompt)
                    
                    # Confidence gate
                    if self.config.confidence_gate_enabled:
                        is_confident, confidence = self.gate.check(reply)
                        self.last_confidence = confidence
                        
                        if is_confident:
                            self.last_provider = "ollama"
                            logger.info(f"Ollama answered (confidence: {confidence:.2f})")
                            return reply, "ollama"
                        else:
                            # Escalate to cloud
                            self.stats["escalations"] += 1
                            self.last_complexity = "escalated"
                            logger.info(f"Ollama uncertain (confidence: {confidence:.2f}), escalating to cloud")
                            try:
                                cloud_reply, provider = await self.query_cloud_with_fallback(
                                    message, history, system_prompt
                                )
                                self.last_provider = provider
                                return cloud_reply, provider
                            except Exception:
                                # Cloud failed too - return Ollama's answer anyway
                                logger.warning("Cloud fallback failed, using Ollama answer")
                                self.last_provider = "ollama (low confidence)"
                                return reply, "ollama"
                    else:
                        self.last_provider = "ollama"
                        self.last_confidence = 1.0
                        return reply, "ollama"
                else:
                    logger.warning("Ollama not available, routing to cloud")
                    # Fall through to cloud
            except Exception as e:
                logger.warning(f"Ollama failed: {e}, routing to cloud")
                # Fall through to cloud
        
        # ── COMPLEX (or Ollama failed) → Cloud with fallback ──
        try:
            reply, provider = await self.query_cloud_with_fallback(
                message, history, system_prompt
            )
            self.last_provider = provider
            self.last_confidence = 1.0
            elapsed = time.time() - start
            logger.info(f"Cloud ({provider}) answered in {elapsed:.1f}s")
            return reply, provider
        except Exception as cloud_error:
            # All cloud providers failed - try Ollama as absolute last resort
            logger.error(f"All cloud providers failed: {cloud_error}")
            try:
                if await self.check_ollama():
                    reply = await self.query_ollama(message, history, system_prompt)
                    self.last_provider = "ollama (fallback)"
                    self.last_confidence = 0.3
                    return reply, "ollama"
            except Exception:
                pass
            
            # Nothing works
            self.last_provider = "error"
            self.last_confidence = 0.0
            return "", "error"
    
    def get_status(self) -> dict:
        """Get router status for debugging."""
        return {
            "providers": {
                "ollama": {"model": self.config.ollama_model, "url": self.config.ollama_url},
                "claude": {"model": self.config.claude_model, "configured": bool(self.config.anthropic_api_key)},
                "gemini": {"model": self.config.gemini_model, "configured": bool(self.config.google_api_key)},
                "chatgpt": {"model": self.config.chatgpt_model, "configured": bool(self.config.openai_api_key)},
            },
            "last": {
                "complexity": self.last_complexity,
                "provider": self.last_provider,
                "confidence": self.last_confidence,
            },
            "stats": self.stats,
        }


# ═══════════════════════════════════════════════════════════════════
# SYNCHRONOUS WRAPPER - For Flask (non-async) backends
# ═══════════════════════════════════════════════════════════════════

class VironAIRouterSync:
    """
    Synchronous wrapper for Flask backends.
    Uses asyncio.run() / event loop to call async methods.
    
    Usage in Flask:
        router = VironAIRouterSync()
        reply, provider = router.chat("Hello!", history=[], system_prompt="...")
    """
    
    def __init__(self, config: Optional[RouterConfig] = None):
        self._async_router = VironAIRouter(config)
        self._loop = None
    
    def _get_loop(self):
        """Get or create an event loop."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an existing async context (like Flask with async)
                # Use nest_asyncio or create new loop in thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool
            return loop
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
    
    def chat(
        self,
        message: str,
        history: list = None,
        system_prompt: str = "",
        force_provider: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Synchronous chat method.
        Returns: (response_text, provider_name)
        """
        if history is None:
            history = []
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context - use thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self._async_router.route(message, history, system_prompt, force_provider)
                    )
                    return future.result(timeout=90)
            else:
                return loop.run_until_complete(
                    self._async_router.route(message, history, system_prompt, force_provider)
                )
        except RuntimeError:
            return asyncio.run(
                self._async_router.route(message, history, system_prompt, force_provider)
            )
    
    def check_ollama_sync(self) -> bool:
        """Check Ollama availability synchronously."""
        try:
            return asyncio.run(self._async_router.check_ollama())
        except Exception:
            return False
    
    @property
    def stats(self):
        return self._async_router.stats
    
    @property
    def last_complexity(self):
        return self._async_router.last_complexity
    
    @property
    def last_provider(self):
        return self._async_router.last_provider
    
    @property
    def last_confidence(self):
        return self._async_router.last_confidence
    
    def get_status(self):
        return self._async_router.get_status()


# ═══════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("  VIRON AI Router - Complexity Test")
    print("=" * 60)
    
    analyzer = ComplexityAnalyzer()
    
    test_cases = [
        # Should be SIMPLE (→ Ollama)
        ("Hello!", "simple"),
        ("Γεια σου!", "simple"),
        ("What is 2+3?", "simple"),
        ("Who is Einstein?", "simple"),
        ("What time is it?", "simple"),
        ("Τι κάνεις;", "simple"),
        ("Play some music", "simple"),
        ("Thanks!", "simple"),
        ("Define gravity", "simple"),
        ("Translate hello to Greek", "simple"),
        
        # Should be COMPLEX (→ Claude Opus)
        ("Explain how photosynthesis works", "complex"),
        ("Εξήγησέ μου τη φωτοσύνθεση", "complex"),
        ("Why does the sky appear blue?", "complex"),
        ("Compare democracy and monarchy", "complex"),
        ("Solve this equation step by step: 2x + 5 = 15", "complex"),
        ("Help me understand derivatives in calculus", "complex"),
        ("Write an essay about climate change", "complex"),
        ("Analyze the themes in Romeo and Juliet", "complex"),
        ("Why is water important for life on Earth and what happens without it?", "complex"),
        ("Debug this Python code for me", "complex"),
        ("Γιατί ο ουρανός είναι μπλε;", "complex"),
        ("Βοήθησέ με με τα μαθηματικά", "complex"),
    ]
    
    correct = 0
    for msg, expected in test_cases:
        result, conf = analyzer.analyze(msg)
        status = "✅" if result == expected else "❌"
        if result == expected:
            correct += 1
        print(f"  {status} [{result:7s} {conf:.2f}] {msg[:50]}")
    
    print(f"\n  Score: {correct}/{len(test_cases)} ({100*correct/len(test_cases):.0f}%)")
    
    print("\n" + "=" * 60)
    print("  Confidence Gate Test")
    print("=" * 60)
    
    gate = ConfidenceGate()
    
    gate_tests = [
        ("Photosynthesis is the process by which plants convert sunlight into energy.", True),
        ("I'm not sure about this, but it might be something related to plants.", False),
        ("I don't know the answer to that question.", False),
        ("", False),
        ("As an AI, I cannot help with that.", False),
        ("The capital of France is Paris. It has been the capital since the 10th century.", True),
        ("Maybe it could be Paris? I'm not entirely certain though.", False),
    ]
    
    for resp, expected in gate_tests:
        confident, score = gate.check(resp)
        status = "✅" if confident == expected else "❌"
        print(f"  {status} [conf={score:.2f}] {resp[:60]}...")
    
    print("\n  Done! Router ready for integration.")
