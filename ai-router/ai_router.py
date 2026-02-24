"""
VIRON AI Router — Smart routing between local LLM and cloud providers.
Simple questions → Ollama (local), Complex questions → Cloud (Claude/Gemini/ChatGPT)
"""

import re, json, time, hashlib, sqlite3
from typing import Optional, Tuple
import httpx
import config as settings


class ComplexityAnalyzer:
    """Determines if a question is simple (local) or complex (cloud)."""

    COMPLEX_INDICATORS = [
        r"explain\s+(how|why|the\s+concept)",
        r"compare\s+and\s+contrast",
        r"what\s+are\s+the\s+(advantages|disadvantages|differences|implications)",
        r"(analyze|evaluate|assess|critique|discuss)",
        r"(calculus|derivative|integral|differential|quantum|thermodynamic)",
        r"(essay|paragraph|composition|report)\s+(about|on)",
        r"step[\s-]by[\s-]step",
        r"(code|program|function|algorithm|debug)\s+",
        r"(prove|theorem|hypothesis|conjecture)",
        r"(philosophy|ethics|moral|existential)",
        r"(history\s+of|evolution\s+of|development\s+of)",
        r"in\s+detail",
        r"(multiple|several|various)\s+(reasons|factors|examples)",
        r"(συγκρ[ίι]ν|αναλ[υύ]|εξ[ήη]γ[ηη]σε\s+πως|αποδε[ίι]ξ|αν[αά]λυσ)",
    ]

    SIMPLE_INDICATORS = [
        r"^(hi|hello|hey|γεια|γεια\s*σου|καλημ[εέ]ρα|τι\s*κ[αά]νεις)",
        r"^(what|who|when|where)\s+is\s+",
        r"^(how\s+much|how\s+many|how\s+old|how\s+far)",
        r"^(yes|no|ok|okay|sure|thanks|ναι|[οό]χι|ευχαριστ[ώω])",
        r"(what|τι)\s+(time|day|date|weather|ώρα|μέρα)",
        r"(play|βάλε|παίξε)\s+.*(song|music|τραγούδι|μουσική)",
        r"^(tell\s+me\s+a\s+joke|πες\s+μου\s+[εέ]να\s+αν[εέ]κδοτο)",
        r"(capital|population)\s+of",
        r"^\d+\s*[\+\-\*\/x×÷]\s*\d+",
        r"(translate|μετάφρασ[εη])",
        r"(define|definition|what\s+does\s+\w+\s+mean)",
        r"(spell|how\s+do\s+you\s+spell)",
    ]

    def analyze(self, message: str, context: str = None) -> Tuple[str, float]:
        message_lower = message.lower().strip()
        word_count = len(message_lower.split())

        if word_count <= 5:
            simple_score = 0.7
        elif word_count <= 15:
            simple_score = 0.5
        else:
            simple_score = 0.3

        complex_hits = sum(1 for p in self.COMPLEX_INDICATORS if re.search(p, message_lower))
        simple_hits = sum(1 for p in self.SIMPLE_INDICATORS if re.search(p, message_lower))

        complex_score = min(complex_hits * 0.3, 1.0)
        simple_bonus = min(simple_hits * 0.2, 0.6)
        final = simple_score + simple_bonus - complex_score

        if context:
            cl = context.lower()
            if any(w in cl for w in ["advanced", "ap ", "college", "university", "exam"]):
                final -= 0.3

        return ("simple", final) if final >= 0.5 else ("complex", 1.0 - final)


class ResponseCache:
    """SQLite cache for AI responses."""

    def __init__(self):
        if not settings.CACHE_ENABLED:
            self.db = None
            return
        self.db = sqlite3.connect(settings.CACHE_DB_PATH, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY, response TEXT, provider TEXT,
            created_at REAL, ttl_hours INTEGER)""")
        self.db.commit()

    def _key(self, message: str, system: str) -> str:
        return hashlib.sha256(f"{message}|{system}".encode()).hexdigest()

    def get(self, message: str, system: str = "") -> Optional[dict]:
        if not self.db:
            return None
        try:
            k = self._key(message, system)
            row = self.db.execute("SELECT response, provider, created_at, ttl_hours FROM cache WHERE key=?", (k,)).fetchone()
            if row and (time.time() - row[2]) < row[3] * 3600:
                return {"response": row[0], "provider": f"{row[1]}(cached)"}
        except:
            pass
        return None

    def set(self, message: str, system: str, response: str, provider: str):
        if not self.db:
            return
        try:
            k = self._key(message, system)
            self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?)",
                           (k, response, provider, time.time(), settings.CACHE_TTL_HOURS))
            self.db.commit()
        except:
            pass


class AIRouter:
    """Routes questions to the best AI provider."""

    def __init__(self):
        self.complexity_analyzer = ComplexityAnalyzer()
        self.cache = ResponseCache()
        self.last_complexity = "simple"
        self.cloud_rotation_index = 0
        self.client = httpx.AsyncClient(timeout=30.0)

    # ─── Local LLM ─────────────────────────────────
    async def check_local(self) -> bool:
        try:
            resp = await self.client.get(f"{settings.OLLAMA_URL}/api/tags")
            return resp.status_code == 200
        except:
            return False

    async def query_local(self, message: str, history: list, system_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for h in history[-settings.CONTEXT_TURNS_LOCAL * 2:]:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": message})

        resp = await self.client.post(
            f"{settings.OLLAMA_URL}/api/chat",
            json={"model": settings.LOCAL_MODEL, "messages": messages, "stream": False,
                  "options": {"temperature": 0.7, "num_predict": 500}},
            timeout=settings.LOCAL_TIMEOUT
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _check_confidence(self, response: str) -> bool:
        """Check if local response seems confident. Low confidence → escalate to cloud."""
        if not settings.CONFIDENCE_GATE:
            return True
        uncertain = [
            r"i('m| am) not (sure|certain)",
            r"i don'?t (know|have|think)",
            r"(maybe|perhaps|possibly|might be)",
            r"(δεν (ξ[εέ]ρω|ε[ίι]μαι σ[ίι]γουρ))",
            r"this (is|might be) (beyond|outside)",
            r"you (should|might want to) (check|verify|ask)",
            r"i (cannot|can't) (help|assist|answer)",
        ]
        lower = response.lower()
        hits = sum(1 for p in uncertain if re.search(p, lower))
        return hits < 2

    # ─── Cloud Providers ──────────────────────────────
    def get_cloud_status(self) -> dict:
        return {
            "claude": bool(settings.ANTHROPIC_API_KEY),
            "gemini": bool(settings.GOOGLE_API_KEY),
            "chatgpt": bool(settings.OPENAI_API_KEY),
        }

    def _pick_cloud(self) -> Optional[str]:
        available = [k for k, v in self.get_cloud_status().items() if v]
        if not available:
            return None
        if settings.CLOUD_STRATEGY == "priority":
            for p in ["claude", "gemini", "chatgpt"]:
                if p in available:
                    return p
        elif settings.CLOUD_STRATEGY == "round_robin":
            provider = available[self.cloud_rotation_index % len(available)]
            self.cloud_rotation_index += 1
            return provider
        elif settings.CLOUD_STRATEGY in available:
            return settings.CLOUD_STRATEGY
        return available[0]

    async def query_claude(self, message: str, history: list, system_prompt: str) -> str:
        msgs = [{"role": h["role"], "content": h["content"]} for h in history[-settings.CONTEXT_TURNS_CLOUD * 2:]]
        msgs.append({"role": "user", "content": message})
        resp = await self.client.post("https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json", "x-api-key": settings.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01"},
            json={"model": settings.CLAUDE_MODEL, "max_tokens": 1500, "system": system_prompt, "messages": msgs},
            timeout=settings.CLOUD_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return "".join(c["text"] for c in data["content"] if c["type"] == "text")

    async def query_gemini(self, message: str, history: list, system_prompt: str) -> str:
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"System: {system_prompt}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        for h in history[-settings.CONTEXT_TURNS_CLOUD * 2:]:
            role = "user" if h["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": h["content"]}]})
        contents.append({"role": "user", "parts": [{"text": message}]})
        resp = await self.client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{settings.GEMINI_MODEL}:generateContent?key={settings.GOOGLE_API_KEY}",
            json={"contents": contents, "generationConfig": {"maxOutputTokens": 1500, "temperature": 0.7}},
            timeout=settings.CLOUD_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    async def query_chatgpt(self, message: str, history: list, system_prompt: str) -> str:
        msgs = [{"role": "system", "content": system_prompt}] if system_prompt else []
        for h in history[-settings.CONTEXT_TURNS_CLOUD * 2:]:
            msgs.append({"role": h["role"], "content": h["content"]})
        msgs.append({"role": "user", "content": message})
        resp = await self.client.post("https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": settings.CHATGPT_MODEL, "messages": msgs, "max_tokens": 1500, "temperature": 0.7},
            timeout=settings.CLOUD_TIMEOUT)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def _query_cloud(self, provider: str, message: str, history: list, system_prompt: str) -> str:
        if provider == "claude":
            return await self.query_claude(message, history, system_prompt)
        elif provider == "gemini":
            return await self.query_gemini(message, history, system_prompt)
        elif provider == "chatgpt":
            return await self.query_chatgpt(message, history, system_prompt)
        raise ValueError(f"Unknown provider: {provider}")

    # ─── Main Routing ──────────────────────────────────
    async def route_and_respond(self, message: str, history: list, system_prompt: str,
                                 force_provider: Optional[str] = None, context: Optional[str] = None) -> Tuple[str, str]:
        # Check cache
        cached = self.cache.get(message, system_prompt)
        if cached:
            return cached["response"], cached["provider"]

        # Forced provider
        if force_provider:
            if force_provider == "local":
                reply = await self.query_local(message, history, system_prompt)
                self.cache.set(message, system_prompt, reply, "local")
                return reply, "local"
            else:
                reply = await self._query_cloud(force_provider, message, history, system_prompt)
                self.cache.set(message, system_prompt, reply, force_provider)
                return reply, force_provider

        # Analyze complexity
        complexity, confidence = self.complexity_analyzer.analyze(message, context)
        self.last_complexity = complexity

        if complexity == "simple":
            # Try local first
            try:
                local_ok = await self.check_local()
                if local_ok:
                    reply = await self.query_local(message, history, system_prompt)
                    # Confidence gate: if response seems uncertain, escalate
                    if not self._check_confidence(reply):
                        cloud = self._pick_cloud()
                        if cloud:
                            try:
                                reply = await self._query_cloud(cloud, message, history, system_prompt)
                                self.cache.set(message, system_prompt, reply, cloud)
                                return reply, f"{cloud}(escalated)"
                            except:
                                pass
                    self.cache.set(message, system_prompt, reply, "local")
                    return reply, "local"
            except:
                pass

            # Local failed, try cloud
            cloud = self._pick_cloud()
            if cloud:
                try:
                    reply = await self._query_cloud(cloud, message, history, system_prompt)
                    self.cache.set(message, system_prompt, reply, f"{cloud}(fallback)")
                    return reply, f"{cloud}(fallback)"
                except:
                    pass
            return "I'm having trouble thinking right now. Try again in a moment!", "error"

        else:
            # Complex → cloud first
            cloud = self._pick_cloud()
            if cloud:
                try:
                    reply = await self._query_cloud(cloud, message, history, system_prompt)
                    self.cache.set(message, system_prompt, reply, cloud)
                    return reply, cloud
                except:
                    pass

            # Cloud failed, try local
            try:
                reply = await self.query_local(message, history, system_prompt)
                self.cache.set(message, system_prompt, reply, "local(fallback)")
                return reply, "local(fallback)"
            except:
                pass
            return "All my brain cells are busy! Try again soon.", "error"
