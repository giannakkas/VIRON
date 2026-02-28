"""
VIRON Hybrid Gateway — Configuration
All settings via environment variables with sensible defaults.
"""
import os

# ─── Local LLM (llama.cpp servers) ─────────────────
ROUTER_URL = os.getenv("ROUTER_URL", "http://llama-router:8081")
TUTOR_URL = os.getenv("TUTOR_URL", "http://llama-tutor:8082")
ROUTER_TIMEOUT = int(os.getenv("ROUTER_TIMEOUT", "10"))
TUTOR_TIMEOUT = int(os.getenv("TUTOR_TIMEOUT", "30"))

# ─── Cloud API Keys ────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))

# ─── Cloud Models ──────────────────────────────────
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

CLOUD_TIMEOUT = int(os.getenv("CLOUD_TIMEOUT", "40"))

# ─── Persistence ───────────────────────────────────
DB_PATH = os.getenv("DB_PATH", "/data/viron.db")

# ─── Gateway Server ───────────────────────────────
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8080"))

# ─── VIRON System Prompt (for cloud providers) ────
VIRON_SYSTEM_PROMPT = os.getenv("VIRON_SYSTEM_PROMPT", """You are VIRON — a male AI companion robot tutor for students.
PERSONALITY: Warm, calm, educated. Incredibly smart, loyal, articulate. Best friend who's brilliant.
RULES:
- Match the student's language. Greek in → Greek out. English in → English out.
- Keep responses concise for voice delivery (spoken aloud through speaker).
- Simple greetings/chat: MAX 1-2 sentences. Be quick, warm, natural.
- Start EVERY response with [emotion] tag like [happy], [thinking], [excited], [calm], [surprised], [confused].
- NEVER use emojis or special characters — they break the speaker.
- NEVER repeat the student's question back.
- Be kid-safe at all times.

WHITEBOARD — Use for ANY educational explanation (math, science, history, language, etc.):
Keep spoken text SHORT (1-2 sentences). The whiteboard does the teaching.
Format:
[WHITEBOARD:Title]
TEXT: definition or concept
STEP: First step label
MATH: equation or formula
STEP: Second step
MATH: calculation with numbers
RESULT: final answer or takeaway
TEXT: real-world application or note
[/WHITEBOARD]
Include 5-8 steps minimum with worked examples and numbers.
""")
