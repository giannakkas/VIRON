"""VIRON AI Router — Configuration"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ─── Local LLM ─────────────────────────────────────
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "phi3")
LOCAL_TIMEOUT = int(os.getenv("LOCAL_TIMEOUT", "5"))  # 5s for local Ollama

# ─── Cloud API Keys ────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", ""))

# ─── Cloud Models ──────────────────────────────────
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")  # Haiku for speed
CHATGPT_MODEL = os.getenv("CHATGPT_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ─── Routing ───────────────────────────────────────
CLOUD_STRATEGY = os.getenv("CLOUD_STRATEGY", "priority")  # priority, round_robin, claude, gemini, chatgpt
CLOUD_TIMEOUT = int(os.getenv("CLOUD_TIMEOUT", "25"))  # 25s for Sonnet
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))

# ─── Confidence Gate ───────────────────────────────
CONFIDENCE_GATE = os.getenv("CONFIDENCE_GATE", "true").lower() == "true"

# ─── Cache ─────────────────────────────────────────
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_DB_PATH = os.getenv("CACHE_DB_PATH", str(Path(__file__).parent / "viron_cache.db"))
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "72"))

# ─── Conversation ──────────────────────────────────
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "20"))
CONTEXT_TURNS_LOCAL = int(os.getenv("CONTEXT_TURNS_LOCAL", "6"))
CONTEXT_TURNS_CLOUD = int(os.getenv("CONTEXT_TURNS_CLOUD", "10"))

# ─── Age Mode ──────────────────────────────────────
DEFAULT_AGE_MODE = os.getenv("DEFAULT_AGE_MODE", "kids")

# ─── Safety ────────────────────────────────────────
SAFETY_FILTER = os.getenv("SAFETY_FILTER", "true").lower() == "true"

# ─── Server ────────────────────────────────────────
API_KEY = os.getenv("VIRON_API_KEY", "change-me-to-a-secret-key")
PORT = int(os.getenv("PORT", "8000"))
