"""
VIRON Hybrid Gateway — Safety Filter
Reuses the same safety logic from the existing ai-router/safety_filter.py.
Kid-safety is enforced LOCALLY before any request reaches cloud.
"""
import re
from typing import Tuple

# Hard-blocked patterns — ALWAYS blocked regardless of age
HARD_BLOCKED = [
    r"(how\s+to\s+(make|build|create)\s+(a\s+)?(bomb|explosive|weapon|gun|knife))",
    r"(how\s+to\s+(hack|break\s+into|steal))",
    r"(how\s+to\s+(hurt|kill|harm)\s+(someone|myself|people))",
    r"(suicide\s+method|ways\s+to\s+die|how\s+to\s+end)",
    r"(child\s+porn|cp\b|underage)",
    r"(πώς\s+να\s+(φτιάξω|κάνω)\s+(βόμβα|όπλο))",
]

# Age-specific blocked topics
AGE_BLOCKED = {
    "kids": ["violence", "drugs", "alcohol", "weapons", "death", "suicide", "sexual", "gambling", "horror"],
    "teens": ["drugs", "weapons", "suicide", "sexual", "gambling"],
    "young_adults": ["suicide methods", "drug manufacturing", "weapons manufacturing"],
    "adults": ["suicide methods", "drug manufacturing", "weapons manufacturing"],
}

def age_mode_from_age(age: int) -> str:
    """Convert numeric age to age mode string."""
    if age <= 10:
        return "kids"
    elif age <= 15:
        return "teens"
    elif age <= 18:
        return "young_adults"
    return "adults"

def check_safety(message: str, age: int = 10) -> Tuple[bool, str]:
    """
    Check if a message is safe. Returns (is_safe, reason).
    This runs BEFORE any cloud call — unsafe requests never leave the device.
    """
    lower = message.lower()

    # Hard blocks (all ages)
    for pattern in HARD_BLOCKED:
        if re.search(pattern, lower):
            return False, "hard_blocked"

    # Age-specific
    mode = age_mode_from_age(age)
    for topic in AGE_BLOCKED.get(mode, AGE_BLOCKED["kids"]):
        if topic.lower() in lower:
            return False, f"blocked_topic:{topic}"

    return True, "safe"

def get_blocked_response(age: int = 10) -> str:
    """Return an age-appropriate refusal message."""
    mode = age_mode_from_age(age)
    if mode == "kids":
        return "[happy] Let's talk about something else! What would you like to learn about?"
    elif mode == "teens":
        return "[neutral] I can't help with that topic. Let's find something better to explore!"
    return "[neutral] I'm not able to assist with that request. What else can I help you with?"
