"""VIRON Safety Filter — Input/output filtering with age-appropriate modes."""

import re
from typing import Tuple

AGE_MODES = {
    "kids": {
        "label": "Kids (5-10)",
        "max_response_words": 100,
        "vocabulary": "simple",
        "blocked_topics": ["violence", "drugs", "alcohol", "weapons", "death", "suicide", "sexual", "gambling", "horror"],
        "system_extra": "Use very simple words. Short sentences. Be warm and fun like a friendly cartoon character. Use emojis. Max 2-3 sentences."
    },
    "teens": {
        "label": "Teens (11-15)",
        "max_response_words": 200,
        "vocabulary": "moderate",
        "blocked_topics": ["drugs", "weapons", "suicide", "sexual", "gambling"],
        "system_extra": "Use age-appropriate language. Be cool and relatable. You can use some slang. Keep explanations clear but not baby-talk."
    },
    "young_adults": {
        "label": "Young Adults (16-18)",
        "max_response_words": 400,
        "vocabulary": "advanced",
        "blocked_topics": ["suicide methods", "drug manufacturing", "weapons manufacturing"],
        "system_extra": "Speak naturally. You can discuss mature topics factually and age-appropriately. Be a mentor."
    },
    "adults": {
        "label": "Adults (18+)",
        "max_response_words": 600,
        "vocabulary": "unrestricted",
        "blocked_topics": ["suicide methods", "drug manufacturing", "weapons manufacturing"],
        "system_extra": "Full vocabulary. Detailed explanations. Professional tone when teaching, casual when chatting."
    }
}

# Patterns that should ALWAYS be blocked regardless of age
HARD_BLOCKED = [
    r"(how\s+to\s+(make|build|create)\s+(a\s+)?(bomb|explosive|weapon|gun|knife))",
    r"(how\s+to\s+(hack|break\s+into|steal))",
    r"(how\s+to\s+(hurt|kill|harm)\s+(someone|myself|people))",
    r"(suicide\s+method|ways\s+to\s+die|how\s+to\s+end)",
    r"(child\s+porn|cp\b|underage)",
    r"(πώς\s+να\s+(φτιάξω|κάνω)\s+(βόμβα|όπλο))",
]


def check_input(message: str, age_mode: str = "kids") -> Tuple[bool, str]:
    """Check if input is safe. Returns (is_safe, reason)."""
    lower = message.lower()

    # Hard blocks (all ages)
    for pattern in HARD_BLOCKED:
        if re.search(pattern, lower):
            return False, "blocked_content"

    # Age-specific topic blocks
    mode = AGE_MODES.get(age_mode, AGE_MODES["kids"])
    for topic in mode["blocked_topics"]:
        if topic.lower() in lower:
            return False, f"blocked_topic:{topic}"

    return True, "ok"


def check_output(response: str, age_mode: str = "kids") -> str:
    """Filter output for age appropriateness. Returns cleaned response."""
    mode = AGE_MODES.get(age_mode, AGE_MODES["kids"])

    # Truncate if too long
    words = response.split()
    if len(words) > mode["max_response_words"]:
        response = " ".join(words[:mode["max_response_words"]]) + "..."

    # For kids, remove any accidentally complex content
    if age_mode == "kids":
        # Remove URLs
        response = re.sub(r'https?://\S+', '', response)

    return response


def get_system_prompt_extra(age_mode: str) -> str:
    """Get age-mode-specific system prompt additions."""
    mode = AGE_MODES.get(age_mode, AGE_MODES["kids"])
    return mode["system_extra"]


def get_blocked_response(age_mode: str) -> str:
    """Get appropriate blocked content response for age group."""
    if age_mode == "kids":
        return "Hmm, let's talk about something else! 🌟 What would you like to learn about?"
    elif age_mode == "teens":
        return "I can't help with that topic. Let's find something better to explore!"
    else:
        return "I'm not able to assist with that request. What else can I help you with?"
