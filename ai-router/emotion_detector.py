"""VIRON Emotion Detector — Maps text to face emotions."""

import re
from typing import Tuple


EMOTION_MAP = {
    # Teaching
    "encouraging": ("hopeful", 0.7),
    "explaining": ("focused", 0.6),
    "correcting": ("thinking", 0.5),
    "celebrating": ("excited", 0.9),
    "praising": ("proud", 0.8),
    # Emotional
    "empathetic": ("worried", 0.7),
    "comforting": ("love", 0.6),
    "playful": ("cheeky", 0.7),
    "joking": ("laughing", 0.8),
    "surprised": ("amazed", 0.8),
    # Conversational
    "greeting": ("excited", 0.7),
    "curious": ("thinking", 0.6),
    "neutral": ("neutral", 0.5),
    "confused": ("confused", 0.6),
}

def detect_emotion(text: str) -> Tuple[str, float]:
    """Detect emotion from response text. Returns (viron_emotion, intensity)."""
    lower = text.lower()

    # Check for explicit emotion tags [emotion_name]
    m = re.match(r'^\[(\w+)\]', text)
    if m:
        emo = m.group(1)
        return (emo, 0.8)

    # Excitement / celebration
    if any(w in lower for w in ["great job", "amazing", "excellent", "μπράβο", "τέλεια", "well done", "correct!"]):
        return ("excited", 0.9)
    if any(w in lower for w in ["🎉", "🌟", "⭐", "💪"]):
        return ("proud", 0.8)

    # Empathy / comfort
    if any(w in lower for w in ["sorry to hear", "that's tough", "i understand", "λυπάμαι", "καταλαβαίνω"]):
        return ("worried", 0.7)

    # Humor
    if any(w in lower for w in ["haha", "lol", "😄", "funny", "joke", "αχαχα"]):
        return ("laughing", 0.8)

    # Question / curiosity
    if text.count("?") >= 2:
        return ("thinking", 0.6)

    # Teaching / explaining
    if any(w in lower for w in ["let me explain", "here's how", "the answer is", "να σου εξηγήσω"]):
        return ("focused", 0.7)

    # Music
    if any(w in lower for w in ["playing", "song", "music", "τραγούδι", "μουσική", "♪"]):
        return ("dreamy", 0.7)

    # Greeting
    if any(w in lower for w in ["hello", "hi!", "hey!", "γεια", "καλημέρα"]):
        return ("happy", 0.7)

    return ("neutral", 0.5)


def clean_for_speech(text: str) -> str:
    """Remove emotion tags and formatting for TTS."""
    text = re.sub(r'^\[\w+\]\s*', '', text)
    text = re.sub(r'\[YOUTUBE:[^\]]+\]', '', text)
    text = re.sub(r'\[WHITEBOARD:.*?\][\s\S]*?\[/WHITEBOARD\]', '', text)
    text = re.sub(r'[*_`#]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
