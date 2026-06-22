"""
Map STT transcript / language hint to TTS prompt language + default speaker.
"""
from __future__ import annotations

import re
from typing import Optional

# language key -> (default_speaker_id, speed_multiplier)
VOICE_DEFAULTS: dict[str, tuple[str, float]] = {
    "hindi": ("159", 1.05),
    "hi": ("159", 1.05),
    "tamil": ("188", 1.10),
    "ta": ("188", 1.10),
    "bengali": ("125", 1.10),
    "bn": ("125", 1.10),
    "malayalam": ("189", 1.10),
    "ml": ("189", 1.10),
    "kannada": ("142", 1.05),
    "kn": ("142", 1.05),
    "telugu": ("69", 1.10),
    "te": ("69", 1.10),
    "punjabi": ("191", 1.08),
    "pa": ("191", 1.08),
    "gujarati": ("62", 1.15),
    "gu": ("62", 1.15),
    "marathi": ("205", 1.05),
    "mr": ("205", 1.05),
    "english": ("159", 1.0),
    "en": ("159", 1.0),
}

# Canonical language names for TTS prompts (lowercase, no spaces)
CANONICAL: dict[str, str] = {
    "hi": "hindi",
    "ta": "tamil",
    "bn": "bengali",
    "ml": "malayalam",
    "kn": "kannada",
    "te": "telugu",
    "pa": "punjabi",
    "gu": "gujarati",
    "mr": "marathi",
    "en": "english",
    "eng": "english",
    "english": "english",
}

SCRIPT_RANGES: list[tuple[str, str]] = [
    ("hindi", r"[\u0900-\u097F]"),
    ("bengali", r"[\u0980-\u09FF]"),
    ("gujarati", r"[\u0A80-\u0AFF]"),
    ("punjabi", r"[\u0A00-\u0A7F]"),
    ("tamil", r"[\u0B80-\u0BFF]"),
    ("telugu", r"[\u0C00-\u0C7F]"),
    ("kannada", r"[\u0C80-\u0CFF]"),
    ("malayalam", r"[\u0D00-\u0D7F]"),
    ("marathi", r"[\u0900-\u097F]"),  # Devanagari — prefer hindi default speaker
]


def _normalize_lang(code: str) -> str:
    key = code.strip().lower().replace("_", "-").split("-")[0]
    return CANONICAL.get(key, key)


def detect_language_from_latin(text: str) -> Optional[str]:
    """Treat mostly-Latin transcripts as English (e.g. 'Hi', 'Hello there')."""
    stripped = text.strip()
    if not stripped:
        return None
    latin = len(re.findall(r"[A-Za-z]", stripped))
    word_chars = len(re.findall(r"\w", stripped, flags=re.UNICODE))
    if latin >= 2 and latin >= 0.5 * max(word_chars, 1):
        return "english"
    return None


def detect_language_from_script(text: str) -> Optional[str]:
    scores: dict[str, int] = {}
    for lang, pattern in SCRIPT_RANGES:
        count = len(re.findall(pattern, text))
        if count:
            scores[lang] = scores.get(lang, 0) + count
    if not scores:
        return None
    # Marathi uses Devanagari — if tied with hindi, keep hindi as safer default
    best = max(scores, key=scores.get)
    if best == "marathi" and "hindi" in scores:
        return "hindi"
    return best


def resolve_voice(
    transcript: str,
    stt_language: Optional[str] = None,
    *,
    default_language: str = "hindi",
    default_speaker: str = "159",
) -> tuple[str, str, float]:
    """
    Return (language, speaker_id, speed) for TTS prompt building.

    Priority: STT language hint → script detection → Latin/English → defaults.
    """
    lang: Optional[str] = None

    if stt_language:
        lang = _normalize_lang(stt_language)

    if not lang or lang not in VOICE_DEFAULTS:
        lang = detect_language_from_script(transcript) or lang

    if not lang or lang not in VOICE_DEFAULTS:
        lang = detect_language_from_latin(transcript) or lang

    if not lang or lang not in VOICE_DEFAULTS:
        lang = _normalize_lang(default_language)

    speaker, speed = VOICE_DEFAULTS.get(lang, (default_speaker, 1.05))
    return lang, speaker, speed


def reply_language_label(language: str) -> str:
    """Human-readable language name for Gemini system instruction."""
    labels = {
        "hindi": "Hindi",
        "tamil": "Tamil",
        "bengali": "Bengali",
        "malayalam": "Malayalam",
        "kannada": "Kannada",
        "telugu": "Telugu",
        "punjabi": "Punjabi",
        "gujarati": "Gujarati",
        "marathi": "Marathi",
        "english": "English",
    }
    return labels.get(language.lower(), language.title())
