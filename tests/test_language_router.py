"""Unit tests for language routing."""

from app.language_router import resolve_voice


def test_english_hi_transcript():
    lang, speaker, speed = resolve_voice("Hi", None)
    assert lang == "english"
    assert speaker == "159"
    assert speed == 1.0


def test_english_stt_code():
    lang, _, _ = resolve_voice("Hello", "en")
    assert lang == "english"


def test_hindi_devanagari():
    lang, speaker, _ = resolve_voice("नमस्ते", None)
    assert lang == "hindi"
    assert speaker == "159"


def test_default_fallback():
    lang, _, _ = resolve_voice("", None, default_language="hindi")
    assert lang == "hindi"
