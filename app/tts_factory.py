"""Factory for TTS backend implementations."""
from __future__ import annotations

from app.config import settings
from app.custom_tts_streamer import CustomTTSStreamer
from app.elevenlabs_tts_streamer import ElevenLabsTTSStreamer
from app.tts_base import TTSStreamer


def create_tts_streamer() -> TTSStreamer:
    if settings.tts_backend == "custom":
        return CustomTTSStreamer()
    return ElevenLabsTTSStreamer()
