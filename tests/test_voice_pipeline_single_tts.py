"""Voice pipeline single-shot TTS tests."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock

import pytest

# Avoid google-genai import when collecting tests outside the project venv.
_mock_genai = MagicMock()
sys.modules.setdefault("google", MagicMock(genai=_mock_genai))
sys.modules.setdefault("google.genai", _mock_genai)

from app.config import settings
from app.voice_pipeline import VoicePipeline


class _MockGemini:
    async def stream(self, user_text: str, reply_language: str | None = None):
        _ = user_text, reply_language
        yield "Hello! "
        yield "How can I help?"


class _MockTTS:
    def __init__(self) -> None:
        self.request_count = 0
        self.stream_text_calls: list[str] = []

    async def stream_text(
        self,
        text: str,
        cancel_event: asyncio.Event,
        *,
        language: str = "hindi",
        speaker_id: str = "159",
        speed: float = 1.05,
    ):
        _ = cancel_event, language, speaker_id, speed
        self.request_count += 1
        self.stream_text_calls.append(text)
        yield b"\x00\x01" * 100

    async def stream(self, segments, cancel_event, **kwargs):
        async for segment in segments:
            async for chunk in self.stream_text(segment, cancel_event, **kwargs):
                yield chunk


@pytest.mark.asyncio
async def test_single_mode_issues_one_tts_request(monkeypatch):
    monkeypatch.setattr(settings, "tts_mode", "single")
    monkeypatch.setattr(settings, "tts_latency_profile", "fast_gpu")

    tts = _MockTTS()
    pipeline = VoicePipeline(_MockGemini(), tts)

    sent_json: list[dict] = []
    sent_bytes: list[bytes] = []

    async def send_json(payload: dict) -> None:
        sent_json.append(payload)

    async def send_bytes(data: bytes) -> None:
        sent_bytes.append(data)

    await pipeline.run(
        "Hi",
        send_json,
        send_bytes,
        asyncio.Event(),
        language="english",
    )

    assert tts.request_count == 1
    assert tts.stream_text_calls == ["Hello! How can I help?"]
    assert any(m.get("event") == "start" for m in sent_json if m.get("type") == "audio")
    assert any(m.get("event") == "end" for m in sent_json if m.get("type") == "audio")
    assert len(sent_bytes) >= 1


@pytest.mark.asyncio
async def test_balanced_profile_buffers_on_client(monkeypatch):
    monkeypatch.setattr(settings, "tts_mode", "single")
    monkeypatch.setattr(settings, "tts_latency_profile", "balanced")

    tts = _MockTTS()
    pipeline = VoicePipeline(_MockGemini(), tts)
    audio_events: list[dict] = []

    async def send_json(payload: dict) -> None:
        if payload.get("type") == "audio":
            audio_events.append(payload)

    async def send_bytes(_data: bytes) -> None:
        pass

    await pipeline.run("Hi", send_json, send_bytes, asyncio.Event())

    starts = [e for e in audio_events if e.get("event") == "start"]
    assert len(starts) == 1
    assert starts[0].get("buffer_until_end") is True
    assert settings.tts_buffer_before_play is False


@pytest.mark.asyncio
async def test_single_mode_buffer_profile_waits_for_all_pcm(monkeypatch):
    monkeypatch.setattr(settings, "tts_mode", "single")
    monkeypatch.setattr(settings, "tts_latency_profile", "slow_gpu")

    tts = _MockTTS()
    pipeline = VoicePipeline(_MockGemini(), tts)
    audio_events: list[dict] = []

    async def send_json(payload: dict) -> None:
        if payload.get("type") == "audio":
            audio_events.append(payload)

    async def send_bytes(_data: bytes) -> None:
        pass

    await pipeline.run("Hi", send_json, send_bytes, asyncio.Event())

    starts = [e for e in audio_events if e.get("event") == "start"]
    assert len(starts) == 1
    assert starts[0].get("buffer_until_end") is True
