"""TTS backend protocol — shared by ElevenLabs and custom sidecar streamers."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Protocol


class TTSStreamer(Protocol):
    request_count: int

    async def stream_text(
        self,
        text: str,
        cancel_event: asyncio.Event,
        *,
        language: str = "hindi",
        speaker_id: str = "159",
        speed: float = 1.05,
    ) -> AsyncIterator[bytes]: ...

    async def stream(
        self,
        segments: AsyncIterator[str],
        cancel_event: asyncio.Event,
        *,
        language: str = "hindi",
        speaker_id: str = "159",
        speed: float = 1.05,
    ) -> AsyncIterator[bytes]: ...
