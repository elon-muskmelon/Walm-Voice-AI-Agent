"""
HTTP client for the GPU TTS sidecar.

Streams Int16 PCM chunks from POST /v1/tts/stream.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import httpx

from app.config import settings
from app.debug_probe import probe

logger = logging.getLogger("custom_tts")


class CustomTTSStreamer:
    def __init__(self) -> None:
        self._base_url = settings.tts_service_url.rstrip("/")
        self._timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
        self.request_count = 0

    async def stream_text(
        self,
        text: str,
        cancel_event: asyncio.Event,
        *,
        language: str = "hindi",
        speaker_id: str = "159",
        speed: float = 1.05,
    ) -> AsyncIterator[bytes]:
        """One POST /v1/tts/stream for the full *text*; yield PCM as it arrives."""
        _ = speed  # custom sidecar does not apply speed client-side
        text = text.strip()
        if not text:
            return

        self.request_count += 1
        payload = {
            "text": text,
            "language": language,
            "speaker_id": speaker_id,
        }
        url = f"{self._base_url}/v1/tts/stream"
        # region agent log
        probe(
            "B",
            "app/custom_tts_streamer.py:request_start",
            "POST /v1/tts/stream",
            {"chars": len(text)},
        )
        # endregion

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        logger.error(
                            "TTS sidecar error status=%s body=%s",
                            response.status_code,
                            body[:200],
                        )
                        return
                    first = True
                    async for chunk in response.aiter_bytes():
                        if cancel_event.is_set():
                            await response.aclose()
                            return
                        if chunk:
                            if first:
                                first = False
                                # region agent log
                                probe(
                                    "B",
                                    "app/custom_tts_streamer.py:first_chunk_in",
                                    "First sidecar HTTP chunk",
                                    {"bytes": len(chunk)},
                                )
                                # endregion
                            yield chunk
            except httpx.HTTPError as exc:
                logger.exception("TTS sidecar request failed: %s", exc)

    async def stream(
        self,
        segments: AsyncIterator[str],
        cancel_event: asyncio.Event,
        *,
        language: str = "hindi",
        speaker_id: str = "159",
        speed: float = 1.05,
    ) -> AsyncIterator[bytes]:
        """Segment mode: one POST per segment from *segments*."""
        async for segment in segments:
            if cancel_event.is_set():
                break
            async for chunk in self.stream_text(
                segment,
                cancel_event,
                language=language,
                speaker_id=speaker_id,
                speed=speed,
            ):
                yield chunk
            if cancel_event.is_set():
                break
            await asyncio.sleep(0)
