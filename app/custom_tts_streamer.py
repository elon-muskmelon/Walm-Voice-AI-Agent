"""
HTTP client for the GPU TTS sidecar.

Streams Int16 PCM chunks from POST /v1/tts/stream for each text segment.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import httpx

from app.config import settings

logger = logging.getLogger("custom_tts")


class CustomTTSStreamer:
    def __init__(self) -> None:
        self._base_url = settings.tts_service_url.rstrip("/")
        self._timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)

    async def stream(
        self,
        segments: AsyncIterator[str],
        cancel_event: asyncio.Event,
        *,
        language: str = "hindi",
        speaker_id: str = "159",
        speed: float = 1.05,
    ) -> AsyncIterator[bytes]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async for segment in segments:
                if cancel_event.is_set():
                    break
                text = segment.strip()
                if not text:
                    continue

                payload = {
                    "text": text,
                    "language": language,
                    "speaker_id": speaker_id,
                }
                url = f"{self._base_url}/v1/tts/stream"

                try:
                    async with client.stream("POST", url, json=payload) as response:
                        if response.status_code != 200:
                            body = await response.aread()
                            logger.error(
                                "TTS sidecar error status=%s body=%s",
                                response.status_code,
                                body[:200],
                            )
                            break
                        async for chunk in response.aiter_bytes():
                            if cancel_event.is_set():
                                await response.aclose()
                                return
                            if chunk:
                                yield chunk
                except httpx.HTTPError as exc:
                    logger.exception("TTS sidecar request failed: %s", exc)
                    break

                if cancel_event.is_set():
                    break

                await asyncio.sleep(0)
