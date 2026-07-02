"""
HTTP client for the GPU TTS sidecar.

Streams Int16 PCM chunks from POST /v1/tts/stream.
Reuses one httpx client across turns to avoid TLS handshake on every request.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator
from urllib.parse import urlparse

import httpx

from app.config import settings
from app.debug_probe import probe

logger = logging.getLogger("custom_tts")


class CustomTTSStreamer:
    def __init__(self) -> None:
        self._timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
        self._client: httpx.AsyncClient | None = None
        self.request_count = 0

    def _base_url(self) -> str:
        return settings.tts_service_url.rstrip("/")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

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
        base = self._base_url()
        url = f"{base}/v1/tts/stream"
        req_start = time.monotonic()
        host = urlparse(base).netloc
        # region agent log
        probe(
            "C",
            "app/custom_tts_streamer.py:request_start",
            "POST /v1/tts/stream",
            {"chars": len(text), "host": host, "url": base},
        )
        # endregion

        client = await self._get_client()
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
                            ttfb_ms = (time.monotonic() - req_start) * 1000.0
                            # region agent log
                            probe(
                                "B",
                                "app/custom_tts_streamer.py:first_chunk_in",
                                "First sidecar HTTP chunk",
                                {
                                    "bytes": len(chunk),
                                    "ttfb_ms": round(ttfb_ms, 1),
                                    "host": host,
                                },
                            )
                            # endregion
                        yield chunk
        except httpx.HTTPError as exc:
            logger.exception("TTS sidecar request failed: %s", exc)
            await self.aclose()

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
