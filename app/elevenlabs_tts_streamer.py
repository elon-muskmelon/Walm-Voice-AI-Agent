"""
ElevenLabs WebSocket TTS streamer.

Opens one WebSocket connection per utterance, streams text in,
yields raw audio bytes out as they arrive.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import AsyncIterator

import websockets
import websockets.exceptions

from app.config import settings

logger = logging.getLogger("elevenlabs_tts")

_BASE_URL = "wss://api.elevenlabs.io/v1/text-to-speech"


class TTSError(RuntimeError):
    """Raised when ElevenLabs returns an error message."""


class ElevenLabsTTSStreamer:
    def __init__(self) -> None:
        self._api_key = settings.elevenlabs_api_key
        self._voice_id = settings.elevenlabs_voice_id
        self._model_id = settings.elevenlabs_model_id
        self._output_format = settings.elevenlabs_tts_output_format
        self._chunk_schedule = settings.elevenlabs_tts_chunk_schedule
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
        """One WebSocket session; send full *text* then drain audio."""
        _ = language, speaker_id, speed
        text = text.strip()
        if not text:
            return

        self.request_count += 1

        url = (
            f"{_BASE_URL}/{self._voice_id}/stream-input"
            f"?model_id={self._model_id}&output_format={self._output_format}"
        )
        headers = {"xi-api-key": self._api_key}

        async with websockets.connect(
            url,
            additional_headers=headers,
            max_queue=None,
        ) as ws:
            await ws.send(
                json.dumps(
                    {
                        "text": " ",
                        "generation_config": {
                            "chunk_length_schedule": self._chunk_schedule
                        },
                    }
                )
            )

            audio_q: asyncio.Queue[bytes | None | Exception] = asyncio.Queue()

            async def _reader() -> None:
                try:
                    async for raw in ws:
                        if isinstance(raw, bytes):
                            await audio_q.put(raw)
                            continue
                        data: dict = json.loads(raw)
                        if "error" in data or data.get("message_type") in (
                            "error",
                            "auth_error",
                            "quota_exceeded",
                        ):
                            await audio_q.put(
                                TTSError(data.get("error") or str(data))
                            )
                            return
                        audio_b64: str | None = data.get("audio")
                        if audio_b64:
                            await audio_q.put(base64.b64decode(audio_b64))
                        if data.get("isFinal"):
                            break
                except websockets.exceptions.ConnectionClosedOK:
                    pass
                except Exception as exc:
                    logger.exception("TTS reader error: %s", exc)
                    await audio_q.put(exc)
                finally:
                    await audio_q.put(None)

            reader_task = asyncio.create_task(_reader())

            try:
                if not cancel_event.is_set():
                    await ws.send(json.dumps({"text": text, "flush": True}))
                    await ws.send(
                        json.dumps({"text": "", "flush": True, "is_final": True})
                    )
            except Exception as exc:
                logger.exception("TTS sender error: %s", exc)
            finally:
                if cancel_event.is_set():
                    reader_task.cancel()
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    await audio_q.put(None)

            while True:
                item = await audio_q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    logger.error("TTS error from ElevenLabs: %s", item)
                    break
                yield item

            try:
                await reader_task
            except asyncio.CancelledError:
                pass

    async def stream(
        self,
        segments: AsyncIterator[str],
        cancel_event: asyncio.Event,
        *,
        language: str = "hindi",
        speaker_id: str = "159",
        speed: float = 1.05,
    ) -> AsyncIterator[bytes]:
        """Segment mode: one WS, multiple flushed text chunks."""
        url = (
            f"{_BASE_URL}/{self._voice_id}/stream-input"
            f"?model_id={self._model_id}&output_format={self._output_format}"
        )
        headers = {"xi-api-key": self._api_key}

        async with websockets.connect(
            url,
            additional_headers=headers,
            max_queue=None,
        ) as ws:
            await ws.send(
                json.dumps(
                    {
                        "text": " ",
                        "generation_config": {
                            "chunk_length_schedule": self._chunk_schedule
                        },
                    }
                )
            )

            audio_q: asyncio.Queue[bytes | None | Exception] = asyncio.Queue()

            async def _reader() -> None:
                try:
                    async for raw in ws:
                        if isinstance(raw, bytes):
                            await audio_q.put(raw)
                            continue
                        data: dict = json.loads(raw)
                        if "error" in data or data.get("message_type") in (
                            "error",
                            "auth_error",
                            "quota_exceeded",
                        ):
                            await audio_q.put(
                                TTSError(data.get("error") or str(data))
                            )
                            return
                        audio_b64: str | None = data.get("audio")
                        if audio_b64:
                            await audio_q.put(base64.b64decode(audio_b64))
                        if data.get("isFinal"):
                            break
                except websockets.exceptions.ConnectionClosedOK:
                    pass
                except Exception as exc:
                    logger.exception("TTS reader error: %s", exc)
                    await audio_q.put(exc)
                finally:
                    await audio_q.put(None)

            reader_task = asyncio.create_task(_reader())

            try:
                async for segment in segments:
                    if cancel_event.is_set():
                        break
                    text = segment.strip()
                    if text:
                        self.request_count += 1
                        await ws.send(json.dumps({"text": text, "flush": True}))

                if not cancel_event.is_set():
                    await ws.send(
                        json.dumps({"text": "", "flush": True, "is_final": True})
                    )
            except Exception as exc:
                logger.exception("TTS sender error: %s", exc)
            finally:
                if cancel_event.is_set():
                    reader_task.cancel()
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    await audio_q.put(None)

            while True:
                item = await audio_q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    logger.error("TTS error from ElevenLabs: %s", item)
                    break
                yield item

            try:
                await reader_task
            except asyncio.CancelledError:
                pass
